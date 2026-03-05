# Node.js Backend Bridge — Express.js Controller Integration

> **Scope**: This document provides the Express.js controller patterns that
> the **travion-backend** (Node.js / TypeScript) service uses to communicate
> with the **ai-agent-engine** (FastAPI / Python) service and relay structured
> responses to the **travion-mobile** (React Native) app.
>
> The ai-engine service is the **single source of truth** for search, weather
> interrupts, and itinerary generation. The Node.js backend acts as a thin
> orchestration layer — proxying, caching `thread_id`, and routing HITL states.

---

## Architecture Overview

```
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────────┐
│  React Native    │──────▶│  Express.js       │──────▶│  ai-agent-engine     │
│  (travion-mobile)│◀──────│  (travion-backend)│◀──────│  (FastAPI + LangGraph)│
└──────────────────┘  WS   └──────────────────┘  HTTP  └──────────────────────┘
     ▲                            │
     │                      MongoDB (thread_id,
     │                       session state)
     └── push notifications ──┘
```

---

## 1. Environment Variables

```env
# .env in travion-backend
AI_ENGINE_BASE_URL=http://localhost:8000   # FastAPI ai-agent-engine
AI_ENGINE_TIMEOUT_MS=30000                 # max wait per call
```

---

## 2. Axios HTTP Client (shared utility)

```typescript
// src/shared/utils/aiEngineClient.ts
import axios from "axios";

export const aiEngine = axios.create({
  baseURL: process.env.AI_ENGINE_BASE_URL || "http://localhost:8000",
  timeout: Number(process.env.AI_ENGINE_TIMEOUT_MS) || 30000,
  headers: { "Content-Type": "application/json" },
});
```

---

## 3. Tour Plan Controller — Full Lifecycle

### 3.1 `POST /api/tour-plan/generate`

Creates a new tour plan. The ai-engine may respond with one of:

- **Normal response** → complete itinerary
- **`SELECTION_REQUIRED`** → graph paused, selection cards returned
- **`USER_PROMPT_REQUIRED`** → graph paused, weather interrupt returned

```typescript
// src/modules/tour-plan/tour-plan.controller.ts
import { Request, Response } from "express";
import { aiEngine } from "../../shared/utils/aiEngineClient";
import { TourSession } from "./tour-session.model"; // Mongoose model

export async function generateTourPlan(req: Request, res: Response) {
  try {
    const { query, locations, startDate, endDate, preferences } = req.body;
    const userId = req.user._id; // from auth middleware

    // 1. Call ai-engine
    const { data } = await aiEngine.post("/api/tour-plan/generate", {
      query,
      target_location: locations?.[0]?.name,
      tour_plan_context: {
        selected_locations: locations,
        start_date: startDate,
        end_date: endDate,
      },
      user_preferences: preferences,
    });

    // 2. Persist thread_id for later resume calls
    await TourSession.findOneAndUpdate(
      { userId },
      {
        threadId: data.thread_id,
        status: data.pending_user_selection
          ? "SELECTION_REQUIRED"
          : data.weather_interrupt
            ? "WEATHER_INTERRUPT"
            : "COMPLETE",
        selectionCards: data.selection_cards || null,
        weatherPrompt: data.weather_interrupt
          ? {
              message: data.weather_prompt_message,
              options: data.weather_prompt_options,
            }
          : null,
        updatedAt: new Date(),
      },
      { upsert: true, new: true },
    );

    // 3. Route response to mobile
    if (data.pending_user_selection) {
      return res.status(200).json({
        status: "SELECTION_REQUIRED",
        threadId: data.thread_id,
        message: data.response,
        selectionCards: data.selection_cards,
        searchMetadata: data.mcp_search_metadata,
      });
    }

    if (data.weather_interrupt) {
      return res.status(200).json({
        status: "WEATHER_INTERRUPT",
        threadId: data.thread_id,
        weatherPromptMessage: data.weather_prompt_message,
        weatherPromptOptions: data.weather_prompt_options,
      });
    }

    // Normal complete response
    return res.status(200).json({
      status: "COMPLETE",
      threadId: data.thread_id,
      response: data.response,
      itinerary: data.itinerary,
      finalItinerary: data.final_itinerary,
      metadata: data.metadata,
      constraints: data.constraints,
      culturalTips: data.cultural_tips,
      warnings: data.warnings,
    });
  } catch (err: any) {
    console.error("Tour plan generation failed:", err.message);
    return res.status(500).json({ error: "Tour plan generation failed" });
  }
}
```

### 3.2 `POST /api/tour-plan/select-candidate`

Resumes the graph after the user selects a candidate from `selectionCards`.

```typescript
export async function selectCandidate(req: Request, res: Response) {
  try {
    const { threadId, candidateId } = req.body;
    const userId = req.user._id;

    // Validate session
    const session = await TourSession.findOne({ userId, threadId });
    if (!session || session.status !== "SELECTION_REQUIRED") {
      return res
        .status(400)
        .json({ error: "No pending selection for this session" });
    }

    // Resume ai-engine graph
    const { data } = await aiEngine.post("/api/tour-plan/resume-selection", {
      thread_id: threadId,
      selected_candidate_id: candidateId,
    });

    // Check if weather interrupt occurs after selection
    if (data.weather_interrupt) {
      await TourSession.updateOne(
        { userId, threadId },
        {
          status: "WEATHER_INTERRUPT",
          weatherPrompt: {
            message: data.weather_prompt_message,
            options: data.weather_prompt_options,
          },
        },
      );
      return res.status(200).json({
        status: "WEATHER_INTERRUPT",
        threadId,
        weatherPromptMessage: data.weather_prompt_message,
        weatherPromptOptions: data.weather_prompt_options,
      });
    }

    // Update session
    await TourSession.updateOne(
      { userId, threadId },
      {
        status: "COMPLETE",
        selectionCards: null,
      },
    );

    return res.status(200).json({
      status: "COMPLETE",
      threadId,
      response: data.response,
      mapReadyItinerary: data.map_ready_itinerary,
      stepResults: data.step_results,
    });
  } catch (err: any) {
    console.error("Candidate selection failed:", err.message);
    return res.status(500).json({ error: "Candidate selection failed" });
  }
}
```

### 3.3 `POST /api/tour-plan/resume-weather`

Resumes the graph after the user responds to a weather interrupt.

```typescript
export async function resumeWeather(req: Request, res: Response) {
  try {
    const { threadId, userChoice } = req.body;
    // userChoice is one of: "switch_indoor", "reschedule", "keep"
    const userId = req.user._id;

    const session = await TourSession.findOne({ userId, threadId });
    if (!session || session.status !== "WEATHER_INTERRUPT") {
      return res
        .status(400)
        .json({ error: "No pending weather prompt for this session" });
    }

    const { data } = await aiEngine.post("/api/tour-plan/resume-weather", {
      thread_id: threadId,
      user_weather_choice: userChoice,
    });

    // After weather resume, check for selection interrupt
    if (data.pending_user_selection) {
      await TourSession.updateOne(
        { userId, threadId },
        {
          status: "SELECTION_REQUIRED",
          selectionCards: data.selection_cards,
          weatherPrompt: null,
        },
      );
      return res.status(200).json({
        status: "SELECTION_REQUIRED",
        threadId,
        selectionCards: data.selection_cards,
      });
    }

    await TourSession.updateOne(
      { userId, threadId },
      { status: "COMPLETE", weatherPrompt: null },
    );

    return res.status(200).json({
      status: "COMPLETE",
      threadId,
      response: data.response,
      itinerary: data.itinerary,
      finalItinerary: data.final_itinerary,
    });
  } catch (err: any) {
    console.error("Weather resume failed:", err.message);
    return res.status(500).json({ error: "Weather resume failed" });
  }
}
```

---

## 4. Mongoose Session Model

```typescript
// src/modules/tour-plan/tour-session.model.ts
import { Schema, model, Document } from "mongoose";

interface ITourSession extends Document {
  userId: Schema.Types.ObjectId;
  threadId: string;
  status: "SELECTION_REQUIRED" | "WEATHER_INTERRUPT" | "COMPLETE";
  selectionCards: any[] | null;
  weatherPrompt: {
    message: string;
    options: { id: string; label: string; description: string }[];
  } | null;
  updatedAt: Date;
}

const tourSessionSchema = new Schema<ITourSession>({
  userId: {
    type: Schema.Types.ObjectId,
    ref: "User",
    required: true,
    index: true,
  },
  threadId: { type: String, required: true, unique: true },
  status: {
    type: String,
    enum: ["SELECTION_REQUIRED", "WEATHER_INTERRUPT", "COMPLETE"],
    default: "COMPLETE",
  },
  selectionCards: { type: Schema.Types.Mixed, default: null },
  weatherPrompt: { type: Schema.Types.Mixed, default: null },
  updatedAt: { type: Date, default: Date.now },
});

// TTL: auto-delete sessions after 24 hours
tourSessionSchema.index({ updatedAt: 1 }, { expireAfterSeconds: 86400 });

export const TourSession = model<ITourSession>(
  "TourSession",
  tourSessionSchema,
);
```

---

## 5. Express Route Registration

```typescript
// src/modules/tour-plan/tour-plan.routes.ts
import { Router } from "express";
import { authMiddleware } from "../../shared/middleware/auth";
import {
  generateTourPlan,
  selectCandidate,
  resumeWeather,
} from "./tour-plan.controller";

const router = Router();

router.post("/generate", authMiddleware, generateTourPlan);
router.post("/select-candidate", authMiddleware, selectCandidate);
router.post("/resume-weather", authMiddleware, resumeWeather);

export default router;
```

---

## 6. React Native Integration Hints

### 6.1 Selection Card UI

When `status === 'SELECTION_REQUIRED'`, render a horizontal `FlatList` of cards:

```tsx
// Pseudocode — travion-mobile
{
  selectionCards.map((card) => (
    <SelectionCard
      key={card.card_id}
      title={card.name}
      badge={card.badge}
      imageUrl={card.image_url}
      photoUrls={card.photo_urls} // carousel
      price={card.price_range}
      rating={card.real_time_rating}
      vibeMatch={card.vibe_match_score}
      description={card.description}
      onSelect={() => handleSelect(card.card_id)}
    />
  ));
}
```

### 6.2 Weather Interrupt UI

When `status === 'WEATHER_INTERRUPT'`, show a bottom-sheet prompt:

```tsx
<WeatherPromptSheet
  message={weatherPromptMessage}
  options={weatherPromptOptions.map((opt) => ({
    id: opt.id,
    label: opt.label,
    description: opt.description,
  }))}
  onSelect={(choiceId) => resumeWeather(threadId, choiceId)}
/>
```

### 6.3 Visual Hierarchy in Itinerary

Each stop in `mapReadyItinerary.stops` now includes:

| Field              | Type          | Description                                                                         |
| ------------------ | ------------- | ----------------------------------------------------------------------------------- |
| `visual_hierarchy` | `1 \| 2 \| 3` | `1` = must-see (large card), `2` = recommended (standard), `3` = optional (compact) |
| `best_for_photos`  | `boolean`     | `true` → show "Best for Photos" badge                                               |
| `photo_urls`       | `string[]`    | Image carousel for the stop                                                         |

### 6.4 Route Geometry for Maps

`mapReadyItinerary.route_geometry` is a GeoJSON `Feature` array ready for
Mapbox `ShapeSource` or Google Maps `Polyline`:

```tsx
// Mapbox
<MapboxGL.ShapeSource
  id="route"
  shape={{ type: "FeatureCollection", features: routeGeometry }}
>
  <MapboxGL.LineLayer
    id="routeLine"
    style={{ lineColor: "#3B82F6", lineWidth: 3 }}
  />
</MapboxGL.ShapeSource>
```

---

## 7. AI-Engine Endpoint Reference

| AI-Engine Endpoint                | Method | Purpose                               |
| --------------------------------- | ------ | ------------------------------------- |
| `/api/tour-plan/generate`         | POST   | Start tour plan generation            |
| `/api/tour-plan/resume-selection` | POST   | Resume after HITL candidate selection |
| `/api/tour-plan/resume-weather`   | POST   | Resume after weather interrupt choice |
| `/api/chat`                       | POST   | Agentic conversational chat           |
| `/api/health`                     | GET    | Health check                          |

---

## 8. State Machine Flow

```
START
  │
  ▼
[generate] ──▶ AI-Engine invoke()
  │
  ├── COMPLETE ──────────────────────────────────▶ Done
  │
  ├── SELECTION_REQUIRED ──▶ [select-candidate]
  │                             │
  │                             ├── COMPLETE ────▶ Done
  │                             └── WEATHER ─────▶ [resume-weather] ──▶ Done
  │
  └── WEATHER_INTERRUPT ───▶ [resume-weather]
                                │
                                ├── COMPLETE ────▶ Done
                                └── SELECTION ───▶ [select-candidate] ──▶ Done
```

---

## Notes

- **Thread ID**: Always store `thread_id` in MongoDB. It's the LangGraph
  checkpoint key and is required for every resume call.
- **Timeout**: The ai-engine may take 10-25 seconds for the initial
  `generate` call (MCP fan-out + LLM ranking). Set client timeout ≥ 30s.
- **Idempotency**: Resuming with the same `thread_id` + `candidateId` /
  `userChoice` is safe — LangGraph checkpoints are deterministic.
- **Error Recovery**: If the ai-engine returns a 5xx, the Node.js backend
  should return the error to the mobile app with a "Retry" button. The
  `thread_id` remains valid for retry.
