# Travion AI Engine

An Agentic RAG (Retrieval-Augmented Generation) system for Sri Lankan tourism, featuring self-correcting responses, cultural context awareness, and multi-constraint optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Travion AI Engine                            │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌───────────┐    ┌────────┐    ┌──────────────┐    │
│  │ Router  │───▶│ Retrieval │───▶│ Grader │───▶│ Web Search   │    │
│  └─────────┘    └───────────┘    └────────┘    │ (if needed)  │    │
│       │              │                │         └──────────────┘    │
│       │              ▼                ▼                │            │
│       │         ChromaDB         Relevance            │            │
│       │        Vector DB          Check               │            │
│       │                                               ▼            │
│  ┌────┴────────────────────────────────────────────────────┐       │
│  │                   Shadow Monitor                         │       │
│  │  ┌──────────────┐ ┌───────────┐ ┌─────────────┐         │       │
│  │  │EventSentinel │ │ CrowdCast │ │ Golden Hour │         │       │
│  │  │ (Poya Days)  │ │(ML Model) │ │ (Sun Calc)  │         │       │
│  │  └──────────────┘ └───────────┘ └─────────────┘         │       │
│  └─────────────────────────────────────────────────────────┘       │
│                              │                                      │
│                              ▼                                      │
│                    ┌─────────────────┐                             │
│                    │    Generator    │◀──────┐                     │
│                    │   (Llama 3.1)   │       │                     │
│                    └─────────────────┘       │                     │
│                              │               │                     │
│                              ▼               │                     │
│                    ┌─────────────────┐       │                     │
│                    │    Verifier     │───────┘                     │
│                    │ (Self-Correct)  │  Loop if constraints fail   │
│                    └─────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### 7 Pillars of Intelligence

| Pillar | Component | Description |
|--------|-----------|-------------|
| 1 | **Specialist Boost** | ChromaDB vector search with 480 tourism documents |
| 2 | **CrowdCast** | ML-based crowd prediction (R² = 0.9982) |
| 3 | **Visual Matcher** | Image-based location matching (future) |
| 4 | **Event Sentinel** | Poya day & holiday detection with constraints |
| 5 | **Golden Hour** | Sun position calculation for photography |
| 6 | **Shadow Monitor** | Multi-constraint validation orchestrator |
| 7 | **Storyteller** | LLM-powered narrative generation |

## Installation

```bash
cd services/ai-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your settings
```

## Configuration

Edit `.env` file:

```env
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# Web Search (Optional)
TAVILY_API_KEY=your_api_key_here

# Server
PORT=8001
DEBUG=true
```

## Running the Server

```bash
# Development mode (with auto-reload)
python run.py

# Production mode
python run.py --production --workers 4

# Custom host/port
python run.py --host 127.0.0.1 --port 8080
```

Server will start at: `http://localhost:8001`

## API Documentation

Interactive docs available at: `http://localhost:8001/docs`

---

## API Endpoints

### 1. Chat Endpoint (Main Conversational AI)

**POST** `/api/v1/chat`

The main endpoint for tourism queries. Uses the full LangGraph workflow with self-correction.

```bash
# Basic query
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about Sigiriya Rock Fortress",
    "session_id": "user-123"
  }'

# Trip planning with date (triggers Shadow Monitor)
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Plan a trip to Jungle Beach next full moon",
    "session_id": "user-123",
    "context": {
      "target_date": "2026-05-11"
    }
  }'

# With location context
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the best times to visit?",
    "session_id": "user-123",
    "context": {
      "current_location": "Galle Fort",
      "preferences": ["photography", "avoid_crowds"]
    }
  }'
```

**Response:**
```json
{
  "response": "Sigiriya Rock Fortress is a UNESCO World Heritage Site...",
  "session_id": "user-123",
  "intent": "TOURISM_QUERY",
  "sources": [
    {"location": "Sigiriya", "aspect": "_history", "relevance": 0.92}
  ],
  "constraints_checked": {
    "poya_day": false,
    "crowd_level": "MODERATE",
    "golden_hour": "16:30-18:00"
  },
  "metadata": {
    "reasoning_loops": 1,
    "model": "llama3.1:8b",
    "processing_time_ms": 1234
  }
}
```

---

### 2. Crowd Prediction Endpoint

**POST** `/api/v1/crowd`

Predict crowd levels at specific locations and times.

```bash
# Basic prediction
curl -X POST http://localhost:8001/api/v1/crowd \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Jungle Beach",
    "location_type": "Beach",
    "target_datetime": "2026-05-11T16:30:00"
  }'

# Find optimal times
curl -X POST http://localhost:8001/api/v1/crowd/optimal \
  -H "Content-Type: application/json" \
  -d '{
    "location_type": "Heritage",
    "date": "2026-07-15",
    "preference": "low_crowd"
  }'

# Compare multiple dates
curl -X POST http://localhost:8001/api/v1/crowd/compare \
  -H "Content-Type: application/json" \
  -d '{
    "location_type": "Nature",
    "dates": ["2026-07-10", "2026-07-11", "2026-07-12"],
    "hour": 9
  }'
```

**Response:**
```json
{
  "location_type": "Beach",
  "datetime": "2026-05-11T16:30:00",
  "crowd_level": 0.72,
  "crowd_percentage": 72.0,
  "crowd_status": "HIGH",
  "recommendation": "Expect queues. Book tickets in advance if possible.",
  "factors": {
    "month": 5,
    "day_of_week": 0,
    "hour": 16,
    "is_weekend": false,
    "is_poya": true,
    "seasonal_factor": 0.6
  },
  "model_type": "ml"
}
```

---

### 3. Event/Holiday Endpoint

**POST** `/api/v1/events`

Check dates for Poya days, holidays, and cultural events.

```bash
# Check specific date
curl -X POST http://localhost:8001/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-05-11"
  }'

# Check activity constraints
curl -X POST http://localhost:8001/api/v1/events/constraints \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-05-11",
    "activity": "nightlife",
    "location_type": "Beach"
  }'

# Get next Poya day
curl -X GET "http://localhost:8001/api/v1/events/next-poya?from_date=2026-05-01"

# Find optimal dates
curl -X POST http://localhost:8001/api/v1/events/optimal \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-05-01",
    "end_date": "2026-05-15",
    "preferences": {
      "avoid_poya": true,
      "needs_alcohol": true,
      "avoid_crowds": true
    }
  }'
```

**Response (event info):**
```json
{
  "date": "2026-05-11",
  "is_poya": true,
  "is_school_holiday": false,
  "is_weekend": false,
  "day_of_week": "Monday",
  "alcohol_allowed": false,
  "special_event": "Vesak Full Moon Poya Day",
  "crowd_impact": "extreme_crowd",
  "warnings": [
    "Alcohol sales banned island-wide on Poya days"
  ],
  "recommendations": [
    "Visit temples early morning (5-7 AM) to avoid crowds"
  ]
}
```

**Response (constraints check):**
```json
{
  "date": "2026-05-11",
  "activity": "nightlife",
  "location_type": "Beach",
  "is_allowed": false,
  "violations": [
    {
      "type": "poya_alcohol",
      "severity": "high",
      "message": "Alcohol activities not available on Vesak Full Moon Poya Day. All alcohol sales are banned island-wide."
    }
  ],
  "suggestions": [
    "Consider visiting a tea plantation or cultural site instead"
  ]
}
```

---

### 4. Golden Hour Endpoint

**POST** `/api/v1/golden-hour`

Calculate optimal photography times based on sun position.

```bash
# Get sun times for location
curl -X POST http://localhost:8001/api/v1/golden-hour \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Sigiriya",
    "date": "2026-07-15"
  }'

# Get lighting quality at specific time
curl -X POST http://localhost:8001/api/v1/golden-hour/quality \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Galle Fort",
    "datetime": "2026-07-15T17:30:00"
  }'

# Get optimal photo times
curl -X POST http://localhost:8001/api/v1/golden-hour/optimal \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Nine Arch Bridge",
    "date": "2026-07-15",
    "preference": "golden_hour"
  }'
```

**Response:**
```json
{
  "location": "Sigiriya",
  "date": "2026-07-15",
  "coordinates": {"lat": 7.957, "lon": 80.7603},
  "sun_times": {
    "sunrise": "06:02",
    "sunset": "18:32",
    "golden_hour_morning": {"start": "06:02", "end": "07:02"},
    "golden_hour_evening": {"start": "17:32", "end": "18:32"},
    "blue_hour_morning": {"start": "05:32", "end": "06:02"},
    "blue_hour_evening": {"start": "18:32", "end": "19:02"}
  },
  "recommendations": [
    "Best morning light: 06:00-07:00 for dramatic shadows on rock face",
    "Best evening light: 17:30-18:30 for warm golden tones"
  ]
}
```

---

### 5. Health Check

**GET** `/api/v1/health`

```bash
curl http://localhost:8001/api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "ollama": "connected",
    "chromadb": "connected",
    "crowdcast_model": "loaded",
    "event_sentinel": "initialized"
  },
  "documents_indexed": 480
}
```

---

### 6. Graph Visualization

**GET** `/api/v1/graph`

Get Mermaid diagram of the LangGraph workflow.

```bash
curl http://localhost:8001/api/v1/graph
```

**Response:**
```json
{
  "mermaid": "graph TD\n    __start__ --> router\n    router --> retrieval...",
  "nodes": ["router", "retrieval", "grader", "web_search", "shadow_monitor", "generator", "verifier"],
  "edges": [...]
}
```

---

## Tools Reference

### EventSentinel

Detects Poya days and cultural constraints.

```python
from app.tools.event_sentinel import get_event_sentinel

sentinel = get_event_sentinel()

# Check if date is Poya
is_poya = sentinel.is_poya_day(datetime(2026, 5, 11))  # True (Vesak)

# Get event info
info = sentinel.get_event_info(datetime(2026, 5, 11))

# Check activity constraints
result = sentinel.check_activity_constraints(
    date=datetime(2026, 5, 11),
    activity="nightlife",
    location_type="Beach"
)
```

### CrowdCast

Predicts crowd density using ML model.

```python
from app.tools.crowdcast import get_crowdcast

crowdcast = get_crowdcast()

# Predict crowd level
prediction = crowdcast.predict(
    location_type="Heritage",
    target_datetime=datetime(2026, 7, 15, 10, 0),
    is_poya=False,
    is_school_holiday=False
)
print(f"Crowd: {prediction['crowd_percentage']}%")

# Find optimal visiting times
slots = crowdcast.find_optimal_time(
    location_type="Beach",
    target_date=datetime(2026, 7, 15),
    preference="low_crowd"
)
```

### GoldenHour

Calculates sun positions for photography.

```python
from app.tools.golden_hour import get_golden_hour

golden = get_golden_hour()

# Get sun times
times = golden.get_sun_times(
    location="Sigiriya",
    date=datetime(2026, 7, 15)
)

# Get lighting quality
quality = golden.get_lighting_quality(
    location="Galle Fort",
    target_time=datetime(2026, 7, 15, 17, 30)
)
```

---

## Data Files

| File | Description |
|------|-------------|
| `data/holidays_2026.json` | Sri Lankan holidays, Poya days, events |
| `models/crowdcast_model.joblib` | Trained Random Forest model |
| `models/label_encoder.joblib` | Location type encoder |
| `vector_db/` | ChromaDB with 480 tourism documents |

---

## Example: Full Trip Planning Flow

```bash
# 1. User asks about trip planning
curl -X POST http://localhost:8001/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Plan a trip to Jungle Beach next full moon",
    "session_id": "trip-001"
  }'

# System Response:
# - Detects "next full moon" → Vesak Poya (May 11, 2026)
# - Checks constraints → Alcohol banned
# - Predicts crowd → HIGH (72%)
# - Calculates golden hour → 16:30-18:00
# - Generates recommendation with alternatives
```

**Expected Output:**
```
I'd be happy to help plan your trip to Jungle Beach! However, I notice
you're planning for the next full moon (May 11, 2026), which is Vesak
Poya - Sri Lanka's most important Buddhist holiday.

⚠️ Important Considerations:
• Alcohol Ban: All alcohol sales are prohibited island-wide
• Expected Crowds: HIGH (72%) - Many locals visiting beaches
• Best Time: 4:30 PM for golden hour photography

✅ Recommended Itinerary:
• Arrive: 4:30 PM (avoid midday heat and crowds)
• Activity: Snorkeling, beach photography
• Sunset: 6:15 PM - Perfect for photos
• Alternative: Visit on May 12 for 35% fewer crowds

Would you like me to suggest nearby restaurants that will be open,
or find an alternative date with better conditions?
```

---

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy app/

# Format code
black app/
```

## License

MIT License - Travion Project
