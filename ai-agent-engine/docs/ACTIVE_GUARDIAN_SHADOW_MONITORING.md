# Active Guardian: Shadow Monitoring System

## Overview

The **Active Guardian Shadow Monitoring System** is a two-phase trip validation and monitoring solution for the Travion AI Tour Guide. It implements a "Digital Twin of Itinerary" concept - continuously simulating trip success against environmental variables.

## Architecture

```
+---------------------------+     +---------------------------+
|   PHASE 1: PRE-TRIP       |     |   PHASE 2: POST-ACCEPT    |
|   (Python/LangGraph)      |     |   (TypeScript/MongoDB)    |
+---------------------------+     +---------------------------+
|                           |     |                           |
|  ShadowMonitorNode        |     |  ShadowWatcher (Worker)   |
|  - Weather Validation     |     |  - Periodic Checks        |
|  - Alert Scanning         |     |  - Delta Plan Generation  |
|  - Holiday Cross-Ref      |     |  - User Notifications     |
|  - APPROVE/REJECT Logic   |     |  - Alert Management       |
|                           |     |                           |
+-------------|-------------+     +-------------|-------------+
              |                               |
              v                               v
      [Self-Correction]              [TripPlan Model]
      [Loop if REJECTED]             [MongoDB Storage]
```

## Phase 1: Pre-Trip Validation (AI Engine)

### Location
`app/graph/nodes/shadow_monitor.py`

### Purpose
Validates itineraries BEFORE presenting to users. Acts as a quality gate ensuring all generated plans are safe and feasible.

### Validation Checks

1. **Weather Validation** (`app/tools/weather_api.py`)
   - OpenWeatherMap 5-day forecast integration
   - Rain probability thresholds (>80% = reject outdoor activities)
   - Wind speed alerts (>60 km/h = reject beach/water activities)
   - Temperature extremes
   - Activity-specific requirements

2. **News Alert Scanning** (`app/tools/news_alert_api.py`)
   - GDELT API integration (free, no API key required)
   - NewsAPI integration (optional, requires key)
   - Categories monitored:
     - Protests and civil unrest
     - Natural disasters (landslides, floods)
     - Road closures
     - Transport disruptions
     - Security incidents
     - Health emergencies

3. **Holiday/Poya Day Validation**
   - Cross-reference with `holidays_2026.json`
   - Alcohol ban detection on Poya days
   - Government office closures
   - Festival crowd predictions

4. **Crowd Predictions** (CrowdCast integration)
   - Location-type based predictions
   - Time-of-day optimization
   - Poya day crowd multipliers

5. **Lighting Conditions** (Golden Hour integration)
   - Photography-optimal times
   - Sunrise/sunset calculations

### Validation Outcomes

```python
class ValidationStatus(str, Enum):
    APPROVED = "APPROVED"                      # All checks pass
    APPROVED_WITH_WARNINGS = "APPROVED_WITH_WARNINGS"  # Minor concerns
    NEEDS_ADJUSTMENT = "NEEDS_ADJUSTMENT"      # Auto-correctable
    REJECTED = "REJECTED"                      # Requires re-planning
```

### Self-Correction Loop

When a plan is REJECTED, the system:
1. Generates `correction_hints` explaining what needs to change
2. Triggers re-generation with constraints
3. Re-validates the new plan
4. Maximum 3 loops to prevent infinite recursion

### Usage Example

```python
from app.graph.nodes.shadow_monitor import validate_trip_plan, ValidationStatus

result = await validate_trip_plan(
    itinerary=[
        {"locationName": "Mirissa Beach", "activity": "Beach day"},
        {"locationName": "Galle Fort", "activity": "Heritage tour"}
    ],
    trip_date=datetime(2026, 5, 11),  # Vesak Poya
    activities=["beach", "nightlife"]
)

if result.status == ValidationStatus.REJECTED:
    print("Plan rejected:", result.correction_hints)
    # Trigger re-generation
elif result.status == ValidationStatus.APPROVED:
    print("Plan approved! Score:", result.overall_score)
```

## Phase 2: Post-Acceptance Active Watcher (Backend)

### Components

#### 1. TripPlan Model (`src/models/TripPlan.ts`)

MongoDB schema with monitoring capabilities:

```typescript
interface ITripPlan {
  // Core trip data
  itinerary: IItineraryItem[];
  startDate: Date;
  endDate: Date;

  // Monitoring state
  monitoringStatus: MonitoringStatus;
  nextScheduledCheck: Date;
  monitoringInterval: number;  // Default: 4 hours

  // Active alerts
  activeAlerts: IActiveAlert[];
  alertHistory: IActiveAlert[];

  // Weather tracking
  weatherForecasts: IWeatherForecast[];

  // Alternative plans
  deltaPlans: IDeltaPlan[];
  currentDeltaPlanId?: string;

  // Notifications
  notifications: INotificationRecord[];
  notificationPreferences: {...};
}
```

#### Monitoring States

```typescript
enum MonitoringStatus {
  NOT_MONITORING = 'NOT_MONITORING',
  ACTIVE_MONITORING = 'ACTIVE_MONITORING',
  ALERT_DETECTED = 'ALERT_DETECTED',
  DELTA_PLAN_GENERATED = 'DELTA_PLAN_GENERATED',
  PAUSED = 'PAUSED',
  COMPLETED = 'COMPLETED',
  CANCELLED = 'CANCELLED',
}
```

#### 2. Shadow Watcher Worker (`src/workers/shadowWatcher.ts`)

Background worker that:
- Runs every 5 minutes to find trips due for checks
- Processes trips in batches (default: 10)
- Calls AI Engine for weather/alert validation
- Generates delta plans when issues detected
- Sends user notifications

```typescript
import { startShadowWatcher } from './workers/shadowWatcher';

// Start in server.ts
startShadowWatcher({
  enabled: true,
  checkIntervalMs: 5 * 60 * 1000,  // 5 minutes
  batchSize: 10,
  weatherCheckEnabled: true,
  alertCheckEnabled: true,
  deltaPlanEnabled: true,
  notificationsEnabled: true,
});
```

#### 3. Trip Monitoring Service (`src/services/TripMonitoringService.ts`)

API layer for controllers:

```typescript
// Create a new monitored trip
const trip = await tripMonitoringService.createTripPlan({
  userId: '...',
  title: 'Sri Lanka Adventure',
  startDate: new Date('2026-05-10'),
  endDate: new Date('2026-05-15'),
  destinations: ['Colombo', 'Mirissa', 'Galle'],
  itinerary: [...]
});

// Start monitoring
await tripMonitoringService.startMonitoring(trip._id);

// Get status
const status = await tripMonitoringService.getMonitoringStatus(trip._id);

// Acknowledge an alert
await tripMonitoringService.acknowledgeAlert(tripId, alertId, 'modify_plan');

// Accept delta plan
await tripMonitoringService.respondToDeltaPlan(tripId, deltaId, true);
```

## API Endpoints

### AI Engine (FastAPI)

```
POST /api/v1/shadow-monitor/validate
  - Validates an itinerary
  - Returns: ValidationResult

POST /api/v1/tools/weather/validate
  - Weather validation for itinerary
  - Returns: WeatherValidationResult

POST /api/v1/tools/alerts/validate
  - News alert scan for itinerary
  - Returns: ItineraryAlertValidation

POST /api/v1/tools/delta-plan/generate
  - Generate alternative plan
  - Returns: DeltaPlan
```

### Backend (Express.js)

```
POST   /api/trips/monitored
  - Create new monitored trip

GET    /api/trips/monitored/:id/status
  - Get monitoring status

POST   /api/trips/monitored/:id/start
  - Start monitoring

POST   /api/trips/monitored/:id/stop
  - Stop monitoring

GET    /api/trips/monitored/:id/alerts
  - Get active alerts

POST   /api/trips/monitored/:id/alerts/:alertId/acknowledge
  - Acknowledge alert

GET    /api/trips/monitored/:id/delta-plan
  - Get current delta plan

POST   /api/trips/monitored/:id/delta-plan/:deltaId/respond
  - Accept/reject delta plan

POST   /api/trips/monitored/:id/check
  - Trigger immediate check

GET    /api/trips/monitored/:id/history
  - Get monitoring history
```

## Configuration

### Environment Variables

```env
# AI Engine
OPENWEATHER_API_KEY=your_api_key
NEWS_API_KEY=your_api_key  # Optional

# Backend
SHADOW_WATCHER_ENABLED=true
SHADOW_WATCHER_INTERVAL_MS=300000
SHADOW_WATCHER_BATCH_SIZE=10
```

### Data Files

1. **holidays_2026.json** - Sri Lanka calendar
   ```json
   {
     "poya_days": [
       {"date": "2026-01-12", "name": "Duruthu Poya"},
       {"date": "2026-05-11", "name": "Vesak Poya"}
     ],
     "public_holidays": [...],
     "special_events": [...]
   }
   ```

2. **locations_metadata.csv** - Location coordinates
   ```csv
   name,latitude,longitude,type,district,activities
   Mirissa,5.9489,80.4589,Beach,Matara,"beach,whale watching"
   Sigiriya,7.9519,80.7603,Heritage,Matale,"hiking,photography"
   ```

## Monitoring Flow

```
User Creates Trip
       |
       v
+------------------+
| Phase 1:         |
| Pre-Validation   |----> REJECTED ----> Re-generate
+------------------+                         |
       |                                     |
       | APPROVED                            |
       v                                     |
+------------------+                         |
| Trip Accepted    |<------------------------+
| by User          |
+------------------+
       |
       v
+------------------+
| Start Monitoring |
+------------------+
       |
       v
+------------------+
| Every 4 Hours:   |
| - Weather Check  |
| - Alert Scan     |
+------------------+
       |
       +--> All OK ----> Continue Monitoring
       |
       +--> Issues Found
              |
              v
       +------------------+
       | Generate Delta   |
       | Plan             |
       +------------------+
              |
              v
       +------------------+
       | Notify User      |
       +------------------+
              |
              v
       User Response:
       - Accept Risk
       - Modify Plan (apply delta)
       - Cancel Trip
```

## Alert Categories

| Category | Severity | Examples |
|----------|----------|----------|
| WEATHER | Medium-Critical | Heavy rain, thunderstorm, high winds |
| PROTEST | High-Critical | Demonstrations, civil unrest |
| STRIKE | Medium-High | Transport strikes, hartals |
| NATURAL_DISASTER | Critical | Landslides, floods, tsunami |
| ROAD_CLOSURE | Medium-High | Blocked routes, accidents |
| TRANSPORT_DISRUPTION | Medium | Flight/train cancellations |
| SECURITY_INCIDENT | High-Critical | Curfews, bomb threats |
| HEALTH_EMERGENCY | High | Disease outbreaks |
| WILDLIFE_DANGER | Medium | Elephant warnings, crocodiles |

## Delta Plan Generation

When issues are detected, the system generates alternative plans:

1. **Weather Issues**: Suggest indoor alternatives or reschedule
2. **Road Closures**: Route alternatives
3. **Alerts**: Location substitutions
4. **Crowd Issues**: Time adjustments

Example Delta Plan:
```json
{
  "deltaId": "uuid",
  "reason": "Heavy rain forecast for Mirissa Beach on May 11",
  "originalItems": [
    {"locationName": "Mirissa Beach", "activity": "Beach day"}
  ],
  "suggestedItems": [
    {"locationName": "Galle Fort", "activity": "Indoor heritage tour"}
  ],
  "impactSummary": "Beach activity replaced with covered heritage site",
  "aiExplanation": "Due to 85% rain probability..."
}
```

## Notification Channels

- **Push Notifications**: Real-time mobile alerts
- **Email**: Detailed summaries with delta plans
- **SMS**: Critical alerts only (optional)

User configurable thresholds:
- `INFO`: All notifications
- `LOW`: Low severity and above
- `MEDIUM`: Medium severity and above (default)
- `HIGH`: High and critical only
- `CRITICAL`: Critical only

## Best Practices

1. **API Key Management**: Store keys in environment variables
2. **Rate Limiting**: Respect API rate limits (especially NewsAPI)
3. **Caching**: Cache weather forecasts for 1 hour
4. **Error Handling**: Graceful degradation if APIs unavailable
5. **Logging**: Comprehensive logging for debugging
6. **Testing**: Mock external APIs in tests

## Troubleshooting

### Common Issues

1. **Weather API Errors**
   - Check OPENWEATHER_API_KEY is set
   - Verify API quota not exceeded

2. **No Alerts Found**
   - GDELT may have regional gaps
   - Consider NewsAPI for better coverage

3. **High Latency**
   - Increase AI Engine timeout
   - Use caching for repeated locations

4. **Missed Checks**
   - Verify Shadow Watcher is running
   - Check MongoDB connection
   - Review batch size settings

## Future Enhancements

1. Machine learning for alert severity prediction
2. User behavior analysis for personalized alerts
3. Integration with travel insurance APIs
4. Real-time traffic data integration
5. Multi-language alert translation
