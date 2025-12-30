"""
Shadow Monitor Node: Active Guardian Multi-Constraint Reasoning for Plan Validation.

This node implements the "Active Guardian Shadow Monitoring" capability - an internal
reasoning step that validates plans against multiple constraint systems:

Phase 1 (Pre-Trip Validation):
1. Event Sentinel: Cultural calendar (Poya days, festivals, holidays)
2. CrowdCast: Predicted crowd levels
3. Golden Hour: Optimal lighting/timing
4. Weather API: Real-time weather forecasts (OpenWeatherMap)
5. News Alert API: Crisis detection (protests, landslides, road closures)

Research Pattern:
    Constraint Satisfaction with Multi-Objective Optimization - The shadow
    monitor doesn't just check for violations; it suggests optimal alternatives
    when constraints conflict with user preferences.

Example Scenario:
    User: "Visit Jungle Beach next full moon at noon"
    Shadow Monitor checks:
    - Event Sentinel: "Full moon = Poya day, alcohol banned"
    - CrowdCast: "Beach at noon on Poya = HIGH (75%)"
    - Golden Hour: "Noon = harsh light, sunset at 6:15 PM"
    - Weather: "80% rain probability - unsuitable for beach"
    - Alerts: "No travel warnings for area"
    Result: REJECT - triggers self-correction loop
    Recommendation: "Shift to 4:30 PM for better photos and lower crowds"

Digital Twin Concept:
    This module simulates the trip success against environmental variables
    before presenting to the user, enabling proactive optimization.
"""

import logging
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

from ..state import (
    GraphState, ShadowMonitorLog, ConstraintViolation, ItinerarySlot
)
from ...tools.event_sentinel import get_event_sentinel
from ...tools.crowdcast import get_crowdcast
from ...tools.golden_hour import get_golden_hour_agent
from ...tools.weather_api import WeatherTool, check_weather_for_trip, WeatherValidationResult
from ...tools.news_alert_api import NewsAlertTool, check_alerts_for_trip, ItineraryAlertValidation

logger = logging.getLogger(__name__)


# ============================================================================
# VALIDATION STATUS TYPES
# ============================================================================

class ValidationStatus(str, Enum):
    """Shadow Monitor validation outcome"""
    APPROVED = "APPROVED"          # Plan passes all checks
    APPROVED_WITH_WARNINGS = "APPROVED_WITH_WARNINGS"  # Minor concerns
    NEEDS_ADJUSTMENT = "NEEDS_ADJUSTMENT"  # Can be fixed automatically
    REJECTED = "REJECTED"          # Requires re-planning


class ConstraintType(str, Enum):
    """Types of constraints checked by Shadow Monitor"""
    POYA_ALCOHOL = "poya_alcohol"
    POYA_ACTIVITY = "poya_activity"
    HOLIDAY_CLOSURE = "holiday_closure"
    WEATHER_RAIN = "weather_rain"
    WEATHER_EXTREME = "weather_extreme"
    WEATHER_WIND = "weather_wind"
    NEWS_ALERT = "news_alert"
    ROAD_CLOSURE = "road_closure"
    CROWD_EXTREME = "crowd_extreme"
    LIGHTING_POOR = "lighting_poor"


class ValidationConstraint(BaseModel):
    """Detailed constraint violation for Shadow Monitor"""
    constraint_type: ConstraintType
    severity: str  # low, medium, high, critical
    description: str
    affected_location: Optional[str] = None
    affected_time: Optional[str] = None
    suggestion: str
    is_blocking: bool = False  # If True, must reject the plan


class ShadowMonitorResult(BaseModel):
    """Complete result from Shadow Monitor validation"""
    status: ValidationStatus
    overall_score: float = Field(ge=0, le=100, description="Overall plan score 0-100")
    constraints: List[ValidationConstraint] = []
    weather_validation: Optional[Dict[str, Any]] = None
    alert_validation: Optional[Dict[str, Any]] = None
    event_validation: Optional[Dict[str, Any]] = None
    crowd_validation: Optional[Dict[str, Any]] = None
    recommendations: List[str] = []
    should_trigger_correction: bool = False
    correction_hints: List[str] = []


# ============================================================================
# DATA LOADERS
# ============================================================================

class HolidayData:
    """Singleton loader for Sri Lanka holidays"""
    _instance = None
    _holidays: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_holidays()
        return cls._instance

    def _load_holidays(self):
        """Load holidays from JSON file"""
        try:
            # Look for holidays file in data directory
            data_paths = [
                Path(__file__).parent.parent.parent / "data" / "holidays_2026.json",
                Path(__file__).parent.parent.parent / "data" / "holidays.json",
                Path(__file__).parent.parent.parent.parent / "data" / "holidays_2026.json",
            ]

            raw_data = None
            for data_path in data_paths:
                if data_path.exists():
                    with open(data_path, 'r', encoding='utf-8') as f:
                        raw_data = json.load(f)
                    logger.info(f"Loaded holidays from {data_path}")
                    break

            if raw_data is None:
                logger.warning("No holidays file found, using empty data")
                self._holidays = {"poya_days": [], "public_holidays": [], "special_events": []}
                return

            # Handle different file formats
            if isinstance(raw_data, dict):
                # Already in expected format
                self._holidays = raw_data
            elif isinstance(raw_data, list):
                # Convert flat list format to expected structure
                # Each item has: summary, categories, start, end
                poya_days = []
                public_holidays = []
                special_events = []

                for item in raw_data:
                    if not isinstance(item, dict):
                        continue

                    categories = item.get("categories", [])
                    if not isinstance(categories, list):
                        categories = []

                    start_date = item.get("start", "")
                    name = item.get("summary", "")

                    holiday_entry = {
                        "date": start_date,
                        "name": name,
                        "closures": []
                    }

                    # Check if it's a Poya day
                    if "Poya" in categories:
                        poya_days.append(holiday_entry)

                    # Check if it's a public holiday
                    if "Public" in categories or "Bank" in categories:
                        public_holidays.append(holiday_entry)

                    # Check for special events (Mercantile holidays, festivals)
                    if "Mercantile" in categories or any(kw in name.lower() for kw in ["festival", "deepavali", "christmas", "new year"]):
                        special_events.append(holiday_entry)

                self._holidays = {
                    "poya_days": poya_days,
                    "public_holidays": public_holidays,
                    "special_events": special_events
                }
                logger.info(f"Converted {len(raw_data)} holidays: {len(poya_days)} Poya days, {len(public_holidays)} public holidays")
            else:
                logger.warning(f"Unexpected holidays data type: {type(raw_data)}")
                self._holidays = {"poya_days": [], "public_holidays": [], "special_events": []}

        except Exception as e:
            logger.error(f"Failed to load holidays: {e}")
            self._holidays = {"poya_days": [], "public_holidays": [], "special_events": []}

    def get_holiday_info(self, date: datetime) -> Dict[str, Any]:
        """Get holiday information for a specific date"""
        date_str = date.strftime("%Y-%m-%d")

        result = {
            "is_poya": False,
            "is_public_holiday": False,
            "is_special_event": False,
            "poya_name": None,
            "holiday_name": None,
            "alcohol_banned": False,
            "closures": [],
        }

        # Check Poya days
        for poya in self._holidays.get("poya_days", []):
            if poya.get("date") == date_str:
                result["is_poya"] = True
                result["poya_name"] = poya.get("name")
                result["alcohol_banned"] = True
                break

        # Check public holidays
        for holiday in self._holidays.get("public_holidays", []):
            if holiday.get("date") == date_str:
                result["is_public_holiday"] = True
                result["holiday_name"] = holiday.get("name")
                result["closures"] = holiday.get("closures", [])
                break

        # Check special events
        for event in self._holidays.get("special_events", []):
            if event.get("date") == date_str:
                result["is_special_event"] = True
                if not result["holiday_name"]:
                    result["holiday_name"] = event.get("name")
                break

        return result


class LocationMetadata:
    """Singleton loader for location metadata"""
    _instance = None
    _locations: Dict[str, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_locations()
        return cls._instance

    def _load_locations(self):
        """Load location metadata from CSV"""
        try:
            data_paths = [
                Path(__file__).parent.parent.parent / "data" / "locations_metadata.csv",
                Path(__file__).parent.parent.parent.parent / "data" / "locations_metadata.csv",
            ]

            for data_path in data_paths:
                if data_path.exists():
                    with open(data_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            name = row.get("name", "").lower()
                            if name:
                                self._locations[name] = {
                                    "latitude": float(row.get("latitude", 0)),
                                    "longitude": float(row.get("longitude", 0)),
                                    "type": row.get("type", "general"),
                                    "district": row.get("district", ""),
                                    "typical_activities": row.get("activities", "").split(","),
                                }
                    logger.info(f"Loaded {len(self._locations)} locations from {data_path}")
                    return

            logger.warning("No locations metadata file found")
        except Exception as e:
            logger.error(f"Failed to load locations: {e}")

    def get_location_info(self, location_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a location"""
        return self._locations.get(location_name.lower())

    def get_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a location"""
        info = self.get_location_info(location_name)
        if info:
            return (info["latitude"], info["longitude"])
        return None


class ShadowMonitor:
    """
    Active Guardian Multi-constraint reasoning engine for trip planning.

    This class orchestrates checks across all constraint systems and
    synthesizes recommendations that optimize multiple objectives.

    Phase 1 (Pre-Trip Validation):
        - Validates itineraries before presenting to users
        - Cross-references weather, news alerts, holidays
        - Triggers self-correction loop on REJECT

    Attributes:
        event_sentinel: Cultural calendar checker
        crowdcast: Crowd prediction engine
        golden_hour: Lighting optimizer
        weather_tool: Real-time weather validation
        news_alert_tool: Crisis/alert detection
        holiday_data: Sri Lanka holidays loader
        location_metadata: Location coordinates and info
    """

    # Thresholds for validation
    RAIN_THRESHOLD_OUTDOOR = 80  # >80% rain = reject outdoor activities
    RAIN_THRESHOLD_WARNING = 50  # >50% rain = warning
    WIND_THRESHOLD_CRITICAL = 60  # >60 km/h = reject beach/water activities
    CROWD_THRESHOLD_EXTREME = 85  # >85% crowd = warning

    def __init__(self):
        """Initialize Shadow Monitor with all sub-systems."""
        self.event_sentinel = get_event_sentinel()
        self.crowdcast = get_crowdcast()
        self.golden_hour = get_golden_hour_agent()

        # Active Guardian components (initialized lazily to handle missing API keys)
        self._weather_tool: Optional[WeatherTool] = None
        self._news_alert_tool: Optional[NewsAlertTool] = None
        self.holiday_data = HolidayData()
        self.location_metadata = LocationMetadata()

        logger.info("ShadowMonitor (Active Guardian) initialized with all constraint systems")

    @property
    def weather_tool(self) -> Optional[WeatherTool]:
        """Lazy initialization of weather tool"""
        if self._weather_tool is None:
            try:
                self._weather_tool = WeatherTool()
                if not self._weather_tool.is_configured():
                    logger.warning("Weather API key not configured - weather validation disabled")
                    return None
            except Exception as e:
                logger.warning(f"Failed to initialize WeatherTool: {e}")
                return None
        return self._weather_tool

    @property
    def news_alert_tool(self) -> Optional[NewsAlertTool]:
        """Lazy initialization of news alert tool"""
        if self._news_alert_tool is None:
            try:
                self._news_alert_tool = NewsAlertTool()
            except Exception as e:
                logger.warning(f"Failed to initialize NewsAlertTool: {e}")
                return None
        return self._news_alert_tool

    def check_constraints(
        self,
        location: str,
        location_type: str,
        target_datetime: datetime,
        planned_activities: List[str] = None
    ) -> Dict:
        """
        Run all constraint checks for a planned visit.

        Args:
            location: Location name
            location_type: One of Heritage, Beach, Nature, Religious, Urban
            target_datetime: When the visit is planned
            planned_activities: List of planned activities

        Returns:
            Dict with all constraint check results

        Example:
            >>> monitor = ShadowMonitor()
            >>> result = monitor.check_constraints(
            ...     "Jungle Beach (Rumassala)",
            ...     "Beach",
            ...     datetime(2026, 5, 11, 12, 0),  # Vesak Poya noon
            ...     ["swimming", "photography"]
            ... )
            >>> print(result["overall_status"])
            'warnings'
        """
        planned_activities = planned_activities or []
        results = {
            "location": location,
            "datetime": target_datetime.isoformat(),
            "checks": {},
            "violations": [],
            "warnings": [],
            "optimizations": [],
            "overall_status": "ok"
        }

        # 1. Event Sentinel Check
        event_check = self._check_event_sentinel(
            target_datetime.date(), location_type, planned_activities
        )
        results["checks"]["event_sentinel"] = event_check

        if event_check.get("violations"):
            results["violations"].extend(event_check["violations"])
            results["overall_status"] = "violations"
        if event_check.get("warnings"):
            results["warnings"].extend(event_check["warnings"])
            if results["overall_status"] == "ok":
                results["overall_status"] = "warnings"

        # 2. CrowdCast Check
        crowd_check = self._check_crowdcast(
            location_type, target_datetime,
            is_poya=event_check.get("is_poya", False)
        )
        results["checks"]["crowdcast"] = crowd_check

        if crowd_check.get("crowd_status") in ["EXTREME", "HIGH"]:
            results["warnings"].append({
                "type": "high_crowd",
                "severity": "medium",
                "message": f"Expected {crowd_check['crowd_status']} crowds ({crowd_check['crowd_percentage']}%)"
            })
            results["optimizations"].append(crowd_check.get("optimal_alternative"))
            if results["overall_status"] == "ok":
                results["overall_status"] = "warnings"

        # 3. Golden Hour Check
        lighting_check = self._check_golden_hour(target_datetime, location)
        results["checks"]["golden_hour"] = lighting_check

        if lighting_check.get("quality") == "harsh":
            results["warnings"].append({
                "type": "poor_lighting",
                "severity": "low",
                "message": "Harsh midday lighting - not ideal for photography"
            })
            results["optimizations"].append({
                "type": "lighting",
                "suggestion": f"Consider visiting during golden hour: {lighting_check['golden_suggestion']}"
            })

        # Generate overall recommendation
        results["recommendation"] = self._synthesize_recommendation(results)

        return results

    def _check_event_sentinel(
        self,
        target_date,
        location_type: str,
        activities: List[str]
    ) -> Dict:
        """Check cultural calendar constraints."""
        try:
            event_info = self.event_sentinel.get_event_info(
                datetime.combine(target_date, datetime.min.time())
            )
        except Exception as e:
            logger.warning(f"Event sentinel check failed: {e}")
            event_info = {}

        # Defensive check: ensure event_info is a dict
        if not isinstance(event_info, dict):
            logger.warning(f"Event sentinel returned non-dict: {type(event_info)}")
            event_info = {}

        result = {
            "is_poya": event_info.get("is_poya", False),
            "is_school_holiday": event_info.get("is_school_holiday", False),
            "special_event": event_info.get("special_event"),
            "alcohol_allowed": event_info.get("alcohol_allowed", True),
            "violations": [],
            "warnings": event_info.get("warnings", []) if isinstance(event_info.get("warnings"), list) else []
        }

        # Check for alcohol-related activity violations
        alcohol_activities = ["nightlife", "bar", "pub", "drinking", "wine"]
        if not result["alcohol_allowed"]:
            for activity in activities:
                if not isinstance(activity, str):
                    continue
                if any(a in activity.lower() for a in alcohol_activities):
                    result["violations"].append({
                        "type": "poya_alcohol",
                        "severity": "critical",
                        "message": f"'{activity}' not available - alcohol banned on Poya days",
                        "suggestion": "Consider cultural activities or nature experiences instead"
                    })

        return result

    def _check_crowdcast(
        self,
        location_type: str,
        target_datetime: datetime,
        is_poya: bool = False
    ) -> Dict:
        """Check crowd predictions."""
        try:
            prediction = self.crowdcast.predict(
                location_type,
                target_datetime,
                is_poya=is_poya
            )
        except Exception as e:
            logger.warning(f"Crowdcast prediction failed: {e}")
            prediction = {}

        # Defensive check: ensure prediction is a dict
        if not isinstance(prediction, dict):
            logger.warning(f"Crowdcast returned non-dict: {type(prediction)}")
            prediction = {"crowd_status": "UNKNOWN", "crowd_percentage": 50}

        # Find optimal alternative time
        try:
            optimal_times = self.crowdcast.find_optimal_time(
                location_type,
                target_datetime,
                is_poya=is_poya,
                preference="low_crowd"
            )
        except Exception as e:
            logger.warning(f"Crowdcast find_optimal_time failed: {e}")
            optimal_times = []

        # Defensive check: ensure optimal_times is a list
        if not isinstance(optimal_times, list):
            optimal_times = []

        # Build suggestion safely
        suggestion = None
        if optimal_times and len(optimal_times) > 0:
            first_time = optimal_times[0]
            if isinstance(first_time, dict):
                time_str = first_time.get('time', 'early morning')
                crowd_pct = first_time.get('crowd_percentage', 'lower')
                suggestion = f"Lower crowds at {time_str} ({crowd_pct}%)"

        result = {
            **prediction,
            "optimal_alternative": {
                "type": "crowd_optimization",
                "suggestion": suggestion
            }
        }

        return result

    def _check_golden_hour(
        self,
        target_datetime: datetime,
        location: str
    ) -> Dict:
        """Check lighting conditions."""
        try:
            lighting = self.golden_hour.get_lighting_quality(target_datetime)
        except Exception as e:
            logger.warning(f"Golden hour check failed: {e}")
            lighting = {}

        # Defensive check: ensure lighting is a dict
        if not isinstance(lighting, dict):
            logger.warning(f"Golden hour returned non-dict: {type(lighting)}")
            lighting = {"quality": "good"}

        sun_times = lighting.get("sun_times", {})
        if not isinstance(sun_times, dict):
            sun_times = {}

        result = {
            **lighting,
            "golden_suggestion": None
        }

        # Suggest golden hour if not already in it
        quality = lighting.get("quality", "good")
        if quality != "golden":
            morning_gh = sun_times.get("golden_hour_morning", {})
            evening_gh = sun_times.get("golden_hour_evening", {})

            if not isinstance(morning_gh, dict):
                morning_gh = {}
            if not isinstance(evening_gh, dict):
                evening_gh = {}

            if target_datetime.hour < 12:
                result["golden_suggestion"] = f"Morning golden hour: {morning_gh.get('start', '06:00')} - {morning_gh.get('end', '07:00')}"
            else:
                result["golden_suggestion"] = f"Evening golden hour: {evening_gh.get('start', '17:30')} - {evening_gh.get('end', '18:30')}"

        return result

    def _synthesize_recommendation(self, results: Dict) -> str:
        """
        Synthesize all checks into a coherent recommendation.

        This is the "reasoning" output that explains the shadow monitoring.
        """
        parts = []

        overall_status = results.get("overall_status", "ok") if isinstance(results, dict) else "ok"

        if overall_status == "violations":
            parts.append("CONSTRAINT VIOLATIONS DETECTED:")
            violations = results.get("violations", []) if isinstance(results, dict) else []
            for v in violations:
                if isinstance(v, dict):
                    parts.append(f"  - {v.get('message', 'Unknown violation')}")
            parts.append("")
            parts.append("Please modify your plans to address these issues.")

        elif overall_status == "warnings":
            parts.append("OPTIMIZATION SUGGESTIONS:")
            warnings = results.get("warnings", []) if isinstance(results, dict) else []
            for w in warnings:
                if isinstance(w, dict):
                    parts.append(f"  - {w.get('message', 'Unknown warning')}")
            parts.append("")

            optimizations = results.get("optimizations", []) if isinstance(results, dict) else []
            if optimizations:
                parts.append("Recommended alternatives:")
                for opt in optimizations:
                    if isinstance(opt, dict) and opt.get("suggestion"):
                        parts.append(f"  - {opt['suggestion']}")

        else:
            parts.append("All constraints satisfied. Your plan looks good!")

        return "\n".join(parts)

    def optimize_itinerary_slot(
        self,
        location: str,
        location_type: str,
        target_date: datetime,
        preferred_time: Optional[int] = None
    ) -> ItinerarySlot:
        """
        Generate an optimized itinerary slot for a location.

        This method finds the best time to visit considering all constraints.

        Args:
            location: Location name
            location_type: Location category
            target_date: Target date
            preferred_time: Preferred hour (optional)

        Returns:
            ItinerarySlot with optimized timing
        """
        # Check if date is Poya
        try:
            event_info = self.event_sentinel.get_event_info(target_date)
        except Exception as e:
            logger.warning(f"Event sentinel failed in optimize_itinerary_slot: {e}")
            event_info = {}

        if not isinstance(event_info, dict):
            event_info = {}
        is_poya = event_info.get("is_poya", False)

        # Get optimal times from CrowdCast
        try:
            optimal_times = self.crowdcast.find_optimal_time(
                location_type, target_date,
                is_poya=is_poya,
                preference="balanced"
            )
        except Exception as e:
            logger.warning(f"Crowdcast find_optimal_time failed: {e}")
            optimal_times = []

        if not isinstance(optimal_times, list):
            optimal_times = []

        # Get sun times for golden hour consideration
        try:
            sun_times = self.golden_hour.get_sun_times(target_date.date())
        except Exception as e:
            logger.warning(f"Golden hour get_sun_times failed: {e}")
            sun_times = {}

        if not isinstance(sun_times, dict):
            sun_times = {}

        # Choose best time
        if preferred_time:
            best_time = preferred_time
        elif optimal_times and len(optimal_times) > 0 and isinstance(optimal_times[0], dict):
            best_time = optimal_times[0].get("hour", 10)
        else:
            best_time = 10  # Default to 10 AM

        # Adjust for photography locations (prefer golden hour)
        if location_type in ["Beach", "Heritage", "Nature"]:
            golden_evening = sun_times.get("golden_hour_evening", {})
            if not isinstance(golden_evening, dict):
                golden_evening = {}
            evening_start_str = golden_evening.get("start", "17:00")
            try:
                evening_start = int(evening_start_str.split(":")[0])
            except (ValueError, AttributeError):
                evening_start = 17
            if best_time >= 14:  # Afternoon visit
                best_time = evening_start - 1  # Arrive before golden hour

        # Get crowd prediction for chosen time
        target_dt = target_date.replace(hour=best_time, minute=30)
        try:
            crowd = self.crowdcast.predict(location_type, target_dt, is_poya)
        except Exception as e:
            logger.warning(f"Crowdcast predict failed: {e}")
            crowd = {}

        if not isinstance(crowd, dict):
            crowd = {"crowd_percentage": 50, "crowd_status": "MODERATE"}

        try:
            lighting = self.golden_hour.get_lighting_quality(target_dt)
        except Exception as e:
            logger.warning(f"Golden hour get_lighting_quality failed: {e}")
            lighting = {}

        if not isinstance(lighting, dict):
            lighting = {"quality": "good"}

        return ItinerarySlot(
            time=f"{best_time:02d}:30",
            location=location,
            activity=f"Visit {location}",
            duration_minutes=90,
            crowd_prediction=crowd.get("crowd_percentage", 50),
            lighting_quality=lighting.get("quality", "good"),
            notes=self._generate_slot_notes(event_info, crowd, lighting)
        )

    def _generate_slot_notes(
        self,
        event_info: Dict,
        crowd: Dict,
        lighting: Dict
    ) -> str:
        """Generate notes for an itinerary slot."""
        notes = []

        # Defensive checks
        if not isinstance(event_info, dict):
            event_info = {}
        if not isinstance(crowd, dict):
            crowd = {}
        if not isinstance(lighting, dict):
            lighting = {}

        if event_info.get("is_poya"):
            notes.append("Poya day - no alcohol available")
        crowd_status = crowd.get("crowd_status", "")
        if crowd_status in ["HIGH", "EXTREME"]:
            notes.append(f"Expected {crowd_status.lower()} crowds")
        if lighting.get("quality") == "golden":
            notes.append("Golden hour - excellent for photos")

        return "; ".join(notes) if notes else None

    # =========================================================================
    # ACTIVE GUARDIAN METHODS
    # =========================================================================

    async def validate_trip_plan_comprehensive(
        self,
        itinerary_items: List[Dict[str, Any]],
        trip_date: datetime,
        planned_activities: List[str] = None
    ) -> ShadowMonitorResult:
        """
        Comprehensive Active Guardian validation for a trip plan.

        This is the main entry point for pre-trip validation that checks:
        1. Weather conditions (rain, wind, temperature)
        2. News alerts (protests, landslides, closures)
        3. Holiday conflicts (Poya days, public holidays)
        4. Crowd predictions
        5. Lighting conditions

        Args:
            itinerary_items: List of itinerary items with location info
            trip_date: The date of the trip
            planned_activities: List of planned activities

        Returns:
            ShadowMonitorResult with validation status and recommendations
        """
        logger.info(f"Active Guardian: Validating trip plan for {trip_date.date()}")

        constraints: List[ValidationConstraint] = []
        recommendations: List[str] = []
        correction_hints: List[str] = []
        overall_score = 100.0

        # Enrich itinerary items with coordinates if missing
        enriched_items = self._enrich_itinerary_with_coordinates(itinerary_items)

        # 1. Weather Validation
        weather_result = None
        if self.weather_tool is not None:
            try:
                weather_result = await self.weather_tool.validate_itinerary_weather(
                    enriched_items, trip_date
                )

                if not weather_result.is_valid:
                    overall_score -= 30
                    for issue in weather_result.blocking_issues:
                        constraints.append(ValidationConstraint(
                            constraint_type=ConstraintType.WEATHER_RAIN,
                            severity="critical",
                            description=issue,
                            suggestion="Consider rescheduling or choosing indoor alternatives",
                            is_blocking=True
                        ))
                        correction_hints.append(f"WEATHER: {issue}")

                for warning in weather_result.warnings:
                    constraints.append(ValidationConstraint(
                        constraint_type=ConstraintType.WEATHER_RAIN,
                        severity="medium",
                        description=warning,
                        suggestion="Have backup plans ready"
                    ))
                    overall_score -= 5

                recommendations.extend(weather_result.recommendations)

            except Exception as e:
                logger.warning(f"Weather validation failed: {e}")
                recommendations.append("Could not verify weather - check forecast manually")
        else:
            logger.info("Weather validation skipped - API key not configured")
            recommendations.append("Weather validation unavailable - set OPENWEATHER_API_KEY to enable")

        # 2. News Alert Validation
        alert_result = None
        if self.news_alert_tool is not None:
            try:
                alert_result = await self.news_alert_tool.validate_itinerary_alerts(
                    enriched_items, days_back=7
                )

                if not alert_result.is_safe:
                    overall_score -= 40
                    for alert in alert_result.blocking_alerts:
                        constraints.append(ValidationConstraint(
                            constraint_type=ConstraintType.NEWS_ALERT,
                            severity="critical",
                            description=f"{alert.title}: {alert.travel_impact}",
                            affected_location=alert.affected_locations[0] if alert.affected_locations else None,
                            suggestion=alert.recommended_action,
                            is_blocking=True
                        ))
                        correction_hints.append(f"ALERT: Avoid {alert.affected_locations} - {alert.category.value}")

                recommendations.extend(alert_result.recommendations)

            except Exception as e:
                logger.warning(f"Alert validation failed: {e}")
                recommendations.append("Could not check news alerts - monitor local news")
        else:
            logger.info("News alert validation skipped - tool not available")

        # 3. Holiday/Poya Day Validation
        event_result = self._validate_holiday_constraints(
            trip_date, enriched_items, planned_activities or []
        )
        constraints.extend(event_result["constraints"])
        recommendations.extend(event_result["recommendations"])
        correction_hints.extend(event_result["correction_hints"])
        overall_score -= event_result["score_penalty"]

        # 4. Crowd Validation (for each location)
        for item in enriched_items:
            location = item.get("locationName", "")
            location_type = self._determine_location_type(location)

            try:
                crowd_check = self._check_crowdcast(
                    location_type, trip_date,
                    is_poya=event_result.get("is_poya", False)
                )

                if crowd_check.get("crowd_percentage", 0) > self.CROWD_THRESHOLD_EXTREME:
                    constraints.append(ValidationConstraint(
                        constraint_type=ConstraintType.CROWD_EXTREME,
                        severity="medium",
                        description=f"Extreme crowds expected at {location} ({crowd_check['crowd_percentage']}%)",
                        affected_location=location,
                        suggestion=crowd_check.get("optimal_alternative", {}).get("suggestion", "Visit early morning")
                    ))
                    overall_score -= 5
            except Exception as e:
                logger.debug(f"Crowd check failed for {location}: {e}")

        # Determine final status
        has_blocking = any(c.is_blocking for c in constraints)
        has_critical = any(c.severity == "critical" for c in constraints)
        has_warnings = any(c.severity in ["medium", "high"] for c in constraints)

        if has_blocking or has_critical:
            status = ValidationStatus.REJECTED
            should_trigger_correction = True
        elif overall_score < 60:
            status = ValidationStatus.NEEDS_ADJUSTMENT
            should_trigger_correction = True
        elif has_warnings:
            status = ValidationStatus.APPROVED_WITH_WARNINGS
            should_trigger_correction = False
        else:
            status = ValidationStatus.APPROVED
            should_trigger_correction = False

        logger.info(f"Active Guardian Result: {status.value} (score: {overall_score:.1f})")

        return ShadowMonitorResult(
            status=status,
            overall_score=max(0, min(100, overall_score)),
            constraints=constraints,
            weather_validation=weather_result.dict() if weather_result else None,
            alert_validation=alert_result.dict() if alert_result else None,
            event_validation=event_result,
            recommendations=recommendations,
            should_trigger_correction=should_trigger_correction,
            correction_hints=correction_hints
        )

    def _enrich_itinerary_with_coordinates(
        self,
        items: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add coordinates to itinerary items if missing."""
        enriched = []
        for item in items:
            # Defensive check: ensure item is a dict
            if not isinstance(item, dict):
                logger.warning(f"Skipping non-dict itinerary item: {type(item)}")
                continue

            enriched_item = dict(item)

            if not enriched_item.get("latitude") or not enriched_item.get("longitude"):
                location_name = enriched_item.get("locationName", "")
                coords = self.location_metadata.get_coordinates(location_name)

                if coords:
                    enriched_item["latitude"] = coords[0]
                    enriched_item["longitude"] = coords[1]
                else:
                    # Use default Sri Lanka center if location unknown
                    logger.debug(f"No coordinates for {location_name}, using defaults")
                    enriched_item["latitude"] = 7.8731
                    enriched_item["longitude"] = 80.7718

            enriched.append(enriched_item)

        return enriched

    def _validate_holiday_constraints(
        self,
        trip_date: datetime,
        items: List[Dict[str, Any]],
        activities: List[str]
    ) -> Dict[str, Any]:
        """Validate against holiday calendar."""
        result = {
            "is_poya": False,
            "is_public_holiday": False,
            "constraints": [],
            "recommendations": [],
            "correction_hints": [],
            "score_penalty": 0
        }

        holiday_info = self.holiday_data.get_holiday_info(trip_date)

        if holiday_info.get("is_poya"):
            result["is_poya"] = True

            # Check for alcohol-related activities
            alcohol_keywords = ["bar", "pub", "nightlife", "drinking", "alcohol", "wine", "beer"]
            for activity in activities:
                # Defensive check: ensure activity is a string
                if not isinstance(activity, str):
                    continue
                if any(kw in activity.lower() for kw in alcohol_keywords):
                    poya_name = holiday_info.get('poya_name', 'Poya day')
                    result["constraints"].append(ValidationConstraint(
                        constraint_type=ConstraintType.POYA_ALCOHOL,
                        severity="critical",
                        description=f"'{activity}' not available on Poya day ({poya_name})",
                        suggestion="Replace with cultural or nature activities",
                        is_blocking=True
                    ))
                    result["correction_hints"].append(f"REMOVE alcohol activity: {activity}")
                    result["score_penalty"] += 20

            poya_name = holiday_info.get('poya_name', 'Poya day')
            result["recommendations"].append(
                f"Note: {trip_date.strftime('%Y-%m-%d')} is {poya_name} - "
                "alcohol sales banned, temples will be busy"
            )

        if holiday_info.get("is_public_holiday"):
            result["is_public_holiday"] = True
            holiday_name = holiday_info.get('holiday_name', 'public holiday')
            result["recommendations"].append(
                f"Public holiday: {holiday_name} - expect closures and crowds"
            )

            # Check for closures
            closures = holiday_info.get("closures", [])
            if isinstance(closures, list):
                for closure in closures:
                    if not isinstance(closure, str):
                        continue
                    for item in items:
                        # Defensive check: ensure item is a dict
                        if not isinstance(item, dict):
                            continue
                        location = item.get("locationName", "").lower() if isinstance(item.get("locationName"), str) else ""
                        if closure.lower() in location:
                            result["constraints"].append(ValidationConstraint(
                                constraint_type=ConstraintType.HOLIDAY_CLOSURE,
                                severity="high",
                                description=f"{item.get('locationName', 'Location')} may be closed on {holiday_name}",
                                affected_location=item.get("locationName"),
                                suggestion="Verify opening hours or choose alternative"
                            ))
                            result["score_penalty"] += 10

        return result

    def _determine_location_type(self, location_name: str) -> str:
        """Determine location type from name."""
        loc_lower = location_name.lower()

        if any(x in loc_lower for x in ["temple", "tooth", "gangaramaya", "kelaniya"]):
            return "Religious"
        elif any(x in loc_lower for x in ["sigiriya", "galle fort", "polonnaruwa", "anuradhapura"]):
            return "Heritage"
        elif any(x in loc_lower for x in ["yala", "horton", "sinharaja", "udawalawe", "wilpattu"]):
            return "Nature"
        elif any(x in loc_lower for x in ["beach", "mirissa", "unawatuna", "hikkaduwa", "arugam"]):
            return "Beach"
        elif any(x in loc_lower for x in ["colombo", "kandy city", "negombo town"]):
            return "Urban"

        return "General"


# Singleton instance
_shadow_monitor: Optional[ShadowMonitor] = None


def get_shadow_monitor() -> ShadowMonitor:
    """Get or create ShadowMonitor singleton."""
    global _shadow_monitor
    if _shadow_monitor is None:
        _shadow_monitor = ShadowMonitor()
    return _shadow_monitor


async def shadow_monitor_node(state: GraphState) -> GraphState:
    """
    Shadow Monitor Node: Active Guardian Multi-constraint validation for generated plans.

    This node orchestrates all constraint checks (Event Sentinel, CrowdCast,
    Golden Hour, Weather API, News Alerts) and synthesizes optimization recommendations.

    Phase 1 (Pre-Trip Validation):
        - Validates itineraries before presenting to users
        - Cross-references weather, news alerts, holidays
        - Triggers self-correction loop on REJECT

    Args:
        state: Current graph state

    Returns:
        Updated GraphState with constraint checks and recommendations

    Research Note:
        The shadow monitor implements "Reflective Planning" - it doesn't
        just detect problems but actively suggests better alternatives,
        enabling the system to self-correct before generating a response.
    """
    logger.info("Shadow Monitor (Active Guardian): Running comprehensive validation...")

    monitor = get_shadow_monitor()

    # Get target info from state
    target_location = state.get("target_location")
    target_date_ref = state.get("target_date")

    # Parse target date
    if target_date_ref == "next_poya":
        target_date, poya_name = monitor.event_sentinel.get_next_poya()
        target_datetime = target_date.replace(hour=10, minute=0)
    elif target_date_ref:
        # Handle other date references
        target_datetime = datetime.now().replace(hour=10, minute=0)
    else:
        target_datetime = datetime.now().replace(hour=10, minute=0)

    # Determine location type (simplified heuristic)
    location_type = monitor._determine_location_type(target_location or "")

    # Build itinerary items from state
    itinerary_items = []
    if target_location:
        itinerary_items.append({
            "locationName": target_location,
            "activity": f"Visit {target_location}"
        })

    # If state has existing itinerary, use that
    if state.get("itinerary"):
        for slot in state["itinerary"]:
            # Defensive check: ensure slot is a dict before calling .get()
            if isinstance(slot, dict):
                itinerary_items.append({
                    "locationName": slot.get("location", slot.get("locationName", "")),
                    "activity": slot.get("activity", ""),
                    "time": slot.get("time", "")
                })
            else:
                logger.warning(f"Unexpected itinerary slot type: {type(slot)}, value: {slot}")

    # Run comprehensive Active Guardian validation
    guardian_result = await monitor.validate_trip_plan_comprehensive(
        itinerary_items=itinerary_items,
        trip_date=target_datetime,
        planned_activities=[]
    )

    # Also run legacy constraint checks for backward compatibility
    constraint_results = monitor.check_constraints(
        location=target_location or "Sri Lanka",
        location_type=location_type,
        target_datetime=target_datetime,
        planned_activities=[]
    )

    # Build constraint violations list (combine legacy + Active Guardian)
    violations = []
    for v in constraint_results.get("violations", []):
        # Defensive check: ensure v is a dict before accessing
        if isinstance(v, dict):
            violations.append(ConstraintViolation(
                constraint_type=v.get("type", "unknown"),
                description=v.get("message", v.get("description", "")),
                severity=v.get("severity", "medium"),
                suggestion=v.get("suggestion", "")
            ))
        else:
            logger.warning(f"Unexpected violation type: {type(v)}, value: {v}")

    # Add Active Guardian constraints
    for constraint in guardian_result.constraints:
        violations.append(ConstraintViolation(
            constraint_type=constraint.constraint_type.value,
            description=constraint.description,
            severity=constraint.severity,
            suggestion=constraint.suggestion
        ))

    # Add shadow monitor log
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="active_guardian",
        input_context={
            "location": target_location,
            "datetime": target_datetime.isoformat(),
            "location_type": location_type,
            "validation_status": guardian_result.status.value,
            "overall_score": guardian_result.overall_score
        },
        result=guardian_result.status.value,
        details="; ".join(guardian_result.recommendations[:3]) if guardian_result.recommendations else "All checks passed",
        action_taken="trigger_correction" if guardian_result.should_trigger_correction else "proceed"
    )

    # Generate optimized itinerary slot if trip planning and approved
    itinerary = None
    if state.get("intent") and state["intent"].value == "trip_planning" and target_location:
        if guardian_result.status in [ValidationStatus.APPROVED, ValidationStatus.APPROVED_WITH_WARNINGS]:
            slot = monitor.optimize_itinerary_slot(
                target_location, location_type, target_datetime
            )
            itinerary = [slot]

    # Determine if we need to trigger self-correction
    needs_correction = guardian_result.should_trigger_correction

    return {
        **state,
        "constraint_violations": violations,
        "itinerary": itinerary,
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry],
        # Store results for generator and correction loop
        "_constraint_results": constraint_results,
        "_guardian_result": guardian_result.dict(),
        "_needs_correction": needs_correction,
        "_correction_hints": guardian_result.correction_hints
    }


# ============================================================================
# CONVENIENCE FUNCTIONS FOR ACTIVE GUARDIAN
# ============================================================================

async def validate_trip_plan(
    itinerary: List[Dict[str, Any]],
    trip_date: datetime,
    activities: List[str] = None
) -> ShadowMonitorResult:
    """
    Convenience function for validating a trip plan with Active Guardian.

    Usage:
        from app.graph.nodes.shadow_monitor import validate_trip_plan

        result = await validate_trip_plan(itinerary, trip_date)
        if result.status == ValidationStatus.REJECTED:
            # Trigger self-correction with correction_hints
            pass
        elif result.status == ValidationStatus.APPROVED:
            # Present to user
            pass

    Args:
        itinerary: List of itinerary items
        trip_date: Date of the trip
        activities: Optional list of planned activities

    Returns:
        ShadowMonitorResult with validation status
    """
    monitor = get_shadow_monitor()
    return await monitor.validate_trip_plan_comprehensive(
        itinerary_items=itinerary,
        trip_date=trip_date,
        planned_activities=activities or []
    )


async def check_location_safety(
    location_name: str,
    trip_date: datetime
) -> Dict[str, Any]:
    """
    Quick safety check for a single location.

    Returns dict with:
        - is_safe: bool
        - weather_ok: bool
        - alerts: List of active alerts
        - recommendations: List of recommendations
    """
    monitor = get_shadow_monitor()

    itinerary = [{"locationName": location_name, "activity": f"Visit {location_name}"}]

    result = await monitor.validate_trip_plan_comprehensive(
        itinerary_items=itinerary,
        trip_date=trip_date
    )

    return {
        "is_safe": result.status in [ValidationStatus.APPROVED, ValidationStatus.APPROVED_WITH_WARNINGS],
        "status": result.status.value,
        "score": result.overall_score,
        "constraints": [c.dict() for c in result.constraints],
        "recommendations": result.recommendations
    }
