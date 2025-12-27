"""
Shadow Monitor Node: Multi-Constraint Reasoning for Plan Validation.

This node implements the "Shadow Monitoring" capability - an internal
reasoning step that validates plans against multiple constraint systems:

1. Event Sentinel: Cultural calendar (Poya days, festivals)
2. CrowdCast: Predicted crowd levels
3. Golden Hour: Optimal lighting/timing

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
    Recommendation: "Shift to 4:30 PM for better photos and lower crowds"
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ..state import (
    GraphState, ShadowMonitorLog, ConstraintViolation, ItinerarySlot
)
from ...tools.event_sentinel import get_event_sentinel
from ...tools.crowdcast import get_crowdcast
from ...tools.golden_hour import get_golden_hour_agent

logger = logging.getLogger(__name__)


class ShadowMonitor:
    """
    Multi-constraint reasoning engine for trip planning.

    This class orchestrates checks across all constraint systems and
    synthesizes recommendations that optimize multiple objectives.

    Attributes:
        event_sentinel: Cultural calendar checker
        crowdcast: Crowd prediction engine
        golden_hour: Lighting optimizer
    """

    def __init__(self):
        """Initialize Shadow Monitor with all sub-systems."""
        self.event_sentinel = get_event_sentinel()
        self.crowdcast = get_crowdcast()
        self.golden_hour = get_golden_hour_agent()

        logger.info("ShadowMonitor initialized with all constraint systems")

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
        event_info = self.event_sentinel.get_event_info(
            datetime.combine(target_date, datetime.min.time())
        )

        result = {
            "is_poya": event_info["is_poya"],
            "is_school_holiday": event_info["is_school_holiday"],
            "special_event": event_info.get("special_event"),
            "alcohol_allowed": event_info["alcohol_allowed"],
            "violations": [],
            "warnings": event_info.get("warnings", [])
        }

        # Check for alcohol-related activity violations
        alcohol_activities = ["nightlife", "bar", "pub", "drinking", "wine"]
        if not event_info["alcohol_allowed"]:
            for activity in activities:
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
        prediction = self.crowdcast.predict(
            location_type,
            target_datetime,
            is_poya=is_poya
        )

        # Find optimal alternative time
        optimal_times = self.crowdcast.find_optimal_time(
            location_type,
            target_datetime,
            is_poya=is_poya,
            preference="low_crowd"
        )

        result = {
            **prediction,
            "optimal_alternative": {
                "type": "crowd_optimization",
                "suggestion": f"Lower crowds at {optimal_times[0]['time']} ({optimal_times[0]['crowd_percentage']}%)" if optimal_times else None
            }
        }

        return result

    def _check_golden_hour(
        self,
        target_datetime: datetime,
        location: str
    ) -> Dict:
        """Check lighting conditions."""
        lighting = self.golden_hour.get_lighting_quality(target_datetime)
        sun_times = lighting.get("sun_times", {})

        result = {
            **lighting,
            "golden_suggestion": None
        }

        # Suggest golden hour if not already in it
        if lighting["quality"] != "golden":
            morning_gh = sun_times.get("golden_hour_morning", {})
            evening_gh = sun_times.get("golden_hour_evening", {})

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

        if results["overall_status"] == "violations":
            parts.append("CONSTRAINT VIOLATIONS DETECTED:")
            for v in results["violations"]:
                parts.append(f"  - {v['message']}")
            parts.append("")
            parts.append("Please modify your plans to address these issues.")

        elif results["overall_status"] == "warnings":
            parts.append("OPTIMIZATION SUGGESTIONS:")
            for w in results["warnings"]:
                parts.append(f"  - {w['message']}")
            parts.append("")

            if results["optimizations"]:
                parts.append("Recommended alternatives:")
                for opt in results["optimizations"]:
                    if opt and opt.get("suggestion"):
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
        event_info = self.event_sentinel.get_event_info(target_date)
        is_poya = event_info["is_poya"]

        # Get optimal times from CrowdCast
        optimal_times = self.crowdcast.find_optimal_time(
            location_type, target_date,
            is_poya=is_poya,
            preference="balanced"
        )

        # Get sun times for golden hour consideration
        sun_times = self.golden_hour.get_sun_times(target_date.date())

        # Choose best time
        if preferred_time:
            best_time = preferred_time
        elif optimal_times:
            best_time = optimal_times[0]["hour"]
        else:
            best_time = 10  # Default to 10 AM

        # Adjust for photography locations (prefer golden hour)
        if location_type in ["Beach", "Heritage", "Nature"]:
            evening_start = int(sun_times.get("golden_hour_evening", {}).get("start", "17:00").split(":")[0])
            if best_time >= 14:  # Afternoon visit
                best_time = evening_start - 1  # Arrive before golden hour

        # Get crowd prediction for chosen time
        target_dt = target_date.replace(hour=best_time, minute=30)
        crowd = self.crowdcast.predict(location_type, target_dt, is_poya)
        lighting = self.golden_hour.get_lighting_quality(target_dt)

        return ItinerarySlot(
            time=f"{best_time:02d}:30",
            location=location,
            activity=f"Visit {location}",
            duration_minutes=90,
            crowd_prediction=crowd["crowd_percentage"],
            lighting_quality=lighting["quality"],
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

        if event_info["is_poya"]:
            notes.append("Poya day - no alcohol available")
        if crowd["crowd_status"] in ["HIGH", "EXTREME"]:
            notes.append(f"Expected {crowd['crowd_status'].lower()} crowds")
        if lighting["quality"] == "golden":
            notes.append("Golden hour - excellent for photos")

        return "; ".join(notes) if notes else None


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
    Shadow Monitor Node: Multi-constraint validation for generated plans.

    This node orchestrates all constraint checks (Event Sentinel, CrowdCast,
    Golden Hour) and synthesizes optimization recommendations.

    Args:
        state: Current graph state

    Returns:
        Updated GraphState with constraint checks and recommendations

    Research Note:
        The shadow monitor implements "Reflective Planning" - it doesn't
        just detect problems but actively suggests better alternatives,
        enabling the system to self-correct before generating a response.
    """
    logger.info("Shadow Monitor: Running constraint checks...")

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
    location_type = "Beach"  # Default
    if target_location:
        loc_lower = target_location.lower()
        if any(x in loc_lower for x in ["temple", "tooth", "gangaramaya"]):
            location_type = "Religious"
        elif any(x in loc_lower for x in ["sigiriya", "galle fort", "polonnaruwa"]):
            location_type = "Heritage"
        elif any(x in loc_lower for x in ["yala", "horton", "sinharaja"]):
            location_type = "Nature"
        elif any(x in loc_lower for x in ["beach", "mirissa", "unawatuna", "jungle"]):
            location_type = "Beach"

    # Run constraint checks
    constraint_results = monitor.check_constraints(
        location=target_location or "Sri Lanka",
        location_type=location_type,
        target_datetime=target_datetime,
        planned_activities=[]
    )

    # Build constraint violations list
    violations = [
        ConstraintViolation(
            constraint_type=v["type"],
            description=v["message"],
            severity=v["severity"],
            suggestion=v.get("suggestion", "")
        )
        for v in constraint_results.get("violations", [])
    ]

    # Add shadow monitor log
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="shadow_monitor",
        input_context={
            "location": target_location,
            "datetime": target_datetime.isoformat(),
            "location_type": location_type
        },
        result=constraint_results["overall_status"],
        details=constraint_results["recommendation"],
        action_taken="generate_with_constraints" if violations else None
    )

    # Generate optimized itinerary slot if trip planning
    itinerary = None
    if state.get("intent") and state["intent"].value == "trip_planning" and target_location:
        slot = monitor.optimize_itinerary_slot(
            target_location, location_type, target_datetime
        )
        itinerary = [slot]

    return {
        **state,
        "constraint_violations": violations,
        "itinerary": itinerary,
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry],
        # Store constraint results for generator
        "_constraint_results": constraint_results
    }
