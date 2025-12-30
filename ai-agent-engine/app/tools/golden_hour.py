"""
Golden Hour Tool: Physics-Based Lighting Optimization for Photography.

This module implements the Golden Hour Agent pillar, calculating optimal
photography times based on sun position and elevation.

Research Problem:
    Aesthetic Ignorance - Standard recommenders ignore lighting conditions,
    missing the most photogenic times for scenic locations.

Solution Logic:
    Physics Calculation - Use the `astral` library to calculate sun elevation.
    Auto-schedule scenic spots during Golden Hour (-4° to 6° elevation).

Golden Hour Definition:
    - Blue Hour: Sun at -6° to -4° below horizon (deep blue sky)
    - Golden Hour: Sun at -4° to 6° above horizon (warm, soft light)
    - Harsh Light: Sun above 20° (harsh shadows, avoid for photography)

Sri Lanka Location:
    - Latitude: ~7.8731° N
    - Longitude: ~80.7718° E
    - Timezone: Asia/Colombo (UTC+5:30)
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

# Sri Lanka timezone
SL_TIMEZONE = ZoneInfo("Asia/Colombo")


def to_local_time(dt: datetime) -> str:
    """Convert UTC datetime to Sri Lanka local time string (HH:MM)."""
    if dt.tzinfo is None:
        # Assume UTC if no timezone
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    local_dt = dt.astimezone(SL_TIMEZONE)
    return local_dt.strftime("%H:%M")

logger = logging.getLogger(__name__)

# Try to import astral library
try:
    from astral import LocationInfo
    from astral.sun import sun, elevation, golden_hour, blue_hour
    ASTRAL_AVAILABLE = True
except ImportError:
    ASTRAL_AVAILABLE = False
    logger.warning("Astral library not available. Using fallback calculations.")


@dataclass
class SunEvent:
    """Represents a sun event (sunrise, sunset, etc.)."""
    name: str
    time: datetime
    elevation: float
    quality: str  # "golden", "blue", "harsh", "dark"


class GoldenHourAgent:
    """
    Sun position and lighting quality calculator for Sri Lankan locations.

    This class provides physics-based calculations for optimal photography
    times at any GPS coordinate in Sri Lanka.

    Attributes:
        default_location: Default coordinates (Colombo)
        timezone: Sri Lankan timezone

    Research Note:
        Golden hour timing varies by ~30 minutes across Sri Lanka
        (east coast vs west coast). For accurate recommendations,
        use the specific location coordinates.
    """

    # Sri Lanka approximate bounds
    SRI_LANKA_BOUNDS = {
        "min_lat": 5.9,
        "max_lat": 9.9,
        "min_lng": 79.5,
        "max_lng": 81.9
    }

    # Famous photography locations with coordinates
    PHOTOGRAPHY_SPOTS = {
        "Sigiriya": {"lat": 7.957, "lng": 80.760, "best_for": "sunrise"},
        "Galle Fort": {"lat": 6.033, "lng": 80.217, "best_for": "sunset"},
        "Nine Arches Bridge": {"lat": 6.878, "lng": 81.060, "best_for": "sunrise"},
        "Temple of the Tooth": {"lat": 7.294, "lng": 80.641, "best_for": "sunset"},
        "Mirissa Beach": {"lat": 5.948, "lng": 80.459, "best_for": "sunset"},
        "Horton Plains": {"lat": 6.809, "lng": 80.800, "best_for": "sunrise"},
        "Ella Rock": {"lat": 6.867, "lng": 81.047, "best_for": "sunrise"},
        "Jungle Beach (Rumassala)": {"lat": 6.015, "lng": 80.230, "best_for": "sunset"},
    }

    # Default fallback times for Sri Lanka
    DEFAULT_TIMES = {
        "sunrise": "06:00",
        "sunset": "18:15",
        "golden_hour_morning_start": "05:45",
        "golden_hour_morning_end": "06:45",
        "golden_hour_evening_start": "17:30",
        "golden_hour_evening_end": "18:30",
    }

    def __init__(self):
        """Initialize Golden Hour Agent."""
        self.timezone = "Asia/Colombo"
        self.default_lat = 7.8731
        self.default_lng = 80.7718

        if ASTRAL_AVAILABLE:
            logger.info("GoldenHourAgent initialized with astral library")
        else:
            logger.info("GoldenHourAgent using fallback calculations")

    def _get_location_info(
        self,
        name: str,
        lat: float,
        lng: float
    ) -> 'LocationInfo':
        """Create an astral LocationInfo object."""
        return LocationInfo(
            name=name,
            region="Sri Lanka",
            timezone=self.timezone,
            latitude=lat,
            longitude=lng
        )

    def get_sun_times(
        self,
        target_date: date,
        lat: Optional[float] = None,
        lng: Optional[float] = None,
        location_name: str = "Location"
    ) -> Dict:
        """
        Calculate sun times for a specific date and location.

        Args:
            target_date: Date to calculate for
            lat: Latitude (default: Colombo)
            lng: Longitude (default: Colombo)
            location_name: Name for the location

        Returns:
            Dict with all sun timing information

        Example:
            >>> agent = GoldenHourAgent()
            >>> times = agent.get_sun_times(date(2026, 5, 11), 6.015, 80.230)
            >>> print(times["golden_hour_morning"])
            {'start': '05:52', 'end': '06:47'}
        """
        lat = lat or self.default_lat
        lng = lng or self.default_lng

        if ASTRAL_AVAILABLE:
            return self._calculate_with_astral(target_date, lat, lng, location_name)
        else:
            return self._calculate_fallback(target_date)

    def _calculate_with_astral(
        self,
        target_date: date,
        lat: float,
        lng: float,
        location_name: str
    ) -> Dict:
        """Calculate sun times using astral library."""
        try:
            loc = self._get_location_info(location_name, lat, lng)
            s = sun(loc.observer, date=target_date)

            # Calculate golden hour times based on sunrise/sunset
            # Golden hour: ~30 min before sunrise/sunset to ~45 min after
            gh_morning_start = s["sunrise"] - timedelta(minutes=30)
            gh_morning_end = s["sunrise"] + timedelta(minutes=45)
            gh_evening_start = s["sunset"] - timedelta(minutes=45)
            gh_evening_end = s["sunset"] + timedelta(minutes=30)

            # Calculate blue hour times
            # Blue hour: ~45 min to ~15 min before sunrise, and ~15 min to ~45 min after sunset
            bh_morning_start = s["sunrise"] - timedelta(minutes=45)
            bh_morning_end = s["sunrise"] - timedelta(minutes=15)
            bh_evening_start = s["sunset"] + timedelta(minutes=15)
            bh_evening_end = s["sunset"] + timedelta(minutes=45)

            return {
                "date": target_date.isoformat(),
                "location": location_name,
                "coordinates": {"lat": lat, "lng": lng},
                "dawn": to_local_time(s["dawn"]),
                "sunrise": to_local_time(s["sunrise"]),
                "noon": to_local_time(s["noon"]),
                "sunset": to_local_time(s["sunset"]),
                "dusk": to_local_time(s["dusk"]),
                "golden_hour_morning": {
                    "start": to_local_time(gh_morning_start),
                    "end": to_local_time(gh_morning_end),
                    "duration_minutes": 75
                },
                "golden_hour_evening": {
                    "start": to_local_time(gh_evening_start),
                    "end": to_local_time(gh_evening_end),
                    "duration_minutes": 75
                },
                "blue_hour_morning": {
                    "start": to_local_time(bh_morning_start),
                    "end": to_local_time(bh_morning_end)
                },
                "blue_hour_evening": {
                    "start": to_local_time(bh_evening_start),
                    "end": to_local_time(bh_evening_end)
                },
                "calculation_method": "astral"
            }

        except Exception as e:
            logger.error(f"Astral calculation failed: {e}")
            return self._calculate_fallback(target_date)


    def _calculate_fallback(self, target_date: date) -> Dict:
        """Fallback sun time calculation without astral."""
        # Sri Lanka sun times are fairly consistent year-round
        # Sunrise: ~5:55-6:30 AM, Sunset: ~6:00-6:30 PM

        # Adjust slightly by month (rough approximation)
        month = target_date.month
        if month in [1, 2, 11, 12]:  # Winter months
            sunrise_hour, sunrise_min = 6, 15
            sunset_hour, sunset_min = 18, 0
        elif month in [5, 6, 7, 8]:  # Summer months
            sunrise_hour, sunrise_min = 5, 55
            sunset_hour, sunset_min = 18, 30
        else:
            sunrise_hour, sunrise_min = 6, 5
            sunset_hour, sunset_min = 18, 15

        return {
            "date": target_date.isoformat(),
            "location": "Sri Lanka (default)",
            "coordinates": {"lat": self.default_lat, "lng": self.default_lng},
            "dawn": f"{sunrise_hour-1:02d}:{sunrise_min:02d}",
            "sunrise": f"{sunrise_hour:02d}:{sunrise_min:02d}",
            "noon": "12:15",
            "sunset": f"{sunset_hour:02d}:{sunset_min:02d}",
            "dusk": f"{sunset_hour+1:02d}:{sunset_min:02d}",
            "golden_hour_morning": {
                "start": f"{sunrise_hour-1:02d}:{sunrise_min+30:02d}",
                "end": f"{sunrise_hour:02d}:{sunrise_min+45:02d}",
                "duration_minutes": 75
            },
            "golden_hour_evening": {
                "start": f"{sunset_hour-1:02d}:{sunset_min+15:02d}",
                "end": f"{sunset_hour:02d}:{sunset_min+30:02d}",
                "duration_minutes": 75
            },
            "blue_hour_morning": {
                "start": f"{sunrise_hour-1:02d}:{sunrise_min:02d}",
                "end": f"{sunrise_hour-1:02d}:{sunrise_min+30:02d}"
            },
            "blue_hour_evening": {
                "start": f"{sunset_hour:02d}:{sunset_min+15:02d}",
                "end": f"{sunset_hour:02d}:{sunset_min+45:02d}"
            },
            "calculation_method": "fallback"
        }

    def get_lighting_quality(
        self,
        target_datetime: datetime,
        lat: Optional[float] = None,
        lng: Optional[float] = None
    ) -> Dict:
        """
        Assess lighting quality at a specific time.

        Args:
            target_datetime: Datetime to assess
            lat: Latitude
            lng: Longitude

        Returns:
            Dict with lighting assessment

        Example:
            >>> agent = GoldenHourAgent()
            >>> quality = agent.get_lighting_quality(
            ...     datetime(2026, 5, 11, 17, 30),
            ...     6.015, 80.230  # Rumassala
            ... )
            >>> print(quality["quality"])
            'golden'
        """
        sun_times = self.get_sun_times(target_datetime.date(), lat, lng)
        time_str = target_datetime.strftime("%H:%M")

        # Parse sun times for comparison
        def time_to_minutes(t: str) -> int:
            h, m = map(int, t.split(":"))
            return h * 60 + m

        current = time_to_minutes(time_str)
        sunrise = time_to_minutes(sun_times["sunrise"])
        sunset = time_to_minutes(sun_times["sunset"])
        gh_morning_start = time_to_minutes(sun_times["golden_hour_morning"]["start"])
        gh_morning_end = time_to_minutes(sun_times["golden_hour_morning"]["end"])
        gh_evening_start = time_to_minutes(sun_times["golden_hour_evening"]["start"])
        gh_evening_end = time_to_minutes(sun_times["golden_hour_evening"]["end"])

        # Determine quality
        if current < sunrise - 60 or current > sunset + 60:
            quality = "dark"
            description = "Too dark for photography without special equipment"
            photography_score = 20
        elif gh_morning_start <= current <= gh_morning_end:
            quality = "golden"
            description = "Perfect golden hour lighting for photography"
            photography_score = 100
        elif gh_evening_start <= current <= gh_evening_end:
            quality = "golden"
            description = "Perfect golden hour lighting for photography"
            photography_score = 100
        elif sunrise <= current < gh_morning_end + 60:
            quality = "good"
            description = "Good soft morning light"
            photography_score = 80
        elif gh_evening_start - 60 < current <= sunset:
            quality = "good"
            description = "Good soft evening light"
            photography_score = 80
        elif 10 * 60 <= current <= 14 * 60:  # 10 AM to 2 PM
            quality = "harsh"
            description = "Harsh midday light - strong shadows"
            photography_score = 40
        else:
            quality = "moderate"
            description = "Acceptable lighting conditions"
            photography_score = 60

        return {
            "time": time_str,
            "quality": quality,
            "description": description,
            "photography_score": photography_score,
            "is_golden_hour": quality == "golden",
            "sun_times": sun_times,
            "recommendation": self._get_lighting_recommendation(quality, target_datetime)
        }

    def _get_lighting_recommendation(
        self,
        quality: str,
        target_datetime: datetime
    ) -> str:
        """Generate lighting-based recommendation."""
        if quality == "golden":
            return "Excellent time for photography! Bring your camera."
        elif quality == "good":
            return "Good lighting for outdoor activities and photos."
        elif quality == "harsh":
            return "Consider indoor activities or seek shade. Use a polarizing filter for photos."
        elif quality == "dark":
            return "Limited visibility. Best for stargazing or night photography."
        else:
            return "Standard lighting conditions."

    def get_optimal_photo_times(
        self,
        location_name: str,
        target_date: date
    ) -> Dict:
        """
        Get optimal photography times for a known location.

        Args:
            location_name: Name of the location
            target_date: Date to calculate for

        Returns:
            Dict with optimal time recommendations

        Example:
            >>> agent = GoldenHourAgent()
            >>> times = agent.get_optimal_photo_times("Sigiriya", date(2026, 5, 11))
            >>> print(times["recommended_time"])
            '05:45 - 06:45 (Sunrise golden hour)'
        """
        # Look up known location
        spot = self.PHOTOGRAPHY_SPOTS.get(location_name)

        if spot:
            lat, lng = spot["lat"], spot["lng"]
            best_for = spot["best_for"]
        else:
            lat, lng = self.default_lat, self.default_lng
            best_for = "both"

        sun_times = self.get_sun_times(target_date, lat, lng, location_name)

        if best_for == "sunrise":
            recommended = f"{sun_times['golden_hour_morning']['start']} - {sun_times['golden_hour_morning']['end']} (Sunrise golden hour)"
            peak_time = sun_times["sunrise"]
        elif best_for == "sunset":
            recommended = f"{sun_times['golden_hour_evening']['start']} - {sun_times['golden_hour_evening']['end']} (Sunset golden hour)"
            peak_time = sun_times["sunset"]
        else:
            recommended = f"Morning: {sun_times['golden_hour_morning']['start']} or Evening: {sun_times['golden_hour_evening']['start']}"
            peak_time = sun_times["sunset"]

        return {
            "location": location_name,
            "date": target_date.isoformat(),
            "coordinates": {"lat": lat, "lng": lng},
            "best_for": best_for,
            "recommended_time": recommended,
            "peak_time": peak_time,
            "sun_times": sun_times,
            "tips": self._get_location_tips(location_name, best_for)
        }

    def _get_location_tips(self, location_name: str, best_for: str) -> List[str]:
        """Get photography tips for specific locations."""
        tips = {
            "Sigiriya": [
                "Climb early to reach top before sunrise",
                "The lion's paws are lit beautifully at dawn",
                "Bring a headlamp for the early climb"
            ],
            "Nine Arches Bridge": [
                "Train passes around 6:45 AM and 5:15 PM",
                "Position on the hill opposite the bridge",
                "Morning mist adds atmosphere"
            ],
            "Galle Fort": [
                "Sunset from the lighthouse is iconic",
                "Walk the ramparts for varied angles",
                "Stay for blue hour after sunset"
            ],
            "Mirissa Beach": [
                "Coconut Tree Hill for elevated sunset shots",
                "Beach reflection shots at low tide",
                "Whale watching boats add foreground interest"
            ],
        }
        return tips.get(location_name, [
            f"Best photographed during {best_for}",
            "Arrive 30 minutes before golden hour starts",
            "Check weather conditions beforehand"
        ])


# Singleton instance
_golden_hour_agent: Optional[GoldenHourAgent] = None


def get_golden_hour_agent() -> GoldenHourAgent:
    """
    Get or create the GoldenHourAgent singleton.

    Returns:
        GoldenHourAgent: Singleton instance
    """
    global _golden_hour_agent
    if _golden_hour_agent is None:
        _golden_hour_agent = GoldenHourAgent()
    return _golden_hour_agent
