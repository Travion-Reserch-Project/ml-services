"""
Weather API Tool for Active Guardian Shadow Monitoring
Integrates with OpenWeatherMap for real-time weather forecasting

Part of the "Digital Twin of Itinerary" system that continuously simulates
trip success against environmental variables.
"""

import os
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import pytz
from functools import lru_cache


# ============================================================================
# CONFIGURATION
# ============================================================================

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
SRI_LANKA_TZ = pytz.timezone("Asia/Colombo")


# ============================================================================
# PYDANTIC MODELS (Schema Validation)
# ============================================================================

class WeatherCondition(str, Enum):
    """Weather condition classifications"""
    CLEAR = "clear"
    CLOUDS = "clouds"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    THUNDERSTORM = "thunderstorm"
    SNOW = "snow"
    MIST = "mist"
    FOG = "fog"
    HAZE = "haze"
    DUST = "dust"
    TORNADO = "tornado"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WeatherAlert(BaseModel):
    """Weather alert model"""
    alert_type: str = Field(..., description="Type of weather alert")
    severity: AlertSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description")
    affected_activities: List[str] = Field(default_factory=list, description="Activities affected by this weather")
    recommendation: str = Field(..., description="Recommended action")
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None


class WeatherForecast(BaseModel):
    """Weather forecast for a specific time"""
    datetime_utc: datetime = Field(..., description="Forecast time in UTC")
    datetime_local: str = Field(..., description="Forecast time in Asia/Colombo")
    temperature_celsius: float = Field(..., description="Temperature in Celsius")
    feels_like_celsius: float = Field(..., description="Feels like temperature")
    humidity_percent: int = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed_kmh: float = Field(..., description="Wind speed in km/h")
    wind_gust_kmh: Optional[float] = Field(None, description="Wind gust speed")
    rain_probability: float = Field(..., ge=0, le=100, description="Probability of rain")
    rain_volume_mm: float = Field(default=0, description="Expected rain in mm")
    cloud_coverage: int = Field(..., ge=0, le=100, description="Cloud coverage percentage")
    condition: WeatherCondition = Field(..., description="Primary weather condition")
    condition_description: str = Field(..., description="Detailed condition description")
    visibility_km: float = Field(..., description="Visibility in kilometers")
    uv_index: Optional[float] = Field(None, description="UV index")
    is_suitable_outdoor: bool = Field(..., description="Whether suitable for outdoor activities")
    alerts: List[WeatherAlert] = Field(default_factory=list, description="Active weather alerts")


class LocationWeatherReport(BaseModel):
    """Complete weather report for a location"""
    location_name: str = Field(..., description="Name of the location")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timezone: str = Field(default="Asia/Colombo")
    current_weather: Optional[WeatherForecast] = None
    forecasts: List[WeatherForecast] = Field(default_factory=list)
    daily_summary: Dict[str, Any] = Field(default_factory=dict)
    trip_suitability_score: float = Field(..., ge=0, le=100, description="Overall suitability for trip")
    critical_alerts: List[WeatherAlert] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class WeatherValidationResult(BaseModel):
    """Result of weather validation for an itinerary"""
    is_valid: bool = Field(..., description="Whether the itinerary passes weather validation")
    overall_risk_level: AlertSeverity = Field(..., description="Overall risk level")
    score: float = Field(..., ge=0, le=100, description="Weather suitability score")
    location_reports: List[LocationWeatherReport] = Field(default_factory=list)
    blocking_issues: List[str] = Field(default_factory=list, description="Issues that block the plan")
    warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings")
    recommendations: List[str] = Field(default_factory=list)


# ============================================================================
# ACTIVITY WEATHER REQUIREMENTS
# ============================================================================

ACTIVITY_WEATHER_REQUIREMENTS = {
    "beach": {
        "max_rain_probability": 30,
        "min_temperature": 24,
        "max_temperature": 35,
        "max_wind_speed": 40,
        "unsuitable_conditions": [WeatherCondition.RAIN, WeatherCondition.THUNDERSTORM, WeatherCondition.FOG],
    },
    "hiking": {
        "max_rain_probability": 40,
        "min_temperature": 18,
        "max_temperature": 32,
        "max_wind_speed": 50,
        "unsuitable_conditions": [WeatherCondition.THUNDERSTORM, WeatherCondition.FOG, WeatherCondition.TORNADO],
    },
    "photography": {
        "max_rain_probability": 20,
        "min_visibility": 5,
        "unsuitable_conditions": [WeatherCondition.RAIN, WeatherCondition.FOG, WeatherCondition.MIST],
    },
    "temple_visit": {
        "max_rain_probability": 60,  # Can visit temples in light rain
        "max_temperature": 38,
        "unsuitable_conditions": [WeatherCondition.THUNDERSTORM],
    },
    "wildlife_safari": {
        "max_rain_probability": 50,
        "min_temperature": 20,
        "max_temperature": 36,
        "unsuitable_conditions": [WeatherCondition.THUNDERSTORM, WeatherCondition.FOG],
    },
    "water_sports": {
        "max_rain_probability": 20,
        "min_temperature": 25,
        "max_wind_speed": 30,
        "unsuitable_conditions": [WeatherCondition.RAIN, WeatherCondition.THUNDERSTORM],
    },
    "city_tour": {
        "max_rain_probability": 70,
        "max_temperature": 36,
        "unsuitable_conditions": [WeatherCondition.THUNDERSTORM, WeatherCondition.TORNADO],
    },
    "outdoor_dining": {
        "max_rain_probability": 30,
        "min_temperature": 22,
        "max_temperature": 34,
        "unsuitable_conditions": [WeatherCondition.RAIN, WeatherCondition.THUNDERSTORM],
    },
}


# ============================================================================
# WEATHER API CLIENT
# ============================================================================

class WeatherAPIClient:
    """Client for OpenWeatherMap API with caching and error handling"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.base_url = OPENWEATHER_BASE_URL
        self.timeout = 10.0  # seconds
        # Don't raise error here - check at request time instead

    def is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with error handling"""
        if not self.api_key:
            raise ValueError("OpenWeatherMap API key not configured. Set OPENWEATHER_API_KEY env var.")

        params["appid"] = self.api_key
        params["units"] = "metric"  # Use Celsius

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(f"{self.base_url}/{endpoint}", params=params)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    raise ValueError("Invalid OpenWeatherMap API key")
                elif e.response.status_code == 404:
                    raise ValueError(f"Location not found: {params}")
                else:
                    raise RuntimeError(f"Weather API error: {e.response.status_code}")
            except httpx.TimeoutException:
                raise RuntimeError("Weather API request timed out")
            except Exception as e:
                raise RuntimeError(f"Weather API request failed: {str(e)}")

    async def get_current_weather(self, lat: float, lng: float) -> Dict[str, Any]:
        """Get current weather for coordinates"""
        return await self._make_request("weather", {"lat": lat, "lon": lng})

    async def get_5day_forecast(self, lat: float, lng: float) -> Dict[str, Any]:
        """Get 5-day/3-hour forecast for coordinates"""
        return await self._make_request("forecast", {"lat": lat, "lon": lng})

    async def get_onecall(self, lat: float, lng: float, exclude: str = "") -> Dict[str, Any]:
        """
        Get One Call API data (requires paid plan for some features)
        Includes: current, minutely, hourly, daily, alerts
        """
        params = {"lat": lat, "lon": lng}
        if exclude:
            params["exclude"] = exclude
        return await self._make_request("onecall", params)


# ============================================================================
# WEATHER TOOL (LangGraph Compatible)
# ============================================================================

class WeatherTool:
    """
    Weather validation tool for the Active Guardian system.
    Integrates with LangGraph nodes for pre-trip validation.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.client = WeatherAPIClient(api_key)

    def is_configured(self) -> bool:
        """Check if weather API is configured"""
        return self.client.is_configured()

    def _parse_condition(self, weather_main: str) -> WeatherCondition:
        """Parse OpenWeatherMap condition to our enum"""
        condition_map = {
            "clear": WeatherCondition.CLEAR,
            "clouds": WeatherCondition.CLOUDS,
            "rain": WeatherCondition.RAIN,
            "drizzle": WeatherCondition.DRIZZLE,
            "thunderstorm": WeatherCondition.THUNDERSTORM,
            "snow": WeatherCondition.SNOW,
            "mist": WeatherCondition.MIST,
            "fog": WeatherCondition.FOG,
            "haze": WeatherCondition.HAZE,
            "dust": WeatherCondition.DUST,
            "tornado": WeatherCondition.TORNADO,
        }
        return condition_map.get(weather_main.lower(), WeatherCondition.CLOUDS)

    def _generate_alerts(
        self,
        forecast: Dict[str, Any],
        activity_type: Optional[str] = None
    ) -> List[WeatherAlert]:
        """Generate weather alerts based on forecast data"""
        alerts = []

        # Extract weather data
        rain_prob = forecast.get("pop", 0) * 100  # Convert to percentage
        wind_speed = forecast.get("wind", {}).get("speed", 0) * 3.6  # m/s to km/h
        wind_gust = forecast.get("wind", {}).get("gust", 0) * 3.6
        temp = forecast.get("main", {}).get("temp", 25)
        weather_main = forecast.get("weather", [{}])[0].get("main", "Clear")

        # Heavy rain alert
        if rain_prob > 80:
            alerts.append(WeatherAlert(
                alert_type="HEAVY_RAIN",
                severity=AlertSeverity.HIGH,
                description=f"High probability of rain ({rain_prob:.0f}%)",
                affected_activities=["beach", "hiking", "photography", "outdoor_dining", "water_sports"],
                recommendation="Consider indoor alternatives or reschedule outdoor activities"
            ))
        elif rain_prob > 50:
            alerts.append(WeatherAlert(
                alert_type="RAIN_POSSIBLE",
                severity=AlertSeverity.MEDIUM,
                description=f"Moderate chance of rain ({rain_prob:.0f}%)",
                affected_activities=["beach", "photography"],
                recommendation="Carry rain gear and have backup plans"
            ))

        # High wind alert
        if wind_speed > 50 or wind_gust > 60:
            alerts.append(WeatherAlert(
                alert_type="HIGH_WIND",
                severity=AlertSeverity.HIGH,
                description=f"Strong winds expected ({wind_speed:.0f} km/h, gusts {wind_gust:.0f} km/h)",
                affected_activities=["beach", "water_sports", "hiking", "photography"],
                recommendation="Avoid exposed areas and water activities"
            ))
        elif wind_speed > 30:
            alerts.append(WeatherAlert(
                alert_type="WINDY",
                severity=AlertSeverity.LOW,
                description=f"Moderately windy ({wind_speed:.0f} km/h)",
                affected_activities=["photography", "water_sports"],
                recommendation="Be cautious with photography equipment"
            ))

        # Extreme temperature alerts
        if temp > 35:
            alerts.append(WeatherAlert(
                alert_type="EXTREME_HEAT",
                severity=AlertSeverity.HIGH,
                description=f"Very hot conditions ({temp:.1f}C)",
                affected_activities=["hiking", "city_tour"],
                recommendation="Stay hydrated, avoid midday sun, seek shade frequently"
            ))
        elif temp > 32:
            alerts.append(WeatherAlert(
                alert_type="HOT",
                severity=AlertSeverity.MEDIUM,
                description=f"Hot conditions ({temp:.1f}C)",
                affected_activities=["hiking"],
                recommendation="Start activities early morning, carry water"
            ))

        # Thunderstorm alert
        if weather_main.lower() == "thunderstorm":
            alerts.append(WeatherAlert(
                alert_type="THUNDERSTORM",
                severity=AlertSeverity.CRITICAL,
                description="Thunderstorm conditions expected",
                affected_activities=["all_outdoor"],
                recommendation="AVOID all outdoor activities. Seek shelter immediately."
            ))

        # Poor visibility
        visibility = forecast.get("visibility", 10000) / 1000  # meters to km
        if visibility < 1:
            alerts.append(WeatherAlert(
                alert_type="POOR_VISIBILITY",
                severity=AlertSeverity.HIGH,
                description=f"Very poor visibility ({visibility:.1f} km)",
                affected_activities=["driving", "photography", "wildlife_safari"],
                recommendation="Exercise extreme caution while driving"
            ))
        elif visibility < 3:
            alerts.append(WeatherAlert(
                alert_type="REDUCED_VISIBILITY",
                severity=AlertSeverity.MEDIUM,
                description=f"Reduced visibility ({visibility:.1f} km)",
                affected_activities=["photography"],
                recommendation="Expect reduced scenic views"
            ))

        return alerts

    def _is_suitable_for_outdoor(
        self,
        forecast: Dict[str, Any],
        activity_type: Optional[str] = None
    ) -> bool:
        """Determine if weather is suitable for outdoor activities"""
        rain_prob = forecast.get("pop", 0) * 100
        wind_speed = forecast.get("wind", {}).get("speed", 0) * 3.6
        weather_main = forecast.get("weather", [{}])[0].get("main", "Clear").lower()
        temp = forecast.get("main", {}).get("temp", 25)

        # Get activity-specific requirements
        if activity_type and activity_type in ACTIVITY_WEATHER_REQUIREMENTS:
            reqs = ACTIVITY_WEATHER_REQUIREMENTS[activity_type]

            if rain_prob > reqs.get("max_rain_probability", 50):
                return False
            if wind_speed > reqs.get("max_wind_speed", 60):
                return False
            if temp < reqs.get("min_temperature", 15) or temp > reqs.get("max_temperature", 40):
                return False

            condition = self._parse_condition(weather_main)
            if condition in reqs.get("unsuitable_conditions", []):
                return False
        else:
            # General outdoor suitability
            if rain_prob > 60 or wind_speed > 50 or weather_main in ["thunderstorm", "tornado"]:
                return False

        return True

    def _parse_forecast_item(
        self,
        item: Dict[str, Any],
        activity_type: Optional[str] = None
    ) -> WeatherForecast:
        """Parse a single forecast item from API response"""
        dt_utc = datetime.utcfromtimestamp(item["dt"])
        dt_local = dt_utc.replace(tzinfo=pytz.UTC).astimezone(SRI_LANKA_TZ)

        weather_data = item.get("weather", [{}])[0]
        main_data = item.get("main", {})
        wind_data = item.get("wind", {})

        alerts = self._generate_alerts(item, activity_type)

        return WeatherForecast(
            datetime_utc=dt_utc,
            datetime_local=dt_local.strftime("%Y-%m-%d %H:%M %Z"),
            temperature_celsius=main_data.get("temp", 25),
            feels_like_celsius=main_data.get("feels_like", 25),
            humidity_percent=main_data.get("humidity", 70),
            wind_speed_kmh=wind_data.get("speed", 0) * 3.6,
            wind_gust_kmh=wind_data.get("gust", 0) * 3.6 if wind_data.get("gust") else None,
            rain_probability=item.get("pop", 0) * 100,
            rain_volume_mm=item.get("rain", {}).get("3h", 0),
            cloud_coverage=item.get("clouds", {}).get("all", 0),
            condition=self._parse_condition(weather_data.get("main", "Clear")),
            condition_description=weather_data.get("description", "clear sky"),
            visibility_km=item.get("visibility", 10000) / 1000,
            is_suitable_outdoor=self._is_suitable_for_outdoor(item, activity_type),
            alerts=alerts
        )

    async def get_location_weather_report(
        self,
        location_name: str,
        latitude: float,
        longitude: float,
        target_date: Optional[datetime] = None,
        activity_type: Optional[str] = None
    ) -> LocationWeatherReport:
        """
        Get comprehensive weather report for a location.

        Args:
            location_name: Human-readable location name
            latitude: Location latitude
            longitude: Location longitude
            target_date: Target date for the visit (filters forecasts)
            activity_type: Type of activity planned (affects suitability scoring)

        Returns:
            LocationWeatherReport with forecasts and alerts
        """
        try:
            # Get 5-day forecast
            forecast_data = await self.client.get_5day_forecast(latitude, longitude)

            forecasts = []
            critical_alerts = []

            for item in forecast_data.get("list", []):
                forecast = self._parse_forecast_item(item, activity_type)

                # Filter by target date if provided
                if target_date:
                    forecast_date = forecast.datetime_utc.date()
                    target_dates = [
                        target_date.date(),
                        (target_date + timedelta(days=1)).date(),
                        (target_date - timedelta(days=1)).date()
                    ]
                    if forecast_date not in target_dates:
                        continue

                forecasts.append(forecast)

                # Collect critical alerts
                for alert in forecast.alerts:
                    if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                        critical_alerts.append(alert)

            # Calculate trip suitability score
            if forecasts:
                suitable_count = sum(1 for f in forecasts if f.is_suitable_outdoor)
                suitability_score = (suitable_count / len(forecasts)) * 100

                # Penalty for critical alerts
                if critical_alerts:
                    suitability_score *= 0.7  # 30% penalty for having critical alerts
            else:
                suitability_score = 50.0  # Default if no forecasts

            # Generate daily summary
            daily_summary = {}
            for forecast in forecasts:
                date_key = forecast.datetime_utc.strftime("%Y-%m-%d")
                if date_key not in daily_summary:
                    daily_summary[date_key] = {
                        "min_temp": forecast.temperature_celsius,
                        "max_temp": forecast.temperature_celsius,
                        "max_rain_prob": forecast.rain_probability,
                        "conditions": []
                    }
                else:
                    daily_summary[date_key]["min_temp"] = min(
                        daily_summary[date_key]["min_temp"],
                        forecast.temperature_celsius
                    )
                    daily_summary[date_key]["max_temp"] = max(
                        daily_summary[date_key]["max_temp"],
                        forecast.temperature_celsius
                    )
                    daily_summary[date_key]["max_rain_prob"] = max(
                        daily_summary[date_key]["max_rain_prob"],
                        forecast.rain_probability
                    )

                if forecast.condition.value not in daily_summary[date_key]["conditions"]:
                    daily_summary[date_key]["conditions"].append(forecast.condition.value)

            return LocationWeatherReport(
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                timezone="Asia/Colombo",
                forecasts=forecasts,
                daily_summary=daily_summary,
                trip_suitability_score=suitability_score,
                critical_alerts=critical_alerts,
                generated_at=datetime.utcnow()
            )

        except Exception as e:
            # Return a report with error status
            return LocationWeatherReport(
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                timezone="Asia/Colombo",
                forecasts=[],
                daily_summary={"error": str(e)},
                trip_suitability_score=0,
                critical_alerts=[
                    WeatherAlert(
                        alert_type="API_ERROR",
                        severity=AlertSeverity.HIGH,
                        description=f"Failed to fetch weather data: {str(e)}",
                        recommendation="Check API configuration or try again later"
                    )
                ],
                generated_at=datetime.utcnow()
            )

    async def validate_itinerary_weather(
        self,
        itinerary_items: List[Dict[str, Any]],
        trip_date: datetime
    ) -> WeatherValidationResult:
        """
        Validate an entire itinerary against weather forecasts.

        This is the main method used by the Shadow Monitor node.

        Args:
            itinerary_items: List of itinerary items with location info
            trip_date: The date of the trip

        Returns:
            WeatherValidationResult with validation status and recommendations
        """
        location_reports = []
        blocking_issues = []
        warnings = []
        recommendations = []

        for item in itinerary_items:
            # Skip items without coordinates
            if not item.get("latitude") or not item.get("longitude"):
                warnings.append(f"No coordinates for {item.get('locationName', 'Unknown')} - skipping weather check")
                continue

            # Determine activity type from activity description
            activity = item.get("activity", "").lower()
            activity_type = None
            for act_type in ACTIVITY_WEATHER_REQUIREMENTS.keys():
                if act_type in activity:
                    activity_type = act_type
                    break

            # Get weather report for this location
            report = await self.get_location_weather_report(
                location_name=item.get("locationName", "Unknown"),
                latitude=item["latitude"],
                longitude=item["longitude"],
                target_date=trip_date,
                activity_type=activity_type
            )

            location_reports.append(report)

            # Check for blocking issues
            for alert in report.critical_alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    blocking_issues.append(
                        f"{report.location_name}: {alert.description}"
                    )
                elif alert.severity == AlertSeverity.HIGH:
                    # Check if this affects the planned activity
                    if activity_type in alert.affected_activities or "all_outdoor" in alert.affected_activities:
                        blocking_issues.append(
                            f"{report.location_name} ({activity}): {alert.description}"
                        )
                    else:
                        warnings.append(
                            f"{report.location_name}: {alert.description}"
                        )

            # Generate recommendations
            if report.trip_suitability_score < 50:
                recommendations.append(
                    f"Consider rescheduling {report.location_name} - weather suitability only {report.trip_suitability_score:.0f}%"
                )
            elif report.trip_suitability_score < 70:
                recommendations.append(
                    f"Have backup plans for {report.location_name} - weather somewhat uncertain"
                )

        # Calculate overall scores
        if location_reports:
            avg_score = sum(r.trip_suitability_score for r in location_reports) / len(location_reports)
        else:
            avg_score = 100  # No locations to check = pass

        # Determine overall risk level
        if blocking_issues:
            risk_level = AlertSeverity.CRITICAL
        elif avg_score < 50:
            risk_level = AlertSeverity.HIGH
        elif avg_score < 70 or warnings:
            risk_level = AlertSeverity.MEDIUM
        else:
            risk_level = AlertSeverity.LOW

        return WeatherValidationResult(
            is_valid=len(blocking_issues) == 0,
            overall_risk_level=risk_level,
            score=avg_score,
            location_reports=location_reports,
            blocking_issues=blocking_issues,
            warnings=warnings,
            recommendations=recommendations
        )


# ============================================================================
# CONVENIENCE FUNCTIONS FOR LANGGRAPH
# ============================================================================

async def check_weather_for_trip(
    itinerary: List[Dict[str, Any]],
    trip_date: datetime,
    api_key: Optional[str] = None
) -> WeatherValidationResult:
    """
    Convenience function for LangGraph nodes.

    Usage in LangGraph:
        from app.tools.weather_api import check_weather_for_trip

        result = await check_weather_for_trip(itinerary, trip_date)
        if not result.is_valid:
            return {"status": "REJECT", "reason": result.blocking_issues}
    """
    tool = WeatherTool(api_key)
    return await tool.validate_itinerary_weather(itinerary, trip_date)


async def get_weather_alerts_for_location(
    location_name: str,
    latitude: float,
    longitude: float,
    activity_type: Optional[str] = None,
    api_key: Optional[str] = None
) -> List[WeatherAlert]:
    """
    Get weather alerts for a specific location.
    Used by the Active Watcher for real-time monitoring.
    """
    tool = WeatherTool(api_key)
    report = await tool.get_location_weather_report(
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        activity_type=activity_type
    )
    return report.critical_alerts
