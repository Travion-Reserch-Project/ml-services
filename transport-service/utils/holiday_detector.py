"""
Holiday and Poya Day Detector for Sri Lanka
Uses public APIs to determine:
- Is weekend (Saturday/Sunday)
- Is public holiday
- Is poya day (Buddhist full moon day)
- Time period classification
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
import requests
from enum import Enum


class DayType(Enum):
    """Classification of day types."""
    REGULAR = "regular"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"
    POYA = "poya"


class TimePeriod(Enum):
    """Time period classification."""
    LATE_NIGHT = "late_night"      # 23:00 - 04:59
    EARLY_MORNING = "early_morning"  # 05:00 - 06:59
    MORNING = "morning"              # 07:00 - 09:59
    DAY = "day"                       # 10:00 - 15:59
    EVENING = "evening"              # 16:00 - 18:59
    NIGHT = "night"                   # 19:00 - 22:59


class HolidayDetector:
    """Detect holidays, poya days, and classify days/times for Sri Lanka."""

    # Sri Lankan public holidays (fixed dates)
    SRI_LANKAN_HOLIDAYS = {
        (1, 14): "Thai Pongal",
        (2, 4): "Independence Day",
        (5, 1): "Labour Day",
        (6, 30): "Bank Holiday",
        (8, 15): "Assumption of Mary",
        (10, 31): "Il Festival",
        (12, 25): "Christmas Day",
    }

    # Poya days for 2025 (Buddhist lunar calendar - can be fetched from API)
    # These are full moon dates
    POYA_DAYS_2025 = [
        datetime(2025, 1, 13),   # January
        datetime(2025, 2, 12),   # February
        datetime(2025, 3, 13),   # March
        datetime(2025, 4, 12),   # April
        datetime(2025, 5, 12),   # May
        datetime(2025, 6, 10),   # June
        datetime(2025, 7, 10),   # July
        datetime(2025, 8, 9),    # August
        datetime(2025, 9, 7),    # September
        datetime(2025, 10, 7),   # October
        datetime(2025, 11, 5),   # November
        datetime(2025, 12, 5),   # December
    ]

    # Poya days for 2026
    POYA_DAYS_2026 = [
        datetime(2026, 1, 3),    # January (Duruthu Poya) - SATURDAY
        datetime(2026, 2, 1),    # February (Navam Poya)
        datetime(2026, 3, 3),    # March (Medin Poya)
        datetime(2026, 4, 1),    # April (Bak Poya)
        datetime(2026, 5, 1),    # May (Vesak Poya) - Also Labour Day
        datetime(2026, 5, 30),   # June (Poson Poya)
        datetime(2026, 6, 29),   # July (Esala Poya)
        datetime(2026, 7, 28),   # August (Nikini Poya)
        datetime(2026, 8, 26),   # September (Binara Poya)
        datetime(2026, 9, 25),   # October (Vap Poya)
        datetime(2026, 10, 25),  # November (Il Poya)
        datetime(2026, 11, 23),  # December (Unduvap Poya)
    ]

    def __init__(self, use_api: bool = True):
        """
        Initialize holiday detector.

        Args:
            use_api: If True, try to fetch holidays from API (fallback to hardcoded)
        """
        self.use_api = use_api
        self._poya_cache = {}
        self._holiday_cache = {}
        self._api_holidays_cache = {}  # Cache holidays fetched from API by year
        self.holiday_api_key = os.getenv("HOLIDAY_API_KEY", "")  # Get from environment
        self._api_holidays_cache = {}  # Cache holidays fetched from API by year
        self.holiday_api_key = os.getenv("HOLIDAY_API_KEY", "")  # Get from environment

    def is_weekend(self, date: datetime) -> bool:
        """Check if date is Saturday (5) or Sunday (6)."""
        return date.weekday() >= 5

    def get_poya_day_from_api(self, year: int) -> list:
        """
        Fetch poya days for a year from API.
        Falls back to hardcoded dates if API fails.
        """
        try:
            # Try to use timeanddate API or similar
            # For now, we'll use a simple lunar calendar library
            response = requests.get(
                f"https://api.aladhan.com/v1/gToH?date=01-01-{year}",
                timeout=5
            )
            if response.status_code == 200:
                # Parse lunar calendar response
                return self._parse_lunar_response(response.json())
        except Exception as e:
            print(f"⚠️ Could not fetch poya days from API: {e}")

        # Fallback to hardcoded
        return [d for d in self.POYA_DAYS_2025 if d.year == year]

    def is_poya_day(self, date: datetime) -> bool:
        """
        Check if date is a poya day (Buddhist full moon).
        Uses hardcoded list for efficiency.
        """
        # Normalize to date only
        date_key = date.date()

        if date_key not in self._poya_cache:
            # Check against hardcoded list for all years
            all_poya_days = self.POYA_DAYS_2025 + self.POYA_DAYS_2026
            is_poya = any(d.date() == date_key for d in all_poya_days)
            self._poya_cache[date_key] = is_poya

        return self._poya_cache[date_key]

    def fetch_holidays_from_api(self, year: int) -> dict:
        """
        Fetch holidays from HolidayAPI for a specific year.
        Returns dict of {date: holiday_name}
        """
        if year in self._api_holidays_cache:
            return self._api_holidays_cache[year]

        holidays = {}
        
        if not self.use_api or not self.holiday_api_key:
            print(f"⚠️ HolidayAPI key not set or API disabled. Using hardcoded holidays.")
            return holidays

        try:
            url = "https://holidayapi.com/v1/holidays"
            params = {
                "country": "LK",
                "year": year,
                "key": self.holiday_api_key,
                "pretty": "true"
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if "holidays" in data:
                    for holiday_data in data["holidays"]:
                        holiday_date = datetime.strptime(holiday_data["date"], "%Y-%m-%d")
                        holidays[holiday_date.date()] = holiday_data["name"]
                    
                    print(f"✅ Fetched {len(holidays)} holidays from HolidayAPI for {year}")
                    self._api_holidays_cache[year] = holidays
                else:
                    print(f"⚠️ No holidays found in API response for {year}")
            else:
                print(f"⚠️ HolidayAPI error: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ Could not fetch holidays from API: {e}")
        
        return holidays

    def is_public_holiday(self, date: datetime) -> bool:
        """
        Check if date is a public holiday in Sri Lanka.
        First tries API (if enabled), then falls back to hardcoded dates.
        """
        date_key = (date.month, date.day)

        if date not in self._holiday_cache:
            is_holiday = False
            
            # Try API first
            if self.use_api and self.holiday_api_key:
                api_holidays = self.fetch_holidays_from_api(date.year)
                if date.date() in api_holidays:
                    is_holiday = True
                    self._holiday_cache[date] = is_holiday
                    return is_holiday
            
            # Fallback to hardcoded fixed holidays
            is_holiday = date_key in self.SRI_LANKAN_HOLIDAYS
            self._holiday_cache[date] = is_holiday

        return self._holiday_cache[date]

    def get_day_type(self, date: datetime) -> DayType:
        """
        Classify day as: poya, holiday, weekend, or regular.
        """
        if self.is_poya_day(date):
            return DayType.POYA
        elif self.is_public_holiday(date):
            return DayType.HOLIDAY
        elif self.is_weekend(date):
            return DayType.WEEKEND
        else:
            return DayType.REGULAR

    def get_day_type_str(self, date: datetime) -> str:
        """Get day type as string."""
        return self.get_day_type(date).value

    @staticmethod
    def get_time_period(time_str: str) -> TimePeriod:
        """
        Classify time as: early_morning, morning, day, evening, night, late_night.

        Args:
            time_str: Time in format "HH:MM" (24-hour)

        Returns:
            TimePeriod enum
        """
        try:
            hour = int(time_str.split(":")[0])

            if 5 <= hour < 7:
                return TimePeriod.EARLY_MORNING
            elif 7 <= hour < 10:
                return TimePeriod.MORNING
            elif 10 <= hour < 16:
                return TimePeriod.DAY
            elif 16 <= hour < 19:
                return TimePeriod.EVENING
            elif 19 <= hour < 23:
                return TimePeriod.NIGHT
            else:  # 23:00 - 04:59
                return TimePeriod.LATE_NIGHT

        except (ValueError, IndexError):
            return TimePeriod.DAY  # Default

    @staticmethod
    def get_time_period_str(time_str: str) -> str:
        """Get time period as string."""
        return HolidayDetector.get_time_period(time_str).value

    def get_temporal_features(self, date: datetime, time_str: str) -> Dict[str, any]:
        """
        Get all temporal features for a given date/time.

        Returns:
            {
                'day_type': 'regular' | 'weekend' | 'holiday' | 'poya',
                'is_weekend': bool,
                'is_holiday': bool,
                'is_poya': bool,
                'time_period': 'early_morning' | 'morning' | 'day' | 'evening' | 'night' | 'late_night',
                'hour': int,
                'day_of_week': int (0=Monday, 6=Sunday),
                'is_crowded_likely': bool (heuristic)
            }
        """
        day_type = self.get_day_type(date)
        time_period = self.get_time_period(time_str)
        hour = int(time_str.split(":")[0])

        # Heuristic: likely crowded during morning/evening commute or on holidays
        is_crowded_likely = (
            (time_period in [TimePeriod.MORNING, TimePeriod.EVENING]) or
            (day_type in [DayType.POYA, DayType.HOLIDAY, DayType.WEEKEND])
        )

        return {
            "day_type": day_type.value,
            "is_weekend": day_type in [DayType.WEEKEND, DayType.POYA],
            "is_holiday": day_type in [DayType.HOLIDAY, DayType.POYA],
            "is_poya": day_type == DayType.POYA,
            "time_period": time_period.value,
            "hour": hour,
            "day_of_week": date.weekday(),
            "is_crowded_likely": is_crowded_likely,
        }

    def _parse_lunar_response(self, response_json: dict) -> list:
        """Parse lunar calendar API response (placeholder)."""
        # This would parse the actual API response
        return []
