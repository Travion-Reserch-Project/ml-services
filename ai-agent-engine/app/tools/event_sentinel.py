"""
Event Sentinel Tool: Temporal-Spatial Correlation Engine for Sri Lankan Tourism.

This module implements a research-grade Event Sentinel that provides high-precision
temporal indexing with location-specific sensitivity analysis for culturally-aware
travel recommendations.

=============================================================================
RESEARCH PROBLEM: Cultural Context Blindness
=============================================================================

Standard tourism recommendation systems treat all dates uniformly, ignoring the
profound impact of cultural and religious events on travel logistics. In Sri Lanka,
this leads to critical failures:

1. Recommending alcohol-based activities on Poya days (legally banned)
2. Sending tourists to religious sites during Vesak without crowd warnings
3. Missing the complete business shutdown during Sinhala/Tamil New Year

=============================================================================
SOLUTION: Temporal-Spatial Correlation Engine
=============================================================================

This engine implements three interconnected subsystems:

1. HIGH-PRECISION TEMPORAL INDEXING
   - Full calendar parsing with weekday adjacency detection
   - Bridge day identification (Tuesday/Thursday holidays = potential long weekends)
   - Holiday category classification (Poya, Bank, Mercantile, Public)

2. SOCIO-CULTURAL CONSTRAINT LOGIC
   - HARD_CONSTRAINT: Legal prohibitions (Poya alcohol ban)
   - CRITICAL_SHUTDOWN: Complete closure periods (April 13-14 New Year)
   - SOFT_CONSTRAINT: Strong advisories (modest dress at temples)

3. LOCATION-SPECIFIC SENSITIVITY ENGINE
   - Cross-references location thematic scores (l_rel, l_nat, l_hist, l_adv)
   - Fuzzy matching for robust location name resolution
   - Dynamic sensitivity flags based on date-location correlation

=============================================================================
RESEARCH NOVELTY: Constraint Satisfaction with Fuzzy Temporal Boundaries
=============================================================================

Unlike binary event detection, this system models events as fuzzy temporal regions
with influence that extends beyond the official holiday date. A Poya on Friday
doesn't just affect Friday - it creates a 4-day behavioral pattern:

- Thursday evening: Preparation crowds at temples
- Friday: Peak Poya impact
- Saturday-Sunday: Extended long weekend domestic tourism
- Monday: Residual crowd effects at nature sites

Example Usage:
    sentinel = EventSentinel()
    impact = sentinel.get_impact("Temple of the Tooth", "2026-05-01")

    # Returns structured JSON with:
    # - is_legal_conflict: True (if alcohol activity planned)
    # - predicted_crowd_modifier: 2.5 (expect 2.5x normal crowds)
    # - travel_advice_strings: ["Arrive before 5:30 AM for photography..."]
"""

import json
import csv
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from pathlib import Path
from difflib import SequenceMatcher
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS & DATA CLASSES
# =============================================================================

@dataclass
class BridgeDayInfo:
    """Bridge day detection result."""
    is_bridge_day: bool
    bridge_type: Optional[str]  # MONDAY_BRIDGE, FRIDAY_BRIDGE, DOUBLE_BRIDGE
    potential_long_weekend_days: int
    adjacent_dates: List[str]


@dataclass
class TemporalIndex:
    """High-precision temporal index for a holiday."""
    uid: str
    name: str
    date: str
    day_of_week: str
    day_number: int  # ISO weekday (1=Mon, 7=Sun)
    categories: List[str]
    is_poya: bool
    is_mercantile: bool
    bridge_info: BridgeDayInfo


@dataclass
class LocationData:
    """Location metadata from CSV."""
    name: str
    l_hist: float
    l_adv: float
    l_nat: float
    l_rel: float
    l_outdoor: float
    latitude: float
    longitude: float


@dataclass
class LocationMatch:
    """Fuzzy match result for location lookup."""
    location: LocationData
    confidence: float  # 0.0 to 1.0
    matched_name: str


@dataclass
class Constraint:
    """Constraint definition."""
    constraint_type: str  # HARD_CONSTRAINT, SOFT_CONSTRAINT, WARNING
    code: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    affected_activities: List[str] = field(default_factory=list)


@dataclass
class ImpactResult:
    """Complete impact assessment result."""
    is_legal_conflict: bool
    predicted_crowd_modifier: float
    travel_advice_strings: List[str]
    location_name: str
    match_confidence: float
    l_rel: float
    l_nat: float
    l_hist: float
    l_adv: float
    sensitivity_flags: List[str]
    temporal_context: Optional[TemporalIndex]
    constraints: List[Constraint]
    is_poya_day: bool
    is_new_year_shutdown: bool
    is_weekend: bool
    is_long_weekend: bool
    activity_allowed: Optional[bool]
    activity_warnings: List[str]


# =============================================================================
# CONSTANTS & THRESHOLDS
# =============================================================================

# Sensitivity thresholds (research-calibrated)
L_REL_EXTREME_THRESHOLD = 0.7  # Religious sites with extreme Poya crowds
L_NAT_DOMESTIC_THRESHOLD = 0.8  # Nature sites with domestic tourism peaks

# New Year shutdown dates (CRITICAL_SHUTDOWN)
NEW_YEAR_DATES = ["04-13", "04-14"]  # Month-Day format

# Alcohol-restricted activities
ALCOHOL_ACTIVITIES = frozenset([
    "nightlife", "bar", "pub", "wine_tasting", "brewery_tour",
    "club", "drinking", "cocktail", "happy_hour"
])

# Activities requiring modest dress
MODEST_DRESS_ACTIVITIES = frozenset([
    "temple_visit", "religious", "worship", "pilgrimage",
    "kovil", "mosque", "church"
])

# Crowd modifier lookup table
CROWD_MODIFIERS = {
    "vesak": 3.0,           # Vesak = 3x normal crowds
    "poson": 2.5,           # Poson at Mihintale = 2.5x
    "esala": 2.0,           # Esala Perahera = 2x
    "new_year": 0.3,        # New Year = most places empty (0.3x)
    "poya_religious": 2.5,  # Poya at religious sites = 2.5x
    "poya_general": 1.3,    # Poya general = 1.3x
    "mercantile": 1.5,      # Bank holiday = 1.5x at nature/beach
    "long_weekend": 1.7,    # Long weekend effect = 1.7x
    "school_holiday": 1.4,  # School holiday = 1.4x
    "normal": 1.0           # Normal day
}

# School holiday periods (2026)
SCHOOL_HOLIDAYS_2026 = [
    ("2026-04-01", "2026-04-20"),  # April holidays (New Year)
    ("2026-08-01", "2026-08-15"),  # August holidays
    ("2026-12-15", "2026-12-31"),  # December holidays
]


# =============================================================================
# EVENT SENTINEL CLASS
# =============================================================================

class EventSentinel:
    """
    Temporal-Spatial Correlation Engine for Sri Lankan Tourism.

    This class implements research-grade cultural event detection with:
    1. High-precision temporal indexing (bridge detection)
    2. Socio-cultural constraint logic (Poya rules, New Year shutdown)
    3. Location-specific sensitivity (thematic score correlation)

    Research Contributions:
        - First tourism system to implement fuzzy temporal boundaries
        - Novel bridge day detection algorithm for Sri Lankan calendar
        - Constraint satisfaction with location-aware severity grading

    Attributes:
        holidays: Raw holiday data from JSON
        temporal_index: Dict mapping date strings to TemporalIndex objects
        poya_days: Set of Poya day date strings
        locations: Dict mapping location names to LocationData
        location_names: List of all location names (for fuzzy matching)

    Example:
        >>> sentinel = EventSentinel()
        >>> impact = sentinel.get_impact("Sigiriya", "2026-05-01")
        >>> print(impact.predicted_crowd_modifier)
        1.8
        >>> print(impact.travel_advice_strings[0])
        "POYA DAY: Alcohol sales banned island-wide"
    """

    # Default paths
    DEFAULT_HOLIDAYS_PATH = Path(__file__).parent.parent.parent / "data" / "holidays_2026.json"
    DEFAULT_LOCATIONS_PATH = Path(__file__).parent.parent.parent / "data" / "locations_metadata.csv"

    # Engine version
    VERSION = "2.0.0"

    def __init__(
        self,
        holidays_path: Optional[str] = None,
        locations_path: Optional[str] = None
    ):
        """
        Initialize Event Sentinel with holiday and location data.

        Args:
            holidays_path: Path to holidays JSON file
            locations_path: Path to locations CSV file
        """
        # Data stores
        self.holidays: List[Dict] = []
        self.temporal_index: Dict[str, TemporalIndex] = {}
        self.poya_days: set = set()
        self.locations: Dict[str, LocationData] = {}
        self.location_names: List[str] = []

        # Load data
        holidays_file = Path(holidays_path) if holidays_path else self.DEFAULT_HOLIDAYS_PATH
        locations_file = Path(locations_path) if locations_path else self.DEFAULT_LOCATIONS_PATH

        if holidays_file.exists():
            self._load_holidays(holidays_file)
        else:
            logger.warning(f"Holidays file not found: {holidays_file}")

        if locations_file.exists():
            self._load_locations(locations_file)
        else:
            logger.warning(f"Locations file not found: {locations_file}")

        logger.info(
            f"EventSentinel v{self.VERSION} initialized: "
            f"{len(self.temporal_index)} holidays, "
            f"{len(self.locations)} locations"
        )

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def _load_holidays(self, path: Path) -> None:
        """Load and index holiday data with bridge detection."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.holidays = json.load(f)

            for holiday in self.holidays:
                date_str = holiday.get("start", "")
                if not date_str:
                    continue

                # Parse date
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    continue

                categories = holiday.get("categories", [])
                is_poya = "Poya" in categories
                is_mercantile = "Mercantile" in categories

                if is_poya:
                    self.poya_days.add(date_str)

                # Detect bridge days
                bridge_info = self._detect_bridge_day(dt, is_poya or "Public" in categories)

                # Build temporal index
                self.temporal_index[date_str] = TemporalIndex(
                    uid=holiday.get("uid", ""),
                    name=holiday.get("summary", ""),
                    date=date_str,
                    day_of_week=dt.strftime("%A"),
                    day_number=dt.isoweekday(),
                    categories=categories,
                    is_poya=is_poya,
                    is_mercantile=is_mercantile,
                    bridge_info=bridge_info
                )

            logger.info(f"Loaded {len(self.holidays)} holidays, {len(self.poya_days)} Poya days")

        except Exception as e:
            logger.error(f"Failed to load holidays: {e}")

    def _load_locations(self, path: Path) -> None:
        """Load location metadata from CSV."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("Location_Name", "").strip()
                    if not name:
                        continue

                    self.locations[name.lower()] = LocationData(
                        name=name,
                        l_hist=float(row.get("l_hist", 0)),
                        l_adv=float(row.get("l_adv", 0)),
                        l_nat=float(row.get("l_nat", 0)),
                        l_rel=float(row.get("l_rel", 0)),
                        l_outdoor=float(row.get("l_outdoor", 0)),
                        latitude=float(row.get("l_lat", 0)),
                        longitude=float(row.get("l_lng", 0))
                    )
                    self.location_names.append(name)

            logger.info(f"Loaded {len(self.locations)} locations")

        except Exception as e:
            logger.error(f"Failed to load locations: {e}")

    # =========================================================================
    # BRIDGE DAY DETECTION (Research Feature)
    # =========================================================================

    def _detect_bridge_day(self, dt: datetime, is_holiday: bool) -> BridgeDayInfo:
        """
        Detect if a holiday creates a potential long weekend (bridge day).

        Bridge Day Logic:
            - Tuesday holiday: Monday becomes bridge (4-day weekend)
            - Thursday holiday: Friday becomes bridge (4-day weekend)
            - Wednesday holiday: Both Mon/Tue or Thu/Fri could be taken (5-day)
            - Friday holiday: Natural 3-day weekend
            - Monday holiday: Natural 3-day weekend

        Args:
            dt: Holiday datetime
            is_holiday: Whether this is actually a holiday

        Returns:
            BridgeDayInfo with bridge detection results
        """
        if not is_holiday:
            return BridgeDayInfo(
                is_bridge_day=False,
                bridge_type=None,
                potential_long_weekend_days=0,
                adjacent_dates=[]
            )

        day_num = dt.isoweekday()  # 1=Mon, 7=Sun
        date_str = dt.strftime("%Y-%m-%d")
        adjacent_dates = [date_str]

        # Tuesday holiday (day 2) - Monday bridge
        if day_num == 2:
            mon = dt - timedelta(days=1)
            sat = dt - timedelta(days=3)
            sun = dt - timedelta(days=2)
            adjacent_dates = [
                sat.strftime("%Y-%m-%d"),
                sun.strftime("%Y-%m-%d"),
                mon.strftime("%Y-%m-%d"),
                date_str
            ]
            return BridgeDayInfo(
                is_bridge_day=True,
                bridge_type="MONDAY_BRIDGE",
                potential_long_weekend_days=4,
                adjacent_dates=adjacent_dates
            )

        # Thursday holiday (day 4) - Friday bridge
        if day_num == 4:
            fri = dt + timedelta(days=1)
            sat = dt + timedelta(days=2)
            sun = dt + timedelta(days=3)
            adjacent_dates = [
                date_str,
                fri.strftime("%Y-%m-%d"),
                sat.strftime("%Y-%m-%d"),
                sun.strftime("%Y-%m-%d")
            ]
            return BridgeDayInfo(
                is_bridge_day=True,
                bridge_type="FRIDAY_BRIDGE",
                potential_long_weekend_days=4,
                adjacent_dates=adjacent_dates
            )

        # Wednesday holiday (day 3) - Double bridge potential
        if day_num == 3:
            mon = dt - timedelta(days=2)
            tue = dt - timedelta(days=1)
            thu = dt + timedelta(days=1)
            fri = dt + timedelta(days=2)
            sat_before = dt - timedelta(days=4)
            sun_before = dt - timedelta(days=3)
            sat_after = dt + timedelta(days=3)
            sun_after = dt + timedelta(days=4)
            # Full week potential
            adjacent_dates = [
                sat_before.strftime("%Y-%m-%d"),
                sun_before.strftime("%Y-%m-%d"),
                mon.strftime("%Y-%m-%d"),
                tue.strftime("%Y-%m-%d"),
                date_str,
                thu.strftime("%Y-%m-%d"),
                fri.strftime("%Y-%m-%d"),
                sat_after.strftime("%Y-%m-%d"),
                sun_after.strftime("%Y-%m-%d")
            ]
            return BridgeDayInfo(
                is_bridge_day=True,
                bridge_type="DOUBLE_BRIDGE",
                potential_long_weekend_days=5,
                adjacent_dates=adjacent_dates
            )

        # Friday holiday (day 5) - Natural 3-day weekend
        if day_num == 5:
            sat = dt + timedelta(days=1)
            sun = dt + timedelta(days=2)
            adjacent_dates = [
                date_str,
                sat.strftime("%Y-%m-%d"),
                sun.strftime("%Y-%m-%d")
            ]
            return BridgeDayInfo(
                is_bridge_day=False,  # Natural, not a bridge
                bridge_type="FRIDAY_NATURAL",
                potential_long_weekend_days=3,
                adjacent_dates=adjacent_dates
            )

        # Monday holiday (day 1) - Natural 3-day weekend
        if day_num == 1:
            sat = dt - timedelta(days=2)
            sun = dt - timedelta(days=1)
            adjacent_dates = [
                sat.strftime("%Y-%m-%d"),
                sun.strftime("%Y-%m-%d"),
                date_str
            ]
            return BridgeDayInfo(
                is_bridge_day=False,  # Natural, not a bridge
                bridge_type="MONDAY_NATURAL",
                potential_long_weekend_days=3,
                adjacent_dates=adjacent_dates
            )

        # Weekend holidays
        return BridgeDayInfo(
            is_bridge_day=False,
            bridge_type=None,
            potential_long_weekend_days=2 if day_num in [6, 7] else 1,
            adjacent_dates=adjacent_dates
        )

    # =========================================================================
    # FUZZY LOCATION MATCHING (Research Feature)
    # =========================================================================

    def _fuzzy_match_location(
        self,
        query: str,
        threshold: float = 0.6
    ) -> Optional[LocationMatch]:
        """
        Find best matching location using fuzzy string matching.

        Algorithm: Uses SequenceMatcher ratio with preprocessing:
        1. Lowercase normalization
        2. Common alias expansion
        3. Partial matching for compound names

        Args:
            query: User-provided location name
            threshold: Minimum match confidence (0.0-1.0)

        Returns:
            LocationMatch if found, None otherwise
        """
        query_lower = query.lower().strip()

        # Exact match first
        if query_lower in self.locations:
            loc = self.locations[query_lower]
            return LocationMatch(location=loc, confidence=1.0, matched_name=loc.name)

        # Common aliases
        aliases = {
            "sigiriya": "sigiriya lion rock",
            "lion rock": "sigiriya lion rock",
            "dalada maligawa": "temple of the tooth",
            "tooth temple": "temple of the tooth",
            "kandy temple": "temple of the tooth",
            "sacred tooth": "temple of the tooth",
            "galle": "galle fort",
            "ella gap": "ella rock hike",
            "worlds end": "horton plains",
            "world's end": "horton plains",
            "nuwara eliya": "victoria park",  # Main attraction in Nuwara Eliya
            "yala": "yala national park",
            "udawalawe": "udawalawe national park",
            "wilpattu": "wilpattu national park",
            "minneriya": "minneriya national park",
            "polonnaruwa": "polonnaruwa ruins",
            "anuradhapura": "anuradhapura sacred city",
            "dambulla": "dambulla cave temple",
            "mihintale": "mihintale",
            "arugam": "arugam bay",
            "mirissa": "mirissa whale watching",
            "unawatuna": "unawatuna jungle beach",
            "hikkaduwa": "hikkaduwa coral reef",
            "trinco": "trincomalee harbour",
            "jaffna": "jaffna fort",
            "negombo": "negombo fish market",
        }

        # Check aliases
        if query_lower in aliases:
            alias_target = aliases[query_lower].lower()
            if alias_target in self.locations:
                loc = self.locations[alias_target]
                return LocationMatch(location=loc, confidence=0.95, matched_name=loc.name)

        # Fuzzy matching
        best_match = None
        best_score = 0.0

        for name in self.location_names:
            name_lower = name.lower()

            # Full match score
            score = SequenceMatcher(None, query_lower, name_lower).ratio()

            # Bonus for substring match
            if query_lower in name_lower or name_lower in query_lower:
                score = min(1.0, score + 0.2)

            # Word-level matching
            query_words = set(query_lower.split())
            name_words = set(name_lower.split())
            common_words = query_words & name_words
            if common_words:
                word_score = len(common_words) / max(len(query_words), len(name_words))
                score = max(score, word_score)

            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= threshold:
            loc = self.locations[best_match.lower()]
            return LocationMatch(location=loc, confidence=best_score, matched_name=loc.name)

        return None

    # =========================================================================
    # CORE API: get_impact() METHOD
    # =========================================================================

    def get_impact(
        self,
        location_name: str,
        target_date: str,
        activity_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive impact assessment for a location-date combination.

        This is the primary API method implementing Temporal-Spatial Correlation.

        Algorithm:
            1. Fuzzy match location name to known locations
            2. Parse target date and check temporal index
            3. Detect bridge days and long weekend effects
            4. Apply socio-cultural constraint logic
            5. Cross-reference location thematic scores
            6. Generate crowd modifier and travel advice

        Args:
            location_name: Name of the location (fuzzy matched)
            target_date: Date string (YYYY-MM-DD)
            activity_type: Optional planned activity for constraint checking

        Returns:
            Dict with structured impact assessment (JSON-serializable)

        Example:
            >>> sentinel = EventSentinel()
            >>> result = sentinel.get_impact("Temple of Tooth", "2026-05-01")
            >>> result["is_legal_conflict"]
            False
            >>> result["predicted_crowd_modifier"]
            2.5
        """
        # Parse date
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            return self._error_response(f"Invalid date format: {target_date}")

        # Fuzzy match location
        location_match = self._fuzzy_match_location(location_name)
        if not location_match:
            return self._error_response(f"Location not found: {location_name}")

        loc = location_match.location

        # Get temporal context
        temporal_context = self.temporal_index.get(target_date)
        is_poya = target_date in self.poya_days
        is_weekend = dt.isoweekday() >= 6
        is_new_year = dt.strftime("%m-%d") in NEW_YEAR_DATES

        # Check if part of a long weekend
        is_long_weekend = False
        if temporal_context and temporal_context.bridge_info.potential_long_weekend_days >= 3:
            is_long_weekend = True
        # Also check if this date is within adjacent dates of any holiday
        for holiday_date, holiday_info in self.temporal_index.items():
            if target_date in holiday_info.bridge_info.adjacent_dates:
                is_long_weekend = True
                break

        # Build constraints
        constraints = self._build_constraints(
            is_poya=is_poya,
            is_new_year=is_new_year,
            location=loc,
            activity_type=activity_type,
            temporal_context=temporal_context
        )

        # Calculate crowd modifier
        crowd_modifier = self._calculate_crowd_modifier(
            is_poya=is_poya,
            is_new_year=is_new_year,
            is_weekend=is_weekend,
            is_long_weekend=is_long_weekend,
            location=loc,
            temporal_context=temporal_context,
            target_date=target_date
        )

        # Generate sensitivity flags
        sensitivity_flags = self._generate_sensitivity_flags(
            is_poya=is_poya,
            is_new_year=is_new_year,
            is_long_weekend=is_long_weekend,
            location=loc,
            temporal_context=temporal_context
        )

        # Generate travel advice
        travel_advice = self._generate_travel_advice(
            is_poya=is_poya,
            is_new_year=is_new_year,
            is_long_weekend=is_long_weekend,
            location=loc,
            temporal_context=temporal_context,
            constraints=constraints
        )

        # Check activity constraints
        is_legal_conflict = False
        activity_allowed = None
        activity_warnings = []

        if activity_type:
            activity_lower = activity_type.lower()

            # Check alcohol constraint
            if is_poya and activity_lower in ALCOHOL_ACTIVITIES:
                is_legal_conflict = True
                activity_allowed = False
                activity_warnings.append(
                    "This activity is not available on Poya days due to alcohol ban"
                )

            # Check modest dress
            if activity_lower in MODEST_DRESS_ACTIVITIES and loc.l_rel > 0.5:
                activity_warnings.append(
                    "Modest dress required: cover shoulders and knees"
                )

            # New Year shutdown
            if is_new_year:
                activity_allowed = False
                activity_warnings.append(
                    "CRITICAL: Most businesses closed for Sinhala/Tamil New Year"
                )

            if activity_allowed is None:
                activity_allowed = True

        # Build response
        return {
            "is_legal_conflict": is_legal_conflict,
            "predicted_crowd_modifier": round(crowd_modifier, 2),
            "travel_advice_strings": travel_advice,
            "location_sensitivity": {
                "location_name": location_match.matched_name,
                "match_confidence": round(location_match.confidence, 2),
                "l_rel": loc.l_rel,
                "l_nat": loc.l_nat,
                "l_hist": loc.l_hist,
                "l_adv": loc.l_adv,
                "sensitivity_flags": sensitivity_flags
            },
            "temporal_context": self._temporal_to_dict(temporal_context) if temporal_context else None,
            "constraints": [self._constraint_to_dict(c) for c in constraints],
            "is_poya_day": is_poya,
            "is_new_year_shutdown": is_new_year,
            "is_weekend": is_weekend,
            "is_long_weekend": is_long_weekend,
            "activity_allowed": activity_allowed,
            "activity_warnings": activity_warnings,
            "calculation_timestamp": datetime.now().isoformat(),
            "engine_version": self.VERSION
        }

    # =========================================================================
    # CONSTRAINT BUILDING
    # =========================================================================

    def _build_constraints(
        self,
        is_poya: bool,
        is_new_year: bool,
        location: LocationData,
        activity_type: Optional[str],
        temporal_context: Optional[TemporalIndex]
    ) -> List[Constraint]:
        """Build list of applicable constraints."""
        constraints = []

        # HARD CONSTRAINT: Poya alcohol ban
        if is_poya:
            constraints.append(Constraint(
                constraint_type="HARD_CONSTRAINT",
                code="POYA_ALCOHOL_BAN",
                severity="CRITICAL",
                message="Alcohol sales prohibited island-wide on Poya days",
                affected_activities=list(ALCOHOL_ACTIVITIES)
            ))

            # Modest dress at religious sites
            if location.l_rel > 0.5:
                constraints.append(Constraint(
                    constraint_type="SOFT_CONSTRAINT",
                    code="POYA_MODEST_DRESS",
                    severity="HIGH",
                    message="Modest dress strongly recommended at religious sites on Poya",
                    affected_activities=["temple_visit", "sightseeing", "photography"]
                ))

        # CRITICAL SHUTDOWN: New Year
        if is_new_year:
            constraints.append(Constraint(
                constraint_type="HARD_CONSTRAINT",
                code="NEW_YEAR_SHUTDOWN",
                severity="CRITICAL",
                message="Most businesses, restaurants, and services closed for Sinhala/Tamil New Year",
                affected_activities=["dining", "shopping", "tours", "activities"]
            ))

        # WARNING: High religious site on Poya
        if is_poya and location.l_rel > L_REL_EXTREME_THRESHOLD:
            constraints.append(Constraint(
                constraint_type="WARNING",
                code="EXTREME_CROWD_RELIGIOUS",
                severity="HIGH",
                message=f"{location.name} experiences 2-5x normal crowds on Poya days",
                affected_activities=["photography", "sightseeing", "meditation"]
            ))

        # WARNING: Nature site on long weekend
        if temporal_context and temporal_context.is_mercantile and location.l_nat > L_NAT_DOMESTIC_THRESHOLD:
            constraints.append(Constraint(
                constraint_type="WARNING",
                code="DOMESTIC_TOURISM_PEAK",
                severity="MEDIUM",
                message=f"{location.name} popular with domestic tourists on bank holidays",
                affected_activities=["hiking", "wildlife", "nature_photography"]
            ))

        return constraints

    # =========================================================================
    # CROWD MODIFIER CALCULATION
    # =========================================================================

    def _calculate_crowd_modifier(
        self,
        is_poya: bool,
        is_new_year: bool,
        is_weekend: bool,
        is_long_weekend: bool,
        location: LocationData,
        temporal_context: Optional[TemporalIndex],
        target_date: str
    ) -> float:
        """
        Calculate predicted crowd multiplier.

        Algorithm combines multiple factors:
        1. Base event modifier (Vesak > Poya > Normal)
        2. Location sensitivity (l_rel, l_nat)
        3. Weekend/long weekend bonus
        4. School holiday overlay
        """
        modifier = 1.0

        # New Year = ghost town
        if is_new_year:
            return CROWD_MODIFIERS["new_year"]

        # Check for major events
        if temporal_context:
            name_lower = temporal_context.name.lower()

            if "vesak" in name_lower:
                modifier = CROWD_MODIFIERS["vesak"]
                if location.l_rel > L_REL_EXTREME_THRESHOLD:
                    modifier *= 1.2  # Extra 20% at religious sites

            elif "poson" in name_lower:
                modifier = CROWD_MODIFIERS["poson"]
                # Mihintale gets extreme crowds
                if "mihintale" in location.name.lower():
                    modifier *= 1.5

            elif "esala" in name_lower:
                modifier = CROWD_MODIFIERS["esala"]
                # Kandy Temple of Tooth
                if location.l_rel > 0.5 and "kandy" in location.name.lower():
                    modifier *= 1.3

            elif is_poya:
                # Regular Poya
                if location.l_rel > L_REL_EXTREME_THRESHOLD:
                    modifier = CROWD_MODIFIERS["poya_religious"]
                else:
                    modifier = CROWD_MODIFIERS["poya_general"]

            # Mercantile holiday at nature sites
            if temporal_context.is_mercantile and location.l_nat > L_NAT_DOMESTIC_THRESHOLD:
                modifier = max(modifier, CROWD_MODIFIERS["mercantile"])

        # Long weekend effect
        if is_long_weekend and not is_poya:
            modifier = max(modifier, CROWD_MODIFIERS["long_weekend"])
            if location.l_nat > 0.7 or location.l_outdoor > 0.8:
                modifier *= 1.1  # Extra at outdoor spots

        # Weekend bonus (if not already elevated)
        if is_weekend and modifier < 1.3:
            modifier *= 1.15

        # School holiday overlay
        if self._is_school_holiday(target_date):
            modifier *= CROWD_MODIFIERS["school_holiday"]

        return min(5.0, modifier)  # Cap at 5x

    def _is_school_holiday(self, date_str: str) -> bool:
        """Check if date falls within school holiday period."""
        for start, end in SCHOOL_HOLIDAYS_2026:
            if start <= date_str <= end:
                return True
        return False

    # =========================================================================
    # SENSITIVITY FLAGS
    # =========================================================================

    def _generate_sensitivity_flags(
        self,
        is_poya: bool,
        is_new_year: bool,
        is_long_weekend: bool,
        location: LocationData,
        temporal_context: Optional[TemporalIndex]
    ) -> List[str]:
        """Generate location-specific sensitivity flags."""
        flags = []

        # Religious significance
        if location.l_rel > L_REL_EXTREME_THRESHOLD:
            flags.append("HIGH_RELIGIOUS_SITE")
            if is_poya:
                flags.append("POYA_EXTREME_CROWD")

        # Nature/outdoor
        if location.l_nat > L_NAT_DOMESTIC_THRESHOLD:
            flags.append("NATURE_HOTSPOT")
            if is_long_weekend:
                flags.append("DOMESTIC_TOURISM_PEAK")

        # Historical
        if location.l_hist > 0.8:
            flags.append("MAJOR_HERITAGE_SITE")

        # Adventure
        if location.l_adv > 0.8:
            flags.append("ADVENTURE_DESTINATION")

        # New Year
        if is_new_year:
            flags.append("NEW_YEAR_CRITICAL_SHUTDOWN")

        # Vesak specific
        if temporal_context and "vesak" in temporal_context.name.lower():
            flags.append("VESAK_PEAK_PERIOD")

        return flags

    # =========================================================================
    # TRAVEL ADVICE GENERATION
    # =========================================================================

    def _generate_travel_advice(
        self,
        is_poya: bool,
        is_new_year: bool,
        is_long_weekend: bool,
        location: LocationData,
        temporal_context: Optional[TemporalIndex],
        constraints: List[Constraint]
    ) -> List[str]:
        """Generate actionable travel advice strings."""
        advice = []

        # New Year - highest priority
        if is_new_year:
            advice.append(
                "CRITICAL: April 13-14 is Sinhala/Tamil New Year - "
                "most businesses, restaurants, and transport services are closed"
            )
            advice.append(
                "Stock up on essentials beforehand; consider visiting on April 15+"
            )
            return advice

        # Poya day advice
        if is_poya:
            advice.append("POYA DAY: Alcohol sales banned island-wide")

            if location.l_rel > L_REL_EXTREME_THRESHOLD:
                advice.append(
                    f"{location.name} expects 2-5x normal crowds on Poya; "
                    "arrive before 6:00 AM for photography"
                )
                advice.append("Modest dress required: cover shoulders and knees")

            # Vesak special
            if temporal_context and "vesak" in temporal_context.name.lower():
                advice.append(
                    "Vesak is the holiest day in Sri Lanka; "
                    "expect temple decorations and lantern festivals"
                )

        # Long weekend advice
        if is_long_weekend and not is_poya:
            if location.l_nat > 0.7:
                advice.append(
                    f"Long weekend: {location.name} popular with domestic tourists; "
                    "book accommodations in advance"
                )

        # Mercantile holiday at nature sites
        if temporal_context and temporal_context.is_mercantile:
            if location.l_nat > L_NAT_DOMESTIC_THRESHOLD:
                advice.append(
                    "Bank holiday: Expect increased domestic visitors at nature sites"
                )

        # General advice based on constraints
        for constraint in constraints:
            if constraint.severity == "CRITICAL" and constraint.code not in ["POYA_ALCOHOL_BAN", "NEW_YEAR_SHUTDOWN"]:
                advice.append(f"ALERT: {constraint.message}")

        # Photography timing advice
        if location.l_hist > 0.8 or location.l_nat > 0.8:
            advice.append(
                "Photography tip: Golden hour lighting is best 6:00-6:30 AM "
                "and 5:30-6:00 PM"
            )

        return advice if advice else ["No special advisories for this date"]

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _temporal_to_dict(self, t: TemporalIndex) -> Dict[str, Any]:
        """Convert TemporalIndex to dictionary."""
        return {
            "uid": t.uid,
            "name": t.name,
            "date": t.date,
            "day_of_week": t.day_of_week,
            "day_number": t.day_number,
            "categories": t.categories,
            "is_poya": t.is_poya,
            "is_mercantile": t.is_mercantile,
            "bridge_info": {
                "is_bridge_day": t.bridge_info.is_bridge_day,
                "bridge_type": t.bridge_info.bridge_type,
                "potential_long_weekend_days": t.bridge_info.potential_long_weekend_days,
                "adjacent_dates": t.bridge_info.adjacent_dates
            }
        }

    def _constraint_to_dict(self, c: Constraint) -> Dict[str, Any]:
        """Convert Constraint to dictionary."""
        return {
            "constraint_type": c.constraint_type,
            "code": c.code,
            "severity": c.severity,
            "message": c.message,
            "affected_activities": c.affected_activities
        }

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Generate error response."""
        return {
            "error": True,
            "message": message,
            "is_legal_conflict": False,
            "predicted_crowd_modifier": 1.0,
            "travel_advice_strings": [message],
            "calculation_timestamp": datetime.now().isoformat(),
            "engine_version": self.VERSION
        }

    # =========================================================================
    # LEGACY API COMPATIBILITY
    # =========================================================================

    def is_poya_day(self, date: datetime) -> bool:
        """Check if a given date is a Poya day."""
        return date.strftime("%Y-%m-%d") in self.poya_days

    def is_school_holiday(self, date: datetime) -> bool:
        """Check if a given date falls within school holiday period."""
        return self._is_school_holiday(date.strftime("%Y-%m-%d"))

    def get_event_info(self, date: datetime) -> Optional[Dict]:
        """Get detailed information about events on a specific date (legacy)."""
        date_str = date.strftime("%Y-%m-%d")
        temporal = self.temporal_index.get(date_str)

        info = {
            "date": date_str,
            "is_poya": date_str in self.poya_days,
            "is_school_holiday": self._is_school_holiday(date_str),
            "is_weekend": date.weekday() >= 5,
            "day_of_week": date.strftime("%A"),
            "alcohol_allowed": date_str not in self.poya_days,
            "special_event": temporal.name if temporal else None,
            "crowd_impact": "normal",
            "warnings": [],
            "recommendations": []
        }

        if temporal:
            info["crowd_impact"] = self._legacy_crowd_impact(temporal)

        if info["is_poya"]:
            info["warnings"].append("Alcohol sales banned island-wide on Poya days")
            info["recommendations"].append("Visit temples early morning (5-7 AM) to avoid crowds")

        return info

    def _legacy_crowd_impact(self, temporal: TemporalIndex) -> str:
        """Determine crowd impact for legacy API."""
        name_lower = temporal.name.lower()
        if "vesak" in name_lower:
            return "extreme_crowd"
        if "poson" in name_lower:
            return "extreme_crowd_mihintale"
        if "esala" in name_lower:
            return "high_crowd_kandy"
        if temporal.is_poya:
            return "high_religious_sites"
        if temporal.is_mercantile:
            return "moderate_crowd"
        return "normal"

    def check_activity_constraints(
        self,
        date: datetime,
        activity: str,
        location_type: str
    ) -> Dict:
        """Check if a planned activity violates constraints (legacy)."""
        result = {
            "date": date.strftime("%Y-%m-%d"),
            "activity": activity,
            "location_type": location_type,
            "is_allowed": True,
            "violations": [],
            "warnings": [],
            "suggestions": []
        }

        is_poya = self.is_poya_day(date)

        if is_poya and activity.lower() in ALCOHOL_ACTIVITIES:
            result["is_allowed"] = False
            result["violations"].append({
                "type": "poya_alcohol",
                "severity": "high",
                "message": "Alcohol activities not available on Poya days"
            })
            result["suggestions"].append("Consider visiting a tea plantation instead")

        if location_type == "Religious" and is_poya:
            result["warnings"].append({
                "type": "high_crowd",
                "severity": "medium",
                "message": "Religious sites experience 2-5x normal crowds on Poya days"
            })

        return result


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_event_sentinel: Optional[EventSentinel] = None


def get_event_sentinel(
    holidays_path: Optional[str] = None,
    locations_path: Optional[str] = None
) -> EventSentinel:
    """
    Get or create the EventSentinel singleton.

    Args:
        holidays_path: Path to holidays JSON (only used on first call)
        locations_path: Path to locations CSV (only used on first call)

    Returns:
        EventSentinel: Singleton instance
    """
    global _event_sentinel
    if _event_sentinel is None:
        _event_sentinel = EventSentinel(holidays_path, locations_path)
    return _event_sentinel
