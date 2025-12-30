"""
News & Alert API Tool for Active Guardian Shadow Monitoring
Integrates with News APIs and GDELT for real-time crisis detection

Monitors for:
- Protests and civil unrest
- Natural disasters (landslides, floods)
- Road closures and transport disruptions
- Emergency situations

Part of the "Digital Twin of Itinerary" system.
"""

import os
import re
import httpx
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum
import pytz
import hashlib


# ============================================================================
# CONFIGURATION
# ============================================================================

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
GDELT_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
NEWS_API_URL = "https://newsapi.org/v2/everything"
SRI_LANKA_TZ = pytz.timezone("Asia/Colombo")


# ============================================================================
# SRI LANKA DISTRICTS AND REGIONS
# ============================================================================

SRI_LANKA_DISTRICTS = {
    "colombo": {"province": "Western", "aliases": ["colombo", "col"]},
    "gampaha": {"province": "Western", "aliases": ["gampaha", "negombo"]},
    "kalutara": {"province": "Western", "aliases": ["kalutara", "panadura"]},
    "kandy": {"province": "Central", "aliases": ["kandy", "peradeniya"]},
    "matale": {"province": "Central", "aliases": ["matale", "dambulla", "sigiriya"]},
    "nuwara eliya": {"province": "Central", "aliases": ["nuwara eliya", "nuwaraeliya", "nuwara-eliya", "ella", "horton plains"]},
    "galle": {"province": "Southern", "aliases": ["galle", "unawatuna", "hikkaduwa"]},
    "matara": {"province": "Southern", "aliases": ["matara", "mirissa", "weligama"]},
    "hambantota": {"province": "Southern", "aliases": ["hambantota", "yala", "tissamaharama", "kataragama"]},
    "jaffna": {"province": "Northern", "aliases": ["jaffna", "point pedro"]},
    "kilinochchi": {"province": "Northern", "aliases": ["kilinochchi"]},
    "mannar": {"province": "Northern", "aliases": ["mannar", "talaimannar"]},
    "vavuniya": {"province": "Northern", "aliases": ["vavuniya"]},
    "mullaitivu": {"province": "Northern", "aliases": ["mullaitivu"]},
    "batticaloa": {"province": "Eastern", "aliases": ["batticaloa", "pasikudah"]},
    "ampara": {"province": "Eastern", "aliases": ["ampara", "arugam bay"]},
    "trincomalee": {"province": "Eastern", "aliases": ["trincomalee", "trinco", "nilaveli", "uppuveli"]},
    "kurunegala": {"province": "North Western", "aliases": ["kurunegala", "yapahuwa"]},
    "puttalam": {"province": "North Western", "aliases": ["puttalam", "kalpitiya", "wilpattu"]},
    "anuradhapura": {"province": "North Central", "aliases": ["anuradhapura", "mihintale"]},
    "polonnaruwa": {"province": "North Central", "aliases": ["polonnaruwa", "minneriya"]},
    "ratnapura": {"province": "Sabaragamuwa", "aliases": ["ratnapura", "sinharaja", "adams peak", "sri pada"]},
    "kegalle": {"province": "Sabaragamuwa", "aliases": ["kegalle", "pinnawala"]},
    "badulla": {"province": "Uva", "aliases": ["badulla", "haputale", "ella"]},
    "monaragala": {"province": "Uva", "aliases": ["monaragala", "wellawaya"]},
}

# Map locations to districts
LOCATION_TO_DISTRICT = {
    "sigiriya": "matale",
    "dambulla": "matale",
    "kandy": "kandy",
    "ella": "badulla",
    "nuwara eliya": "nuwara eliya",
    "galle": "galle",
    "mirissa": "matara",
    "unawatuna": "galle",
    "hikkaduwa": "galle",
    "colombo": "colombo",
    "negombo": "gampaha",
    "anuradhapura": "anuradhapura",
    "polonnaruwa": "polonnaruwa",
    "trincomalee": "trincomalee",
    "arugam bay": "ampara",
    "yala": "hambantota",
    "udawalawe": "ratnapura",
    "sinharaja": "ratnapura",
    "horton plains": "nuwara eliya",
    "adams peak": "ratnapura",
    "pinnawala": "kegalle",
    "bentota": "galle",
    "tangalle": "hambantota",
    "jaffna": "jaffna",
    "wilpattu": "puttalam",
    "minneriya": "polonnaruwa",
    "kaudulla": "polonnaruwa",
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AlertCategory(str, Enum):
    """Categories of alerts that affect travel"""
    PROTEST = "protest"
    STRIKE = "strike"
    NATURAL_DISASTER = "natural_disaster"
    LANDSLIDE = "landslide"
    FLOOD = "flood"
    ROAD_CLOSURE = "road_closure"
    TRANSPORT_DISRUPTION = "transport_disruption"
    SECURITY_INCIDENT = "security_incident"
    HEALTH_EMERGENCY = "health_emergency"
    WEATHER_EMERGENCY = "weather_emergency"
    WILDLIFE_DANGER = "wildlife_danger"
    GENERAL_WARNING = "general_warning"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TravelAlert(BaseModel):
    """Model for a travel-relevant alert"""
    id: str = Field(..., description="Unique alert identifier")
    category: AlertCategory = Field(..., description="Alert category")
    severity: AlertSeverity = Field(..., description="Severity level")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")
    affected_district: Optional[str] = Field(None, description="Affected district")
    affected_locations: List[str] = Field(default_factory=list)
    source: str = Field(..., description="News source")
    source_url: Optional[str] = Field(None, description="URL to original article")
    published_at: datetime = Field(..., description="When the alert was published")
    expires_at: Optional[datetime] = Field(None, description="When the alert expires")
    is_active: bool = Field(True, description="Whether alert is currently active")
    travel_impact: str = Field(..., description="Impact on travel plans")
    recommended_action: str = Field(..., description="Recommended action for travelers")
    keywords_matched: List[str] = Field(default_factory=list)


class AlertScanResult(BaseModel):
    """Result of scanning for alerts"""
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)
    location_queried: str = Field(..., description="Location that was queried")
    district: Optional[str] = Field(None)
    alerts_found: int = Field(0)
    alerts: List[TravelAlert] = Field(default_factory=list)
    has_critical_alerts: bool = Field(False)
    has_blocking_alerts: bool = Field(False)
    overall_risk_level: AlertSeverity = Field(AlertSeverity.INFO)
    summary: str = Field("", description="Human-readable summary")


class ItineraryAlertValidation(BaseModel):
    """Result of validating an itinerary against alerts"""
    is_safe: bool = Field(..., description="Whether itinerary is safe to proceed")
    overall_risk: AlertSeverity = Field(..., description="Overall risk assessment")
    affected_items: List[Dict[str, Any]] = Field(default_factory=list)
    all_alerts: List[TravelAlert] = Field(default_factory=list)
    blocking_alerts: List[TravelAlert] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# KEYWORD PATTERNS FOR ALERT DETECTION
# ============================================================================

ALERT_KEYWORDS = {
    AlertCategory.PROTEST: [
        "protest", "demonstration", "rally", "march", "strike", "harthal",
        "bandh", "civil unrest", "riot", "clash", "tear gas", "water cannon"
    ],
    AlertCategory.STRIKE: [
        "strike", "harthal", "work stoppage", "trade union", "labor dispute",
        "transport strike", "bus strike", "railway strike"
    ],
    AlertCategory.NATURAL_DISASTER: [
        "earthquake", "tsunami", "cyclone", "hurricane", "tornado",
        "disaster", "emergency", "evacuation"
    ],
    AlertCategory.LANDSLIDE: [
        "landslide", "mudslide", "rockfall", "earth slip", "slope failure",
        "hillside collapse"
    ],
    AlertCategory.FLOOD: [
        "flood", "flooding", "flash flood", "overflow", "waterlogging",
        "dam overflow", "river overflow", "inundation"
    ],
    AlertCategory.ROAD_CLOSURE: [
        "road closure", "road closed", "road block", "highway closure",
        "route blocked", "traffic diversion", "bridge closure"
    ],
    AlertCategory.TRANSPORT_DISRUPTION: [
        "train cancelled", "flight cancelled", "bus cancelled",
        "transport suspended", "ferry cancelled", "airport closure"
    ],
    AlertCategory.SECURITY_INCIDENT: [
        "bomb threat", "security alert", "curfew", "military operation",
        "security concern", "checkpoint"
    ],
    AlertCategory.HEALTH_EMERGENCY: [
        "outbreak", "epidemic", "pandemic", "dengue", "health emergency",
        "quarantine", "hospital emergency"
    ],
    AlertCategory.WEATHER_EMERGENCY: [
        "severe weather", "storm warning", "cyclone warning", "heavy rain warning",
        "wind warning", "amber alert", "red alert"
    ],
    AlertCategory.WILDLIFE_DANGER: [
        "elephant attack", "crocodile attack", "wild animal", "leopard sighting",
        "wildlife warning", "animal danger"
    ],
}


# ============================================================================
# NEWS ALERT API CLIENT
# ============================================================================

class NewsAlertClient:
    """Client for fetching and processing news alerts"""

    def __init__(self, news_api_key: Optional[str] = None):
        self.news_api_key = news_api_key or NEWS_API_KEY
        self.timeout = 15.0

    def _get_district_for_location(self, location_name: str) -> Optional[str]:
        """Get district name for a location"""
        location_lower = location_name.lower().strip()

        # Direct mapping
        if location_lower in LOCATION_TO_DISTRICT:
            return LOCATION_TO_DISTRICT[location_lower]

        # Check district names and aliases
        for district, info in SRI_LANKA_DISTRICTS.items():
            if location_lower in info["aliases"] or location_lower == district:
                return district

        return None

    def _classify_alert(self, text: str) -> tuple[Optional[AlertCategory], List[str]]:
        """Classify text into alert category and return matched keywords"""
        text_lower = text.lower()
        matched_keywords = []

        for category, keywords in ALERT_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    matched_keywords.append(keyword)

        if not matched_keywords:
            return None, []

        # Determine primary category based on matched keywords
        category_scores = {}
        for category, keywords in ALERT_KEYWORDS.items():
            score = sum(1 for kw in matched_keywords if kw in keywords)
            if score > 0:
                category_scores[category] = score

        if category_scores:
            primary_category = max(category_scores.items(), key=lambda x: x[1])[0]
            return primary_category, matched_keywords

        return AlertCategory.GENERAL_WARNING, matched_keywords

    def _determine_severity(
        self,
        category: AlertCategory,
        keywords: List[str],
        text: str
    ) -> AlertSeverity:
        """Determine severity based on category and content"""
        text_lower = text.lower()

        # Critical indicators
        critical_words = ["death", "fatal", "killed", "emergency", "evacuation",
                         "curfew", "tsunami", "critical", "severe"]
        if any(word in text_lower for word in critical_words):
            return AlertSeverity.CRITICAL

        # High severity indicators
        high_words = ["major", "significant", "widespread", "blocked", "closed",
                     "cancelled", "suspended", "danger"]
        if any(word in text_lower for word in high_words):
            return AlertSeverity.HIGH

        # Category-based defaults
        if category in [AlertCategory.NATURAL_DISASTER, AlertCategory.SECURITY_INCIDENT,
                       AlertCategory.HEALTH_EMERGENCY]:
            return AlertSeverity.HIGH

        if category in [AlertCategory.LANDSLIDE, AlertCategory.FLOOD,
                       AlertCategory.ROAD_CLOSURE]:
            return AlertSeverity.MEDIUM

        return AlertSeverity.LOW

    def _generate_travel_impact(self, category: AlertCategory, location: str) -> str:
        """Generate travel impact description"""
        impacts = {
            AlertCategory.PROTEST: f"Access to {location} may be restricted. Expect traffic disruptions and potential safety concerns.",
            AlertCategory.STRIKE: f"Transportation services in {location} may be unavailable. Plan alternative arrangements.",
            AlertCategory.NATURAL_DISASTER: f"Travel to {location} is dangerous. Area may be inaccessible.",
            AlertCategory.LANDSLIDE: f"Roads to {location} may be blocked. Risk of further slides.",
            AlertCategory.FLOOD: f"Roads to {location} may be flooded and impassable. Risk to safety.",
            AlertCategory.ROAD_CLOSURE: f"Direct routes to {location} may be unavailable. Expect detours and delays.",
            AlertCategory.TRANSPORT_DISRUPTION: f"Public transport to {location} may be affected. Arrange private transport.",
            AlertCategory.SECURITY_INCIDENT: f"Security situation in {location} is concerning. Avoid unnecessary travel.",
            AlertCategory.HEALTH_EMERGENCY: f"Health risks in {location}. Take necessary precautions.",
            AlertCategory.WEATHER_EMERGENCY: f"Dangerous weather conditions expected in {location}.",
            AlertCategory.WILDLIFE_DANGER: f"Wildlife safety concerns in {location}. Exercise extreme caution.",
            AlertCategory.GENERAL_WARNING: f"General advisory for {location}. Stay informed of developments.",
        }
        return impacts.get(category, f"Travel advisory for {location}")

    def _generate_recommended_action(
        self,
        category: AlertCategory,
        severity: AlertSeverity
    ) -> str:
        """Generate recommended action based on alert"""
        if severity == AlertSeverity.CRITICAL:
            return "AVOID travel to this area. Seek safe alternatives immediately."

        if severity == AlertSeverity.HIGH:
            return "Postpone travel if possible. Monitor situation closely before proceeding."

        actions = {
            AlertCategory.PROTEST: "Avoid protest areas. Use alternative routes.",
            AlertCategory.STRIKE: "Arrange private transportation. Confirm bookings.",
            AlertCategory.NATURAL_DISASTER: "Follow official evacuation orders. Stay away from affected areas.",
            AlertCategory.LANDSLIDE: "Avoid mountain roads. Use main highways only.",
            AlertCategory.FLOOD: "Do not attempt to cross flooded areas. Wait for waters to recede.",
            AlertCategory.ROAD_CLOSURE: "Check for alternative routes before departure.",
            AlertCategory.TRANSPORT_DISRUPTION: "Book private transport. Confirm all reservations.",
            AlertCategory.SECURITY_INCIDENT: "Stay in secure areas. Follow local authority guidance.",
            AlertCategory.HEALTH_EMERGENCY: "Take health precautions. Carry necessary supplies.",
            AlertCategory.WEATHER_EMERGENCY: "Stay indoors during severe weather. Have emergency supplies ready.",
            AlertCategory.WILDLIFE_DANGER: "Stay in vehicles. Do not approach wildlife.",
            AlertCategory.GENERAL_WARNING: "Stay informed and exercise caution.",
        }
        return actions.get(category, "Monitor situation and proceed with caution.")

    def _generate_alert_id(self, title: str, published_at: datetime) -> str:
        """Generate unique alert ID"""
        content = f"{title}:{published_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def fetch_news_api_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI"""
        if not self.news_api_key:
            return []

        params = {
            "apiKey": self.news_api_key,
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
        }

        if from_date:
            params["from"] = from_date.strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(NEWS_API_URL, params=params)
                response.raise_for_status()
                data = response.json()

                # Handle different response formats
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get("articles", [])
                else:
                    print(f"NewsAPI returned unexpected type: {type(data)}")
                    return []
            except Exception as e:
                print(f"NewsAPI error: {e}")
                return []

    async def fetch_gdelt_articles(
        self,
        query: str,
        from_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Fetch articles from GDELT (free, no API key required)"""
        params = {
            "query": f"{query} sourcelang:eng",
            "mode": "artlist",
            "maxrecords": 20,
            "format": "json",
            "sort": "datedesc",
        }

        if from_date:
            params["startdatetime"] = from_date.strftime("%Y%m%d%H%M%S")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.get(GDELT_API_URL, params=params)
                response.raise_for_status()

                # Get raw text first for debugging
                raw_text = response.text

                # Handle empty response
                if not raw_text or raw_text.strip() == "":
                    print("GDELT returned empty response")
                    return []

                try:
                    data = response.json()
                except Exception as json_err:
                    print(f"GDELT JSON parse error: {json_err}, raw: {raw_text[:200]}")
                    return []

                # Debug logging
                print(f"GDELT response type: {type(data)}")

                # GDELT API can return different formats:
                # - A dict with "articles" key
                # - A list of articles directly
                # - An empty response or error message
                if data is None:
                    return []
                elif isinstance(data, list):
                    # Check if it's a list of article dicts or a wrapper list
                    if len(data) > 0 and isinstance(data[0], dict):
                        # Could be list of articles OR list containing a wrapper dict
                        if "articles" in data[0]:
                            # It's a wrapper: [{"articles": [...]}]
                            articles = data[0].get("articles", [])
                            return articles if isinstance(articles, list) else []
                        else:
                            # It's a direct list of articles
                            return data
                    return []
                elif isinstance(data, dict):
                    articles = data.get("articles", [])
                    return articles if isinstance(articles, list) else []
                else:
                    print(f"GDELT returned unexpected type: {type(data)}")
                    return []
            except Exception as e:
                print(f"GDELT error: {e}")
                return []

    async def scan_for_alerts(
        self,
        location_name: str,
        days_back: int = 7
    ) -> AlertScanResult:
        """
        Scan for alerts affecting a specific location.

        Args:
            location_name: Name of the location to check
            days_back: How many days back to search

        Returns:
            AlertScanResult with found alerts
        """
        try:
            return await self._scan_for_alerts_impl(location_name, days_back)
        except Exception as e:
            print(f"scan_for_alerts error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe default
            return AlertScanResult(
                scan_timestamp=datetime.utcnow(),
                location_queried=location_name,
                district=None,
                alerts_found=0,
                alerts=[],
                has_critical_alerts=False,
                has_blocking_alerts=False,
                overall_risk_level=AlertSeverity.INFO,
                summary=f"Alert scan failed for {location_name}. Please check manually."
            )

    async def _scan_for_alerts_impl(
        self,
        location_name: str,
        days_back: int = 7
    ) -> AlertScanResult:
        """Internal implementation of scan_for_alerts."""
        district = self._get_district_for_location(location_name)
        from_date = datetime.utcnow() - timedelta(days=days_back)

        # Build search queries
        search_terms = [location_name, "Sri Lanka"]
        if district:
            search_terms.append(district)

        # Add alert keywords to search
        alert_keywords_flat = [kw for kws in ALERT_KEYWORDS.values() for kw in kws[:3]]
        base_query = f"({' OR '.join(search_terms)}) AND ({' OR '.join(alert_keywords_flat[:10])})"

        alerts = []

        # Fetch from GDELT (free)
        gdelt_articles = await self.fetch_gdelt_articles(base_query, from_date)
        for article in gdelt_articles:
            # Ensure article is a dictionary
            if not isinstance(article, dict):
                continue

            title = article.get("title", "")
            url = article.get("url", "")
            source = article.get("domain", "GDELT")

            # Parse date
            try:
                date_str = article.get("seendate", "")
                if date_str:
                    pub_date = datetime.strptime(date_str[:14], "%Y%m%d%H%M%S")
                else:
                    pub_date = datetime.utcnow()
            except:
                pub_date = datetime.utcnow()

            # Classify the article
            category, keywords = self._classify_alert(title)
            if not category:
                continue

            # Check if it actually relates to the location
            title_lower = title.lower()
            location_match = any(
                term.lower() in title_lower
                for term in search_terms
            )
            if not location_match:
                continue

            severity = self._determine_severity(category, keywords, title)

            alert = TravelAlert(
                id=self._generate_alert_id(title, pub_date),
                category=category,
                severity=severity,
                title=title,
                description=title,
                affected_district=district,
                affected_locations=[location_name],
                source=source,
                source_url=url,
                published_at=pub_date,
                is_active=True,
                travel_impact=self._generate_travel_impact(category, location_name),
                recommended_action=self._generate_recommended_action(category, severity),
                keywords_matched=keywords
            )
            alerts.append(alert)

        # Fetch from NewsAPI if key available
        if self.news_api_key:
            news_articles = await self.fetch_news_api_articles(base_query, from_date)
            for article in news_articles:
                # Ensure article is a dictionary
                if not isinstance(article, dict):
                    continue

                title = article.get("title", "")
                description = article.get("description", "")
                full_text = f"{title} {description}"

                category, keywords = self._classify_alert(full_text)
                if not category:
                    continue

                # Check location match
                text_lower = full_text.lower()
                location_match = any(
                    term.lower() in text_lower
                    for term in search_terms
                )
                if not location_match:
                    continue

                try:
                    pub_date = datetime.fromisoformat(article.get("publishedAt", "").replace("Z", "+00:00"))
                except:
                    pub_date = datetime.utcnow()

                severity = self._determine_severity(category, keywords, full_text)

                # Safely get source name
                source_info = article.get("source")
                if isinstance(source_info, dict):
                    source_name = source_info.get("name", "NewsAPI")
                elif isinstance(source_info, str):
                    source_name = source_info
                else:
                    source_name = "NewsAPI"

                alert = TravelAlert(
                    id=self._generate_alert_id(title, pub_date),
                    category=category,
                    severity=severity,
                    title=title,
                    description=description or title,
                    affected_district=district,
                    affected_locations=[location_name],
                    source=source_name,
                    source_url=article.get("url"),
                    published_at=pub_date,
                    is_active=True,
                    travel_impact=self._generate_travel_impact(category, location_name),
                    recommended_action=self._generate_recommended_action(category, severity),
                    keywords_matched=keywords
                )
                alerts.append(alert)

        # Deduplicate by ID
        seen_ids = set()
        unique_alerts = []
        for alert in alerts:
            if alert.id not in seen_ids:
                seen_ids.add(alert.id)
                unique_alerts.append(alert)

        # Sort by severity and date
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }
        unique_alerts.sort(key=lambda a: (severity_order[a.severity], -a.published_at.timestamp()))

        # Determine overall risk
        has_critical = any(a.severity == AlertSeverity.CRITICAL for a in unique_alerts)
        has_high = any(a.severity == AlertSeverity.HIGH for a in unique_alerts)
        has_blocking = has_critical or has_high

        if has_critical:
            overall_risk = AlertSeverity.CRITICAL
        elif has_high:
            overall_risk = AlertSeverity.HIGH
        elif unique_alerts:
            overall_risk = AlertSeverity.MEDIUM
        else:
            overall_risk = AlertSeverity.INFO

        # Generate summary
        if not unique_alerts:
            summary = f"No significant alerts found for {location_name}."
        elif has_critical:
            summary = f"CRITICAL: {len(unique_alerts)} alert(s) found for {location_name}. Travel not recommended."
        elif has_high:
            summary = f"WARNING: {len(unique_alerts)} alert(s) found for {location_name}. Exercise caution."
        else:
            summary = f"{len(unique_alerts)} minor alert(s) found for {location_name}. Monitor situation."

        return AlertScanResult(
            scan_timestamp=datetime.utcnow(),
            location_queried=location_name,
            district=district,
            alerts_found=len(unique_alerts),
            alerts=unique_alerts,
            has_critical_alerts=has_critical,
            has_blocking_alerts=has_blocking,
            overall_risk_level=overall_risk,
            summary=summary
        )


# ============================================================================
# NEWS ALERT TOOL (LangGraph Compatible)
# ============================================================================

class NewsAlertTool:
    """
    News alert tool for the Active Guardian system.
    Scans for real-time crisis situations affecting travel.
    """

    def __init__(self, news_api_key: Optional[str] = None):
        self.client = NewsAlertClient(news_api_key)

    async def validate_itinerary_alerts(
        self,
        itinerary_items: List[Dict[str, Any]],
        days_back: int = 7
    ) -> ItineraryAlertValidation:
        """
        Validate an entire itinerary against news alerts.

        This is the main method used by the Shadow Monitor node.

        Args:
            itinerary_items: List of itinerary items with location info
            days_back: How many days back to search for alerts

        Returns:
            ItineraryAlertValidation with validation status
        """
        try:
            return await self._validate_itinerary_alerts_impl(itinerary_items, days_back)
        except Exception as e:
            print(f"validate_itinerary_alerts error: {e}")
            import traceback
            traceback.print_exc()
            # Return safe default
            return ItineraryAlertValidation(
                is_safe=True,
                overall_risk=AlertSeverity.INFO,
                affected_items=[],
                all_alerts=[],
                blocking_alerts=[],
                recommendations=["Alert validation failed. Please check local news for safety updates."],
                scan_timestamp=datetime.utcnow()
            )

    async def _validate_itinerary_alerts_impl(
        self,
        itinerary_items: List[Dict[str, Any]],
        days_back: int = 7
    ) -> ItineraryAlertValidation:
        """Internal implementation of validate_itinerary_alerts."""
        all_alerts = []
        blocking_alerts = []
        affected_items = []
        recommendations = []

        # Collect unique locations
        locations = set()
        for item in itinerary_items:
            # Ensure item is a dictionary
            if not isinstance(item, dict):
                continue
            loc_name = item.get("locationName", "")
            if loc_name:
                locations.add(loc_name)

        # Scan each location
        for location in locations:
            scan_result = await self.client.scan_for_alerts(location, days_back)

            for alert in scan_result.alerts:
                all_alerts.append(alert)

                if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
                    blocking_alerts.append(alert)

                    # Find affected itinerary items
                    for item in itinerary_items:
                        # Ensure item is a dictionary
                        if not isinstance(item, dict):
                            continue
                        if item.get("locationName", "").lower() == location.lower():
                            affected_items.append({
                                "item": item,
                                "alert": alert.dict(),
                                "recommendation": alert.recommended_action
                            })

            # Add location-specific recommendations
            if scan_result.has_blocking_alerts:
                recommendations.append(
                    f"AVOID {location}: {scan_result.alerts[0].travel_impact}"
                )
            elif scan_result.alerts:
                recommendations.append(
                    f"CAUTION for {location}: Monitor developing situations"
                )

        # Determine overall safety
        is_safe = len(blocking_alerts) == 0

        if blocking_alerts:
            critical_count = sum(1 for a in blocking_alerts if a.severity == AlertSeverity.CRITICAL)
            if critical_count > 0:
                overall_risk = AlertSeverity.CRITICAL
            else:
                overall_risk = AlertSeverity.HIGH
        elif all_alerts:
            overall_risk = AlertSeverity.MEDIUM
        else:
            overall_risk = AlertSeverity.INFO

        return ItineraryAlertValidation(
            is_safe=is_safe,
            overall_risk=overall_risk,
            affected_items=affected_items,
            all_alerts=all_alerts,
            blocking_alerts=blocking_alerts,
            recommendations=recommendations,
            scan_timestamp=datetime.utcnow()
        )


# ============================================================================
# CONVENIENCE FUNCTIONS FOR LANGGRAPH
# ============================================================================

async def check_alerts_for_trip(
    itinerary: List[Dict[str, Any]],
    days_back: int = 7,
    news_api_key: Optional[str] = None
) -> ItineraryAlertValidation:
    """
    Convenience function for LangGraph nodes.

    Usage in LangGraph:
        from app.tools.news_alert_api import check_alerts_for_trip

        result = await check_alerts_for_trip(itinerary)
        if not result.is_safe:
            return {"status": "REJECT", "reason": result.blocking_alerts}
    """
    tool = NewsAlertTool(news_api_key)
    return await tool.validate_itinerary_alerts(itinerary, days_back)


async def get_location_alerts(
    location_name: str,
    days_back: int = 7,
    news_api_key: Optional[str] = None
) -> AlertScanResult:
    """
    Get alerts for a specific location.
    Used by the Active Watcher for real-time monitoring.
    """
    client = NewsAlertClient(news_api_key)
    return await client.scan_for_alerts(location_name, days_back)
