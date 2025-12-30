"""
Simple API Endpoints for Current Day Predictions.

These endpoints provide simplified access to CrowdCast and GoldenHour tools
by accepting just a location name and using the current date.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from difflib import get_close_matches

from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def get_location_type_from_scores(preference_scores: dict) -> str:
    """
    Determine location type from preference scores.
    
    Maps preference scores to CrowdCast location types:
    - High l_rel (religious) -> Religious
    - High l_hist (historical) -> Heritage
    - High l_nat (nature) -> Nature
    - High l_adv (adventure) -> Beach (most adventure activities)
    - Otherwise -> Urban
    """
    l_rel = preference_scores.get("relaxation", 0) or preference_scores.get("l_rel", 0)
    l_hist = preference_scores.get("history", 0) or preference_scores.get("l_hist", 0)
    l_nat = preference_scores.get("nature", 0) or preference_scores.get("l_nat", 0)
    l_adv = preference_scores.get("adventure", 0) or preference_scores.get("l_adv", 0)
    
    scores = {
        "Religious": l_rel,
        "Heritage": l_hist,
        "Nature": l_nat,
        "Beach": l_adv,
    }
    max_type = max(scores, key=scores.get)
    
    if scores[max_type] >= 0.5:
        return max_type
    return "Urban"


# Score-based content for location descriptions
SCORE_BASED_CONTENT = {
    "nature": {
        "focus": ["flora and fauna", "ecosystems", "wildlife", "natural beauty", "landscapes"],
        "activities": ["Bird watching", "Nature photography", "Wildlife spotting", "Botanical exploration"],
        "best_time": "Early morning (6-8 AM) for wildlife activity and cooler temperatures"
    },
    "history": {
        "focus": ["ancient history", "archaeological significance", "historical events", "cultural heritage"],
        "activities": ["Guided historical tours", "Archaeological site visits", "Museum exploration", "Historical photography"],
        "best_time": "Mid-morning (9-11 AM) when guides are available and lighting is good"
    },
    "adventure": {
        "focus": ["physical challenges", "climbing", "hiking", "adrenaline activities"],
        "activities": ["Rock climbing", "Hiking trails", "Trekking", "Exploring hidden areas"],
        "best_time": "Early morning (5-7 AM) to beat the heat during physical activities"
    },
    "relaxation": {
        "focus": ["spiritual significance", "meditation spots", "peaceful atmosphere", "tranquil settings"],
        "activities": ["Meditation", "Relaxation", "Peaceful walks", "Temple visits"],
        "best_time": "Early morning (5-6 AM) or late afternoon for peaceful atmosphere"
    }
}

# Legacy string-based preferences
PREFERENCE_CONTENT = {
    "nature lover": SCORE_BASED_CONTENT["nature"],
    "history buff": SCORE_BASED_CONTENT["history"],
    "adventure seeker": SCORE_BASED_CONTENT["adventure"],
    "spiritual traveler": SCORE_BASED_CONTENT["relaxation"],
    "photography enthusiast": {
        "focus": ["scenic views", "lighting conditions", "composition opportunities", "unique angles"],
        "activities": ["Sunrise/sunset photography", "Landscape shots", "Architectural photography", "Wildlife photography"],
        "best_time": "Golden hours (6-7 AM and 5-6 PM) for best lighting conditions"
    },
    "cultural explorer": {
        "focus": ["local traditions", "cultural practices", "art and architecture", "community life"],
        "activities": ["Cultural tours", "Local craft workshops", "Traditional performances", "Community interactions"],
        "best_time": "Morning and late afternoon when local activities are most vibrant"
    }
}


def get_primary_focus_from_scores(scores: Dict[str, float]) -> str:
    """Determine primary focus from preference scores."""
    focus_map = {
        "nature": scores.get("nature", 0.0),
        "history": scores.get("history", 0.0),
        "adventure": scores.get("adventure", 0.0),
        "relaxation": scores.get("relaxation", 0.0)
    }
    return max(focus_map, key=focus_map.get)


def get_content_from_scores(scores: Dict[str, float]) -> Dict[str, Any]:
    """Get content based on preference scores."""
    primary = get_primary_focus_from_scores(scores)
    return SCORE_BASED_CONTENT.get(primary, SCORE_BASED_CONTENT["nature"])


def get_highlights_from_scores(scores: Dict[str, float]) -> List[str]:
    """Generate highlights based on preference scores."""
    highlights = []

    if scores.get("nature", 0) >= 0.5:
        highlights.extend([
            "Diverse wildlife and bird species",
            "Natural landscapes and ecosystems"
        ])

    if scores.get("history", 0) >= 0.5:
        highlights.extend([
            "Ancient historical significance",
            "Archaeological and cultural heritage"
        ])

    if scores.get("adventure", 0) >= 0.5:
        highlights.extend([
            "Physical challenges and activities",
            "Hiking and exploration opportunities"
        ])

    if scores.get("relaxation", 0) >= 0.5:
        highlights.extend([
            "Peaceful and serene atmosphere",
            "Meditation and spiritual experiences"
        ])

    if not highlights:
        primary = get_primary_focus_from_scores(scores)
        if primary == "nature":
            highlights = ["Natural beauty and landscapes", "Wildlife viewing opportunities"]
        elif primary == "history":
            highlights = ["Historical significance", "Cultural heritage sites"]
        elif primary == "adventure":
            highlights = ["Adventure activities", "Exploration opportunities"]
        else:
            highlights = ["Peaceful atmosphere", "Relaxation experiences"]

    return highlights[:4]


def get_highlights_for_preference(preference: str) -> List[str]:
    """Generate highlights based on string user preference (legacy)."""
    if preference == "nature lover" or preference == "nature":
        return [
            "Diverse wildlife and bird species",
            "Natural landscapes and ecosystems",
            "Botanical diversity and rare plants",
            "Scenic natural views"
        ]
    elif preference == "history buff" or preference == "history":
        return [
            "Ancient historical significance",
            "Archaeological wonders",
            "Cultural heritage sites",
            "Historical artifacts and inscriptions"
        ]
    elif preference == "adventure seeker" or preference == "adventure":
        return [
            "Physical challenges and climbs",
            "Hiking and trekking opportunities",
            "Off-the-beaten-path exploration",
            "Adrenaline-pumping activities"
        ]
    elif preference == "photography enthusiast":
        return [
            "Stunning golden hour lighting",
            "Unique compositional opportunities",
            "Panoramic viewpoints",
            "Architectural and landscape subjects"
        ]
    elif preference == "spiritual traveler" or preference == "relaxation":
        return [
            "Sacred and peaceful atmosphere",
            "Meditation and reflection spots",
            "Religious and spiritual heritage",
            "Tranquil and serene settings"
        ]
    else:
        return [
            "Rich cultural traditions",
            "Local arts and crafts",
            "Traditional practices and customs",
            "Community and local life"
        ]
