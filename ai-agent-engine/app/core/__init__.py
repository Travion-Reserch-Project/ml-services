"""
Core module for Travion Recommendation Engine.

Contains mathematical logic for:
- Cosine Similarity (preference matching)
- Haversine Distance (geospatial filtering)
- Hybrid Recommendation Pipeline
"""

from .recommender import (
    HybridRecommender,
    cosine_similarity,
    haversine_distance,
    LocationCandidate,
)

__all__ = [
    "HybridRecommender",
    "cosine_similarity",
    "haversine_distance",
    "LocationCandidate",
]
