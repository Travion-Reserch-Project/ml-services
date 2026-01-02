"""
Hybrid Recommendation Engine: Two-Stage Recommender System.

Stage 1: Candidate Generation (Retrieval)
    - Content-Based Filtering with Cosine Similarity
    - Geospatial Filtering with Haversine Distance

Stage 2: Agentic Re-ranking (handled by LangGraph agents)

Research Pattern:
    This implements a "Retrieve-then-Rerank" architecture where:
    1. Fast mathematical retrieval narrows candidates
    2. LLM reasoning provides contextual optimization

Mathematical Foundations:
    Cosine Similarity: sim(A,B) = (A . B) / (||A|| * ||B||)
    Haversine Distance: d = 2r * arcsin(sqrt(sin^2((lat2-lat1)/2) +
                         cos(lat1)*cos(lat2)*sin^2((lng2-lng1)/2)))
"""

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371.0

# Default path to locations metadata
DEFAULT_LOCATIONS_PATH = Path(__file__).parent.parent.parent / "data" / "locations_metadata.csv"


@dataclass
class LocationCandidate:
    """
    Represents a location candidate with scores.

    Attributes:
        name: Location name
        lat: Latitude coordinate
        lng: Longitude coordinate
        preference_scores: Dict with hist, adv, nat, rel scores
        similarity_score: Cosine similarity to user preferences
        distance_km: Haversine distance from user location
        combined_score: Weighted combination of similarity and proximity
        is_outdoor: Whether location is outdoor (affects weather sensitivity)
        metadata: Additional location metadata
    """
    name: str
    lat: float
    lng: float
    preference_scores: Dict[str, float]
    similarity_score: float = 0.0
    distance_km: float = 0.0
    combined_score: float = 0.0
    is_outdoor: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "preference_scores": self.preference_scores,
            "similarity_score": round(self.similarity_score, 4),
            "distance_km": round(self.distance_km, 2),
            "combined_score": round(self.combined_score, 4),
            "is_outdoor": self.is_outdoor,
            "metadata": self.metadata
        }


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Mathematical Definition:
        sim(A, B) = (A . B) / (||A|| * ||B||)

    Where:
        A . B = sum(a_i * b_i) for all i
        ||A|| = sqrt(sum(a_i^2)) for all i

    Args:
        vec_a: First vector (e.g., user preferences)
        vec_b: Second vector (e.g., location scores)

    Returns:
        Similarity score between 0 and 1

    Example:
        >>> user_pref = [0.8, 0.3, 0.6, 0.2]  # [hist, adv, nat, rel]
        >>> location = [0.9, 0.2, 0.7, 0.1]
        >>> sim = cosine_similarity(user_pref, location)
        >>> print(f"Similarity: {sim:.3f}")
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(f"Vector dimensions must match: {len(vec_a)} != {len(vec_b)}")

    # Convert to numpy for efficient computation
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)

    # Calculate dot product
    dot_product = np.dot(a, b)

    # Calculate magnitudes
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)

    # Handle zero vectors
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    # Calculate cosine similarity
    similarity = dot_product / (magnitude_a * magnitude_b)

    # Clamp to [0, 1] (numerical precision)
    return max(0.0, min(1.0, similarity))


def haversine_distance(
    lat1: float, lng1: float,
    lat2: float, lng2: float
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula for accuracy on spherical surfaces.

    Mathematical Definition:
        a = sin^2((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin^2((lng2-lng1)/2)
        c = 2 * arcsin(sqrt(a))
        d = R * c

    Where R is Earth's radius (6371 km).

    Args:
        lat1, lng1: First point coordinates (degrees)
        lat2, lng2: Second point coordinates (degrees)

    Returns:
        Distance in kilometers

    Example:
        >>> # Colombo to Sigiriya
        >>> dist = haversine_distance(6.9271, 79.8612, 7.957, 80.7603)
        >>> print(f"Distance: {dist:.1f} km")  # ~110 km
    """
    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)

    # Haversine formula
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2)

    c = 2 * math.asin(math.sqrt(a))

    # Distance in kilometers
    distance = EARTH_RADIUS_KM * c

    return distance


class HybridRecommender:
    """
    Two-Stage Hybrid Recommendation Engine.

    Stage 1: Candidate Generation
        - Content-based filtering using Cosine Similarity
        - Geospatial filtering using Haversine Distance
        - Combined scoring with configurable weights

    Stage 2: Agentic Re-ranking (external)
        - Candidates passed to LangGraph for LLM reasoning
        - Contextual constraints applied (weather, crowds, etc.)

    Attributes:
        locations_df: DataFrame with location metadata
        similarity_weight: Weight for preference similarity (0-1)
        proximity_weight: Weight for distance proximity (0-1)
        max_distance_km: Maximum distance to consider

    Example:
        >>> recommender = HybridRecommender()
        >>> candidates = recommender.get_candidates(
        ...     user_preferences=[0.8, 0.3, 0.6, 0.2],
        ...     user_lat=7.2906, user_lng=80.6337,  # Kandy
        ...     top_k=10
        ... )
    """

    # Preference dimension mapping
    PREFERENCE_COLUMNS = ["l_hist", "l_adv", "l_nat", "l_rel"]

    def __init__(
        self,
        locations_path: Optional[str] = None,
        similarity_weight: float = 0.6,
        proximity_weight: float = 0.4,
        max_distance_km: float = 200.0
    ):
        """
        Initialize the Hybrid Recommender.

        Args:
            locations_path: Path to locations_metadata.csv
            similarity_weight: Weight for preference similarity
            proximity_weight: Weight for distance proximity
            max_distance_km: Maximum distance to consider for candidates
        """
        self.similarity_weight = similarity_weight
        self.proximity_weight = proximity_weight
        self.max_distance_km = max_distance_km
        self.locations_df = None

        # Load locations data
        path = Path(locations_path) if locations_path else DEFAULT_LOCATIONS_PATH
        self._load_locations(path)

        logger.info(
            f"HybridRecommender initialized: {len(self.locations_df)} locations, "
            f"weights=(sim={similarity_weight}, prox={proximity_weight})"
        )

    def _load_locations(self, path: Path) -> None:
        """Load and validate locations metadata."""
        if not path.exists():
            logger.warning(f"Locations file not found: {path}")
            self.locations_df = pd.DataFrame()
            return

        try:
            self.locations_df = pd.read_csv(path)

            # Validate required columns
            required_cols = ["Location_Name", "l_lat", "l_lng"] + self.PREFERENCE_COLUMNS
            missing = set(required_cols) - set(self.locations_df.columns)
            if missing:
                raise ValueError(f"Missing columns in locations file: {missing}")

            # Handle outdoor column
            if "l_outdoor" not in self.locations_df.columns:
                self.locations_df["l_outdoor"] = 1

            logger.info(f"Loaded {len(self.locations_df)} locations from {path}")

        except Exception as e:
            logger.error(f"Failed to load locations: {e}")
            self.locations_df = pd.DataFrame()

    def get_candidates(
        self,
        user_preferences: List[float],
        user_lat: float,
        user_lng: float,
        top_k: int = 10,
        max_distance_km: Optional[float] = None,
        outdoor_only: Optional[bool] = None,
        exclude_locations: Optional[List[str]] = None
    ) -> List[LocationCandidate]:
        """
        Generate top-K location candidates using hybrid filtering.

        Algorithm:
            1. Calculate cosine similarity for each location
            2. Calculate haversine distance for each location
            3. Normalize distance to [0, 1] using max_distance
            4. Compute combined score: sim_weight * sim + prox_weight * (1 - norm_dist)
            5. Sort and return top-K candidates

        Args:
            user_preferences: 4D vector [hist, adv, nat, rel] with values 0-1
            user_lat: User's current latitude
            user_lng: User's current longitude
            top_k: Number of candidates to return
            max_distance_km: Maximum distance to consider (default: self.max_distance_km)
            outdoor_only: Filter for outdoor locations only
            exclude_locations: List of location names to exclude

        Returns:
            List of LocationCandidate objects sorted by combined score
        """
        # Use provided max_distance or default
        max_dist = max_distance_km if max_distance_km is not None else self.max_distance_km
        if self.locations_df.empty:
            logger.warning("No locations data available")
            return []

        # Validate preferences
        if len(user_preferences) != 4:
            raise ValueError(f"User preferences must be 4D vector, got {len(user_preferences)}")

        candidates = []
        exclude_set = set(exclude_locations or [])

        for _, row in self.locations_df.iterrows():
            name = row["Location_Name"]

            # Skip excluded locations
            if name in exclude_set:
                continue

            # Filter outdoor if specified
            is_outdoor = bool(row.get("l_outdoor", 1))
            if outdoor_only is not None and outdoor_only != is_outdoor:
                continue

            # Extract location preference scores
            location_scores = [
                float(row["l_hist"]),
                float(row["l_adv"]),
                float(row["l_nat"]),
                float(row["l_rel"])
            ]

            # Calculate similarity
            similarity = cosine_similarity(user_preferences, location_scores)

            # Calculate distance
            loc_lat = float(row["l_lat"])
            loc_lng = float(row["l_lng"])
            distance = haversine_distance(user_lat, user_lng, loc_lat, loc_lng)

            # Skip if beyond max distance
            if distance > max_dist:
                continue

            # Normalize distance to [0, 1] (0 = far, 1 = close)
            proximity_score = 1.0 - (distance / max_dist)

            # Calculate combined score
            combined = (
                self.similarity_weight * similarity +
                self.proximity_weight * proximity_score
            )

            # Create candidate
            candidate = LocationCandidate(
                name=name,
                lat=loc_lat,
                lng=loc_lng,
                preference_scores={
                    "history": location_scores[0],
                    "adventure": location_scores[1],
                    "nature": location_scores[2],
                    "relaxation": location_scores[3]
                },
                similarity_score=similarity,
                distance_km=distance,
                combined_score=combined,
                is_outdoor=is_outdoor,
                metadata={
                    "proximity_score": round(proximity_score, 4)
                }
            )
            candidates.append(candidate)

        # Sort by combined score (descending)
        candidates.sort(key=lambda x: x.combined_score, reverse=True)

        # Add diversity: Group candidates by score buckets and shuffle within buckets
        # This prevents always returning the exact same results
        import random
        if len(candidates) > top_k:
            # Take top candidates with some buffer for diversity
            buffer_size = min(len(candidates), top_k * 3)
            candidate_pool = candidates[:buffer_size]
            
            # Group by score ranges (0.05 buckets)
            from collections import defaultdict
            score_buckets = defaultdict(list)
            for c in candidate_pool:
                bucket = round(c.combined_score * 20) / 20  # 0.05 granularity
                score_buckets[bucket].append(c)
            
            # Shuffle within each bucket to add variety
            for bucket_candidates in score_buckets.values():
                random.shuffle(bucket_candidates)
            
            # Reconstruct sorted list with shuffled buckets
            diversified = []
            for bucket_key in sorted(score_buckets.keys(), reverse=True):
                diversified.extend(score_buckets[bucket_key])
            
            top_candidates = diversified[:top_k]
        else:
            top_candidates = candidates[:top_k]

        logger.info(
            f"Generated {len(top_candidates)} candidates from {len(candidates)} "
            f"within {max_dist}km (with diversity)"
        )

        return top_candidates

    def get_nearest_locations(
        self,
        user_lat: float,
        user_lng: float,
        top_k: int = 5,
        max_distance_km: Optional[float] = None
    ) -> List[LocationCandidate]:
        """
        Get nearest locations by distance only.

        Useful for "nearby attractions" queries without preference filtering.

        Args:
            user_lat: User's latitude
            user_lng: User's longitude
            top_k: Number of locations to return
            max_distance_km: Override max distance

        Returns:
            List of nearest LocationCandidate objects
        """
        if self.locations_df.empty:
            return []

        max_dist = max_distance_km or self.max_distance_km
        candidates = []

        for _, row in self.locations_df.iterrows():
            loc_lat = float(row["l_lat"])
            loc_lng = float(row["l_lng"])
            distance = haversine_distance(user_lat, user_lng, loc_lat, loc_lng)

            if distance <= max_dist:
                location_scores = [
                    float(row["l_hist"]),
                    float(row["l_adv"]),
                    float(row["l_nat"]),
                    float(row["l_rel"])
                ]

                candidate = LocationCandidate(
                    name=row["Location_Name"],
                    lat=loc_lat,
                    lng=loc_lng,
                    preference_scores={
                        "history": location_scores[0],
                        "adventure": location_scores[1],
                        "nature": location_scores[2],
                        "relaxation": location_scores[3]
                    },
                    similarity_score=0.0,
                    distance_km=distance,
                    combined_score=1.0 - (distance / max_dist),
                    is_outdoor=bool(row.get("l_outdoor", 1))
                )
                candidates.append(candidate)

        # Sort by distance (ascending)
        candidates.sort(key=lambda x: x.distance_km)

        return candidates[:top_k]

    def find_similar_locations(
        self,
        location_name: str,
        top_k: int = 5
    ) -> List[LocationCandidate]:
        """
        Find locations similar to a given location.

        Uses the location's preference scores as the reference vector.

        Args:
            location_name: Name of reference location
            top_k: Number of similar locations to return

        Returns:
            List of similar LocationCandidate objects
        """
        if self.locations_df.empty:
            return []

        # Find reference location
        ref_row = self.locations_df[
            self.locations_df["Location_Name"].str.lower() == location_name.lower()
        ]

        if ref_row.empty:
            logger.warning(f"Location not found: {location_name}")
            return []

        ref_row = ref_row.iloc[0]
        ref_scores = [
            float(ref_row["l_hist"]),
            float(ref_row["l_adv"]),
            float(ref_row["l_nat"]),
            float(ref_row["l_rel"])
        ]
        ref_lat = float(ref_row["l_lat"])
        ref_lng = float(ref_row["l_lng"])

        # Get candidates excluding the reference
        candidates = self.get_candidates(
            user_preferences=ref_scores,
            user_lat=ref_lat,
            user_lng=ref_lng,
            top_k=top_k + 1,
            exclude_locations=[location_name]
        )

        return candidates[:top_k]

    def get_location_info(self, location_name: str) -> Optional[LocationCandidate]:
        """
        Get detailed information for a specific location.

        Args:
            location_name: Name of the location

        Returns:
            LocationCandidate with full details, or None if not found
        """
        if self.locations_df.empty:
            return None

        row = self.locations_df[
            self.locations_df["Location_Name"].str.lower() == location_name.lower()
        ]

        if row.empty:
            return None

        row = row.iloc[0]
        location_scores = [
            float(row["l_hist"]),
            float(row["l_adv"]),
            float(row["l_nat"]),
            float(row["l_rel"])
        ]

        return LocationCandidate(
            name=row["Location_Name"],
            lat=float(row["l_lat"]),
            lng=float(row["l_lng"]),
            preference_scores={
                "history": location_scores[0],
                "adventure": location_scores[1],
                "nature": location_scores[2],
                "relaxation": location_scores[3]
            },
            similarity_score=1.0,
            distance_km=0.0,
            combined_score=1.0,
            is_outdoor=bool(row.get("l_outdoor", 1))
        )


# Singleton instance
_recommender: Optional[HybridRecommender] = None


def get_recommender() -> HybridRecommender:
    """Get or create the HybridRecommender singleton."""
    global _recommender
    if _recommender is None:
        _recommender = HybridRecommender()
    return _recommender
