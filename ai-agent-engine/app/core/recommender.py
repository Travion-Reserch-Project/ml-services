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
import hashlib
import uuid
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


def weighted_preference_match(
    user_preferences: List[float],
    location_scores: List[float],
    confidence_weights: Optional[List[float]] = None,
    favorite_categories: Optional[List[str]] = None,
    avoid_categories: Optional[List[str]] = None
) -> float:
    """
    Calculate weighted preference match score with proper reward/penalty logic.

    IMPROVED ALGORITHM:
    - For categories user IS interested in (pref >= 0.5): 
      REWARD locations that have this category strongly.
    - For categories user is NOT interested in (pref < 0.5):
      PENALIZE locations that have what user doesn't want.

    Args:
        user_preferences: User's preference vector [hist, adv, nat, rel] (0-1)
        location_scores: Location's category scores [hist, adv, nat, rel] (0-1)
        confidence_weights: Optional confidence for each preference (0-1)
        favorite_categories: Optional list of categories to boost
        avoid_categories: Optional list of categories to penalize

    Returns:
        Match score between 0 and 1

    Example:
        >>> # User wants adventure (0.9) and nature (0.9)
        >>> user = [0.5, 0.9, 0.9, 0.5]
        >>> # Location is an adventure/nature spot
        >>> location = [0.3, 0.8, 0.9, 0.2]
        >>> score = weighted_preference_match(user, location)
        >>> # Score will be HIGH because location matches user's interests
    """
    if len(user_preferences) != len(location_scores):
        raise ValueError("Preference vectors must have same length")

    if len(user_preferences) != 4:
        raise ValueError("Expected 4 preference dimensions [hist, adv, nat, rel]")

    # Default confidence weights (1.0 = full confidence)
    if confidence_weights is None:
        confidence_weights = [1.0, 1.0, 1.0, 1.0]

    # Category names for favorite/avoid matching
    category_names = ["history", "adventure", "nature", "relaxation"]

    # Convert favorite/avoid to sets for O(1) lookup
    favorites_set = set(favorite_categories or [])
    avoid_set = set(avoid_categories or [])

    weighted_match = 0.0
    total_weight = 0.0

    for i in range(4):
        user_pref = user_preferences[i]
        loc_score = location_scores[i]
        confidence = confidence_weights[i] if i < len(confidence_weights) else 1.0
        category_name = category_names[i]

        if user_pref >= 0.5:
            # User IS interested in this category
            interest_level = (user_pref - 0.5) * 2  # Scale to 0-1
            
            # Reward: how well does location satisfy this interest?
            satisfaction = loc_score
            
            # Boost for strong matches (both user and location high)
            if loc_score >= 0.5:
                boost = 1.0 + (interest_level * (loc_score - 0.5) * 0.5)
                satisfaction *= boost
            
            # Apply favorite category boost
            if category_name in favorites_set:
                satisfaction = min(1.0, satisfaction * 1.2)
            
            # Apply avoid category penalty
            if category_name in avoid_set:
                satisfaction = max(0.0, satisfaction * 0.5)
            
            weight = user_pref * confidence
            weighted_match += satisfaction * weight
            total_weight += weight
        else:
            # User is NOT interested (pref < 0.5)
            disinterest_level = (0.5 - user_pref) * 2  # Scale to 0-1
            
            if loc_score >= 0.5:
                # Location has something user doesn't want - penalty!
                penalty = loc_score * disinterest_level
                penalty_factor = math.exp(-2.5 * penalty)
                category_score = penalty_factor
            else:
                # Location doesn't have what user doesn't want - neutral/slight positive
                category_score = 0.7
            
            # Apply avoid category penalty (stronger)
            if category_name in avoid_set:
                if loc_score >= 0.5:
                    category_score = max(0.0, category_score * 0.3)
            
            weight = (1.0 - user_pref) * confidence
            weighted_match += category_score * weight
            total_weight += weight

    if total_weight == 0:
        return 0.5  # Neutral score if no significant preferences

    # Normalize
    raw_score = weighted_match / total_weight

    # Apply sigmoid smoothing for better score distribution
    k = 5.0  # Steepness parameter
    smoothed_score = 1.0 / (1.0 + math.exp(-k * (raw_score - 0.5)))

    return max(0.0, min(1.0, smoothed_score))


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
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lng1_rad = math.radians(lng1)
    lat2_rad = math.radians(lat2)
    lng2_rad = math.radians(lng2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def vectorized_haversine(user_lat: float, user_lng: float, 
                          lats: np.ndarray, lngs: np.ndarray) -> np.ndarray:
    """
    Vectorized haversine distance calculation for all locations at once.
    
    Args:
        user_lat: User latitude in degrees
        user_lng: User longitude in degrees
        lats: Array of location latitudes in degrees
        lngs: Array of location longitudes in degrees
    
    Returns:
        Array of distances in kilometers
    """
    # Convert to radians
    lat1_rad = np.radians(user_lat)
    lng1_rad = np.radians(user_lng)
    lats_rad = np.radians(lats)
    lngs_rad = np.radians(lngs)
    
    # Haversine formula
    dlat = lats_rad - lat1_rad
    dlng = lngs_rad - lng1_rad
    
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlng / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return EARTH_RADIUS_KM * c


def vectorized_weighted_match(user_prefs: np.ndarray, 
                               location_scores: np.ndarray) -> np.ndarray:
    """
    Vectorized weighted preference match for all locations at once.
    
    IMPROVED ALGORITHM:
    - For categories user IS interested in (pref >= 0.5): 
      Reward locations that HAVE this category. Score = loc_score scaled by user interest.
    - For categories user is NOT interested in (pref < 0.5):
      Penalize locations that have what user doesn't want.
    
    Args:
        user_prefs: User preferences [4] array
        location_scores: Location scores [n_locations, 4] array
    
    Returns:
        Array of similarity scores for each location (0.0 to 1.0)
    """
    n_locations = location_scores.shape[0]
    scores = np.zeros(n_locations)
    
    for i in range(n_locations):
        loc_scores = location_scores[i]
        weighted_match = 0.0
        total_interest_weight = 0.0
        
        for j in range(4):
            user_pref = user_prefs[j]
            loc_score = loc_scores[j]
            
            if user_pref >= 0.5:
                # User IS interested in this category
                # Weight by how interested the user is (0.5 = neutral, 1.0 = very interested)
                interest_level = (user_pref - 0.5) * 2  # Scale to 0-1
                
                # Reward: how well does location satisfy this interest?
                # If user wants adventure (0.9) and location has adventure (0.8), that's great!
                satisfaction = loc_score  # Direct: location's strength in this category
                
                # Boost for strong matches (both user and location high)
                if loc_score >= 0.5:
                    # Both interested and location has it - boost!
                    boost = 1.0 + (interest_level * (loc_score - 0.5) * 0.5)
                    satisfaction *= boost
                
                weighted_match += satisfaction * user_pref  # Weight by user's preference strength
                total_interest_weight += user_pref
            else:
                # User is NOT interested (pref < 0.5)
                # Penalize if location has what user doesn't want
                disinterest_level = (0.5 - user_pref) * 2  # Scale to 0-1
                
                if loc_score >= 0.5:
                    # Location has something user doesn't want - penalty!
                    penalty = loc_score * disinterest_level
                    penalty_factor = math.exp(-2.5 * penalty)  # Exponential penalty
                    weighted_match += penalty_factor * (1.0 - user_pref)
                else:
                    # Location doesn't have what user doesn't want - neutral/slight positive
                    weighted_match += 0.7 * (1.0 - user_pref)
                
                total_interest_weight += (1.0 - user_pref)
        
        if total_interest_weight > 0:
            raw_score = weighted_match / total_interest_weight
            # Sigmoid smoothing for better distribution
            # Center around 0.5 with moderate steepness
            scores[i] = 1.0 / (1.0 + math.exp(-5.0 * (raw_score - 0.5)))
        else:
            scores[i] = 0.5
    
    return np.clip(scores, 0.0, 1.0)


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
        """Load and validate locations metadata with comprehensive validation."""
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

            # ==================== COMPREHENSIVE VALIDATION ====================

            validation_issues = []
            rows_to_drop = []

            for idx, row in self.locations_df.iterrows():
                name = row["Location_Name"]

                # Validate coordinates
                lat = row["l_lat"]
                lng = row["l_lng"]

                if pd.isna(lat) or pd.isna(lng):
                    validation_issues.append(f"{name}: Missing coordinates")
                    rows_to_drop.append(idx)
                    continue

                # Sri Lanka bounds check (5.5-10.0 lat, 79.0-82.5 lng)
                if not (5.0 <= lat <= 10.5) or not (78.5 <= lng <= 83.0):
                    validation_issues.append(f"{name}: Coordinates outside Sri Lanka ({lat}, {lng})")

                # Validate preference scores are in [0, 1]
                for col in self.PREFERENCE_COLUMNS:
                    score = row[col]
                    if pd.isna(score):
                        validation_issues.append(f"{name}: Missing {col} score, defaulting to 0.5")
                        self.locations_df.at[idx, col] = 0.5
                    elif not (0 <= score <= 1):
                        validation_issues.append(f"{name}: {col}={score} out of range [0,1], clamping")
                        self.locations_df.at[idx, col] = max(0, min(1, score))

                # Check for empty names
                if pd.isna(name) or str(name).strip() == "":
                    validation_issues.append(f"Row {idx}: Empty location name")
                    rows_to_drop.append(idx)

            # Drop invalid rows
            if rows_to_drop:
                self.locations_df = self.locations_df.drop(rows_to_drop)
                logger.warning(f"Dropped {len(rows_to_drop)} invalid location entries")

            # Remove duplicates by name (keep first occurrence)
            duplicate_mask = self.locations_df.duplicated(subset=["Location_Name"], keep="first")
            duplicates = self.locations_df[duplicate_mask]["Location_Name"].tolist()
            if duplicates:
                validation_issues.append(f"Removed duplicate entries: {duplicates[:5]}...")
                self.locations_df = self.locations_df[~duplicate_mask]

            # Log validation summary
            if validation_issues:
                logger.warning(f"Location data validation found {len(validation_issues)} issues")
                for issue in validation_issues[:10]:  # Log first 10 issues
                    logger.warning(f"  - {issue}")
                if len(validation_issues) > 10:
                    logger.warning(f"  ... and {len(validation_issues) - 10} more issues")

            # Reset index after dropping rows
            self.locations_df = self.locations_df.reset_index(drop=True)

            logger.info(f"Loaded and validated {len(self.locations_df)} locations from {path}")

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
        exclude_locations: Optional[List[str]] = None,
        visited_locations: Optional[List[str]] = None,
        favorite_categories: Optional[List[str]] = None,
        avoid_categories: Optional[List[str]] = None,
        confidence_weights: Optional[List[float]] = None,
        search_history_boost: Optional[Dict[str, float]] = None,
        diversity_seed: Optional[str] = None
    ) -> List[LocationCandidate]:
        """
        Generate top-K location candidates using hybrid filtering with advanced features.

        Algorithm:
            1. Calculate weighted preference match (with exponential penalty)
            2. Calculate haversine distance for each location
            3. Apply category boosting (favorites/avoids)
            4. Apply search history boost (clicked locations rank higher)
            5. Normalize distance to [0, 1] using max_distance
            6. Compute combined score: sim_weight * sim + prox_weight * (1 - norm_dist)
            7. Apply diversity shuffling within score buckets
            8. Sort and return top-K candidates

        Args:
            user_preferences: 4D vector [hist, adv, nat, rel] with values 0-1
            user_lat: User's current latitude
            user_lng: User's current longitude
            top_k: Number of candidates to return
            max_distance_km: Maximum distance to consider (default: self.max_distance_km)
            outdoor_only: Filter for outdoor locations only
            exclude_locations: List of location names to exclude
            visited_locations: List of already visited locations (auto-excluded or de-boosted)
            favorite_categories: Categories to boost ("history", "adventure", "nature", "relaxation")
            avoid_categories: Categories to penalize
            confidence_weights: Confidence in each preference dimension [0-1]
            search_history_boost: Dict mapping location names to boost scores from search clicks
            diversity_seed: Seed for diversity randomization (default: UUID-based)

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

        exclude_set = set(exclude_locations or [])
        visited_set = set(visited_locations or [])
        history_boost = search_history_boost or {}

        # ==================== DETERMINISTIC DIVERSITY SEED ====================
        if diversity_seed is None:
            seed_input = f"{user_lat}:{user_lng}:{user_preferences}:{uuid.uuid4()}"
            seed_hash = hashlib.md5(seed_input.encode()).hexdigest()
            seed_value = int(seed_hash[:8], 16)
        else:
            seed_value = int(hashlib.md5(diversity_seed.encode()).hexdigest()[:8], 16)

        # ==================== VECTORIZED COMPUTATION (OPTIMIZED) ====================
        df = self.locations_df.copy()
        
        # Filter excluded locations
        if exclude_set:
            df = df[~df["Location_Name"].isin(exclude_set)]
        
        # Filter outdoor if specified
        if outdoor_only is not None:
            df = df[df["l_outdoor"].astype(bool) == outdoor_only]
        
        if df.empty:
            return []
        
        # Extract arrays for vectorized computation
        names = df["Location_Name"].values
        lats = df["l_lat"].values.astype(np.float64)
        lngs = df["l_lng"].values.astype(np.float64)
        outdoor_flags = df["l_outdoor"].values.astype(bool)
        
        # Location scores matrix [n_locations, 4]
        location_scores = np.column_stack([
            df["l_hist"].values.astype(np.float64),
            df["l_adv"].values.astype(np.float64),
            df["l_nat"].values.astype(np.float64),
            df["l_rel"].values.astype(np.float64)
        ])
        
        # Vectorized distance calculation
        distances = vectorized_haversine(user_lat, user_lng, lats, lngs)
        
        # Filter by max distance
        distance_mask = distances <= max_dist
        
        # Apply mask to all arrays
        names = names[distance_mask]
        lats = lats[distance_mask]
        lngs = lngs[distance_mask]
        outdoor_flags = outdoor_flags[distance_mask]
        location_scores = location_scores[distance_mask]
        distances = distances[distance_mask]
        
        if len(names) == 0:
            return []
        
        # Vectorized similarity calculation
        user_prefs = np.array(user_preferences, dtype=np.float64)
        similarities = vectorized_weighted_match(user_prefs, location_scores)
        
        # Normalize distance to [0, 1] (0 = far, 1 = close)
        proximity_scores = 1.0 - (distances / max_dist)
        
        # Calculate combined scores
        combined_scores = (
            self.similarity_weight * similarities +
            self.proximity_weight * proximity_scores
        )
        
        # Apply history boost and visited penalty
        for i, name in enumerate(names):
            if name in history_boost:
                combined_scores[i] = min(1.0, combined_scores[i] + min(0.15, history_boost[name]))
            if name in visited_set:
                combined_scores[i] *= 0.7
        
        # Build candidates list
        candidates = []
        for i in range(len(names)):
            candidate = LocationCandidate(
                name=names[i],
                lat=float(lats[i]),
                lng=float(lngs[i]),
                preference_scores={
                    "history": float(location_scores[i, 0]),
                    "adventure": float(location_scores[i, 1]),
                    "nature": float(location_scores[i, 2]),
                    "relaxation": float(location_scores[i, 3])
                },
                similarity_score=float(similarities[i]),
                distance_km=float(distances[i]),
                combined_score=float(combined_scores[i]),
                is_outdoor=bool(outdoor_flags[i]),
                metadata={
                    "proximity_score": round(float(proximity_scores[i]), 4),
                    "is_visited": names[i] in visited_set
                }
            )
            candidates.append(candidate)

        # Sort by combined score (descending)
        candidates.sort(key=lambda x: x.combined_score, reverse=True)

        # ==================== IMPROVED DIVERSITY SHUFFLING ====================
        # Use deterministic seed based on request parameters
        random.seed(seed_value)

        if len(candidates) >= top_k:
            # Take top candidates with buffer for diversity
            buffer_size = min(len(candidates), top_k * 3)
            candidate_pool = candidates[:buffer_size]

            # Group by score ranges (0.02 buckets for finer granularity)
            score_buckets = defaultdict(list)
            for c in candidate_pool:
                # Use finer granularity (0.02) for better diversity
                bucket = round(c.combined_score * 50) / 50
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
            # If we have fewer candidates than top_k, use light shuffling
            shuffled = candidates.copy()
            # Only shuffle bottom 50% to maintain top quality
            mid = len(shuffled) // 2
            bottom_half = shuffled[mid:]
            random.shuffle(bottom_half)
            shuffled = shuffled[:mid] + bottom_half
            top_candidates = shuffled[:top_k]

        logger.info(
            f"Generated {len(top_candidates)} candidates from {len(candidates)} "
            f"within {max_dist}km (diversity_seed={seed_value})"
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
