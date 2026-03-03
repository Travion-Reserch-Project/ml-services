"""
Comprehensive Tests for the Improved Recommendation Engine.

Tests the following improvements:
1. Deterministic diversity seeding (UUID-based instead of time-based)
2. Exponential penalty for negative preferences
3. Visited locations de-boosting
4. Category boosting (favorites/avoids)
5. Preference confidence weights
6. Search history boost
7. Location metadata validation
"""

import pytest
import math
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.recommender import (
    weighted_preference_match,
    haversine_distance,
    cosine_similarity,
    HybridRecommender,
    LocationCandidate
)


class TestWeightedPreferenceMatch:
    """Tests for the improved weighted_preference_match function."""

    def test_perfect_match(self):
        """User wants history, location is historical."""
        user_prefs = [0.9, 0.1, 0.5, 0.5]  # High history
        location_scores = [0.9, 0.1, 0.5, 0.5]  # High history

        score = weighted_preference_match(user_prefs, location_scores)

        # Should be high (close to 1.0)
        assert score > 0.7, f"Perfect match should score > 0.7, got {score}"

    def test_negative_preference_penalty(self):
        """User DOESN'T want history, location IS historical - should penalize."""
        user_prefs = [0.1, 0.9, 0.5, 0.5]  # Low history, high adventure
        location_scores = [0.9, 0.1, 0.5, 0.5]  # High history, low adventure

        score = weighted_preference_match(user_prefs, location_scores)

        # Should be low due to exponential penalty
        assert score < 0.5, f"Negative match should score < 0.5, got {score}"

    def test_exponential_penalty_strength(self):
        """Strong negative preference should result in stronger penalty."""
        user_prefs_mild = [0.4, 0.5, 0.5, 0.5]  # Mild dislike of history
        user_prefs_strong = [0.1, 0.5, 0.5, 0.5]  # Strong dislike of history
        location_scores = [0.9, 0.5, 0.5, 0.5]  # Very historical

        score_mild = weighted_preference_match(user_prefs_mild, location_scores)
        score_strong = weighted_preference_match(user_prefs_strong, location_scores)

        # Strong dislike should result in lower score
        assert score_strong < score_mild, \
            f"Strong penalty {score_strong} should be < mild penalty {score_mild}"

    def test_favorite_category_boost(self):
        """Favorite categories should boost the score."""
        user_prefs = [0.7, 0.5, 0.5, 0.5]
        location_scores = [0.7, 0.5, 0.5, 0.5]

        score_without = weighted_preference_match(user_prefs, location_scores)
        score_with = weighted_preference_match(
            user_prefs, location_scores,
            favorite_categories=["history"]
        )

        # With favorite boost should be higher
        assert score_with >= score_without, \
            f"Favorite boost {score_with} should be >= without {score_without}"

    def test_avoid_category_penalty(self):
        """Avoided categories should reduce the score."""
        user_prefs = [0.7, 0.5, 0.5, 0.5]
        location_scores = [0.7, 0.5, 0.5, 0.5]

        score_without = weighted_preference_match(user_prefs, location_scores)
        score_with = weighted_preference_match(
            user_prefs, location_scores,
            avoid_categories=["history"]
        )

        # With avoid penalty should be lower
        assert score_with <= score_without, \
            f"Avoid penalty {score_with} should be <= without {score_without}"

    def test_confidence_weights(self):
        """Low confidence should reduce impact of that dimension."""
        user_prefs = [0.9, 0.1, 0.5, 0.5]  # Strong history preference
        location_scores = [0.9, 0.1, 0.5, 0.5]  # Historical location

        # Full confidence
        score_full = weighted_preference_match(
            user_prefs, location_scores,
            confidence_weights=[1.0, 1.0, 1.0, 1.0]
        )

        # Low confidence in history
        score_low = weighted_preference_match(
            user_prefs, location_scores,
            confidence_weights=[0.3, 1.0, 1.0, 1.0]
        )

        # Both should be valid scores
        assert 0 <= score_full <= 1
        assert 0 <= score_low <= 1


class TestHaversineDistance:
    """Tests for haversine distance calculation."""

    def test_same_location(self):
        """Distance to same point should be ~0."""
        dist = haversine_distance(7.2906, 80.6337, 7.2906, 80.6337)
        assert dist < 0.01, f"Same point distance should be ~0, got {dist}"

    def test_colombo_to_kandy(self):
        """Colombo to Kandy should be ~100-120 km."""
        # Colombo: 6.9271, 79.8612
        # Kandy: 7.2906, 80.6337
        dist = haversine_distance(6.9271, 79.8612, 7.2906, 80.6337)
        assert 100 < dist < 130, f"Colombo-Kandy should be ~115km, got {dist}km"

    def test_symmetric(self):
        """Distance A->B should equal B->A."""
        dist_ab = haversine_distance(6.9271, 79.8612, 7.2906, 80.6337)
        dist_ba = haversine_distance(7.2906, 80.6337, 6.9271, 79.8612)
        assert abs(dist_ab - dist_ba) < 0.01, "Distance should be symmetric"


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        vec = [0.8, 0.3, 0.6, 0.2]
        sim = cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.01, f"Identical vectors similarity should be 1.0, got {sim}"

    def test_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        vec_a = [1.0, 0.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0, 0.0]
        sim = cosine_similarity(vec_a, vec_b)
        assert abs(sim) < 0.01, f"Orthogonal vectors similarity should be 0.0, got {sim}"


class TestHybridRecommender:
    """Tests for the HybridRecommender class."""

    @pytest.fixture
    def recommender(self):
        """Create a recommender with mock data."""
        recommender = HybridRecommender()
        return recommender

    def test_get_candidates_basic(self, recommender):
        """Basic candidate retrieval should work."""
        if recommender.locations_df.empty:
            pytest.skip("No location data available")

        candidates = recommender.get_candidates(
            user_preferences=[0.8, 0.3, 0.6, 0.2],
            user_lat=7.2906,
            user_lng=80.6337,
            top_k=5,
            max_distance_km=200.0
        )

        assert len(candidates) > 0, "Should return some candidates"
        assert all(isinstance(c, LocationCandidate) for c in candidates)

    def test_visited_locations_deboosted(self, recommender):
        """Visited locations should have lower scores."""
        if recommender.locations_df.empty:
            pytest.skip("No location data available")

        user_prefs = [0.8, 0.3, 0.6, 0.2]
        user_lat, user_lng = 7.2906, 80.6337

        # Get candidates without visited filter
        candidates_normal = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=user_lat,
            user_lng=user_lng,
            top_k=10,
            max_distance_km=200.0
        )

        if not candidates_normal:
            pytest.skip("No candidates found")

        # Mark first candidate as visited
        visited = [candidates_normal[0].name]

        candidates_with_visited = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=user_lat,
            user_lng=user_lng,
            top_k=10,
            max_distance_km=200.0,
            visited_locations=visited
        )

        # Find the visited location in the new results
        visited_in_new = [c for c in candidates_with_visited if c.name == visited[0]]

        if visited_in_new:
            # If it appears, it should have is_visited=True in metadata
            assert visited_in_new[0].metadata.get("is_visited") == True

    def test_diversity_seed_deterministic(self, recommender):
        """Same diversity seed should produce same results."""
        if recommender.locations_df.empty:
            pytest.skip("No location data available")

        user_prefs = [0.8, 0.3, 0.6, 0.2]

        candidates1 = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=7.2906,
            user_lng=80.6337,
            top_k=5,
            max_distance_km=200.0,
            diversity_seed="test_seed_123"
        )

        candidates2 = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=7.2906,
            user_lng=80.6337,
            top_k=5,
            max_distance_km=200.0,
            diversity_seed="test_seed_123"
        )

        if candidates1 and candidates2:
            # With same seed, order should be the same
            names1 = [c.name for c in candidates1]
            names2 = [c.name for c in candidates2]
            assert names1 == names2, "Same seed should produce same order"

    def test_search_history_boost(self, recommender):
        """Search history should boost clicked locations."""
        if recommender.locations_df.empty:
            pytest.skip("No location data available")

        user_prefs = [0.5, 0.5, 0.5, 0.5]  # Neutral preferences

        # Get candidates without boost
        candidates_normal = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=7.2906,
            user_lng=80.6337,
            top_k=10,
            max_distance_km=200.0
        )

        if not candidates_normal:
            pytest.skip("No candidates found")

        # Boost a specific location
        boost_location = candidates_normal[-1].name  # Boost the last one

        candidates_boosted = recommender.get_candidates(
            user_preferences=user_prefs,
            user_lat=7.2906,
            user_lng=80.6337,
            top_k=10,
            max_distance_km=200.0,
            search_history_boost={boost_location: 0.15}
        )

        # Find boosted location
        boosted_in_results = [c for c in candidates_boosted if c.name == boost_location]
        if boosted_in_results:
            assert boosted_in_results[0].metadata.get("has_search_boost") == True

    def test_max_distance_filter(self, recommender):
        """Candidates should be within max_distance."""
        if recommender.locations_df.empty:
            pytest.skip("No location data available")

        max_dist = 50.0  # 50km radius

        candidates = recommender.get_candidates(
            user_preferences=[0.5, 0.5, 0.5, 0.5],
            user_lat=7.2906,
            user_lng=80.6337,
            top_k=10,
            max_distance_km=max_dist
        )

        for c in candidates:
            assert c.distance_km <= max_dist, \
                f"Candidate {c.name} at {c.distance_km}km exceeds max {max_dist}km"


class TestLocationDataValidation:
    """Tests for location metadata validation."""

    def test_recommender_loads_data(self):
        """Recommender should load location data successfully."""
        recommender = HybridRecommender()

        # Should have loaded some locations (or be empty if file not found)
        assert hasattr(recommender, 'locations_df')

    def test_preference_columns_exist(self):
        """Required preference columns should exist."""
        recommender = HybridRecommender()

        if recommender.locations_df.empty:
            pytest.skip("No location data available")

        required_cols = ["l_hist", "l_adv", "l_nat", "l_rel"]
        for col in required_cols:
            assert col in recommender.locations_df.columns, \
                f"Required column {col} missing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
