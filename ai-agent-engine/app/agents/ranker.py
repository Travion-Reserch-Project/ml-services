"""
LangGraph Ranker Agent: Contextual Re-ranking with Self-Correction.

This agent implements Stage 2 of the Hybrid Recommendation System:
1. Receives top-K candidates from Stage 1 (mathematical filtering)
2. Applies contextual constraints (crowds, weather, holidays)
3. Uses LLM for reasoning and explanation generation
4. Implements self-correction loop for blocked/overcrowded locations

Research Pattern:
    Self-Correcting RAG with Constraint Satisfaction
    - Each candidate is evaluated against multiple constraints
    - If top candidate fails, agent "self-corrects" by:
      a) Modifying visit time
      b) Requesting alternative candidate
      c) Adding warnings

Graph Architecture:
    ┌──────────────┐
    │   START      │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Check        │
    │ Constraints  │◄─────────────────┐
    └──────┬───────┘                  │
           │                          │
    ┌──────▼───────┐                  │
    │ Evaluate     │                  │
    │ Candidates   │                  │
    └──────┬───────┘                  │
           │                          │
    ┌──────▼───────┐      ┌───────────┴────┐
    │ Self-Correct │──────►   Need More    │
    │ Check        │      │   Candidates?  │
    └──────┬───────┘      └────────────────┘
           │ (ok)
    ┌──────▼───────┐
    │ Generate     │
    │ Reasoning    │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │    END       │
    └──────────────┘
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Literal
from datetime import datetime
from dataclasses import dataclass

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    ChatOpenAI = None

from ..core.recommender import LocationCandidate, get_recommender
from ..config import settings

logger = logging.getLogger(__name__)

# Maximum self-correction iterations
MAX_SELF_CORRECTIONS = 3


class RankerState(TypedDict, total=False):
    """
    State for the Ranker Agent graph.

    Tracks candidates, constraints, and reasoning through the pipeline.
    """
    # Input
    candidates: List[Dict[str, Any]]
    user_lat: float
    user_lng: float
    user_preferences: List[float]
    target_datetime: Optional[str]

    # Constraint results
    constraint_results: Dict[str, Dict[str, Any]]
    blocked_locations: List[str]

    # Re-ranking
    ranked_candidates: List[Dict[str, Any]]
    self_correction_count: int
    needs_more_candidates: bool

    # Output
    final_recommendations: List[Dict[str, Any]]
    reasoning_logs: List[Dict[str, Any]]
    overall_reasoning: str


@dataclass
class ConstraintResult:
    """Result of evaluating a constraint for a location."""
    constraint_type: str
    location_name: str
    status: Literal["ok", "warning", "blocked"]
    value: Any
    message: str
    suggested_time: Optional[str] = None


class RerankerAgent:
    """
    LangGraph-based Re-ranking Agent.

    Implements contextual re-ranking with self-correction loop.

    Features:
        - Constraint checking (CrowdCast, GoldenHour, EventSentinel)
        - LLM-powered reasoning generation
        - Self-correction for blocked locations
        - Optimal time suggestions

    Example:
        >>> agent = RerankerAgent()
        >>> result = await agent.rerank(
        ...     candidates=[...],
        ...     user_lat=7.29, user_lng=80.63,
        ...     target_datetime=datetime.now()
        ... )
    """

    def __init__(self):
        """Initialize the Reranker Agent."""
        self.llm = None
        self.graph = None
        self.crowdcast = None
        self.golden_hour = None
        self.event_sentinel = None

        # Initialize LLM
        self._init_llm()

        # Initialize tools
        self._init_tools()

        # Build graph
        if LANGGRAPH_AVAILABLE:
            self._build_graph()

        logger.info("RerankerAgent initialized")

    def _init_llm(self):
        """Initialize the LLM for reasoning."""
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"Ranker LLM initialized: {settings.OPENAI_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize LLM: {e}")

    def _init_tools(self):
        """Initialize constraint-checking tools."""
        # Import CrowdCast
        try:
            from ..tools.crowdcast import get_crowdcast
            self.crowdcast = get_crowdcast()
            logger.info("CrowdCast tool connected")
        except Exception as e:
            logger.warning(f"CrowdCast not available: {e}")

        # Import GoldenHour
        try:
            from ..tools.golden_hour import get_golden_hour
            self.golden_hour = get_golden_hour()
            logger.info("GoldenHour tool connected")
        except Exception as e:
            logger.warning(f"GoldenHour not available: {e}")

        # Import EventSentinel
        try:
            from ..tools.event_sentinel import get_event_sentinel
            self.event_sentinel = get_event_sentinel()
            logger.info("EventSentinel tool connected")
        except Exception as e:
            logger.warning(f"EventSentinel not available: {e}")

    def _build_graph(self):
        """Build the LangGraph state machine."""
        workflow = StateGraph(RankerState)

        # Add nodes
        workflow.add_node("check_constraints", self._check_constraints_node)
        workflow.add_node("evaluate_candidates", self._evaluate_candidates_node)
        workflow.add_node("self_correct", self._self_correct_node)
        workflow.add_node("generate_reasoning", self._generate_reasoning_node)

        # Set entry point
        workflow.set_entry_point("check_constraints")

        # Add edges
        workflow.add_edge("check_constraints", "evaluate_candidates")
        workflow.add_conditional_edges(
            "evaluate_candidates",
            self._should_self_correct,
            {
                "self_correct": "self_correct",
                "generate": "generate_reasoning"
            }
        )
        workflow.add_conditional_edges(
            "self_correct",
            self._after_self_correct,
            {
                "retry": "check_constraints",
                "generate": "generate_reasoning"
            }
        )
        workflow.add_edge("generate_reasoning", END)

        # Compile
        self.graph = workflow.compile()
        logger.info("Ranker graph compiled")

    async def _check_constraints_node(self, state: RankerState) -> RankerState:
        """
        Check constraints for all candidates.

        Evaluates:
        - Crowd levels (CrowdCast)
        - Lighting conditions (GoldenHour)
        - Holiday/Poya status (EventSentinel)
        """
        candidates = state.get("candidates", [])
        target_dt = state.get("target_datetime")

        if target_dt and isinstance(target_dt, str):
            target_dt = datetime.fromisoformat(target_dt)
        else:
            target_dt = datetime.now()

        constraint_results = {}
        blocked_locations = []
        logs = state.get("reasoning_logs", [])

        # Get event info once for the date (applies to all locations)
        event_info = None
        is_poya = False
        is_school_holiday = False
        if self.event_sentinel:
            try:
                event_info = self.event_sentinel.get_event_info(target_dt)
                is_poya = event_info.get("is_poya", False)
                is_school_holiday = event_info.get("is_school_holiday", False)
            except Exception as e:
                logger.warning(f"Event check failed: {e}")

        for candidate in candidates:
            name = candidate.get("name", "Unknown")
            lat = candidate.get("lat", 0)
            lng = candidate.get("lng", 0)
            is_outdoor = candidate.get("is_outdoor", True)

            location_constraints = {
                "crowd": None,
                "lighting": None,
                "holiday": None
            }

            # Determine location type from preference scores
            pref_scores = candidate.get("preference_scores", {})
            location_type = self._infer_location_type(pref_scores)

            # Check crowd levels using CrowdCast
            if self.crowdcast:
                try:
                    crowd_result = self.crowdcast.predict(
                        location_type=location_type,
                        target_datetime=target_dt,
                        is_poya=is_poya,
                        is_school_holiday=is_school_holiday
                    )
                    crowd_level = crowd_result.get("crowd_percentage", 50)
                    location_constraints["crowd"] = {
                        "status": "blocked" if crowd_level > 90 else ("warning" if crowd_level > 70 else "ok"),
                        "value": crowd_level,
                        "message": f"Expected crowd: {crowd_level}%"
                    }
                    if crowd_level > 90:
                        blocked_locations.append(name)
                except Exception as e:
                    logger.warning(f"Crowd check failed for {name}: {e}")

            # Check lighting (for outdoor locations)
            if self.golden_hour and is_outdoor:
                try:
                    lighting = self.golden_hour.get_lighting_quality(
                        target_datetime=target_dt,
                        lat=lat,
                        lng=lng
                    )
                    # quality is a string like "golden", "good", "harsh", "dark"
                    quality_str = lighting.get("quality", "moderate")
                    photo_score = lighting.get("photography_score", 50)
                    is_good = quality_str in ["golden", "good"]
                    location_constraints["lighting"] = {
                        "status": "ok" if is_good else "warning",
                        "value": photo_score / 100,  # Normalize to 0-1
                        "message": lighting.get("description", "Moderate lighting")
                    }
                except Exception as e:
                    logger.warning(f"Lighting check failed for {name}: {e}")

            # Add holiday/Poya info from event_info
            if event_info:
                special_event = event_info.get("special_event")
                location_constraints["holiday"] = {
                    "status": "warning" if is_poya else "ok",
                    "value": {"is_poya": is_poya, "event": special_event},
                    "message": special_event if special_event else ("Poya Day" if is_poya else "No special events")
                }

            constraint_results[name] = location_constraints

        logs.append({
            "timestamp": datetime.now().isoformat(),
            "node": "check_constraints",
            "candidates_checked": len(candidates),
            "blocked_count": len(blocked_locations)
        })

        return {
            **state,
            "constraint_results": constraint_results,
            "blocked_locations": blocked_locations,
            "reasoning_logs": logs
        }

    def _infer_location_type(self, pref_scores: Dict[str, float]) -> str:
        """Infer location type from preference scores for CrowdCast."""
        if not pref_scores:
            return "Heritage"

        # Find highest scoring category
        max_score = 0
        max_cat = "history"
        for cat, score in pref_scores.items():
            if score > max_score:
                max_score = score
                max_cat = cat

        # Map to CrowdCast location types
        type_mapping = {
            "history": "Heritage",
            "adventure": "Nature",
            "nature": "Nature",
            "relaxation": "Beach"
        }
        return type_mapping.get(max_cat, "Heritage")

    async def _evaluate_candidates_node(self, state: RankerState) -> RankerState:
        """
        Evaluate and rank candidates based on constraints.

        Applies penalty scores for warnings and removes blocked locations.
        """
        candidates = state.get("candidates", [])
        constraint_results = state.get("constraint_results", {})
        blocked = set(state.get("blocked_locations", []))
        logs = state.get("reasoning_logs", [])

        ranked = []
        for candidate in candidates:
            name = candidate.get("name")
            if name in blocked:
                continue

            # Calculate adjusted score
            base_score = candidate.get("combined_score", 0.5)
            constraints = constraint_results.get(name, {})

            penalty = 0.0
            warnings = []

            # Apply crowd penalty
            crowd = constraints.get("crowd", {})
            if crowd.get("status") == "warning":
                penalty += 0.1
                warnings.append(crowd.get("message", "Moderate crowds"))

            # Apply lighting penalty
            lighting = constraints.get("lighting", {})
            if lighting.get("status") == "warning":
                penalty += 0.05
                warnings.append(lighting.get("message", "Suboptimal lighting"))

            # Apply holiday boost (temples may be busier but more atmospheric)
            holiday = constraints.get("holiday", {})
            if holiday.get("status") == "warning":
                warnings.append(holiday.get("message", "Holiday period"))

            adjusted_score = max(0, base_score - penalty)

            ranked.append({
                **candidate,
                "adjusted_score": adjusted_score,
                "constraint_checks": constraints,
                "warnings": warnings
            })

        # Sort by adjusted score
        ranked.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)

        logs.append({
            "timestamp": datetime.now().isoformat(),
            "node": "evaluate_candidates",
            "ranked_count": len(ranked),
            "top_score": ranked[0].get("adjusted_score", 0) if ranked else 0
        })

        return {
            **state,
            "ranked_candidates": ranked,
            "reasoning_logs": logs
        }

    def _should_self_correct(self, state: RankerState) -> str:
        """Determine if self-correction is needed."""
        ranked = state.get("ranked_candidates", [])
        corrections = state.get("self_correction_count", 0)

        # Need at least some candidates
        if len(ranked) < 1 and corrections < MAX_SELF_CORRECTIONS:
            return "self_correct"

        # Top candidate has too many warnings
        if ranked:
            top = ranked[0]
            if len(top.get("warnings", [])) > 2 and corrections < MAX_SELF_CORRECTIONS:
                return "self_correct"

        return "generate"

    async def _self_correct_node(self, state: RankerState) -> RankerState:
        """
        Self-correction: request more candidates or modify constraints.

        Implements the research-grade self-correction loop.
        """
        corrections = state.get("self_correction_count", 0) + 1
        logs = state.get("reasoning_logs", [])

        # Try to get more candidates from recommender
        recommender = get_recommender()
        user_lat = state.get("user_lat", 7.0)
        user_lng = state.get("user_lng", 80.0)
        user_prefs = state.get("user_preferences", [0.5, 0.5, 0.5, 0.5])
        blocked = state.get("blocked_locations", [])

        try:
            new_candidates = recommender.get_candidates(
                user_preferences=user_prefs,
                user_lat=user_lat,
                user_lng=user_lng,
                top_k=5,
                exclude_locations=blocked
            )

            new_candidate_dicts = [c.to_dict() for c in new_candidates]

            logs.append({
                "timestamp": datetime.now().isoformat(),
                "node": "self_correct",
                "correction_number": corrections,
                "new_candidates": len(new_candidate_dicts)
            })

            return {
                **state,
                "candidates": new_candidate_dicts,
                "self_correction_count": corrections,
                "needs_more_candidates": False,
                "reasoning_logs": logs
            }

        except Exception as e:
            logger.error(f"Self-correction failed: {e}")
            logs.append({
                "timestamp": datetime.now().isoformat(),
                "node": "self_correct",
                "error": str(e)
            })
            return {
                **state,
                "self_correction_count": corrections,
                "needs_more_candidates": False,
                "reasoning_logs": logs
            }

    def _after_self_correct(self, state: RankerState) -> str:
        """Determine next step after self-correction."""
        corrections = state.get("self_correction_count", 0)
        candidates = state.get("candidates", [])

        if candidates and corrections < MAX_SELF_CORRECTIONS:
            return "retry"
        return "generate"

    async def _generate_reasoning_node(self, state: RankerState) -> RankerState:
        """
        Generate LLM reasoning for recommendations.

        Uses the LLM to explain why each location is recommended.
        """
        ranked = state.get("ranked_candidates", [])
        logs = state.get("reasoning_logs", [])

        final_recommendations = []
        overall_reasoning = ""

        if self.llm:
            try:
                # Generate reasoning for top 3
                for i, candidate in enumerate(ranked[:3]):
                    name = candidate.get("name", "Unknown")
                    score = candidate.get("adjusted_score", 0)
                    prefs = candidate.get("preference_scores", {})
                    warnings = candidate.get("warnings", [])
                    distance = candidate.get("distance_km", 0)

                    # Create reasoning prompt
                    prompt = f"""Generate a brief, helpful recommendation explanation for a tourist.

Location: {name}
Match Score: {score:.2f}
Distance: {distance:.1f} km
Preference Match:
- History: {prefs.get('history', 0):.1f}
- Adventure: {prefs.get('adventure', 0):.1f}
- Nature: {prefs.get('nature', 0):.1f}
- Relaxation: {prefs.get('relaxation', 0):.1f}
Warnings: {', '.join(warnings) if warnings else 'None'}

Write 2-3 sentences explaining why this is a good choice and any tips for visiting.
Be specific to Sri Lankan tourism context."""

                    response = await self.llm.ainvoke([
                        {"role": "system", "content": "You are Travion, an expert Sri Lankan tour guide. Be concise and helpful."},
                        {"role": "user", "content": prompt}
                    ])

                    reasoning = response.content.strip()

                    final_recommendations.append({
                        **candidate,
                        "rank": i + 1,
                        "reasoning": reasoning,
                        "optimal_visit_time": self._suggest_optimal_time(candidate)
                    })

                # Generate overall summary
                summary_prompt = f"""Summarize the top 3 recommendations for a tourist:
{', '.join(r.get('name', '') for r in final_recommendations)}

Write 1-2 sentences about why these locations complement each other."""

                summary_response = await self.llm.ainvoke([
                    {"role": "system", "content": "You are Travion, an expert Sri Lankan tour guide."},
                    {"role": "user", "content": summary_prompt}
                ])

                overall_reasoning = summary_response.content.strip()

            except Exception as e:
                logger.error(f"LLM reasoning failed: {e}")
                # Fallback to template reasoning
                for i, candidate in enumerate(ranked[:3]):
                    final_recommendations.append({
                        **candidate,
                        "rank": i + 1,
                        "reasoning": self._generate_fallback_reasoning(candidate),
                        "optimal_visit_time": self._suggest_optimal_time(candidate)
                    })
                overall_reasoning = "These recommendations are based on your preferences and current location."

        else:
            # No LLM available - use templates
            for i, candidate in enumerate(ranked[:3]):
                final_recommendations.append({
                    **candidate,
                    "rank": i + 1,
                    "reasoning": self._generate_fallback_reasoning(candidate),
                    "optimal_visit_time": self._suggest_optimal_time(candidate)
                })
            overall_reasoning = "These recommendations are based on your preferences and proximity."

        logs.append({
            "timestamp": datetime.now().isoformat(),
            "node": "generate_reasoning",
            "recommendations_generated": len(final_recommendations)
        })

        return {
            **state,
            "final_recommendations": final_recommendations,
            "overall_reasoning": overall_reasoning,
            "reasoning_logs": logs
        }

    def _suggest_optimal_time(self, candidate: Dict) -> str:
        """Suggest optimal visit time based on constraints."""
        constraints = candidate.get("constraint_checks", {})
        is_outdoor = candidate.get("is_outdoor", True)

        # Check crowd patterns
        crowd = constraints.get("crowd", {})
        crowd_level = crowd.get("value", 50)

        if crowd_level > 70:
            return "Early morning (07:00-09:00) to avoid crowds"
        elif is_outdoor:
            return "Morning (09:00-11:00) or late afternoon (15:00-17:00)"
        else:
            return "Any time during opening hours"

    def _generate_fallback_reasoning(self, candidate: Dict) -> str:
        """Generate template-based reasoning when LLM unavailable."""
        name = candidate.get("name", "This location")
        score = candidate.get("similarity_score", 0.5)
        distance = candidate.get("distance_km", 0)
        prefs = candidate.get("preference_scores", {})

        # Find best matching category
        best_cat = max(prefs.items(), key=lambda x: x[1])[0] if prefs else "tourism"

        return (
            f"{name} is a {score:.0%} match for your preferences, "
            f"particularly for {best_cat}. Located {distance:.1f} km away, "
            f"it offers an excellent experience for visitors."
        )

    async def rerank(
        self,
        candidates: List[LocationCandidate],
        user_lat: float,
        user_lng: float,
        user_preferences: List[float],
        target_datetime: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Execute the re-ranking pipeline.

        Args:
            candidates: List of LocationCandidate from Stage 1
            user_lat: User's latitude
            user_lng: User's longitude
            user_preferences: 4D preference vector
            target_datetime: Target visit datetime

        Returns:
            Dict with ranked recommendations and reasoning
        """
        # Convert candidates to dicts
        candidate_dicts = [c.to_dict() for c in candidates]

        # If graph is not available, use fallback
        if not self.graph:
            logger.warning("Graph not initialized, using fallback ranking")
            return self._fallback_rerank(candidate_dicts)

        # Create initial state
        initial_state: RankerState = {
            "candidates": candidate_dicts,
            "user_lat": user_lat,
            "user_lng": user_lng,
            "user_preferences": user_preferences,
            "target_datetime": target_datetime.isoformat() if target_datetime else None,
            "constraint_results": {},
            "blocked_locations": [],
            "ranked_candidates": [],
            "self_correction_count": 0,
            "needs_more_candidates": False,
            "final_recommendations": [],
            "reasoning_logs": [],
            "overall_reasoning": ""
        }

        try:
            # Execute graph
            final_state = await self.graph.ainvoke(initial_state)

            recommendations = final_state.get("final_recommendations", [])

            # If no recommendations from graph, use fallback
            if not recommendations:
                logger.warning("Graph returned no recommendations, using fallback")
                return self._fallback_rerank(candidate_dicts)

            return {
                "success": True,
                "recommendations": recommendations,
                "overall_reasoning": final_state.get("overall_reasoning", ""),
                "reasoning_logs": final_state.get("reasoning_logs", []),
                "self_corrections": final_state.get("self_correction_count", 0)
            }

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}, using fallback")
            return self._fallback_rerank(candidate_dicts)

    def _fallback_rerank(self, candidates: List[Dict]) -> Dict[str, Any]:
        """Fallback ranking when graph is not available."""
        # Sort by combined_score (already calculated in Stage 1)
        sorted_candidates = sorted(
            candidates,
            key=lambda x: x.get("combined_score", 0),
            reverse=True
        )

        recommendations = []
        for i, candidate in enumerate(sorted_candidates[:3]):
            recommendations.append({
                **candidate,
                "rank": i + 1,
                "reasoning": self._generate_fallback_reasoning(candidate),
                "optimal_visit_time": self._suggest_optimal_time(candidate),
                "constraint_checks": {},
                "warnings": []
            })

        return {
            "success": True,
            "recommendations": recommendations,
            "overall_reasoning": "Recommendations based on your preferences and proximity.",
            "reasoning_logs": [{"note": "Used fallback ranking (graph unavailable)"}],
            "self_corrections": 0
        }


# Singleton instance
_ranker_agent: Optional[RerankerAgent] = None


def get_ranker_agent() -> RerankerAgent:
    """Get or create the RerankerAgent singleton."""
    global _ranker_agent
    if _ranker_agent is None:
        _ranker_agent = RerankerAgent()
    return _ranker_agent
