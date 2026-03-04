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

# Gemini LLM (Primary - Free tier available)
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

# OpenAI LLM (Fallback)
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
        """Initialize the LLM for reasoning (Gemini primary, OpenAI fallback)."""
        provider = settings.LLM_PROVIDER

        # Try Gemini first (if configured as primary or if available)
        if provider == "gemini" and GEMINI_AVAILABLE and settings.GOOGLE_API_KEY:
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=settings.GEMINI_MODEL,
                    temperature=0.7,
                    google_api_key=settings.GOOGLE_API_KEY,
                    convert_system_message_to_human=True
                )
                logger.info(f"Ranker LLM initialized with Gemini: {settings.GEMINI_MODEL}")
                return
            except Exception as e:
                logger.warning(f"Could not initialize Gemini: {e}")

        # Fallback to OpenAI
        if OPENAI_AVAILABLE and settings.OPENAI_API_KEY:
            try:
                self.llm = ChatOpenAI(
                    model=settings.OPENAI_MODEL,
                    temperature=0.7,
                    api_key=settings.OPENAI_API_KEY
                )
                logger.info(f"Ranker LLM initialized with OpenAI: {settings.OPENAI_MODEL}")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI LLM: {e}")

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
            from ..tools.golden_hour import get_golden_hour_agent
            self.golden_hour = get_golden_hour_agent()
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
            # IMPROVED: Use soft penalty instead of hard blocking
            if self.crowdcast:
                try:
                    crowd_result = self.crowdcast.predict(
                        location_type=location_type,
                        target_datetime=target_dt,
                        is_poya=is_poya,
                        is_school_holiday=is_school_holiday
                    )
                    crowd_level = crowd_result.get("crowd_percentage", 50)

                    # Soft penalty system instead of hard blocking
                    # - >90%: severe warning (heavy penalty) but NOT blocked
                    # - >80%: high warning
                    # - >70%: moderate warning
                    # - >50%: light warning
                    # - <=50%: ok
                    if crowd_level > 90:
                        status = "severe_warning"
                        message = f"Very crowded ({crowd_level}%) - consider early morning visit"
                    elif crowd_level > 80:
                        status = "high_warning"
                        message = f"High crowds ({crowd_level}%) - best to visit early"
                    elif crowd_level > 70:
                        status = "warning"
                        message = f"Moderate crowds ({crowd_level}%)"
                    elif crowd_level > 50:
                        status = "light_warning"
                        message = f"Some crowds expected ({crowd_level}%)"
                    else:
                        status = "ok"
                        message = f"Low crowds expected ({crowd_level}%)"

                    location_constraints["crowd"] = {
                        "status": status,
                        "value": crowd_level,
                        "message": message,
                        "optimal_time": crowd_result.get("optimal_time", "07:00-09:00")
                    }

                    # Note: We no longer block locations, just apply penalties in evaluate
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
        """
        Infer location type from preference scores for CrowdCast.

        IMPROVED: Uses multi-label classification instead of single max score.
        Considers combinations of scores for more accurate type inference.
        """
        if not pref_scores:
            return "Heritage"

        history = pref_scores.get("history", 0)
        adventure = pref_scores.get("adventure", 0)
        nature = pref_scores.get("nature", 0)
        relaxation = pref_scores.get("relaxation", 0)

        # Multi-label classification with thresholds
        # Priority order matters for combined types

        # Religious/Heritage site (high history + moderate/high relaxation)
        if history >= 0.7 and relaxation >= 0.4:
            return "Religious"

        # Pure Heritage site (high history, low nature)
        if history >= 0.7:
            return "Heritage"

        # Beach/Coastal (high relaxation + low-moderate nature)
        if relaxation >= 0.7 and nature <= 0.6:
            return "Beach"

        # Nature Reserve (high nature + moderate-high adventure)
        if nature >= 0.8 and adventure >= 0.5:
            return "Nature Reserve"

        # Adventure/Outdoor (high adventure)
        if adventure >= 0.7:
            return "Adventure"

        # Nature (high nature)
        if nature >= 0.7:
            return "Nature"

        # Scenic Viewpoint (moderate nature + moderate history)
        if nature >= 0.5 and history >= 0.3:
            return "Scenic"

        # Default based on highest score
        scores = [
            (history, "Heritage"),
            (adventure, "Adventure"),
            (nature, "Nature"),
            (relaxation, "Beach")
        ]
        max_score, max_type = max(scores, key=lambda x: x[0])
        return max_type

    async def _evaluate_candidates_node(self, state: RankerState) -> RankerState:
        """
        Evaluate and rank candidates based on constraints.

        IMPROVED: Uses graduated penalty system instead of blocking.
        Provides better recommendations by keeping all options available.
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
            tips = []

            # ==================== IMPROVED CROWD PENALTY SYSTEM ====================
            crowd = constraints.get("crowd", {})
            crowd_status = crowd.get("status", "ok")
            crowd_value = crowd.get("value", 50)

            if crowd_status == "severe_warning":
                penalty += 0.20  # Heavy but not blocking
                warnings.append(crowd.get("message", "Very crowded"))
                tips.append(f"Optimal time: {crowd.get('optimal_time', '07:00-09:00')}")
            elif crowd_status == "high_warning":
                penalty += 0.15
                warnings.append(crowd.get("message", "High crowds expected"))
                tips.append(f"Consider visiting: {crowd.get('optimal_time', '07:00-09:00')}")
            elif crowd_status == "warning":
                penalty += 0.10
                warnings.append(crowd.get("message", "Moderate crowds"))
            elif crowd_status == "light_warning":
                penalty += 0.05
                # Don't add warning for light crowds

            # ==================== LIGHTING PENALTY ====================
            lighting = constraints.get("lighting", {})
            if lighting.get("status") == "warning":
                penalty += 0.05
                warnings.append(lighting.get("message", "Suboptimal lighting"))

            # ==================== HOLIDAY/POYA HANDLING ====================
            holiday = constraints.get("holiday", {})
            if holiday.get("status") == "warning":
                # Poya days: busier but can be more atmospheric
                warnings.append(holiday.get("message", "Holiday period"))
                # Slight penalty for crowds but note the cultural experience
                penalty += 0.03
                tips.append("Poya day: temples may be busier but offer authentic cultural experience")

            # Calculate adjusted score (min 0.1 to keep all candidates viable)
            adjusted_score = max(0.1, base_score - penalty)

            ranked.append({
                **candidate,
                "adjusted_score": adjusted_score,
                "constraint_checks": constraints,
                "warnings": warnings,
                "visit_tips": tips,
                "crowd_penalty_applied": penalty
            })

        # Sort by adjusted score
        ranked.sort(key=lambda x: x.get("adjusted_score", 0), reverse=True)

        logs.append({
            "timestamp": datetime.now().isoformat(),
            "node": "evaluate_candidates",
            "ranked_count": len(ranked),
            "top_score": ranked[0].get("adjusted_score", 0) if ranked else 0,
            "max_penalty_applied": max((r.get("crowd_penalty_applied", 0) for r in ranked), default=0)
        })

        return {
            **state,
            "ranked_candidates": ranked,
            "reasoning_logs": logs
        }

    def _should_self_correct(self, state: RankerState) -> str:
        """
        Determine if self-correction is needed.

        IMPROVED: More nuanced decision making based on candidate quality.
        """
        ranked = state.get("ranked_candidates", [])
        corrections = state.get("self_correction_count", 0)

        # Need at least some candidates
        if len(ranked) < 1 and corrections < MAX_SELF_CORRECTIONS:
            return "self_correct"

        # Check top candidates quality
        if ranked:
            top = ranked[0]
            top_score = top.get("adjusted_score", 0)

            # If top score is too low (below 0.3), try to find better
            if top_score < 0.3 and corrections < MAX_SELF_CORRECTIONS:
                return "self_correct"

            # If too many severe warnings in top 3
            severe_warning_count = sum(
                1 for r in ranked[:3]
                if "severe_warning" in str(r.get("constraint_checks", {}).get("crowd", {}).get("status", ""))
            )
            if severe_warning_count >= 2 and corrections < MAX_SELF_CORRECTIONS:
                return "self_correct"

        return "generate"

    async def _self_correct_node(self, state: RankerState) -> RankerState:
        """
        Self-correction: adaptive re-ranking with intelligent candidate selection.

        IMPROVED:
        1. Expands search radius if candidates have low scores
        2. Adjusts preference weights if results are suboptimal
        3. Suggests alternative times for crowded locations
        4. Provides clear feedback on why correction was needed
        """
        corrections = state.get("self_correction_count", 0) + 1
        logs = state.get("reasoning_logs", [])

        # Try to get more candidates from recommender
        recommender = get_recommender()
        user_lat = state.get("user_lat", 7.0)
        user_lng = state.get("user_lng", 80.0)
        user_prefs = state.get("user_preferences", [0.5, 0.5, 0.5, 0.5])
        blocked = state.get("blocked_locations", [])
        ranked = state.get("ranked_candidates", [])

        # Analyze why we need correction
        correction_reason = "insufficient_candidates"
        expanded_search = False
        relaxed_preferences = False

        if ranked:
            top_score = ranked[0].get("adjusted_score", 0) if ranked else 0
            if top_score < 0.3:
                correction_reason = "low_quality_matches"

            # Check for crowd issues
            crowd_issues = [r for r in ranked[:3] if r.get("crowd_penalty_applied", 0) > 0.1]
            if len(crowd_issues) >= 2:
                correction_reason = "crowd_constraints"

        try:
            # Adaptive search parameters based on correction reason
            max_distance = 200.0  # Default

            if correction_reason == "low_quality_matches":
                # Expand search radius progressively
                max_distance = 200.0 + (corrections * 50)  # 250km, 300km, 350km...
                expanded_search = True

            if correction_reason == "insufficient_candidates":
                max_distance = 250.0
                expanded_search = True

            # Get excluded locations (previous candidates + blocked)
            previous_names = [c.get("name") for c in ranked[:5]] if ranked else []
            all_excluded = list(set(blocked + previous_names))

            new_candidates = recommender.get_candidates(
                user_preferences=user_prefs,
                user_lat=user_lat,
                user_lng=user_lng,
                top_k=8,  # Get more candidates for better selection
                max_distance_km=max_distance,
                exclude_locations=all_excluded
            )

            new_candidate_dicts = [c.to_dict() for c in new_candidates]

            # Generate adaptive suggestions based on correction
            suggestions = []
            if correction_reason == "crowd_constraints":
                suggestions.append("Consider visiting early morning (07:00-09:00) to avoid crowds")
                suggestions.append("Weekdays typically have fewer visitors than weekends")
            if expanded_search:
                suggestions.append(f"Expanded search to {max_distance:.0f}km radius for more options")

            logs.append({
                "timestamp": datetime.now().isoformat(),
                "node": "self_correct",
                "correction_number": corrections,
                "reason": correction_reason,
                "new_candidates": len(new_candidate_dicts),
                "search_radius_km": max_distance,
                "expanded_search": expanded_search,
                "suggestions": suggestions
            })

            return {
                **state,
                "candidates": new_candidate_dicts,
                "self_correction_count": corrections,
                "needs_more_candidates": False,
                "correction_suggestions": suggestions,
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
        """Suggest optimal visit time based on constraints and location type."""
        constraints = candidate.get("constraint_checks", {})
        is_outdoor = candidate.get("is_outdoor", True)
        prefs = candidate.get("preference_scores", {})

        # Check crowd patterns
        crowd = constraints.get("crowd", {})
        crowd_level = crowd.get("value", 50)
        optimal_from_crowd = crowd.get("optimal_time", None)

        # Location type specific suggestions
        history = prefs.get("history", 0)
        nature = prefs.get("nature", 0)
        relaxation = prefs.get("relaxation", 0)

        if crowd_level > 80:
            return optimal_from_crowd or "Early morning (06:00-08:00) to avoid peak crowds"
        elif crowd_level > 70:
            return optimal_from_crowd or "Early morning (07:00-09:00) to avoid crowds"
        elif history >= 0.7:
            # Heritage sites: cooler morning hours
            return "Morning (08:00-11:00) for comfortable temple/monument exploration"
        elif nature >= 0.8:
            # Wildlife: early morning or late afternoon
            return "Early morning (06:00-08:00) or late afternoon (16:00-18:00) for wildlife sightings"
        elif relaxation >= 0.7 and is_outdoor:
            # Beaches: avoid midday sun
            return "Morning (08:00-10:00) or late afternoon (16:00-18:00) to avoid harsh sun"
        elif is_outdoor:
            return "Morning (09:00-11:00) or late afternoon (15:00-17:00) for best lighting"
        else:
            return "Any time during opening hours (typically 09:00-17:00)"

    def _generate_fallback_reasoning(self, candidate: Dict) -> str:
        """
        Generate rich template-based reasoning when LLM unavailable.

        IMPROVED: Location-specific templates with contextual information.
        """
        name = candidate.get("name", "This location")
        score = candidate.get("similarity_score", 0.5)
        distance = candidate.get("distance_km", 0)
        prefs = candidate.get("preference_scores", {})
        warnings = candidate.get("warnings", [])
        visit_tips = candidate.get("visit_tips", [])

        # Find top 2 matching categories
        sorted_prefs = sorted(prefs.items(), key=lambda x: x[1], reverse=True) if prefs else []
        best_cat = sorted_prefs[0][0] if sorted_prefs else "tourism"
        second_cat = sorted_prefs[1][0] if len(sorted_prefs) > 1 else None

        # Category descriptions
        category_descriptions = {
            "history": "historical and cultural significance",
            "adventure": "adventure activities and outdoor experiences",
            "nature": "natural beauty and wildlife",
            "relaxation": "peaceful and relaxing atmosphere"
        }

        # Category-specific location insights
        location_insights = {
            "history": "rich cultural heritage and ancient architecture",
            "adventure": "thrilling outdoor activities and scenic trails",
            "nature": "diverse flora and fauna in pristine surroundings",
            "relaxation": "serene environment perfect for unwinding"
        }

        # Build reasoning
        primary_desc = category_descriptions.get(best_cat, "unique attractions")
        insight = location_insights.get(best_cat, "memorable experiences")

        # Determine match quality
        if score >= 0.8:
            match_quality = "excellent"
        elif score >= 0.6:
            match_quality = "great"
        elif score >= 0.4:
            match_quality = "good"
        else:
            match_quality = "suitable"

        # Build base reasoning
        reasoning = f"{name} is an {match_quality} match ({score:.0%}) for your preferences"

        # Add category details
        if second_cat and sorted_prefs[1][1] >= 0.5:
            second_desc = category_descriptions.get(second_cat, "various activities")
            reasoning += f", offering {primary_desc} combined with {second_desc}"
        else:
            reasoning += f", known for its {insight}"

        # Add distance context
        if distance < 10:
            reasoning += f". Conveniently located just {distance:.1f} km away"
        elif distance < 30:
            reasoning += f". A short {distance:.1f} km drive from your location"
        else:
            reasoning += f". Located {distance:.1f} km away, worth the journey"

        # Add tips if available
        if visit_tips:
            reasoning += f". Tip: {visit_tips[0]}"
        elif warnings:
            # Convert warning to actionable tip
            if "crowd" in warnings[0].lower():
                reasoning += ". Best visited early morning to avoid crowds"

        reasoning += "."
        return reasoning

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
