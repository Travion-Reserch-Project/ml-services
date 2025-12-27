"""
Verifier Node: Final Self-Correction Check.

This node performs a final validation of the generated response to ensure
it correctly addresses the user's query and doesn't violate any constraints.

Research Pattern:
    Self-Correcting Agent Loop - The verifier can reject a response and
    trigger regeneration with correction instructions, implementing a
    closed-loop feedback mechanism.

Verification Checks:
    1. Constraint Compliance: Response doesn't suggest banned activities
    2. Query Alignment: Response actually addresses the user's question
    3. Accuracy Check: No obvious factual errors in the response
    4. Completeness: Response includes key requested information
"""

import logging
import re
from datetime import datetime
from typing import Dict, Optional

from ..state import GraphState, ShadowMonitorLog

logger = logging.getLogger(__name__)

# Verification prompt
VERIFIER_SYSTEM_PROMPT = """You are a response verifier for a Sri Lankan tourism assistant.

Check if the response:
1. Answers the user's question
2. Respects cultural constraints (if any mentioned)
3. Provides accurate information
4. Is helpful and complete

If the response is good, respond: APPROVED
If there's an issue, respond: NEEDS_CORRECTION: [brief explanation]

Be strict about constraint violations (e.g., suggesting alcohol on Poya day)."""


def verify_constraints(response: str, constraints: list) -> Dict:
    """
    Check if response violates any detected constraints.

    Args:
        response: Generated response text
        constraints: List of ConstraintViolation objects

    Returns:
        Dict with verification results
    """
    response_lower = response.lower()
    violations = []

    for constraint in constraints:
        constraint_type = constraint.get("constraint_type", "")

        # Check for Poya alcohol violations
        if constraint_type == "poya_alcohol":
            alcohol_terms = ["bar", "pub", "nightclub", "drinking", "alcohol", "beer", "wine", "cocktail"]
            for term in alcohol_terms:
                if term in response_lower and "not available" not in response_lower and "banned" not in response_lower:
                    violations.append({
                        "type": "constraint_violation",
                        "message": f"Response mentions '{term}' without noting Poya restrictions",
                        "severity": "high"
                    })
                    break

    return {
        "has_violations": len(violations) > 0,
        "violations": violations
    }


def verify_query_alignment(query: str, response: str) -> Dict:
    """
    Check if response aligns with the user's query.

    Args:
        query: Original user query
        response: Generated response

    Returns:
        Dict with alignment assessment
    """
    query_lower = query.lower()
    response_lower = response.lower()

    # Extract key entities from query
    key_terms = []

    # Location names to check
    locations = [
        "sigiriya", "galle", "kandy", "ella", "mirissa", "trincomalee",
        "jungle beach", "rumassala", "yala", "horton", "temple of the tooth",
        "nine arches", "dambulla", "polonnaruwa", "anuradhapura"
    ]

    for loc in locations:
        if loc in query_lower:
            key_terms.append(loc)

    # Check if key terms appear in response
    missing_terms = [term for term in key_terms if term not in response_lower]

    if missing_terms:
        return {
            "aligned": False,
            "missing_terms": missing_terms,
            "message": f"Response doesn't mention: {', '.join(missing_terms)}"
        }

    return {
        "aligned": True,
        "missing_terms": [],
        "message": "Response aligns with query"
    }


def verify_completeness(intent_value: str, response: str) -> Dict:
    """
    Check if response is complete for the intent type.

    Args:
        intent_value: Intent type (string)
        response: Generated response

    Returns:
        Dict with completeness assessment
    """
    response_len = len(response)

    if intent_value == "greeting":
        # Greetings should be short but welcoming
        is_complete = 50 < response_len < 500
        message = "Greeting is appropriate" if is_complete else "Greeting may be too short/long"

    elif intent_value == "trip_planning":
        # Trip plans should have times and locations
        has_times = bool(re.search(r'\d{1,2}:\d{2}|\d{1,2}\s*(am|pm|AM|PM)', response))
        has_structure = ":" in response or "-" in response or "•" in response
        is_complete = has_times or has_structure or response_len > 200
        message = "Itinerary appears complete" if is_complete else "Itinerary may need more detail"

    else:
        # General queries should be informative
        is_complete = response_len > 100
        message = "Response is informative" if is_complete else "Response may be too brief"

    return {
        "complete": is_complete,
        "message": message
    }


async def verifier_node(state: GraphState, llm=None) -> GraphState:
    """
    Verifier Node: Final validation and self-correction check.

    This node validates the generated response against:
    1. Detected constraints
    2. Query alignment
    3. Completeness
    4. (Optional) LLM-based quality check

    If verification fails, it can trigger regeneration with corrections.

    Args:
        state: Current graph state
        llm: Optional LLM for quality verification

    Returns:
        Updated GraphState with verification results

    Research Note:
        The verifier implements "Constitutional AI" principles by checking
        responses against explicit rules (constraints) before delivery.
    """
    response = state.get("generated_response", "")
    query = state["user_query"]
    intent = state.get("intent")
    constraints = state.get("constraint_violations", [])
    loops = state.get("reasoning_loops", 0)

    logger.info(f"Verifier checking response (loop {loops + 1})...")

    # Check constraints
    constraint_check = verify_constraints(response, [
        {"constraint_type": c["constraint_type"], "description": c["description"]}
        for c in constraints
    ] if constraints else [])

    # Check query alignment
    alignment_check = verify_query_alignment(query, response)

    # Check completeness
    completeness_check = verify_completeness(
        intent.value if intent else "tourism_query",
        response
    )

    # Aggregate verification results
    issues = []
    if constraint_check["has_violations"]:
        issues.extend([v["message"] for v in constraint_check["violations"]])
    if not alignment_check["aligned"]:
        issues.append(alignment_check["message"])
    if not completeness_check["complete"]:
        issues.append(completeness_check["message"])

    # Determine if regeneration needed
    needs_correction = len(issues) > 0 and loops < 2  # Max 2 correction loops

    # Log verification
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="verifier",
        input_context={
            "response_length": len(response),
            "constraint_count": len(constraints),
            "loop_number": loops
        },
        result="needs_correction" if needs_correction else "approved",
        details="; ".join(issues) if issues else "All checks passed",
        action_taken="regenerate" if needs_correction else "finalize"
    )

    if needs_correction:
        # Prepare correction instructions for regeneration
        correction_prompt = f"Please regenerate addressing these issues: {'; '.join(issues)}"
        return {
            **state,
            "reasoning_loops": loops + 1,
            "_correction_instructions": correction_prompt,
            "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry]
        }
    else:
        # Response approved - set as final
        return {
            **state,
            "final_response": response,
            "reasoning_loops": loops,
            "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry]
        }


def route_after_verification(state: GraphState) -> str:
    """
    Routing function: Decide whether to finalize or regenerate.

    Args:
        state: Current graph state

    Returns:
        Next node name: "generate" (regenerate) or "__end__" (finalize)
    """
    if state.get("final_response"):
        return "__end__"
    elif state.get("_correction_instructions"):
        return "generate"  # Loop back for correction
    else:
        return "__end__"
