"""
Grader Node: Self-Correction through Document Relevance Assessment.

This node implements the critical self-reflection capability of the agentic
system. It evaluates whether retrieved documents are sufficient and relevant
to answer the user's query.

Research Pattern:
    Self-Correcting RAG - The grader enables the agent to recognize when
    its knowledge is insufficient and trigger fallback mechanisms (web search).

Grading Criteria:
    1. Relevance: Do documents address the query topic?
    2. Completeness: Is there enough information to answer fully?
    3. Recency: For time-sensitive queries, is the info current?
    4. Specificity: Do documents match the specific location/context?
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from ..state import GraphState, DocumentRelevance, RetrievedDocument

logger = logging.getLogger(__name__)

# Grading system prompt
GRADER_SYSTEM_PROMPT = """You are a document relevance grader for a Sri Lankan tourism assistant.

Given a user question and a retrieved document, determine if the document is relevant.

Grade based on:
1. Does the document address the topic of the question?
2. Does it provide useful information for answering?
3. Is it about the correct location (if specified)?

Respond with ONLY one of these: RELEVANT, PARTIAL, IRRELEVANT

Examples:
Question: "What is the history of Sigiriya?"
Document: "Sigiriya was built in the 5th century by King Kashyapa..."
Grade: RELEVANT

Question: "Best time to visit Mirissa Beach"
Document: "Temple of the Tooth in Kandy houses the sacred relic..."
Grade: IRRELEVANT

Question: "Things to do in Ella"
Document: "Ella is known for hiking and tea plantations but this doesn't cover specific activities..."
Grade: PARTIAL
"""


def requires_external_info(query: str) -> bool:
    """
    Check if query requires information not in the knowledge base.

    These topics need web search:
    - Hotels, accommodations, resorts
    - Restaurants, food places
    - Current prices, fees
    - Transportation, tickets
    - Weather, current conditions
    - Recent events, news

    Args:
        query: User's query string

    Returns:
        True if web search is needed
    """
    import re
    query_lower = query.lower()

    # Topics that require web search (not in knowledge base)
    external_topics = [
        r"hotel[s]?", r"resort[s]?", r"accommodation[s]?", r"stay", r"lodge",
        r"hostel[s]?", r"guesthouse[s]?", r"airbnb", r"booking",
        r"restaurant[s]?", r"food", r"eat", r"dining", r"cafe",
        r"price[s]?", r"cost[s]?", r"fee[s]?", r"ticket[s]?", r"budget",
        r"transport", r"bus", r"train", r"taxi", r"uber", r"tuk.?tuk",
        r"weather", r"forecast", r"rain", r"temperature",
        r"open", r"closed", r"hours", r"timing",
        r"current", r"today", r"now", r"latest", r"recent",
        r"book", r"reserve", r"available", r"availability",
        r"near", r"nearby", r"nearest", r"close to", r"around"
    ]

    for pattern in external_topics:
        if re.search(pattern, query_lower):
            return True

    return False


def calculate_relevance_score(
    query: str,
    documents: List[RetrievedDocument],
    target_location: Optional[str] = None
) -> Dict:
    """
    Calculate overall relevance score for retrieved documents.

    This function uses heuristics when LLM is unavailable:
    1. Check similarity scores from retrieval
    2. Verify location matching
    3. Assess document count and diversity
    4. Check if query needs external information

    Args:
        query: User's original query
        documents: List of retrieved documents
        target_location: Expected location (if specified)

    Returns:
        Dict with relevance assessment
    """
    # Check if query requires external info (hotels, prices, etc.)
    needs_external = requires_external_info(query)

    if not documents:
        return {
            "relevance": DocumentRelevance.INSUFFICIENT,
            "score": 0.0,
            "reason": "No documents retrieved",
            "needs_web_search": True
        }

    # Check if documents are from fallback (mock data)
    is_fallback = all(d.get("source") == "fallback" for d in documents)

    # Calculate average similarity
    avg_score = sum(d["relevance_score"] for d in documents) / len(documents)
    top_score = max(d["relevance_score"] for d in documents)

    # Check location matching
    if target_location:
        location_matches = sum(
            1 for d in documents
            if target_location.lower() in d.get("metadata", {}).get("location", "").lower()
        )
        location_match_ratio = location_matches / len(documents)
    else:
        location_match_ratio = 1.0  # No location constraint

    # Combined scoring
    combined_score = (avg_score * 0.4 + top_score * 0.4 + location_match_ratio * 0.2)

    # Force web search for external topics or fallback data
    if needs_external:
        return {
            "relevance": DocumentRelevance.PARTIAL,
            "score": combined_score,
            "avg_similarity": avg_score,
            "top_similarity": top_score,
            "location_match_ratio": location_match_ratio,
            "document_count": len(documents),
            "reason": "Query requires external information (hotels/prices/nearby/etc.)",
            "needs_web_search": True
        }

    if is_fallback:
        return {
            "relevance": DocumentRelevance.INSUFFICIENT,
            "score": combined_score,
            "avg_similarity": avg_score,
            "top_similarity": top_score,
            "location_match_ratio": location_match_ratio,
            "document_count": len(documents),
            "reason": "Using fallback data - vector DB not available",
            "needs_web_search": True
        }

    # Determine relevance level
    if combined_score >= 0.75 and len(documents) >= 3:
        relevance = DocumentRelevance.RELEVANT
        needs_web_search = False
        reason = "Documents are highly relevant and sufficient"
    elif combined_score >= 0.6:
        relevance = DocumentRelevance.PARTIAL
        needs_web_search = False
        reason = "Documents provide partial information"
    elif combined_score >= 0.4:
        relevance = DocumentRelevance.PARTIAL
        needs_web_search = True
        reason = "Documents may need supplementation from web search"
    else:
        relevance = DocumentRelevance.IRRELEVANT
        needs_web_search = True
        reason = "Documents are not relevant enough"

    return {
        "relevance": relevance,
        "score": combined_score,
        "avg_similarity": avg_score,
        "top_similarity": top_score,
        "location_match_ratio": location_match_ratio,
        "document_count": len(documents),
        "reason": reason,
        "needs_web_search": needs_web_search
    }


async def grade_documents_llm(
    query: str,
    documents: List[RetrievedDocument],
    llm
) -> List[Dict]:
    """
    Use LLM to grade individual documents.

    Args:
        query: User's query
        documents: Documents to grade
        llm: LangChain LLM instance

    Returns:
        List of grading results
    """
    grades = []

    for doc in documents[:5]:  # Grade top 5 only
        try:
            prompt = f"""Question: {query}

Document:
{doc['content'][:500]}...

Grade this document's relevance (RELEVANT, PARTIAL, or IRRELEVANT):"""

            response = await llm.ainvoke([
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])

            grade_str = response.content.strip().upper()
            grades.append({
                "document": doc,
                "grade": grade_str,
                "graded_by": "llm"
            })
        except Exception as e:
            logger.warning(f"LLM grading failed: {e}")
            grades.append({
                "document": doc,
                "grade": "PARTIAL",  # Default to partial
                "graded_by": "fallback"
            })

    return grades


async def grader_node(state: GraphState, llm=None) -> GraphState:
    """
    Grader Node: Assess relevance of retrieved documents.

    This node implements the self-correction mechanism by evaluating
    whether the retrieved documents are sufficient to answer the query.
    If not, it sets the needs_web_search flag to trigger fallback.

    Args:
        state: Current graph state
        llm: Optional LLM for detailed grading

    Returns:
        Updated GraphState with relevance assessment

    Research Note:
        The grader is a key component of "Reflective RAG" - it enables
        the agent to recognize its own knowledge gaps and take corrective
        action rather than generating hallucinated responses.
    """
    query = state["user_query"]
    documents = state.get("retrieved_documents", [])
    target_location = state.get("target_location")

    logger.info(f"Grading {len(documents)} documents for query: {query[:50]}...")

    # Calculate heuristic relevance score
    relevance_result = calculate_relevance_score(query, documents, target_location)

    # If LLM available, do detailed grading (but don't override external info requirement)
    if llm and documents and not requires_external_info(query):
        try:
            llm_grades = await grade_documents_llm(query, documents, llm)

            # Count relevant documents
            relevant_count = sum(1 for g in llm_grades if g["grade"] == "RELEVANT")
            partial_count = sum(1 for g in llm_grades if g["grade"] == "PARTIAL")

            # Override heuristic if LLM finds good documents
            if relevant_count >= 2:
                relevance_result["relevance"] = DocumentRelevance.RELEVANT
                relevance_result["needs_web_search"] = False
            elif relevant_count >= 1 or partial_count >= 2:
                relevance_result["relevance"] = DocumentRelevance.PARTIAL
                relevance_result["needs_web_search"] = False

            relevance_result["llm_grades"] = llm_grades

        except Exception as e:
            logger.warning(f"LLM grading failed: {e}")

    logger.info(f"Relevance: {relevance_result['relevance'].value}, "
               f"Score: {relevance_result['score']:.3f}, "
               f"Needs web search: {relevance_result['needs_web_search']}")

    # Update state
    return {
        **state,
        "document_relevance": relevance_result["relevance"],
        "needs_web_search": relevance_result["needs_web_search"],
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [{
            "timestamp": datetime.now().isoformat(),
            "check_type": "grader",
            "input_context": {
                "query": query,
                "document_count": len(documents),
                "target_location": target_location
            },
            "result": relevance_result["relevance"].value,
            "details": relevance_result["reason"],
            "action_taken": "trigger_web_search" if relevance_result["needs_web_search"] else None
        }]
    }


def route_after_grading(state: GraphState) -> str:
    """
    Routing function: Decide next step based on grading result.

    Args:
        state: Current graph state

    Returns:
        Next node name: "web_search" or "shadow_monitor"
    """
    if state.get("needs_web_search", False):
        return "web_search"
    else:
        return "shadow_monitor"
