"""
Generator Node: LLM Response Generation with Context Injection.

This node synthesizes all gathered information (retrieved documents,
web search results, constraint checks) into a coherent response.

Research Pattern:
    Context-Aware Generation - The generator receives structured context
    from multiple sources and produces a response that addresses the
    user's query while respecting detected constraints.

Response Types:
    - GREETING: Casual, friendly response
    - TOURISM_QUERY: Informative response with retrieved knowledge
    - TRIP_PLANNING: Structured itinerary with optimizations
    - OFF_TOPIC: Polite redirect to tourism topics
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional

# Retry logic for reliability
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

# Tracing
try:
    from ...utils.tracing import trace_node
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False
    def trace_node(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from ..state import GraphState, IntentType, ShadowMonitorLog
from ...config import settings

logger = logging.getLogger(__name__)

# System prompts for different response types
SYSTEM_PROMPTS = {
    "tourism_guide": """You are Travion 🇱🇰, an expert AI tour guide for Sri Lanka. Think of yourself as a knowledgeable local friend who loves sharing insider tips — warm, conversational, and genuinely excited about Sri Lanka.

🚫 STRICT SCOPE RULE:
You ONLY answer questions related to Sri Lanka tourism — destinations, culture, food, transport, accommodation, weather, activities, safety, visas, and travel planning within Sri Lanka.
If a question is NOT related to Sri Lanka tourism, politely decline and redirect. Do NOT answer questions about other countries, coding, finance, politics, recipes (unless Sri Lankan food), or any non-travel topic.

💬 Conversation Style:
- Talk like a friendly, knowledgeable local guide — not a textbook or Wikipedia article
- Use "you", "I'd suggest", "honestly", "trust me", "one thing I love about this place" — make it feel personal
- Keep it conversational and engaging, not formal or robotic
- If the user seems excited, match their energy! If they're asking practical questions, be helpful and direct

📋 Response Format (use markdown — this app renders it):
- Use **bold** for place names, key tips, and important warnings
- Use bullet points (•) for lists of tips, options, or steps
- Use relevant emojis naturally throughout: 🏛️ temples, 🌅 sunrise/sunset, ⏰ timing, 💡 pro tips, ⚠️ warnings, 🎫 tickets, 👕 dress code, 📸 photo spots, 🌧️ weather, 👥 crowds, 💰 costs, 🚗 transport, 🍛 food, 🏨 hotels, 🐘 wildlife
- Keep responses focused — 3-6 bullet points for practical questions, a short paragraph for simple ones
- End with a friendly follow-up nudge like "Want me to suggest the best time to visit?" or "Shall I add this to your itinerary?"

Current context will be provided. Only use facts from the context — do not invent details.""",

    "tourism_guide_location": """You are Travion 🇱🇰, a friendly AI tour guide helping the user explore **{location_name}**. You're like that one local friend who knows every hidden gem, the best time to visit, and what to absolutely avoid.

🚫 STRICT SCOPE RULE:
You ONLY answer questions related to Sri Lanka tourism — and primarily about {location_name} in this conversation. If asked about other countries or non-travel topics, politely decline and redirect back to {location_name} or Sri Lanka.

💬 Conversation Style:
- Be personal and conversational — "I'd really recommend...", "Honestly, the best time is...", "Here's a tip most tourists miss..."
- Match the user's tone — excited, curious, or practical
- Reference previous messages in the conversation naturally

📋 Response Format (use markdown — this app renders it):
- Use **bold** for {location_name} highlights, must-see spots, and warnings
- Use bullet points for tips, options, timings, and comparisons
- Use emojis throughout: 🏛️ 🌅 ⏰ 💡 ⚠️ 🎫 👕 📸 🌧️ 👥 💰 🚗 🍛 🏨 🐘
- Keep it tight — 3-6 bullets for practical info, 2-3 sentences for simple answers
- End with a helpful follow-up like "Want me to help plan your full day at {location_name}?" or "Should I check crowd levels for your visit date?"

Only use facts from the provided context. Do not make up opening hours, prices, or distances.""",

    "trip_planner": """You are Travion 🇱🇰, a friendly Sri Lanka travel planner who creates practical, well-timed itineraries. You think about golden hour photography, crowd levels, Poya days, and travel time between stops.

🚫 STRICT SCOPE RULE:
Only plan trips within Sri Lanka. If asked to plan trips to other countries, politely decline.

💬 Style:
- Be enthusiastic but practical — "This is a great combo!", "Start early — crowds build up fast here"
- Give honest advice: "Skip X if you're short on time", "This one is worth the detour"

📋 Itinerary Format:
- Use **bold** for location names and key timings
- Use emojis for each stop: 🌅 early morning, ☀️ morning, 🌞 midday, 🌆 afternoon, 🌙 evening
- ⏰ for specific times, ⚠️ for Poya/crowd warnings, 💡 for insider tips
- Number each day clearly: **Day 1**, **Day 2**, etc.
- Format stops as: `⏰ 6:30 AM — 🏛️ **Sigiriya Rock** — beat the heat and crowds`
- End with a **Quick Tips** section covering: best transport option, what to pack, and one thing not to miss

Focus on practical, optimised plans. Real timings, real tips.""",

    "greeting": """You are Travion 🇱🇰, a friendly AI tour guide for Sri Lanka.

Respond warmly and briefly (2-3 sentences max).
Use a welcoming emoji like 👋 or 🙏
If the user mentions a specific Sri Lankan location (e.g. Sigiriya, Kandy, Galle, Ella, Colombo, Mirissa), reference that location by name and one iconic thing about it.
Otherwise, mention one famous Sri Lankan experience to spark curiosity.
Ask what they'd like help with — planning a trip, finding a location, or learning about Sri Lanka.
Use markdown: **bold** for location names, a friendly conversational tone.""",

    "greeting_location": """You are Travion 🇱🇰, a friendly AI tour guide for Sri Lanka.

The user is interested in **{location_name}**. Respond warmly in 2-3 sentences.
Reference **{location_name}** directly — mention one iconic, specific thing about it (ancient ruins, golden beach, wildlife, tea estates, fort, etc.).
Use 👋 or 🙏 emoji.
Invite them to ask anything about {location_name} — best time to visit, what to see, how to get there, etc.
Use **bold** for {location_name} and key highlights.""",

    "off_topic": """You are Travion 🇱🇰, an AI tour guide specialised exclusively in Sri Lanka tourism.

The user has asked something outside your scope. Respond briefly (2-3 sentences):
- Politely explain you only cover Sri Lanka travel topics
- Use a friendly, non-judgmental tone with 😊 or 🙏
- Immediately suggest a related Sri Lanka topic they might enjoy
- Use **bold** for the Sri Lanka topic you suggest

Do NOT answer the off-topic question at all — just redirect warmly.""",

    "image_query": """You are Travion 🇱🇰, an expert AI tour guide for Sri Lanka with visual search capabilities. The user is looking for images or has asked about the visual appearance of a destination.

You have been provided with IMAGE SEARCH RESULTS from a CLIP-based visual search engine that matched the user's query against a curated collection of Sri Lankan tourism images.

📸 Response Guidelines:
- Present the matched images naturally — describe what the user will see at each location
- For EACH image result, include the image URL using markdown image syntax: ![Location Name](image_url)
- Highlight the **top match** and explain why it's the best result
- Add practical tips: best time for photos, lighting conditions, what to bring
- If multiple locations matched, briefly compare them
- Use emojis: 📸 photos, 🌅 golden hour, 🏛️ heritage, 🏖️ beach, 🌿 nature, 🐘 wildlife
- End with a suggestion like "Want me to plan a photography trip here?" or "Should I find more images of this area?"

💡 Image Presentation Format:
For each image result, use this format:
📍 **Location Name** (match: X%)
Description of what the user will see
![Location Name](image_url)
💡 Pro tip about photography/visiting

Only use facts from the provided context. Do not invent image URLs or descriptions.""",

    "image_query_location": """You are Travion 🇱🇰, a friendly AI tour guide helping the user explore **{location_name}** with visual search.

The user is looking for images of {location_name} or has uploaded a photo related to this location. You have been provided with IMAGE SEARCH RESULTS from a CLIP-based visual search.

📸 Response Guidelines:
- Present the matched images of **{location_name}** — describe what each photo shows
- For EACH image result, include the image URL using markdown: ![{location_name}](image_url)
- Add photography tips specific to {location_name}: best angles, time of day, viewpoints
- Use emojis: 📸 🌅 🏛️ 🏖️ 🌿 🐘
- End with a follow-up like "Want me to suggest the best photo spots at {location_name}?"

Only use facts from the provided context. Do not invent image URLs.""",

    "image_upload": """You are Travion 🇱🇰, an expert AI tour guide for Sri Lanka with visual recognition capabilities.

The user has UPLOADED a photo. You matched it against the Sri Lankan tourism image collection using CLIP visual embeddings. The IMAGE SEARCH RESULTS show the most visually similar destinations.

📸 Response Guidelines:
- Tell the user what their photo looks like / what destination it matches
- Present the top matches: "Your photo looks most like **Location Name**! Here's what I found..."
- For each match, show the reference image: ![Location Name](image_url)
- Explain what makes the match (architecture, landscape, colours, features)
- Add practical visit info for the top match: how to get there, best time, entry fees
- Use emojis: 📸 🔍 🏛️ 🏖️ 🌿 ✨
- End with "Want to plan a visit to this place?" or "Should I find more similar spots?"

Only use facts from the provided context. Do not invent details.""",

    "image_rejected": """You are Travion 🇱🇰, a friendly AI tour guide for Sri Lanka.

The user uploaded an image that does NOT appear to be a Sri Lankan tourism destination. Respond politely:
- Acknowledge that you received their image
- Explain that you specialise in Sri Lankan tourism destinations and the image didn't match
- Suggest what kinds of images work: photos of temples, beaches, historical sites, nature, wildlife, landscapes in Sri Lanka
- Offer to help with text-based search instead: "Try asking me to show photos of a specific place!"
- Keep it brief, warm, and helpful (3-4 sentences)
- Use 📸 and 😊 emojis"""
}


def build_context_string(state: GraphState) -> str:
    """
    Build a context string from all available information.

    Args:
        state: Current graph state

    Returns:
        Formatted context string for LLM injection
    """
    parts = []

    # Add retrieved documents
    docs = state.get("retrieved_documents", [])
    if docs:
        parts.append("=== KNOWLEDGE BASE ===")
        for i, doc in enumerate(docs[:5], 1):
            location = doc.get("metadata", {}).get("location", "Unknown")
            parts.append(f"\n[{i}] {location}:")
            parts.append(doc["content"][:400])
        parts.append("")

    # Add web search results
    web_results = state.get("web_search_results", [])
    if web_results:
        parts.append("=== WEB SEARCH RESULTS ===")
        for r in web_results[:3]:
            parts.append(f"\n- {r.get('title', 'Untitled')}:")
            parts.append(f"  {r.get('content', '')[:300]}")
        parts.append("")

    # Add constraint information
    constraints = state.get("_constraint_results")
    if constraints and isinstance(constraints, dict):
        parts.append("=== CONSTRAINT ANALYSIS ===")
        parts.append(constraints.get("recommendation", ""))
        parts.append("")

    # Add image search results
    image_results = state.get("image_search_results", [])
    if image_results:
        parts.append("=== IMAGE SEARCH RESULTS ===")
        for i, img in enumerate(image_results[:5], 1):
            parts.append(
                f"\n[Image {i}] {img['location_name']} "
                f"(similarity: {img['similarity_score']:.3f})"
            )
            parts.append(f"  Description: {img['description'][:200]}")
            if img.get("image_url"):
                parts.append(f"  Image URL: {img['image_url']}")
            if img.get("tags"):
                parts.append(f"  Tags: {img['tags']}")
        parts.append("")

    # Add itinerary if generated
    itinerary = state.get("itinerary")
    if itinerary:
        parts.append("=== OPTIMIZED SCHEDULE ===")
        for slot in itinerary:
            parts.append(f"- {slot['time']}: {slot['location']}")
            parts.append(f"  Crowd: {slot['crowd_prediction']}%, Lighting: {slot['lighting_quality']}")
            if slot.get("notes"):
                parts.append(f"  Note: {slot['notes']}")
        parts.append("")

    return "\n".join(parts)


def build_source_attribution(state: GraphState) -> str:
    """
    Build a user-friendly source attribution footer for the response.

    Shows the user where the information came from:
    - Vector DB (knowledge base documents)
    - Web search results with links
    - Real-time API data

    Args:
        state: Current graph state

    Returns:
        Formatted source attribution string (empty string if no sources)
    """
    sources = []

    # Vector DB / knowledge base sources
    docs = state.get("retrieved_documents", [])
    kb_locations = []
    for doc in docs[:5]:
        metadata = doc.get("metadata", {})
        location = metadata.get("location", "")
        doc_source = metadata.get("source", "")
        if location and location not in kb_locations:
            kb_locations.append(location)
    if kb_locations:
        sources.append(f"🗄️ **Knowledge Base:** {', '.join(kb_locations)}")

    # Web search sources with links
    web_results = state.get("web_search_results", [])
    if web_results:
        sources.append("🌐 **Web Sources:**")
        for r in web_results[:3]:
            title = r.get("title", "Web Result")
            url = r.get("url", r.get("link", ""))
            if url:
                sources.append(f"   • [{title}]({url})")
            else:
                sources.append(f"   • {title}")

    # Weather / real-time API data
    weather_data = state.get("weather_data")
    if weather_data:
        sources.append("🌤️ **Live Weather:** OpenWeatherMap API")

    # Image search results (CLIP visual search)
    image_results = state.get("image_search_results", [])
    if image_results:
        img_locations = list(set(r.get("location_name", "") for r in image_results[:5]))
        sources.append(f"📸 **Image Search (CLIP):** {', '.join(img_locations)}")

    # Hotel / search candidates (MCP search)
    candidates = state.get("search_candidates", [])
    if candidates:
        sources.append("📍 **Place Data:** Google Maps / Tavily Search")

    if not sources:
        return ""

    return "\n\n---\n📚 **Sources**\n" + "\n".join(sources)


def get_system_prompt(
    intent: Optional[IntentType],
    target_location: Optional[str] = None,
    has_uploaded_image: bool = False,
    image_rejected: bool = False,
) -> str:
    """Get appropriate system prompt based on intent and location context."""
    if intent == IntentType.GREETING:
        if target_location:
            return SYSTEM_PROMPTS["greeting_location"].format(location_name=target_location)
        return SYSTEM_PROMPTS["greeting"]
    elif intent == IntentType.OFF_TOPIC:
        return SYSTEM_PROMPTS["off_topic"]
    elif intent == IntentType.IMAGE_QUERY or has_uploaded_image:
        # Image query variants
        if image_rejected:
            return SYSTEM_PROMPTS["image_rejected"]
        if has_uploaded_image:
            return SYSTEM_PROMPTS["image_upload"]
        if target_location:
            return SYSTEM_PROMPTS["image_query_location"].format(location_name=target_location)
        return SYSTEM_PROMPTS["image_query"]
    elif intent == IntentType.TRIP_PLANNING:
        return SYSTEM_PROMPTS["trip_planner"]
    else:
        # For tourism queries, use location-specific prompt if we have a target location
        if target_location:
            return SYSTEM_PROMPTS["tourism_guide_location"].format(location_name=target_location)
        return SYSTEM_PROMPTS["tourism_guide"]


@trace_node("generator", run_type="llm")
async def generator_node(state: GraphState, llm=None) -> GraphState:
    """
    Generator Node: Produce final response using LLM.

    This node synthesizes all gathered context into a coherent response.
    It handles different intents appropriately and incorporates constraint
    information when relevant.

    Args:
        state: Current graph state
        llm: LangChain LLM instance

    Returns:
        Updated GraphState with generated response

    Research Note:
        The generator implements "Grounded Generation" - it strictly uses
        provided context to reduce hallucination risk while maintaining
        natural, helpful responses.
    """
    import time as _time
    _start = _time.time()
    query = state["user_query"]
    intent = state.get("intent", IntentType.TOURISM_QUERY)
    target_location = state.get("target_location")
    correction_instructions = state.get("_correction_instructions")

    # Image query flags
    has_uploaded_image = bool(state.get("uploaded_image_base64"))
    image_rejected = state.get("uploaded_image_validated") is False
    has_image_results = bool(state.get("image_search_results"))

    logger.info(
        f"Generator processing intent: {intent.value}, target_location: {target_location}, "
        f"has_uploaded_image: {has_uploaded_image}, image_rejected: {image_rejected}, "
        f"has_image_results: {has_image_results}"
    )
    if correction_instructions:
        logger.info(f"Generator applying corrections: {correction_instructions[:100]}...")

    # Build context from all sources
    context = build_context_string(state)

    # Get appropriate system prompt (with location + image context)
    system_prompt = get_system_prompt(
        intent,
        target_location,
        has_uploaded_image=has_uploaded_image,
        image_rejected=image_rejected,
    )

    # Build user message with context
    if image_rejected:
        # Short-circuit: image was rejected by validator
        rejection_msg = state.get("image_validation_message", "The uploaded image doesn't appear to be a Sri Lankan tourism destination.")
        user_message = f"The user uploaded an image. Validation result: {rejection_msg}\n\nUser message: {query}"

    elif context and intent not in [IntentType.GREETING, IntentType.OFF_TOPIC]:
        # If we have a target location, explicitly mention it in the context
        location_context = f"\nYou are answering questions about: {target_location}\n" if target_location else ""

        # Include correction instructions if this is a regeneration
        correction_context = ""
        if correction_instructions:
            correction_context = f"\n\n=== IMPORTANT CORRECTIONS NEEDED ===\n{correction_instructions}\n"

        # Image-specific framing
        image_framing = ""
        if has_image_results and has_uploaded_image:
            image_framing = "\nThe user uploaded a photo. The IMAGE SEARCH RESULTS below show the most visually similar Sri Lankan destinations.\n"
        elif has_image_results:
            image_framing = "\nThe user is looking for images. The IMAGE SEARCH RESULTS below were found via CLIP visual search.\n"

        user_message = f"""Context Information:
{location_context}{image_framing}{context}{correction_context}
User Question: {query}

Please provide a helpful response based on the context above.{' Address the correction issues mentioned.' if correction_instructions else ''}"""
    else:
        user_message = query

    # Build conversation history from state messages
    # This enables multi-turn conversation memory
    messages = state.get("messages", [])
    conversation_messages = [{"role": "system", "content": system_prompt}]

    # Add previous messages from history (excluding the current query)
    # Messages are stored as [user, assistant, user, assistant, ...]
    # We include previous turns to maintain conversation context
    if len(messages) > 1:
        # Include up to last 10 messages (5 turns) for context
        history_messages = messages[:-1][-10:]
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["user", "assistant"] and content:
                conversation_messages.append({"role": role, "content": content})

    # Add current user message (with context if applicable)
    conversation_messages.append({"role": "user", "content": user_message})

    # Generate response with retry logic for reliability
    if llm:
        max_retries = getattr(settings, 'LLM_MAX_RETRIES', 3)
        retry_delay = getattr(settings, 'LLM_RETRY_DELAY', 1.0)
        generated_text = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await llm.ainvoke(conversation_messages)
                generated_text = response.content
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                logger.warning(f"LLM generation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    import asyncio
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        if generated_text is None:
            logger.error(f"LLM generation failed after {max_retries} attempts: {last_error}")
            generated_text = generate_fallback_response(state)
    else:
        generated_text = generate_fallback_response(state)

    # Sources are passed via API metadata (not injected inline into response text)

    # Log generation
    log_entry = ShadowMonitorLog(
        timestamp=datetime.now().isoformat(),
        check_type="generator",
        input_context={
            "intent": intent.value,
            "context_length": len(context),
            "has_constraints": bool(state.get("constraint_violations"))
        },
        result="generated",
        details=f"Response length: {len(generated_text)} chars",
        action_taken=None
    )

    # Add assistant message to conversation history for future turns
    # This enables the conversation memory to persist across messages
    assistant_message = {"role": "assistant", "content": generated_text}

    _duration_ms = (_time.time() - _start) * 1000
    is_correction = bool(correction_instructions)
    return {
        **state,
        "generated_response": generated_text,
        "messages": [assistant_message],  # Will be appended via operator.add
        "step_results": [{
            "node": "generator",
            "status": "success",
            "summary": f"Generated {len(generated_text)} chars | Intent: {intent.value} | Location: {target_location or 'general'} | Context: {len(context)} chars | Correction: {is_correction}",
            "duration_ms": round(_duration_ms, 2),
        }],
        "shadow_monitor_logs": state.get("shadow_monitor_logs", []) + [log_entry]
    }


def generate_fallback_response(state: GraphState) -> str:
    """
    Generate a response without LLM (rule-based fallback).

    Args:
        state: Current graph state

    Returns:
        Fallback response string
    """
    intent = state.get("intent", IntentType.TOURISM_QUERY)
    target_location = state.get("target_location")
    image_rejected = state.get("uploaded_image_validated") is False
    image_results = state.get("image_search_results", [])

    # Handle image rejection
    if image_rejected:
        rejection_msg = state.get("image_validation_message", "")
        return (
            f"📸 Thanks for sharing that image! Unfortunately, it doesn't appear to be "
            f"a Sri Lankan tourism destination.\n\n"
            f"{rejection_msg}\n\n"
            f"💡 **Try these instead:**\n"
            f"• Upload a photo of a temple, beach, fortress, or nature spot in Sri Lanka\n"
            f"• Or ask me to show photos — e.g., *\"Show me photos of Sigiriya\"*\n\n"
            f"I'm happy to help you explore Sri Lanka visually! 😊"
        )

    # Handle image query with results
    if intent == IntentType.IMAGE_QUERY and image_results:
        parts = ["📸 Here are the best visual matches I found!\n"]
        for i, img in enumerate(image_results[:5], 1):
            score_pct = int(img["similarity_score"] * 100)
            parts.append(f"**{i}. {img['location_name']}** (match: {score_pct}%)")
            if img.get("description"):
                parts.append(f"   {img['description'][:150]}")
            if img.get("image_url"):
                parts.append(f"   ![{img['location_name']}]({img['image_url']})")
            parts.append("")
        parts.append("Want me to tell you more about any of these places? 🌴")
        return "\n".join(parts)

    if intent == IntentType.GREETING:
        if target_location:
            location_greetings = {
                "sigiriya": ("Ayubowan! 🦁 Sigiriya awaits — the ancient Lion Rock fortress "
                             "rising 200m above the jungle is truly unforgettable. "
                             "I'm Travion, your AI guide. How can I help you plan your Sigiriya visit?"),
                "kandy": ("Ayubowan! 🙏 Kandy, Sri Lanka's cultural capital and home of the "
                          "sacred Temple of the Tooth Relic, is a wonderful destination. "
                          "I'm Travion — let me help you make the most of your time there!"),
                "galle": ("Ayubowan! 🏰 Galle's 17th-century Dutch fort is one of Asia's "
                          "best-preserved colonial landmarks. I'm Travion, ready to guide "
                          "you through every corner of this historic coastal gem!"),
                "ella": ("Ayubowan! 🌿 Ella's misty tea highlands, the iconic Nine Arches Bridge, "
                         "and stunning mountain hikes are waiting for you. "
                         "I'm Travion — let's plan your perfect Ella escape!"),
                "colombo": ("Ayubowan! 🌆 Colombo blends colonial heritage with a buzzing modern "
                            "city vibe — great food, markets, and the sea. "
                            "I'm Travion, here to help you discover the best of the capital!"),
                "mirissa": ("Ayubowan! 🐋 Mirissa is Sri Lanka's top spot for whale watching "
                            "and dreamy palm-fringed beaches. "
                            "I'm Travion — let me help you plan the perfect coastal getaway!"),
                "dambulla": ("Ayubowan! ⛩️ The Dambulla Cave Temples are a breathtaking UNESCO "
                             "World Heritage Site filled with ancient Buddha statues and murals. "
                             "I'm Travion — ready to guide your visit!"),
                "nuwara eliya": ("Ayubowan! 🍵 Nuwara Eliya's rolling tea estates and cool mountain "
                                 "air make it Sri Lanka's 'Little England'. "
                                 "I'm Travion — let's explore the hill country together!"),
                "trincomalee": ("Ayubowan! 🌊 Trincomalee's crystal-clear waters, coral reefs, "
                                "and beautiful beaches are some of Sri Lanka's best-kept secrets. "
                                "I'm Travion — let me help plan your east coast adventure!"),
                "jaffna": ("Ayubowan! 🏺 Jaffna offers a unique window into Sri Lanka's rich "
                           "Tamil heritage, ancient temples, and vibrant culture. "
                           "I'm Travion — let's discover the north together!"),
            }
            location_key = target_location.lower()
            for key, message in location_greetings.items():
                if key in location_key or location_key in key:
                    return message
            # Generic location-specific fallback
            return (f"Ayubowan! 👋 {target_location} is a wonderful destination in Sri Lanka. "
                    f"I'm Travion, your AI tour guide — let me help you plan an unforgettable "
                    f"visit to {target_location}!")
        return ("Ayubowan! 👋 Welcome to Travion, your AI guide to Sri Lanka. "
                "From ancient Sigiriya to the misty hills of Ella and the golden shores of Mirissa — "
                "I'm here to help you explore it all. Where would you like to go?")

    elif intent == IntentType.OFF_TOPIC:
        return (
            "😊 Hey! I'm **Travion**, your dedicated Sri Lanka travel guide — "
            "I'm only set up to help with Sri Lanka tourism topics like destinations, "
            "culture, transport, accommodation, and trip planning.\n\n"
            "💡 **How about we explore Sri Lanka instead?** I can help you with:\n"
            "• 🏛️ Must-see cultural sites like **Sigiriya** or **Kandy**\n"
            "• 🏖️ Best beaches — **Mirissa**, **Unawatuna**, or **Arugam Bay**\n"
            "• 🗺️ Planning a custom itinerary for your trip\n\n"
            "What would you like to discover? 🇱🇰"
        )

    elif intent == IntentType.TRIP_PLANNING:
        itinerary = state.get("itinerary", [])
        constraints = state.get("_constraint_results", {})
        location_label = f"**{target_location}**" if target_location else "**Sri Lanka**"

        response_parts = [f"🗺️ Here's your optimised plan for visiting {location_label}!\n"]

        if constraints.get("recommendation"):
            response_parts.append(f"⚠️ **Heads up:** {constraints['recommendation']}\n")

        if itinerary:
            response_parts.append("**Suggested Schedule:**\n")
            for slot in itinerary:
                line = f"⏰ **{slot['time']}** — 📍 **{slot['location']}**"
                if slot.get("notes"):
                    line += f" _(💡 {slot['notes']})_"
                response_parts.append(line)
        else:
            response_parts.append(
                f"I'd love to build you a full itinerary for {location_label}! "
                "Just let me know how many days you have and what you enjoy most — "
                "culture, beaches, wildlife, or hiking? 🏔️"
            )

        return "\n".join(response_parts)

    else:
        # Tourism query - use retrieved docs
        docs = state.get("retrieved_documents", [])
        if docs:
            best_doc = docs[0]
            location = best_doc.get("metadata", {}).get("location", target_location or "Sri Lanka")
            content_preview = best_doc["content"][:400].strip()
            return (
                f"Here's what I know about **{location}** 📍\n\n"
                f"{content_preview}\n\n"
                f"💡 Want me to go deeper on any of this — best time to visit, "
                f"how to get there, or what to eat nearby?"
            )
        else:
            loc_label = f"**{target_location}**" if target_location else "**Sri Lanka**"
            return (
                f"Great question about {loc_label}! 🌴 I want to give you the most "
                f"accurate answer — could you be a little more specific about what "
                f"you're looking for?\n\n"
                f"For example:\n"
                f"• 🕐 Best time to visit?\n"
                f"• 🎫 Entry fees and tickets?\n"
                f"• 🚗 How to get there?\n"
                f"• 📸 Best photo spots?\n\n"
                f"Just let me know and I'll help you out! 😊"
            )
