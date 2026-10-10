# import logfire
# from langchain_groq import ChatGroq
# from nemoguardrails import RailsConfig, LLMRails

# from app.config import settings
# from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


# _rails: LLMRails | None = None


# def initialize_rails() -> None:
#     """
#     Build the NeMo LLMRails singleton at app startup.
#     Uses llama-3.1-8b-instant for fast intent classification at the gate —
#     the heavier llama-3.3-70b-versatile is reserved for the RAG pipeline.
#     """
#     global _rails

#     guard_llm = ChatGroq(
#         api_key=settings.GROQ_API_KEY,
#         model="openai/gpt-oss-20b",
#         temperature=0
#     )

#     config = RailsConfig.from_content(
#         colang_content=COLANG_CONTENT,
#         yaml_content=YAML_CONTENT
#     )

#     _rails = LLMRails(config, llm=guard_llm)
#     logfire.info("🛡️ NeMo Guardrails initialised (openai/gpt-oss-20b).")
    
    


# def guard(message: str) -> tuple[bool, str | None]:
#     """
#     Run a user message through the NeMo rails gate.

#     Returns:
#         (True,  rail_response) — a rail fired; return this response immediately,
#                                 skip the RAG pipeline entirely.
#         (False, None)          — message is clean; proceed to LangGraph.
#     """
#     if _rails is None:
#         logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
#         return False, None

#     with logfire.span("🛡️ Guardrails Check"):
#         result = _rails.generate(messages=[{"role": "user", "content": message}])

#         # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
#         content = result.get("content", "") if isinstance(result, dict) else str(result)

#         fired = any(indicator in content for indicator in RAIL_INDICATORS)

#         if fired:
#             logfire.info(f"🛡️ Guardrails fired | query='{message[:80]}'")
#             return True, content

#         logfire.info("✅ Guardrails passed.")
#         return False, None









import logfire

from app.guardrails.classifier import (
    classify_message,
    initialize_classifier,
)
from app.guardrails.schemas import GuardDecision


FIXED_RESPONSES = {
    "greeting": (
        "Hello! I'm your Enterprise IT Assistant. "
        "I specialise in Kubernetes, Intel hardware, "
        "and enterprise networking. What can I help you with today?"
    ),
    "capabilities": (
        "I'm your Enterprise IT Assistant. I can help with "
        "Kubernetes deployment, scaling and networking; "
        "Intel CPUs, FPGAs, NICs and SR-IOV; and enterprise "
        "networking concepts such as SDN, VLANs, BGP and routing."
    ),
    "farewell": (
        "Goodbye! Feel free to return whenever you have more "
        "enterprise IT questions. Have a great day!"
    ),
    "jailbreak": (
        "I can't follow instructions that attempt to override "
        "my operating rules. I can still help with enterprise IT."
    ),
    "out_of_scope": (
        "I'm an Enterprise IT Assistant focused on Kubernetes, "
        "Intel hardware, and enterprise networking. "
        "Please ask me a question in one of those areas."
    ),
    "ambiguous": (
        "I couldn't confidently classify that request. "
        "Please rephrase it as a clear enterprise IT question."
    ),
}


def initialize_rails() -> None:
    initialize_classifier()
    logfire.info("Guard and classifier initialized.")


def guard(message: str) -> tuple[GuardDecision, str | None]:
    """
    Returns:
        (decision, response)

    response is populated when the request should be handled
    here instead of being sent to LangGraph.
    """

    decision = classify_message(message)

    logfire.info(
        "Guard decision",
        category=decision.category,
        action=decision.action,
        reason=decision.reason,
    )

    # These messages are answered directly.
    if decision.category in {
        "greeting",
        "capabilities",
        "farewell",
    }:
        return decision, FIXED_RESPONSES[decision.category]

    # Blocked or uncertain requests must not reach the Planner.
    if decision.action in {"block", "review"}:
        response = FIXED_RESPONSES.get(
            decision.category,
            FIXED_RESPONSES["ambiguous"],
        )
        return decision, response

    # In-scope technical questions continue to LangGraph.
    return decision, None
