import logfire
from app.agents.state import AgentState
from app.gateway.client import portkey_client, extract_cache_status


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state.get("documents", []):
            remaining = max_context_chars - len(full_context)

            if remaining <= 0:
                logfire.warning("Context limit reached.")
                break

            if len(doc) + 2 <= remaining:
                full_context += doc + "\n\n"
            else:
                full_context += doc[:max(0, remaining - 2)] + "\n\n"
                logfire.warning("Last document partially truncated.")
                break

        logfire.info(
            "Generation context inspection",
            query=query,
            document_count=len(state.get("documents", [])),
            context_length=len(full_context),
            documents=[
                {
                    "index": i + 1,
                    "length": len(doc),
                    "content": doc[:1000],
                }
                for i, doc in enumerate(state.get("documents", []))
            ],
        )

        
        prompt = f"""
        You are a Senior Technical Architect answering questions using retrieved technical documentation.

        IMPORTANT ANSWERING RULES:
        1. Treat the TECHNICAL CONTEXT as the primary source of truth.
        2. Answer the user's exact question directly before adding extra details.
        3. If the context provides specific commands, code, configuration,
        or procedures, prioritize those exact instructions.
        4. Do not replace documented commands with alternative implementations
        unless the user explicitly asks for alternatives.
        5. Do not invent commands, configuration values, or facts that are
        unsupported by the retrieved context.
        6. If the context does not contain enough information to answer,
        clearly state what is missing.
        7. Keep answers concise and relevant. Avoid unnecessary tutorials.
        8. Use conversation history only to understand the user's intent
        and previous messages. Do not treat it as verified documentation.
        9. Do not claim a command or fact appears in the context unless
        it actually appears there.
        10. If the user asks for exact steps, present them as a numbered list
            or code block where appropriate.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"

        Provide a direct, context-grounded answer.
        """



    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = portkey_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = response.choices[0].message.content
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e