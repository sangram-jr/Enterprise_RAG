# # ============================================================
# # CRITICAL: logfire MUST be configured before ALL other imports
# # so that spans from all modules are captured from the start.
# # ============================================================
# import logfire
# import os
# from dotenv import load_dotenv

# load_dotenv()
# logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

# # Now safe to import app modules - logfire is already active
# from fastapi import FastAPI, Response
# from app.agents.graph import rag_agent
# from app.guardrails.rails import initialize_rails, guard

# from pydantic import BaseModel
# from typing import Optional


# # Initialize FastAPI
# app = FastAPI(title="Enterprise Agentic RAG API")


# @app.on_event("startup")
# def startup_event():
#     initialize_rails()

# class QueryRequest(BaseModel):
#     q: str
#     thread_id: Optional[str] = "default_user"
    
    
# @app.get("/")
# def home():
#     return {"message": "Enterprise LangGraph RAG API is live."}


# @app.get("/graph")
# def get_graph_image():
#     """
#     Returns the Mermaid image of the agent's workflow.
#     """
#     try:
#         png_bytes = rag_agent.get_graph().draw_mermaid_png()
#         return Response(content=png_bytes, media_type="image/png")
#     except Exception as e:
#         return {"error": f"Could not generate graph image: {e}"}
    
    
# @app.post("/query")
# def query(request: QueryRequest):
#     """
#     Executes the LangGraph RAG flow with memory using a POST request.
#     """
#     q = request.q
#     thread_id = request.thread_id

#     initial_state = {
#         "messages": [{"role": "user", "content": q}],
#         "current_query": q,
#         "documents": [],
#         "plan": ["Start"],
#         "status": "Initializing Graph..."
#     }
    
#     # Configuration for Memory (Thread ID)
#     config = {"configurable": {"thread_id": thread_id}}
    
#     try:
#         # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
#         rail_fired, rail_response = guard(q)
#         if rail_fired:
#             logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
#             return {
#                 "question": q,
#                 "answer": rail_response,
#                 "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
#                 "status": "Blocked by guardrails.",
#                 "sources": []
#             }

#         # Gate 2: LangGraph RAG pipeline
#         # Run the graph synchronously to preserve Logfire context variables
#         final_output = rag_agent.invoke(initial_state, config=config)
        
#         return {
#             "question": q,
#             "answer": final_output.get("final_answer"),
#             "thought_process": final_output.get("plan"),
#             "status": final_output.get("status"),
#             "sources": final_output.get("documents", [])
#         }
#     except Exception as e:
#         logfire.error(f"❌ Backend Execution Failed: {e}")
#         return {
#             "question": q,
#             "answer": "I apologize, but I encountered an internal error while processing your request. Please try again later.",
#             "thought_process": ["Error encountered during execution."],
#             "status": "error",
#             "sources": []
#         }






# ============================================================
# CRITICAL:
# logfire MUST be configured before importing app modules.
# This ensures spans from imported modules are captured.
# ============================================================

import os

import logfire
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")

if LOGFIRE_TOKEN:
    logfire.configure(
        token=LOGFIRE_TOKEN
    )
else:
    # Allows the application to run locally even when
    # Logfire token is not configured.
    logfire.configure(
        send_to_logfire=False
    )


# ============================================================
# APPLICATION IMPORTS
# ============================================================

from fastapi import FastAPI, Response
from pydantic import BaseModel
from typing import Optional

from app.agents.graph import rag_agent
from app.guardrails.rails import (
    initialize_rails,
    guard,
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Enterprise Agentic RAG backend with Guardrails and LangGraph.",
    version="1.0.0",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    logfire.info(
        "🚀 Starting Enterprise Agentic RAG API..."
    )

    # Initialize NeMo Guardrails once
    initialize_rails()

    logfire.info(
        "🛡️ Guardrails initialized successfully."
    )

    logfire.info(
        "🚀 Enterprise Agentic RAG API is ready."
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class QueryRequest(BaseModel):

    q: str

    thread_id: Optional[str] = "default_user"


# ============================================================
# HEALTH / HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Enterprise LangGraph RAG API is live.",
        "status": "healthy",
    }


# ============================================================
# GRAPH VISUALIZATION
# ============================================================

@app.get("/graph")
def get_graph_image():

    """
    Returns the Mermaid-generated PNG image
    of the LangGraph workflow.
    """

    try:

        png_bytes = (
            rag_agent
            .get_graph()
            .draw_mermaid_png()
        )

        return Response(
            content=png_bytes,
            media_type="image/png",
        )

    except Exception as e:

        logfire.error(
            f"❌ Could not generate graph image: {e}"
        )

        return {
            "error": f"Could not generate graph image: {e}"
        }



# ============================================================
# QUERY ENDPOINT
# ============================================================

@app.post("/query")
def query(request: QueryRequest):
    """
    Executes the Enterprise Agentic RAG workflow.

    Flow:
        User Query
             |
             v
        Guard Classifier
             |
        +----+-------------------+
        |                        |
        v                        v
    Direct Response          Technical Query
    or Block                      |
        |                         v
        v                     LangGraph
      Return                      |
                                  v
                               Planner
                                  |
                                  v
                              Retrieval
                                  |
                                  v
                               Reranking
                                  |
                                  v
                              Generation
                                  |
                                  v
                               Response
    """

    # --------------------------------------------------------
    # 1. Extract request
    # --------------------------------------------------------

    q = request.q.strip()
    thread_id = request.thread_id or "default_user"

    # --------------------------------------------------------
    # 2. Validate request
    # --------------------------------------------------------

    if not q:
        return {
            "question": q,
            "answer": "Please provide a question.",
            "thought_process": ["Invalid empty query."],
            "status": "invalid_request",
            "sources": [],
        }

    logfire.info(
        f"📥 Query received | thread={thread_id} | query='{q[:100]}'"
    )

    try:

        # ====================================================
        # GATE 1: GUARD CLASSIFIER
        # ====================================================

        with logfire.span("🛡️ Guardrails Gate"):
            decision, guard_response = guard(q)

        # ----------------------------------------------------
        # 3. Handle greetings, capabilities, farewells,
        #    blocked requests, and requests needing review.
        # ----------------------------------------------------

        if guard_response is not None:

            direct_categories = {
                "greeting",
                "capabilities",
                "farewell",
            }

            is_direct_response = (
                decision.category in direct_categories
                and decision.action == "allow"
            )

            if is_direct_response:
                status = "Handled directly by guardrails."
                logfire.info(
                    f"💬 Direct response | category={decision.category} "
                    f"| thread={thread_id}"
                )
            else:
                status = "Blocked by guardrails."
                logfire.warning(
                    f"🛑 Request blocked | category={decision.category} "
                    f"| action={decision.action} | thread={thread_id}"
                )

            return {
                "question": q,
                "answer": guard_response,
                "thought_process": [
                    f"Intent: {decision.category}",
                    f"Decision: {decision.action}",
                    "Planner: Skipped",
                    "Retrieval: Skipped",
                ],
                "status": status,
                "sources": [],
            }

        # ----------------------------------------------------
        # 4. Only explicitly allowed technical queries
        #    are permitted to reach LangGraph.
        # ----------------------------------------------------

        if (
            decision.action != "allow"
            or decision.category != "in_scope"
        ):
            logfire.warning(
                f"🛑 Request not authorized for RAG | "
                f"category={decision.category} | "
                f"action={decision.action}"
            )

            return {
                "question": q,
                "answer": (
                    "I couldn't safely classify that request. "
                    "Please rephrase it as a clear enterprise IT question."
                ),
                "thought_process": [
                    f"Intent: {decision.category}",
                    "Planner: Skipped",
                    "Retrieval: Skipped",
                ],
                "status": "Blocked by guardrails.",
                "sources": [],
            }

        # ====================================================
        # GATE 2: LANGGRAPH RAG PIPELINE
        # ====================================================

        initial_state = {
            "messages": [
                {
                    "role": "user",
                    "content": q,
                }
            ],
            "current_query": q,
            "documents": [],
            "plan": ["Start"],
            "status": "Initializing Graph...",
        }

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        logfire.info(
            f"✅ Guard passed | category={decision.category} "
            f"| thread={thread_id}"
        )

        with logfire.span("🤖 LangGraph RAG Pipeline"):
            final_output = rag_agent.invoke(
                initial_state,
                config=config,
            )

        # ====================================================
        # 5. BUILD FINAL RESPONSE
        # ====================================================

        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan", []),
            "status": final_output.get("status", "completed"),
            "sources": final_output.get("documents", []),
        }

    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception:
        logfire.exception(
            f"❌ Backend execution failed | thread={thread_id}"
        )

        return {
            "question": q,
            "answer": (
                "I apologize, but I encountered an internal error "
                "while processing your request. Please try again later."
            ),
            "thought_process": [
                "Error encountered during execution."
            ],
            "status": "error",
            "sources": [],
        }