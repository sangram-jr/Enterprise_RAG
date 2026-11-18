import logfire

from app.agents.state import AgentState
#from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.hybrid_search import hybrid_search
from app.services.retrieval.ranking_service import rerank_documents


def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking
    for technical queries, with retrieval diagnostics.
    """

    query = state["current_query"]

    with logfire.span("Knowledge Retrieval"):

        logfire.info("Starting retrieval", query=query)

        # Phase 1: Vector Search
        #raw_results = search_enterprise_knowledge(query, limit=15)
        raw_results= hybrid_search(
            query,
            dense_limit=15,
            bm25_limit=15,
            final_limit=20
        )
        

        logfire.info(
            "Qdrant retrieval completed",
            candidate_count=len(raw_results),
            candidates=[
                {
                    "source": doc.get("source"),
                    "score": doc.get("score"),
                    "content_length": len(doc.get("content", "")),
                }
                for doc in raw_results
            ],
        )

        # Remove empty content
        doc_contents = [
            doc["content"]
            for doc in raw_results
            if doc.get("content", "").strip()
        ]

        # Phase 2: Semantic Reranking
        if doc_contents:
            with logfire.span("Semantic Reranking"):

                reranked_contents = rerank_documents(
                    query,
                    doc_contents,
                    top_n=8,
                )

                logfire.info(
                "FINAL RERANKED DOCUMENTS",
                query=query,
                input_count=len(doc_contents),
                output_count=len(reranked_contents),
                documents=[
                    {
                        "rank": i + 1,
                        "length": len(doc),
                        "preview": doc[:1500],
                        "contains_redis_pod": "redis-pod.yaml" in doc,
                        "contains_redis_service": "redis-service.yaml" in doc,
                        "contains_redis_cli": "kubectl run -i" in doc,
                    }
                    for i, doc in enumerate(reranked_contents)
                ],
            )
        else:
            reranked_contents = []

            logfire.warning(
                "No usable documents received from Qdrant"
            )

        # Phase 3: Format documents
        formatted_docs = [
            f"CONTENT: {doc}"
            for doc in reranked_contents
        ]

        logfire.info(
            "Retrieval node completed",
            final_document_count=len(formatted_docs),
        )

    return {
        "documents": formatted_docs,
        "status": (
            f"Found {len(formatted_docs)} technical contexts."
            if formatted_docs
            else "No relevant technical context found."
        ),
        "plan": state["plan"] + ["Context Retrieved"],
    }
