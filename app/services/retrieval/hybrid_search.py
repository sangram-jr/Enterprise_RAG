# step1: Run Qdrant dense retrieval.
# step 2: Run BM25 retrieval.
# step 3: Combine the rankings using RRF.
# step 4: Return a unified list for FlashRank.



import logfire

from app.services.retrieval.qdrant_service import (
    search_enterprise_knowledge,
)
from app.services.retrieval.bm25_service import search_bm25


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    k: int = 60,
    limit: int = 20,
) -> list[dict]:
    """
    Combine dense and BM25 rankings using Reciprocal Rank Fusion.

    Both result lists must contain a common chunk_id.
    """
    fused = {}

    result_lists = [
        ("dense", dense_results),
        ("bm25", bm25_results),
    ]

    for retrieval_type, results in result_lists:
        for rank, document in enumerate(results, start=1):
            chunk_id = document.get("chunk_id")

            # Cannot reliably fuse a document without a common identity.
            if not chunk_id:
                continue

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "content": document["content"],
                    "source": document.get("source", "Unknown"),
                    "source_type": document.get("source_type", "general"),
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)

            if retrieval_type == "dense":
                fused[chunk_id]["dense_score"] = document.get("score")
            else:
                fused[chunk_id]["bm25_score"] = document.get("score")

    ranked_results = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked_results[:limit]


def hybrid_search(
    query: str,
    dense_limit: int = 15,
    bm25_limit: int = 15,
    final_limit: int = 20,
) -> list[dict]:
    """Execute dense + sparse retrieval and fuse the results."""

    with logfire.span("Hybrid Retrieval", query=query):
        dense_results = search_enterprise_knowledge(
            query,
            limit=dense_limit,
        )

        bm25_results = search_bm25(
            query,
            limit=bm25_limit,
        )

        logfire.info(
            "Hybrid retrieval completed",
            dense_count=len(dense_results),
            bm25_count=len(bm25_results),
        )

        fused_results = reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            limit=final_limit,
        )

        logfire.info(
            "RRF fusion completed",
            fused_count=len(fused_results),
        )

        return fused_results