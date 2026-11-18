import json
import re
import threading
from pathlib import Path

import logfire
from rank_bm25 import BM25Okapi


# Keep the BM25 corpus alongside your processed ingestion data.
PROCESSED_DATA_DIR = Path("processed_data")
BM25_CORPUS_PATH = PROCESSED_DATA_DIR / "bm25_corpus.json"

_lock = threading.RLock()

_documents: list[dict] = []
_bm25_index = None
_initialized = False


def _tokenize(text: str) -> list[str]:
    """Normalize text into tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def _save_corpus():
    """Persist the complete BM25 document corpus atomically."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = BM25_CORPUS_PATH.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(_documents, file, ensure_ascii=False, indent=2)

    temp_path.replace(BM25_CORPUS_PATH)


def _load_existing_processed_data() -> list[dict]:
    """
    Bootstrap BM25 from existing processed_data JSON files
    when the dedicated BM25 corpus does not exist yet.
    """
    documents = []

    if not PROCESSED_DATA_DIR.exists():
        return documents

    for file_path in PROCESSED_DATA_DIR.rglob("*.json"):
        if file_path.name == BM25_CORPUS_PATH.name:
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            filename = data.get("filename", file_path.stem)
            source_type = data.get("source_type", "general")

            for index, chunk in enumerate(data.get("chunks", [])):
                if not isinstance(chunk, str) or not chunk.strip():
                    continue

                documents.append({
                    "chunk_id": f"{source_type}:{filename}:{index}",
                    "content": chunk,
                    "source": filename,
                    "source_type": source_type,
                })

        except Exception as exc:
            logfire.warning(
                f"Could not load processed file {file_path}: {exc}"
            )

    return documents


def _initialize():
    """Load or bootstrap the corpus and build the BM25 index."""
    global _documents, _bm25_index, _initialized

    with _lock:
        if _initialized:
            return

        if BM25_CORPUS_PATH.exists():
            with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as file:
                _documents = json.load(file)

            # Migrate the old ID field to chunk_id.
            for document in _documents:
                if not document.get("chunk_id"):
                    document["chunk_id"] = document.get("id")

                document.pop("id", None)

            _save_corpus()
        else:
            _documents = _load_existing_processed_data()
            _save_corpus()

        _rebuild_index()
        _initialized = True

        logfire.info(
            "BM25 initialized",
            document_count=len(_documents),
        )


def _rebuild_index():
    """Rebuild the BM25 index from the current corpus."""
    global _bm25_index

    if not _documents:
        _bm25_index = None
        return

    tokenized_corpus = [
        _tokenize(document["content"])
        for document in _documents
    ]

    _bm25_index = BM25Okapi(tokenized_corpus)


def update_bm25_documents(
    chunks: list[str],
    filename: str,
    source_type: str,
):
    """
    Add or replace all chunks belonging to one source document.

    Call this after successfully indexing the document in Qdrant.
    """
    global _documents

    with _lock:
        _initialize()

        # Remove the previous version of this source.
        _documents = [
            doc for doc in _documents
            if not (
                doc["source"] == filename
                and doc["source_type"] == source_type
            )
        ]

        # Add the latest chunks.
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, str) or not chunk.strip():
                continue

            _documents.append({
                "chunk_id": f"{source_type}:{filename}:{index}",
                "content": chunk,
                "source": filename,
                "source_type": source_type,
            })

        _save_corpus()
        _rebuild_index()

        logfire.info(
            "BM25 corpus updated",
            source=filename,
            source_type=source_type,
            total_documents=len(_documents),
        )


def search_bm25(query: str, limit: int = 15) -> list[dict]:
    """Retrieve the most relevant documents using BM25."""
    _initialize()

    with _lock:
        if not _bm25_index or not query.strip():
            return []

        query_tokens = _tokenize(query)

        if not query_tokens:
            return []

        scores = _bm25_index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]

        results = []

        for index in ranked_indices:
            document = _documents[index]

            results.append({
                **document,
                "score": float(scores[index]),
            })

        return results