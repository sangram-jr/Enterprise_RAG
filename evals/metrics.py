"""
Phase 2 — RAGAS + Tool Correctness metrics.
Uses JUDGE_GROQ key so production GROQ_API_KEY is never exhausted by eval runs.
All LLM-based metrics run in batches of 5 with 30s cooldowns between sub-batches
and 60s cooldowns between experiments — calibrated for Groq's 6,000 TPM on_demand tier.
Contexts are truncated to 300 chars (2 chunks max) so no single request exceeds the limit.
"""


import os
import asyncio
import logfire
import pandas as pd
from openai import AsyncOpenAI

import json
import hashlib
from pathlib import Path


from ragas.llms import llm_factory
from ragas.embeddings import HuggingfaceEmbeddings
from ragas import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
)
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import AnswerSimilarity

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_MODEL = "openai/gpt-oss-120b"
COOLDOWN_STANDARD = 62
COOLDOWN_MINI = 40       # between individual samples — lets sliding TPM window recover (~2,800 tok/sample)
GENERAL_BATCH_SIZE = 1  # one sample at a time: abatch_score fires calls concurrently per sample,
                         # so batch>1 stacks multiple samples' async calls inside the same second
CONTEXT_TRUNCATE = 300  # chars per context chunk — reduces single request from ~7,700 to ~400 tokens
CONTEXT_LIMIT = 2       # number of context chunks passed to RAGAS per sample





# ── Persistent evaluation checkpoint ──────────────────────────────

CHECKPOINT_VERSION = "v1"

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "metric_checkpoint.json"
)


def _load_checkpoint() -> dict:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.exists():
        return {"metrics": {}}

    try:
        with CHECKPOINT_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(
            f"Cannot read evaluation checkpoint: {CHECKPOINT_PATH}. "
            "Back up the file before repairing it."
        ) from exc

    if not isinstance(data, dict) or not isinstance(
        data.get("metrics"), dict
    ):
        raise RuntimeError(
            f"Invalid checkpoint structure: {CHECKPOINT_PATH}"
        )

    return data


def _save_checkpoint(data: dict) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temp_path = CHECKPOINT_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=True)
        file.flush()
        os.fsync(file.fileno())

    # Atomic replacement helps protect against incomplete writes.
    temp_path.replace(CHECKPOINT_PATH)


def _make_dataset_hash(inputs: list, metric_key: str) -> str:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "metric": metric_key,
        "judge_model": JUDGE_MODEL,
        "inputs": inputs,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()








def _build_judge():
    api_key = os.getenv("JUDGE_GROQ") or os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing judge API key. Set JUDGE_GROQ or GROQ_API_KEY in your .env file."
        )

    # Ragas llm_factory creates ChatOpenAI internally.
    # Groq supports the OpenAI-compatible API format.
    os.environ["OPENAI_API_KEY"] = api_key

    llm = llm_factory(
        model=JUDGE_MODEL,
        base_url=GROQ_BASE_URL,
    )

    # Increase the output-token budget for Ragas structured responses.
    if hasattr(llm, "langchain_llm"):
        llm.langchain_llm.max_tokens = 2048
        print("Ragas wrapper:", type(llm).__name__)
        print("Underlying model:", type(llm.langchain_llm).__name__)
        print("Configured max_tokens:", llm.langchain_llm.max_tokens)
    else:
        raise RuntimeError(
            "Cannot locate the underlying LangChain LLM; "
            "inspect the returned Ragas wrapper before configuring tokens."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return llm, embeddings

async def _cooldown(seconds: int, label: str, status_cb=None):
    msg = f"⏳ {seconds}s cooldown after {label} (Groq TPM buffer)..."
    if status_cb:
        status_cb(msg)
    for _ in range(seconds // 10):
        await asyncio.sleep(10)
    if status_cb:
        status_cb(f"✅ Ready — starting next experiment.")
        
        
def _prep_samples(golden_dataset: dict) -> list:
    """
    Returns only samples with actual_response populated.
    Truncates contexts to CONTEXT_TRUNCATE chars and limits to CONTEXT_LIMIT chunks
    so a single RAGAS LLM call stays well under the 6,000 TPM ceiling.
    (Live contexts from Qdrant are ~1,500 chars each — without truncation a single
    Faithfulness request exceeds 7,000 tokens which hard-fails on the on_demand tier.)
    """
    valid = []
    for s in golden_dataset["rag_samples"]:
        response = s.get("actual_response", "").strip()
        if not response:
            continue
        raw_contexts = s.get("actual_contexts") or s.get("relevant_contexts") or []
        contexts = [c[:CONTEXT_TRUNCATE] for c in raw_contexts[:CONTEXT_LIMIT]]
        valid.append({**s, "actual_contexts": contexts})
    return valid


# def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
#     return pd.DataFrame([
#         {"question": s["question"][:65], metric_key: round(float(r.value), 3)}
#         for s, r in zip(samples, scores)
#     ])



# def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
#     return pd.DataFrame([
#         {
#             "question": sample["question"][:65],
#             metric_key: round(
#                 float(score.value if hasattr(score, "value") else score),
#                 3,
#             ),
#         }
#         for sample, score in zip(samples, scores)
#     ])



def _score_df(metric_key: str, samples: list, scores) -> pd.DataFrame:
    rows = []

    for sample, score in zip(samples, scores):
        value = score.value if hasattr(score, "value") else score

        rows.append({
            "question": sample["question"][:65],
            metric_key: round(float(value), 3),
        })

    return pd.DataFrame(rows)



# async def _batched_score(metric, inputs: list, samples: list, status_cb=None, label: str = "") -> list:
#     """
#     Runs abatch_score in chunks of GENERAL_BATCH_SIZE with cooldowns between chunks.
#     Keeps each burst under 6,000 TPM on Groq's on_demand tier.
#     """
#     all_scores = []
#     batches = [inputs[i : i + GENERAL_BATCH_SIZE] for i in range(0, len(inputs), GENERAL_BATCH_SIZE)]
#     for b_idx, batch in enumerate(batches):
#         if b_idx > 0:
#             await _cooldown(COOLDOWN_MINI, f"{label} batch {b_idx}", status_cb)
#         scores = await metric.abatch_score(batch)
#         all_scores.extend(scores)
#     return all_scores




# async def _batched_score(
#     metric,
#     inputs: list,
#     samples: list,
#     status_cb=None,
#     label: str = "",
# ) -> list:
#     """Score inputs using the Ragas 0.3.1 single-turn API."""
#     all_scores = []

#     batches = [
#         inputs[i:i + GENERAL_BATCH_SIZE]
#         for i in range(0, len(inputs), GENERAL_BATCH_SIZE)
#     ]

#     for b_idx, batch in enumerate(batches):
#         if b_idx > 0:
#             await _cooldown(
#                 COOLDOWN_MINI,
#                 f"{label} batch {b_idx}",
#                 status_cb,
#             )

#         # Ragas 0.3.1 expects SingleTurnSample, not dictionaries.
#         ragas_samples = [
#             SingleTurnSample(**item)
#             for item in batch
#         ]

#         scores = await asyncio.gather(
#             *(
#                 metric.single_turn_ascore(sample)
#                 for sample in ragas_samples
#             )
#         )

#         all_scores.extend(scores)

#     return all_scores





async def _batched_score(
    metric,
    inputs,
    samples,
    status_cb=None,
    label="",
):
    """
    Score Ragas samples with persistent, per-sample checkpoints.

    Successful scores are saved immediately. Failed samples are retried
    on the next run; previously saved samples are skipped.
    """

    metric_key = label or metric.__class__.__name__
    dataset_hash = _make_dataset_hash(inputs, metric_key)

    checkpoint = _load_checkpoint()
    saved = checkpoint["metrics"].get(metric_key, {})

    # Do not reuse scores when the input dataset has changed.
    if saved.get("dataset_hash") != dataset_hash:
        saved = {
            "dataset_hash": dataset_hash,
            "scores": {},
        }

    saved_scores = saved["scores"]
    checkpoint["metrics"][metric_key] = saved

    all_scores = [None] * len(inputs)
    batch_size = max(1, GENERAL_BATCH_SIZE)

    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))

        for i in range(start, end):
            key = str(i)

            # Reuse scores from earlier runs.
            if key in saved_scores:
                all_scores[i] = float(saved_scores[key])
                continue

            if status_cb:
                status_cb(
                    f"🧪 {metric_key}: sample {i + 1}/{len(inputs)} "
                    f"(resuming from checkpoint)"
                )

            # Preserve the cooldown policy for new API calls.
            if i > 0:
                await _cooldown(
                    COOLDOWN_MINI,
                    f"{metric_key}, sample {i + 1}",
                    status_cb,
                )

            item = inputs[i]

            ragas_sample = (
                item
                if isinstance(item, SingleTurnSample)
                else SingleTurnSample(**item)
            )

            # If this raises an exception, the failed score is NOT saved.
            # All earlier successful scores remain on disk.
            score = await metric.single_turn_ascore(ragas_sample)
            score = float(score)

            saved_scores[key] = score
            all_scores[i] = score

            # Persist immediately after each successful sample.
            _save_checkpoint(checkpoint)

    return all_scores




async def run_all_metrics(golden_dataset: dict, status_cb=None) -> dict:
    """
    Runs all 6 experiments. Returns dict keyed by metric name → DataFrame.
    status_cb(message: str) is called for live UI updates.
    """
    judge_llm, ragas_embeddings = _build_judge()
    samples = _prep_samples(golden_dataset)

    if not samples:
        raise ValueError("No samples with actual_response found. Run Phase 1 first.")

    results = {}

    with logfire.span("🧪 Eval Phase 2 — All Metrics", total_samples=len(samples)):

        # ── Exp 1: Faithfulness ───────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 1/6 — Faithfulness ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 1 — Faithfulness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(Faithfulness(llm=judge_llm), inputs, samples, status_cb, "Faithfulness")
            df = _score_df("faithfulness", samples, scores)
            results["faithfulness"] = df
            logfire.info("🧪 Faithfulness done", avg=round(df["faithfulness"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Faithfulness", status_cb)

        # ── Exp 2: Answer Relevancy ───────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 2/6 — Answer Relevancy ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 2 — Answer Relevancy"):
            inputs = [
                {"user_input": s["question"], "response": s["actual_response"]}
                for s in samples
            ]
            scores = await _batched_score(
                AnswerRelevancy(llm=judge_llm, embeddings=ragas_embeddings),
                inputs, samples, status_cb, "Answer Relevancy"
            )
            df = _score_df("answer_relevancy", samples, scores)
            results["answer_relevancy"] = df
            logfire.info("🧪 Answer Relevancy done", avg=round(df["answer_relevancy"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Answer Relevancy", status_cb)

        # ── Exp 3: Context Precision ──────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 3/6 — Context Precision ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 3 — Context Precision"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextPrecision(llm=judge_llm), inputs, samples, status_cb, "Context Precision")
            df = _score_df("context_precision", samples, scores)
            results["context_precision"] = df
            logfire.info("🧪 Context Precision done", avg=round(df["context_precision"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Precision", status_cb)

        # ── Exp 4: Context Recall ─────────────────────────────────────────────
        if status_cb:
            status_cb(f"🧪 Exp 4/6 — Context Recall ({len(samples)} samples)...")
        with logfire.span("🧪 Exp 4 — Context Recall"):
            inputs = [
                {
                    "user_input": s["question"],
                    "reference": s["reference"],
                    "retrieved_contexts": s["actual_contexts"],
                }
                for s in samples
            ]
            scores = await _batched_score(ContextRecall(llm=judge_llm), inputs, samples, status_cb, "Context Recall")
            df = _score_df("context_recall", samples, scores)
            results["context_recall"] = df
            logfire.info("🧪 Context Recall done", avg=round(df["context_recall"].mean(), 3))

        await _cooldown(COOLDOWN_STANDARD, "Context Recall", status_cb)

        
        # ── Exp 5: Answer Correctness ────────────────────────────────────
        if status_cb:
            status_cb(
                f"🧪 Exp 5/6 — Answer Correctness ({len(samples)} samples)..."
            )

        with logfire.span("🧪 Exp 5 — Answer Correctness"):
            inputs = [
                {
                    "user_input": s["question"],
                    "response": s["actual_response"],
                    "reference": s["reference"],
                }
                for s in samples
            ]

            # Adapt LangChain embeddings to the interface expected by Ragas
            similarity_embeddings = LangchainEmbeddingsWrapper(
                ragas_embeddings
            )

            similarity_metric = AnswerSimilarity(
                embeddings=similarity_embeddings
            )

            correctness_metric = AnswerCorrectness(
                llm=judge_llm,
                embeddings=ragas_embeddings,
                answer_similarity=similarity_metric,
            )

            all_scores = await _batched_score(
                correctness_metric,
                inputs,
                samples,
                status_cb,
                "Answer Correctness",
            )

            df = _score_df(
                "answer_correctness",
                samples,
                all_scores,
            )
            results["answer_correctness"] = df

            logfire.info(
                "🧪 Answer Correctness done",
                avg=round(df["answer_correctness"].mean(), 3),
            )


        # ── Exp 6: Tool Correctness (no LLM — Jaccard) ───────────────────────
        if status_cb:
            status_cb("⚡ Exp 6/6 — Tool Correctness (zero LLM calls)...")
        with logfire.span("🧪 Exp 6 — Tool Correctness"):
            tool_rows = []
            for s in samples:
                called = set(s.get("actual_tools_called") or [])
                expected = set(s.get("expected_tools") or [])
                union = len(called | expected)
                score = len(called & expected) / union if union > 0 else 0.0
                tool_rows.append({"question": s["question"][:65], "tool_correctness": round(score, 3)})
            df = pd.DataFrame(tool_rows)
            results["tool_correctness"] = df
            logfire.info("🧪 Tool Correctness done", avg=round(df["tool_correctness"].mean(), 3))

        if status_cb:
            status_cb("✅ All 6 experiments complete!")

    return results 