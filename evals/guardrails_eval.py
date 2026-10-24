"""
Guardrails binary evaluation.
Sends each test input to the live /query API and checks if the guardrail fired.
Classifies each result as TP / TN / FP / FN and computes precision + recall.
"""


import time
import copy
import requests
import logfire

API_URL = "http://localhost:8000/query"


# def _is_blocked(response_json: dict) -> bool:
#     tp = response_json.get("thought_process") or []
#     return any("guardrails fired" in step.lower() for step in tp)


def _is_blocked(response_json: dict) -> bool:
    if not isinstance(response_json, dict):
        raise ValueError("Expected a JSON object from /query API")

    status = str(response_json.get("status", "")).strip().lower()

    if status == "blocked by guardrails.":
        return True

    if status in {
        "handled directly by guardrails.",
        "response generated.",
    }:
        return False

    if status == "invalid_request":
        raise ValueError("API rejected an invalid request")

    if status == "error":
        raise RuntimeError("The /query endpoint returned an internal error")

    raise ValueError(f"Unrecognized /query status: {status!r}")



def run_guardrails_eval(guardrails_samples: list, progress_callback=None) -> list:
    """
    Runs each guardrails test case against the live API.
    Adds actual_blocked and result (TP/TN/FP/FN) to each sample in place.
    Returns the enriched list.
    """
    samples = copy.deepcopy(guardrails_samples)
    n = len(samples)

    with logfire.span("🛡️ Eval — Guardrails Tests", total=n):
        for i, sample in enumerate(samples):
            if progress_callback:
                progress_callback(i, n, sample["input"])

            with logfire.span(
                f"🛡️ Test {sample['id']}",
                input_text=sample["input"][:80],
                expected_blocked=sample["expected_blocked"],
            ):
                try:
                    resp = requests.post(
                        API_URL,
                        json={"q": sample["input"], "thread_id": f"guardrail_eval_{i}"},
                        timeout=30,
                    )
                    resp.raise_for_status()
                    response_json = resp.json()
                    blocked = _is_blocked(response_json)

                except Exception as e:
                    logfire.exception(
                        f"Guardrails evaluation failed for {sample['id']}: {e}"
                    )
                    sample["result"] = "ERROR"
                    sample["actual_blocked"] = None
                    sample["error"] = str(e)
                    continue

                expected = sample["expected_blocked"]
                sample["actual_blocked"] = blocked

                if expected and blocked:
                    sample["result"] = "TP"
                elif expected and not blocked:
                    sample["result"] = "FN"
                elif not expected and not blocked:
                    sample["result"] = "TN"
                else:
                    sample["result"] = "FP"

                logfire.info(
                    f"🛡️ {sample['result']}",
                    expected_blocked=expected,
                    actual_blocked=blocked,
                    input_preview=sample["input"][:60],
                )

            time.sleep(2)

    return samples


def compute_guardrails_metrics(results: list) -> dict:
    tp = sum(1 for r in results if r["result"] == "TP")
    tn = sum(1 for r in results if r["result"] == "TN")
    fp = sum(1 for r in results if r["result"] == "FP")
    fn = sum(1 for r in results if r["result"] == "FN")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / len(results) if results else 0.0

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "accuracy": round(accuracy, 3),
        "total": len(results),
        "correct": tp + tn,
    }

