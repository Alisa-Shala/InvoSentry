"""
Evaluation harness: runs both the multi-agent pipeline and the baseline
single-LLM-call system over a labeled test set, computes
precision/recall/F1/accuracy for each, and dumps per-invoice error
analysis (false positives / false negatives) for the thesis's
"Error Analysis and Limitations" chapter.

Usage:
    python -m src.evaluate --n-legit 30 --n-fraud 20 --system both
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

from src.data_loader import generate_sample_dataset, train_test_split_invoices
from src.orchestrator import run_pipeline
from src.baseline import run_baseline
from src.agents.validation_agent import reset_duplicate_registry

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def evaluate_system(name: str, run_fn, test_invoices: list) -> dict:
    y_true, y_pred, records = [], [], []

    for invoice in test_invoices:
        start = time.time()
        try:
            prediction = run_fn(invoice)
        except Exception as e:  # keep evaluation going even if one call fails
            print(f"[{name}] GABIM te {invoice.invoice_id}: {e}")
            continue
        elapsed = time.time() - start

        y_true.append(1 if invoice.label == "fraud" else 0)
        y_pred.append(1 if prediction.predicted_label == "fraud" else 0)

        records.append({
            "invoice_id": invoice.invoice_id,
            "true_label": invoice.label,
            "true_fraud_type": invoice.fraud_type,
            "predicted_label": prediction.predicted_label,
            "risk_score": prediction.risk_score,
            "reasons": prediction.reasons,
            "correct": invoice.label == prediction.predicted_label,
            "latency_sec": round(elapsed, 2),
        })

    metrics = {
        "system": name,
        "n_samples": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 4) if y_true else None,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4) if y_true else None,
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4) if y_true else None,
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4) if y_true else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist() if y_true else None,
    }

    false_positives = [r for r in records if r["true_label"] == "legit" and r["predicted_label"] == "fraud"]
    false_negatives = [r for r in records if r["true_label"] == "fraud" and r["predicted_label"] == "legit"]

    return {
        "metrics": metrics,
        "records": records,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-legit", type=int, default=30)
    parser.add_argument("--n-fraud", type=int, default=20)
    parser.add_argument("--system", choices=["pipeline", "baseline", "both"], default="both")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    dataset = generate_sample_dataset(n_legit=args.n_legit, n_fraud=args.n_fraud)
    _, test_set = train_test_split_invoices(dataset, test_ratio=0.4)
    print(f"Testing on {len(test_set)} invoices "
          f"({sum(1 for i in test_set if i.label == 'fraud')} fraud, "
          f"{sum(1 for i in test_set if i.label == 'legit')} legit)")

    all_results = {}

    if args.system in ("pipeline", "both"):
        reset_duplicate_registry()
        print("\n=== Multi-Agent Pipeline ===")
        all_results["multi_agent_pipeline"] = evaluate_system("multi_agent_pipeline", run_pipeline, test_set)
        print(json.dumps(all_results["multi_agent_pipeline"]["metrics"], indent=2, ensure_ascii=False))

    if args.system in ("baseline", "both"):
        print("\n=== Baseline (single LLM call) ===")
        all_results["baseline"] = evaluate_system("baseline", run_baseline, test_set)
        print(json.dumps(all_results["baseline"]["metrics"], indent=2, ensure_ascii=False))

    out_path = RESULTS_DIR / "evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nRezultatet e plota u ruajtën në: {out_path}")


if __name__ == "__main__":
    main()
