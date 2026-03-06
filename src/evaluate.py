"""Model evaluation utilities and metrics reporting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import setup_logging

logger = setup_logging("evaluate")


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute classification metrics for a fitted pipeline.

    Returns a dict with accuracy, precision, recall, f1, roc_auc and
    the full sklearn classification report string.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info(
        "Evaluation — accuracy=%.4f | precision=%.4f | recall=%.4f | "
        "f1=%.4f | roc_auc=%s",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
    )
    return metrics


def compare_models(results: dict) -> str:
    """Given a dict of {model_name: metrics_dict}, return the best model name by F1."""
    best = max(results, key=lambda name: results[name]["f1"])
    logger.info("Best model by F1: %s (f1=%.4f)", best, results[best]["f1"])
    return best


def print_summary(results: dict) -> None:
    """Print a formatted comparison table."""
    header = f"{'Model':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        roc = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "   N/A"
        print(
            f"{name:<30} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} {m['f1']:>10.4f} {roc:>10}"
        )
