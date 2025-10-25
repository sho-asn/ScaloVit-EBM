"""Classification metrics helpers for anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def get_confusion_matrix_components(ground_truth: np.ndarray, predictions: np.ndarray) -> Tuple[int, int, int, int]:
    """Return TP, TN, FP, FN counts for binary labels."""

    tp = int(np.sum((predictions == 1) & (ground_truth == 1)))
    tn = int(np.sum((predictions == 0) & (ground_truth == 0)))
    fp = int(np.sum((predictions == 1) & (ground_truth == 0)))
    fn = int(np.sum((predictions == 0) & (ground_truth == 1)))
    return tp, tn, fp, fn


def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def calculate_sensitivity(tp: int, fn: int) -> float:
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0


def calculate_specificity(tn: int, fp: int) -> float:
    denominator = tn + fp
    return tn / denominator if denominator > 0 else 0.0


def calculate_fpr(fp: int, tn: int) -> float:
    denominator = fp + tn
    return fp / denominator if denominator > 0 else 0.0


def calculate_precision(tp: int, fp: int) -> float:
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0.0


def calculate_f1_score(precision: float, sensitivity: float) -> float:
    denominator = precision + sensitivity
    return 2 * (precision * sensitivity) / denominator if denominator > 0 else 0.0


def calculate_roc_auc(ground_truth: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(ground_truth)) < 2:
        return 0.0
    return float(roc_auc_score(ground_truth, scores))


def compute_metrics_from_cm(tp: int, tn: int, fp: int, fn: int) -> Dict[str, Any]:
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    sensitivity = calculate_sensitivity(tp, fn)
    specificity = calculate_specificity(tn, fp)
    fpr = calculate_fpr(fp, tn)
    precision = calculate_precision(tp, fp)
    f1 = calculate_f1_score(precision, sensitivity)

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Sensitivity (Recall/TPR)": sensitivity,
        "Specificity (TNR)": specificity,
        "FPR": fpr,
        "Precision": precision,
        "F1 Score": f1,
    }


def compute_all_metrics(ground_truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    tp, tn, fp, fn = get_confusion_matrix_components(ground_truth, predictions)
    metrics = compute_metrics_from_cm(tp, tn, fp, fn)
    metrics["ROC_AUC"] = calculate_roc_auc(ground_truth, scores)
    return metrics
