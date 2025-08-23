import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Any

def get_confusion_matrix_components(ground_truth: np.ndarray, predictions: np.ndarray) -> tuple[int, int, int, int]:
    """
    Calculates the components of the confusion matrix.

    Args:
        ground_truth: A numpy array of the true labels (0s and 1s).
        predictions: A numpy array of the predicted labels (0s and 1s).

    Returns:
        A tuple containing (TP, TN, FP, FN).
    """
    tp = np.sum((predictions == 1) & (ground_truth == 1))
    tn = np.sum((predictions == 0) & (ground_truth == 0))
    fp = np.sum((predictions == 1) & (ground_truth == 0))
    fn = np.sum((predictions == 0) & (ground_truth == 1))
    return int(tp), int(tn), int(fp), int(fn)

def calculate_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Calculates accuracy."""
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0

def calculate_sensitivity(tp: int, fn: int) -> float:
    """Calculates sensitivity, also known as recall or True Positive Rate (TPR)."""
    denominator = tp + fn
    return tp / denominator if denominator > 0 else 0.0

def calculate_specificity(tn: int, fp: int) -> float:
    """Calculates specificity or True Negative Rate (TNR)."""
    denominator = tn + fp
    return tn / denominator if denominator > 0 else 0.0

def calculate_fpr(fp: int, tn: int) -> float:
    """Calculates the False Positive Rate (FPR)."""
    denominator = fp + tn
    return fp / denominator if denominator > 0 else 0.0

def calculate_precision(tp: int, fp: int) -> float:
    """Calculates precision."""
    denominator = tp + fp
    return tp / denominator if denominator > 0 else 0.0

def calculate_f1_score(precision: float, sensitivity: float) -> float:
    """Calculates the F1 score."""
    denominator = precision + sensitivity
    return 2 * (precision * sensitivity) / denominator if denominator > 0 else 0.0

def calculate_roc_auc(ground_truth: np.ndarray, scores: np.ndarray) -> float:
    """Calculates the ROC AUC score."""
    if len(np.unique(ground_truth)) < 2:
        return 0.0
    return roc_auc_score(ground_truth, scores)

def compute_all_metrics(ground_truth: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    """
    Computes and returns a dictionary of all relevant classification metrics.

    Args:
        ground_truth: A numpy array of the true labels (0s and 1s).
        predictions: A numpy array of the predicted labels (0s and 1s) after applying a threshold.
        scores: A numpy array of the raw anomaly scores from the model (before thresholding).

    Returns:
        A dictionary containing all calculated metrics.
    """
    tp, tn, fp, fn = get_confusion_matrix_components(ground_truth, predictions)
    
    accuracy = calculate_accuracy(tp, tn, fp, fn)
    sensitivity = calculate_sensitivity(tp, fn)
    specificity = calculate_specificity(tn, fp)
    fpr = calculate_fpr(fp, tn)
    precision = calculate_precision(tp, fp)
    f1 = calculate_f1_score(precision, sensitivity)
    roc_auc = calculate_roc_auc(ground_truth, scores)
    
    metrics = {
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
        "ROC_AUC": roc_auc
    }
    return metrics
