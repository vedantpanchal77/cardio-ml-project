"""Evaluation metrics written with NumPy (cross-checked against scikit-learn in tests/).

For a threshold t a patient is predicted "diseased" when p >= t.

    TP  predicted diseased, really diseased      FP  predicted diseased, really healthy ("false alarm")
    TN  predicted healthy,  really healthy       FN  predicted healthy,  really diseased ("missed case")

    accuracy    = (TP + TN) / all
    precision   = TP / (TP + FP)    of those flagged, how many really have the disease
    recall      = TP / (TP + FN)    of those with the disease, how many were caught (sensitivity)
    specificity = TN / (TN + FP)    of the healthy, how many were left alone
    F1          = 2PR / (P + R)     harmonic mean of precision and recall
"""
from __future__ import annotations

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * precision * recall, precision + recall),
    }


def roc_curve(y_true, scores):
    """False-positive rate and true-positive rate at every distinct threshold."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")            # highest score first
    y_sorted, s_sorted = y_true[order], scores[order]
    tps = np.cumsum(y_sorted)                                # true positives if we cut after row i
    fps = np.cumsum(1 - y_sorted)
    last_of_tie = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]  # one point per distinct score
    tpr = np.r_[0.0, tps[last_of_tie] / tps[-1]]
    fpr = np.r_[0.0, fps[last_of_tie] / fps[-1]]
    return fpr, tpr


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve by the trapezoid rule."""
    fpr, tpr = roc_curve(y_true, scores)
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def roc_points(y_true, scores, max_points: int = 250) -> dict:
    """ROC curve thinned to at most ``max_points`` points, for sending to the browser."""
    fpr, tpr = roc_curve(y_true, scores)
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": np.round(fpr, 5).tolist(), "tpr": np.round(tpr, 5).tolist()}


def log_loss(y_true, proba) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
