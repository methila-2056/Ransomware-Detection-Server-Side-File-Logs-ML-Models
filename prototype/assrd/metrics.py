"""Evaluation metrics for cost-sensitive ransomware detection.

Every number reported by Phase 2 is computed here, from a confusion matrix, so
that the headline figures cannot drift apart from each other.

The cost model
--------------
A missed attack and a false alarm are not interchangeable. This prototype prices
them explicitly (Elkan's formulation, reference [15] in the Phase I report):

    C_FP = 1     cost of predicting attack for a benign window
    C_FN = 10    cost of missing a real attack window

    total_cost = C_FP * FP + C_FN * FN

``C_FP`` and ``C_FN`` are module constants, overridable per call, so that a
sensitivity analysis over other cost ratios does not have to edit this file.

Conventions
-----------
* positive class = attack (``label == 1``);
* ``y_pred`` is a *hard* decision produced by applying a threshold to a
  probability, never ``predict()``, so that the operating point is explicit and
  reported;
* ``roc_auc`` is computed from probabilities and is threshold-independent. It is
  ``None`` when either class is absent or when the scores are degenerate, and
  callers must treat ``None`` as "not measurable", not as zero;
* division guards return ``None`` rather than 0.0 for undefined rates (e.g.
  precision with no predicted positives). A zero would silently look like a
  result.
"""

from __future__ import annotations

import numpy as np

C_FP = 1
C_FN = 10

ATTACK = 1
BENIGN = 0

#: Confusion-matrix row/column order used everywhere in this module.
LABELS = (BENIGN, ATTACK)


def _safe_div(numerator, denominator):
    """``numerator / denominator`` or ``None`` when undefined."""
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def confusion_counts(y_true, y_pred):
    """Return ``(tn, fp, fn, tp)`` as plain ints."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    tn = int(np.sum((y_true == BENIGN) & (y_pred == BENIGN)))
    fp = int(np.sum((y_true == BENIGN) & (y_pred == ATTACK)))
    fn = int(np.sum((y_true == ATTACK) & (y_pred == BENIGN)))
    tp = int(np.sum((y_true == ATTACK) & (y_pred == ATTACK)))
    return tn, fp, fn, tp


def total_cost(tn, fp, fn, tp, c_fp=C_FP, c_fn=C_FN):
    """``C_FP * FP + C_FN * FN``. Correct negatives are free."""
    return float(c_fp) * float(fp) + float(c_fn) * float(fn)


def cost_from_predictions(y_true, y_pred, c_fp=C_FP, c_fn=C_FN):
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return total_cost(tn, fp, fn, tp, c_fp, c_fn)


def classification_metrics(y_true, y_pred, y_proba=None, *, c_fp=C_FP, c_fn=C_FN):
    """Full metric bundle for one (labels, decisions, probabilities) triple.

    ``y_proba`` is the probability of the *attack* class. It is optional; when
    omitted, ``roc_auc`` and ``average_precision`` are ``None``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity) \
        if (precision is not None and sensitivity is not None) else None
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    cost_fp = float(c_fp) * fp
    cost_fn = float(c_fn) * fn
    cost = cost_fp + cost_fn
    n = tp + tn + fp + fn

    roc_auc = average_precision = threshold_free_auc = None
    if y_proba is not None:
        scores = np.asarray(y_proba, dtype=float)
        if scores.shape != y_true.shape:
            raise ValueError(
                f"shape mismatch: y_true {y_true.shape} vs y_proba {scores.shape}")
        if 0 < int(y_true.sum()) < n:
            threshold_free_auc = _roc_auc(y_true, scores)
            average_precision = _average_precision(y_true, scores)

    return {
        "n": int(n),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows": "actual",
            "matrix": [[tn, fp], [fn, tp]],
        },
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": npv,
        "f1": f1,
        "accuracy": accuracy,
        "balanced_accuracy": (
            None if (sensitivity is None or specificity is None)
            else (sensitivity + specificity) / 2.0),
        "roc_auc": threshold_free_auc,
        "average_precision": average_precision,
        "support_attack": int(tp + fn),
        "support_benign": int(tn + fp),
        "prevalence": _safe_div(tp + fn, n),
        "predicted_positive_rate": _safe_div(tp + fp, n),
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "total_cost": cost,
        "mean_cost": _safe_div(cost, n),
        "cost_fpratio": (c_fn / c_fp) if c_fp else None,
    }


# ---------------------------------------------------------------------------
# Threshold-free curves, implemented locally to keep the dependency surface and
# the tie-handling explicit.
# ---------------------------------------------------------------------------
def _roc_auc(y_true, scores):
    """Area under the ROC curve via the rank identity (handles ties exactly)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    # Average rank handles ties the way the ROC definition requires.
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision(y_true, scores):
    """Average precision (area under the precision/recall envelope)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.arange(1, len(y_sorted) + 1) - cum_tp
    total_pos = int(y_true.sum())
    if total_pos == 0:
        return None

    # Evaluate only at the last index of each tied-score block.
    keep = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    precision = cum_tp[keep] / (cum_tp[keep] + cum_fp[keep])
    recall = cum_tp[keep] / total_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))