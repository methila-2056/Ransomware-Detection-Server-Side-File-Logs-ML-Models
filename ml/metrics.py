"""Evaluation metrics for the ransomware detection pipeline.

Extends the metric set reported by the base paper (Aranyi, Miseta & Szucs, 2026,
J. Inf. Secur. 8) with the metrics the paper names but does not report, plus the
early-phase detection measurement the paper claims without quantifying.

Base paper coverage
-------------------
Reported  : accuracy, F1, sensitivity (TPR), specificity (TNR), confusion matrix,
            cost-sensitive total cost, prediction latency, ROC point, PR point.
Added here: specificity (TNR) and negative predictive value (NPV) named in Sec. 4.4.1,
            full precision-recall curve and average precision (paper reports only a
            single PR operating point), balanced accuracy, Matthews correlation
            coefficient, per-class F1, and time-to-detect (Sec. 5.3 "early phases").

All functions take plain numpy arrays and return JSON-serialisable python types so
results can be written straight to ``results/``.
"""

import time

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Cost schema from base paper Sect. 3.8 / Table 8:
# C_TP = 0, C_TN = 0, C_FP = 1, C_FN = 10.
COST_MATRIX = {"C_TP": 0, "C_TN": 0, "C_FP": 1, "C_FN": 10}


def _rate(numerator, denominator):
    """Ratio that returns 0.0 instead of nan when the denominator is empty."""
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_cost(tp, tn, fp, fn):
    """Total cost under the paper's asymmetric cost matrix."""
    return (
        COST_MATRIX["C_TP"] * tp
        + COST_MATRIX["C_TN"] * tn
        + COST_MATRIX["C_FP"] * fp
        + COST_MATRIX["C_FN"] * fn
    )


def confusion_counts(y_true, y_pred):
    """Return ``(tn, fp, fn, tp)`` in the sklearn ``confusion_matrix`` order.

    ``confusion_matrix`` for a binary problem lays out rows as true classes and
    columns as predicted classes, so ``ravel()`` yields ``tn, fp, fn, tp``.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def binary_metrics(y_true, y_pred):
    """Threshold-dependent metrics, all derived from one confusion matrix.

    The base paper reports accuracy, F1, sensitivity and specificity but omits
    specificity from some tables and never reports NPV. Both are computed here so
    the two rates describing benign traffic are always present together.

    Total cost scales with the number of evaluated windows, so it cannot be
    compared between datasets of different sizes. The base paper's published cost
    of 175 is computed over 209 attack seconds; this implementation's cost is
    computed over whatever the test split contains, which is 1,300 attack windows
    at 20k scale and 42,700 at 2.5M scale. The per-window figures are therefore
    reported alongside the total so a comparison against the paper is meaningful
    rather than an artefact of dataset size.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)

    pos_support = tp + fn
    neg_support = tn + fp
    total_cost = int(compute_cost(tp, tn, fp, fn))

    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "Balanced Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 6),
        "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "Sensitivity": round(_rate(tp, pos_support), 6),
        "Specificity": round(_rate(tn, neg_support), 6),
        "NPV": round(_rate(tn, tn + fn), 6),
        "F1 Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "F1 Benign": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 6),
        "F1 Attack": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 6),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 6),
        "FPR": round(_rate(fp, neg_support), 6),
        "FNR": round(_rate(fn, pos_support), 6),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Support Attack": pos_support,
        "Support Benign": neg_support,
        "Confusion Matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "Total Cost": total_cost,
        # Size-normalised cost. "Cost per 1000 windows" is the unit the earlier
        # evaluation plan proposed and is what makes cross-scale comparison valid.
        "Cost Per 1000 Windows": round(
            total_cost * 1000.0 / (pos_support + neg_support), 6) if (pos_support + neg_support) else 0.0,
        "Cost Per Attack Window": round(
            total_cost / pos_support, 6) if pos_support else 0.0,
        "Attack Windows Evaluated": pos_support,
    }


def _clean_roc(y_true, y_proba):
    """ROC curve with the leading ``inf`` threshold dropped so it stays JSON-safe.

    ``roc_curve`` returns ``threshold[0] == inf`` (the point above all scores).
    JSON has no infinity literal, so it is removed and the caller is expected to
    plot the resulting points, which still span the full curve.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    thresholds = np.asarray(thresholds, dtype=float)
    finite = np.isfinite(thresholds)
    return (
        np.asarray(fpr)[finite].astype(float),
        np.asarray(tpr)[finite].astype(float),
        thresholds[finite].astype(float),
    )


def roc_analysis(y_true, y_proba):
    """Full ROC curve plus AUC.

    The paper reports only a single ROC *point* because every model was evaluated
    at a fixed 0.5 threshold (Sect. 3.6). The curve is computed here as well so the
    point can be located on it and threshold choice justified rather than assumed.
    """
    fpr, tpr, thresholds = _clean_roc(y_true, y_proba)
    return {
        "AUC": round(float(roc_auc_score(y_true, y_proba)), 6),
        "FPR": [round(float(v), 6) for v in fpr],
        "TPR": [round(float(v), 6) for v in tpr],
        "Thresholds": [round(float(v), 6) for v in thresholds],
    }


def pr_analysis(y_true, y_proba):
    """Precision-recall curve plus average precision.

    Completes the paper's PR *point* analysis (Figs. 8 and 9) with the full curve
    and AP, which is the more informative summary when classes are imbalanced --
    the condition the paper explicitly claims to operate under.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return {
        "Average Precision": round(float(average_precision_score(y_true, y_proba)), 6),
        "Precision": [round(float(v), 6) for v in precision],
        "Recall": [round(float(v), 6) for v in recall],
        "Thresholds": [round(float(v), 6) for v in thresholds],
    }


def measure_latency(predict_fn, x_probe, repeats=20, warmup=3):
    """Per-call prediction latency in milliseconds.

    The base paper reports a single elapsed time per model. This averages over
    ``repeats`` calls after ``warmup`` untimed calls, which removes first-call
    import/JIT overhead from the figure.
    """
    probe = x_probe[:1]
    for _ in range(warmup):
        predict_fn(probe)

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict_fn(probe)
        samples.append((time.perf_counter() - start) * 1000.0)

    samples = np.asarray(samples, dtype=float)
    return {
        "Avg Latency (ms)": round(float(samples.mean()), 4),
        "Median Latency (ms)": round(float(np.median(samples)), 4),
        "P95 Latency (ms)": round(float(np.percentile(samples, 95)), 4),
        "Min Latency (ms)": round(float(samples.min()), 4),
        "Latency Repeats": int(repeats),
    }


def early_detection_metrics(y_true, y_pred, group_ids=None):
    """Detection rate and time-to-detect within each attack episode.

    The paper's abstract claims the model "identified ransomware activity during
    the early phases of execution in the majority of test cases" but reports no
    number supporting it. This treats consecutive attack windows as one episode,
    finds the offset of the first correctly detected window inside it, and
    reports that offset as the detection lag.

    ``group_ids`` is unused for now and reserved for grouping windows back into
    the runs or families they came from, which episode-level analysis needs.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    hits = (y_true == 1) & (y_pred == 1)

    episodes = []
    start = None
    for idx, is_attack in enumerate(y_true == 1):
        if is_attack and start is None:
            start = idx
        elif not is_attack and start is not None:
            episodes.append((start, idx))
            start = None
    if start is not None:
        episodes.append((start, len(y_true)))

    detected_episodes = 0
    lags = []
    for begin, end in episodes:
        window = hits[begin:end]
        if window.any():
            detected_episodes += 1
            lags.append(int(np.argmax(window)))

    lags_arr = np.asarray(lags, dtype=float)
    total = len(episodes)
    return {
        "Attack Episodes": total,
        "Episodes Detected": int(detected_episodes),
        "Episode Detection Rate": round(_rate(detected_episodes, total), 6),
        "Median Lag (windows)": round(float(np.median(lags_arr)), 4) if lags else None,
        "Mean Lag (windows)": round(float(lags_arr.mean()), 4) if lags else None,
        "P90 Lag (windows)": round(float(np.percentile(lags_arr, 90)), 4) if lags else None,
        "Max Lag (windows)": round(float(lags_arr.max()), 4) if lags else None,
    }


def evaluate_all(y_true, y_pred, y_proba, x_probe=None, predict_fn=None,
                 group_ids=None, repeats=20):
    """Single entry point returning every metric the paper and this build report."""
    result = {}
    result.update(binary_metrics(y_true, y_pred))
    result.update(roc_analysis(y_true, y_proba))
    result.update(pr_analysis(y_true, y_proba))
    result.update(early_detection_metrics(y_true, y_pred, group_ids))
    if x_probe is not None and predict_fn is not None:
        result.update(measure_latency(predict_fn, x_probe, repeats=repeats))
    return result


def threshold_sweep(y_true, y_proba, thresholds=None):
    """Accuracy, sensitivity and cost across decision thresholds.

    The paper fixes 0.5 (Sect. 3.6) and never examines alternatives. Sweeping the
    threshold shows how much of the reported result is an artefact of that choice
    and is the input needed for the alarm policy in ``ml.alarm_policy``.
    """
    if thresholds is None:
        thresholds = [round(v, 3) for v in np.arange(0.05, 0.96, 0.05)]
    rows = []
    for thr in thresholds:
        y_pred = (np.asarray(y_proba) >= thr).astype(int)
        m = binary_metrics(y_true, y_pred)
        rows.append({
            "Threshold": float(thr),
            "Accuracy": m["Accuracy"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Precision": m["Precision"],
            "Total Cost": m["Total Cost"],
        })
    return rows


def separability_report(x, y, feature_names=None):
    """Describe how easily a linear rule separates the two classes.

    Included as a dataset sanity check rather than a model metric. On a synthetic
    generator the two classes can be separated by a single count threshold, which
    is the reason five classifiers all score 1.000. Reporting the gap makes that
    condition visible instead of letting a perfect score look like a strong result.

    ``feature_names`` must be passed whenever ``x`` has more than three columns.
    Without it a wider matrix would be reported under the three base count names,
    labelling rolling aggregates as "create_count" and producing figures that
    cannot be reconciled with the underlying one-second windows.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)

    if feature_names is None:
        if x.shape[1] != 3:
            raise ValueError(
                f"separability_report received {x.shape[1]} columns but no "
                f"feature_names. Pass the names that produced these columns, or "
                f"restrict the call to the three base count columns."
            )
        feature_names = ["create_count", "rename_count", "delete_count"]

    per_feature = {}
    for i, name in enumerate(feature_names):
        if i >= x.shape[1]:
            break
        benign = x[y == 0, i]
        attack = x[y == 1, i]
        gap = float(attack.min() - benign.max())
        per_feature[name] = {
            "Benign Min": float(benign.min()),
            "Benign Max": float(benign.max()),
            "Attack Min": float(attack.min()),
            "Attack Max": float(attack.max()),
            "Separable By This Feature": bool(gap > 0),
            "Gap": round(gap, 4),
            "Overlap Windows": int(max(0, int(benign.max()) - int(attack.min()) + 1)),
        }
    return {
        "Per Feature": per_feature,
        "Perfectly Linearly Separable": bool(
            all(v["Separable By This Feature"] for v in per_feature.values())
        ),
    }