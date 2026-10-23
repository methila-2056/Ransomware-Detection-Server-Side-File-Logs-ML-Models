"""Decision-threshold selection on a held-out validation split.

The base paper fixes the classification threshold at 0.5 for every model and
states this explicitly (Sect. 3.6, repeated in Sect. 4.4.2), which is why each
model appears as a single operating point in its ROC and PR analysis.

On this project's data that choice is expensive. A Random Forest reaches AUROC
0.95 on held-out attack windows, which means it ranks attack windows ahead of
benign ones very reliably, yet its sensitivity at threshold 0.5 is only 0.52.
The same model reaches 0.80 sensitivity at threshold 0.1. Ranking quality and
sensitivity diverge because the score distribution is skewed towards the benign
class, so 0.5 falls in the wrong place.

That is a property of the data, not a defect in the estimator, and the paper's
fixed-threshold protocol cannot adapt to it. Selecting the threshold on a
validation split, against the paper's own cost matrix, removes the dependence on
an arbitrary constant without touching the test split.
"""

import numpy as np

from ml.metrics import compute_cost


def select_threshold(y_val, y_proba, cost_fp=1, cost_fn=10,
                     grid=None, default=0.5, protect_sensitivity=None):
    """Pick the threshold minimising total cost on the validation split.

    Parameters
    ----------
    protect_sensitivity
        Optional floor on sensitivity. Among thresholds whose cost lies within
        ``tolerance`` of the minimum, the one meeting this floor is preferred.
        Used where an operationally unacceptable miss rate matters more than the
        smallest arithmetic cost. ``None`` means strict cost minimisation.

    Returns a dict with the chosen threshold, the full sweep, and the cost at the
    paper's fixed 0.5 for comparison.
    """
    if grid is None:
        grid = np.round(np.arange(0.02, 0.99, 0.02), 4)

    y_val = np.asarray(y_val)
    y_proba = np.asarray(y_proba)

    sweep = []
    for thr in grid:
        pred = (y_proba >= thr).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())

        pos = tp + fn
        neg = tn + fp
        cost = compute_cost(tp, tn, fp, fn)

        sweep.append({
            "Threshold": float(thr),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "Sensitivity": (tp / pos) if pos else 0.0,
            "Specificity": (tn / neg) if neg else 0.0,
            "Accuracy": (tp + tn) / len(y_val) if len(y_val) else 0.0,
            "Precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
            "Total Cost": int(cost),
        })

    if not sweep:
        return {"Threshold": default, "Sweep": [], "Reason": "empty sweep"}

    min_cost = min(row["Total Cost"] for row in sweep)

    if protect_sensitivity is None:
        # Strict cost minimisation; break ties towards the middle of the plateau
        # so the choice is not balanced on the cliff edge of a single window.
        candidates = [r for r in sweep if r["Total Cost"] == min_cost]
        chosen = max(candidates, key=lambda r: r["Threshold"])
        reason = "minimum total cost on validation split"
    else:
        tolerance = max(1, int(0.02 * len(y_val)))
        pool = [r for r in sweep
                if r["Total Cost"] <= min_cost + tolerance
                and r["Sensitivity"] >= protect_sensitivity]
        if pool:
            chosen = max(pool, key=lambda r: r["Specificity"])
            reason = (f"cost within {tolerance} of minimum while holding "
                      f"sensitivity >= {protect_sensitivity}")
        else:
            feasible = [r for r in sweep if r["Sensitivity"] >= protect_sensitivity]
            if feasible:
                chosen = min(feasible, key=lambda r: r["Total Cost"])
                reason = ("no threshold meets both criteria; cheapest threshold "
                          f"meeting the sensitivity floor {protect_sensitivity}")
            else:
                chosen = max(sweep, key=lambda r: r["Sensitivity"])
                reason = "sensitivity floor unattainable; threshold maximising sensitivity"

    fixed = min(sweep, key=lambda r: abs(r["Threshold"] - default))

    return {
        "Threshold": chosen["Threshold"],
        "Reason": reason,
        "Selected": chosen,
        "Cost At Paper Threshold": fixed["Total Cost"],
        "Paper Threshold": default,
        "Cost Improvement": int(fixed["Total Cost"] - chosen["Total Cost"]),
        "Sensitivity Gain": round(
            chosen["Sensitivity"] - fixed["Sensitivity"], 6),
        "Sweep": sweep,
    }


def compare_to_paper_threshold(result):
    """Head-to-head row for the validation split."""
    chosen = result["Selected"]
    fixed_cost = result["Cost At Paper Threshold"]
    return {
        "Chosen Threshold": result["Threshold"],
        "Sensitivity": round(chosen["Sensitivity"], 6),
        "Specificity": round(chosen["Specificity"], 6),
        "Accuracy": round(chosen["Accuracy"], 6),
        "Total Cost": chosen["Total Cost"],
        "Paper Threshold Cost": fixed_cost,
        "Cost Improvement": result["Cost Improvement"],
        "Sensitivity Gain": result["Sensitivity Gain"],
    }


def threshold_from_sweep(sweep, target_sensitivity):
    """Threshold required to reach a target sensitivity on the given sweep.

    Returns ``None`` when the sweep cannot reach the target, which is the honest
    answer when a model genuinely cannot detect fast enough.
    """
    feasible = [r for r in sweep if r["Sensitivity"] >= target_sensitivity]
    if not feasible:
        best = max(sweep, key=lambda r: r["Sensitivity"])
        return {"Achievable": False,
                "Best Sensitivity": round(best["Sensitivity"], 6),
                "Best Threshold": best["Threshold"],
                "Best Total Cost": best["Total Cost"]}

    chosen = min(feasible, key=lambda r: (r["Total Cost"], -r["Threshold"]))
    return {
        "Achievable": True,
        "Threshold": chosen["Threshold"],
        "Sensitivity": round(chosen["Sensitivity"], 6),
        "Specificity": round(chosen["Specificity"], 6),
        "Total Cost": chosen["Total Cost"],
    }