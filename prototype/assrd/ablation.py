"""Phase 3 — feature-set ablation arms, fixed-hyperparameter fitting, robustness.

This module is the analytical half of Phase 3; ``assrd.phase3`` is the driver.

Feature arms
------------
The arms are **strictly nested** so every comparison changes exactly one group of
columns, and so a reader can see what each group is worth on top of the last:

    A_raw_only          3    the base paper's create/rename/delete counts
    B_trailing_level   15  + ops_sum/mean/max over 3/5/10/30 s
    C_plus_activity    19  + active_frac over 3/5/10/30 s
    D_plus_volatility  27  + ops_absdelta_mean and ops_cv
    E_full             29  = MODEL_FEATURE_COLUMNS
    T_temporal_only    26  the 26 temporal features *without* the raw counts

T is deliberately not part of the nested chain: it tests whether the raw counts
still contribute once the aggregates are present.

Fixed hyperparameters
---------------------
Phase 2 stored the winning parameters for each model in
``results/phase2_results.json``. Phase 3 applies them directly
(:func:`load_phase2_params`) instead of re-searching, which both isolates the
feature-set variable and removes the dominant cost of a run. The values were
selected on a *different* trace; that transfer assumption is documented and can
be probed with ``--retune-full``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from assrd.features import MODEL_FEATURE_COLUMNS, RAW_COUNT_COLUMNS
from assrd.features import TEMPORAL_FEATURES, TRAILING_WINDOWS
from assrd.metrics import C_FN, C_FP, classification_metrics
from assrd.models import fit_fixed, select_threshold
from assrd.splits import support_flag

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

_LEVEL = tuple(f"ops_{m}_{w}s" for w in TRAILING_WINDOWS
               for m in ("sum", "mean", "max"))
_ACTIVITY = tuple(f"active_frac_{w}s" for w in TRAILING_WINDOWS)
_ABS_DELTA = tuple(f"ops_absdelta_mean_{w}s" for w in TRAILING_WINDOWS)
_CV = tuple(f"ops_cv_{w}s" for w in TRAILING_WINDOWS)
_WINDOWLESS = ("active_run_length", "burst_ratio")

_E_FULL = tuple(MODEL_FEATURE_COLUMNS)
_D_LEVEL = tuple(c for c in _E_FULL if c not in _WINDOWLESS)
_C_LEVEL = tuple(c for c in _D_LEVEL if c not in _ABS_DELTA and c not in _CV)
_B_LEVEL = tuple(c for c in _C_LEVEL if c not in _ACTIVITY)

#: Ordered arm -> exact column tuple. Order matters for reporting, not fitting.
ABLATION_ARMS = {
    "A_raw_only": tuple(RAW_COUNT_COLUMNS),
    "B_trailing_level": _B_LEVEL,
    "C_plus_activity": _C_LEVEL,
    "D_plus_volatility": _D_LEVEL,
    "E_full": _E_FULL,
    "T_temporal_only": tuple(TEMPORAL_FEATURES),
}

#: The nested chain used to build the cumulative comparison (C1).
NESTED_ARMS = ("A_raw_only", "B_trailing_level", "C_plus_activity",
               "D_plus_volatility", "E_full")


def arm_columns(arm):
    """Exact model-input columns for an ablation arm."""
    try:
        return ABLATION_ARMS[arm]
    except KeyError:
        raise KeyError(f"unknown arm {arm!r}; expected one of "
                       f"{tuple(ABLATION_ARMS)}") from None


def validate_arms():
    """Structural invariants the ablation relies on; raises on violation."""
    ordered = [set(arm_columns(a)) for a in NESTED_ARMS]
    for smaller, larger in zip(ordered, ordered[1:]):
        if not smaller < larger:
            raise AssertionError("ablation arms are not strictly nested")
    if ordered[-1] != set(MODEL_FEATURE_COLUMNS):
        raise AssertionError("E_full must equal the 29 model inputs")
    if set(arm_columns("T_temporal_only")) != set(TEMPORAL_FEATURES):
        raise AssertionError("T_temporal_only must equal the 26 temporal features")
    if set(arm_columns("A_raw_only")) != set(RAW_COUNT_COLUMNS):
        raise AssertionError("A_raw_only must equal the 3 raw counts")
    return True


# ---------------------------------------------------------------------------
# Fixed hyperparameters carried over from Phase 2
# ---------------------------------------------------------------------------
PHASE2_FALLBACK_PARAMS = {
    "decision_tree": {"clf__criterion": "entropy", "clf__max_depth": 3,
                      "clf__min_samples_leaf": 1, "clf__min_samples_split": 2,
                      "clf__class_weight": "balanced"},
    "random_forest": {"clf__n_estimators": 50, "clf__max_depth": 5,
                      "clf__min_samples_leaf": 1, "clf__min_samples_split": 2,
                      "clf__class_weight": "balanced_subsample"},
    "svm_calibrated": {"clf__estimator__C": 1.0, "clf__estimator__kernel": "rbf",
                       "clf__estimator__gamma": "auto",
                       "clf__estimator__class_weight": "balanced"},
    "adaboost": {"clf__n_estimators": 150, "clf__learning_rate": 0.5},
    "xgboost": {"clf__n_estimators": 50, "clf__max_depth": 3,
                "clf__learning_rate": 0.1, "clf__subsample": 1.0,
                "clf__colsample_bytree": 1.0, "clf__min_child_weight": 1},
}


def coerce_param_value(value):
    """Recover a typed value from the stringly-typed Phase 2 JSON."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if text == "None":
        return None
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def load_phase2_params(path=None):
    """Recorded Phase 2 ``best_params`` per model, coerced to real types.

    Falls back to :data:`PHASE2_FALLBACK_PARAMS` for any model missing from the
    artefact, so Phase 3 can run without the file but never silently invents a
    value it did not record.
    """
    path = Path(path) if path is not None else RESULTS_DIR / "phase2_results.json"
    params = {k: dict(v) for k, v in PHASE2_FALLBACK_PARAMS.items()}
    source = {"path": str(path), "found": False, "models_from_file": []}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        source["found"] = True
        for name, block in payload.get("models", {}).items():
            raw = block.get("best_params")
            if raw:
                params[name] = {k: coerce_param_value(v) for k, v in raw.items()}
                source["models_from_file"].append(name)
    return params, source


# fit_fixed lives in assrd.models (next to build_estimator) and is re-exported
# here because the ablation is its only Phase 3 caller.
__all__ = [
    "ABLATION_ARMS", "NESTED_ARMS", "arm_columns", "validate_arms",
    "PHASE2_FALLBACK_PARAMS", "coerce_param_value", "load_phase2_params",
    "fit_fixed", "subgroup_eval", "robustness_rows", "threshold_stability",
]


# ---------------------------------------------------------------------------
# Robustness: per-family, stealth/non-stealth, threshold stability
# ---------------------------------------------------------------------------
def subgroup_eval(meta, y_true, y_pred, proba, *, scope, value, c_fp=C_FP,
                  c_fn=C_FN):
    """Metrics for one subgroup, scored against all benign windows.

    Restricting to ``benign | subgroup-attack`` is what makes the figure
    interpretable: sensitivity answers "of this family's attacks, how many were
    caught", while the false alarms remain comparable across subgroups. Every
    rate that is undefined (no predicted positives, single class) is left as
    ``None`` by :func:`assrd.metrics.classification_metrics`.
    """
    y_true = np.asarray(y_true).astype(int)
    family = meta["family"].to_numpy()
    stealth = meta["is_stealth"].to_numpy().astype(bool)
    episode = meta["episode_id"].to_numpy()

    if scope == "family":
        attack = (y_true == 1) & (family == value)
    elif scope == "stealth":
        attack = (y_true == 1) & (stealth == (value == "stealth"))
    else:
        raise KeyError(f"unknown scope {scope!r}")

    mask = (y_true == 0) | attack
    metrics = classification_metrics(y_true[mask], y_pred[mask], proba[mask],
                                     c_fp=c_fp, c_fn=c_fn)
    n_episodes = int(np.unique(episode[attack]).size) if attack.any() else 0
    metrics.update({
        "scope": scope,
        "group": value,
        "support_attack_windows": int(attack.sum()),
        "support_episodes": n_episodes,
        "support_flag": support_flag(n_episodes),
    })
    return metrics


def robustness_rows(meta, y_true, y_pred, proba, *, family_names,
                    c_fp=C_FP, c_fn=C_FN):
    """Per-family plus stealth/non-stealth rows for one primary cell."""
    rows = [subgroup_eval(meta, y_true, y_pred, proba, scope="family", value=fam,
                          c_fp=c_fp, c_fn=c_fn) for fam in family_names]
    rows += [subgroup_eval(meta, y_true, y_pred, proba, scope="stealth", value=key,
                           c_fp=c_fp, c_fn=c_fn)
             for key in ("stealth", "non_stealth")]
    return rows


def threshold_stability(y_val, proba_val, groups, *, n_boot=100, seed=42,
                        c_fp=C_FP, c_fn=C_FN, n_grid=201):
    """Bootstrap the validation-selected threshold by **episode/block**.

    Resampling windows would treat correlated windows of one episode as
    independent evidence and understate the spread. Each bootstrap draw samples
    whole groups, so the reported spread reflects how few independent episodes
    really are. The base threshold is the un-resampled validation optimum; a wide
    interval means the operating point is a plateau, not an optimum.
    """
    y_val = np.asarray(y_val).astype(int)
    proba_val = np.asarray(proba_val, dtype=float)
    groups = np.asarray(groups)
    if y_val.shape != proba_val.shape or groups.shape != y_val.shape:
        raise ValueError("y_val, proba_val and groups must be the same shape")

    base = select_threshold(y_val, proba_val, c_fp=c_fp, c_fn=c_fn,
                            n_grid=n_grid)["threshold"]
    unique_groups = np.unique(groups)
    rng = np.random.RandomState(seed)
    thresholds = []
    for _ in range(int(n_boot)):
        draw = rng.choice(unique_groups, size=unique_groups.size, replace=True)
        indices = np.concatenate([np.flatnonzero(groups == g) for g in draw])
        if indices.size == 0:
            continue
        thresholds.append(select_threshold(
            y_val[indices], proba_val[indices], c_fp=c_fp, c_fn=c_fn,
            n_grid=n_grid)["threshold"])

    arr = np.asarray(thresholds, dtype=float)
    if arr.size == 0:
        return {"base_threshold": base, "n_boot": 0, "median": None, "iqr": None,
                "min": None, "max": None, "p10": None, "p90": None,
                "share_equal_to_base": None}
    return {
        "base_threshold": float(base),
        "n_boot": int(arr.size),
        "median": float(np.median(arr)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "share_equal_to_base": float(np.mean(arr == base)),
        "grid_points": int(n_grid),
    }
