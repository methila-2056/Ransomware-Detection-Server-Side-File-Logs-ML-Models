"""Feature engineering: 3 raw baseline counts vs 26 proposed temporal features.

The abstract's central claim is that the base paper's *three per-window counts*
have a structural ceiling, and that *26 trailing temporal features* over 3, 5,
10 and 30-second windows remove it. This module makes that boundary explicit in
code so it cannot drift out of sync with the prose:

* :data:`RAW_COUNT_COLUMNS`  — the 3 baseline counts (create/rename/delete)
* :data:`TEMPORAL_FEATURES`  — the 26 proposed trailing features
* :data:`MODEL_FEATURE_COLUMNS` — raw + temporal = 29 model inputs

Three derivation helpers (``ops_total``, ``is_active_window``, ``ops_delta``)
are produced but deliberately *excluded* from both tuples: they are intermediate
quantities, not model inputs, and counting them would turn "26" into a lie.

Definitions
-----------
Let ``c_i, r_i, d_i`` be the create/rename/delete counts in window ``i`` and

    T_i = c_i + r_i + d_i            (total operations in window i)
    A_i = 1 if T_i > 0 else 0        (active indicator)
    D_i = T_i - T_{i-1}             (first difference, D_0 = 0)

For each window width ``w`` in {3, 5, 10, 30}, over the trailing ``w`` windows
including the current one (``min_periods=1``):

    ops_sum_w        = sum_{j=max(0,i-w+1)..i} T_j
    ops_mean_w       = mean of the same window
    ops_max_w        = max  of the same window
    active_frac_w    = mean of A_j over the same window
    ops_absdelta_w   = mean of |D_j| over the same window
    ops_cv_w         = std(T)/mean(T) over the same window  (0 when mean = 0)

plus two windowless features:

    active_run_length = length of the current run of consecutive active windows
                        (0 at the first inactive window, resets after a gap)
    burst_ratio       = ops_mean_{3s} / mean(T) over the last 60 windows,
                        clipped to [0, 20]  (short-term rate vs long baseline)

Feature count: 4 widths x 6 families = 24, plus ``active_run_length`` and
``burst_ratio`` = 26.

All aggregates include the current window. That is correct for operational
detection — the window has elapsed before it is classified — and, because the
same rolling operator is applied identically at training and inference time,
nothing leaks across a train/test boundary. The features must be computed on the
whole trace *in temporal order* before splitting; ``add_temporal_features`` is a
whole-trace operation for that reason.
"""

import numpy as np
import pandas as pd

RAW_COUNT_COLUMNS = ["create_count", "rename_count", "delete_count"]

TRAILING_WINDOWS = (3, 5, 10, 30)
BASELINE_WINDOW = 60

# Helpers produced by add_temporal_features but never used as model inputs.
_HELPER_COLUMNS = ("ops_total", "is_active_window", "ops_delta")


def temporal_feature_names(trailing=TRAILING_WINDOWS):
    """The 26 temporal feature names for the given window widths."""
    names = []
    for w in trailing:
        names += [
            f"ops_sum_{w}s",
            f"ops_mean_{w}s",
            f"ops_max_{w}s",
            f"active_frac_{w}s",
            f"ops_absdelta_mean_{w}s",
            f"ops_cv_{w}s",
        ]
    names += ["active_run_length", "burst_ratio"]
    return tuple(names)


TEMPORAL_FEATURES = temporal_feature_names()
MODEL_FEATURE_COLUMNS = tuple(RAW_COUNT_COLUMNS) + TEMPORAL_FEATURES


def _run_length(active):
    """Length of the current run of consecutive active windows."""
    values = np.asarray(active, dtype=int)
    out = np.zeros(len(values), dtype=int)
    current = 0
    for i, v in enumerate(values):
        current = current + 1 if v else 0
        out[i] = current
    return out


def add_temporal_features(df, trailing=TRAILING_WINDOWS,
                          baseline_window=BASELINE_WINDOW,
                          count_columns=None):
    """Return a copy of ``df`` augmented with the 26 temporal features.

    ``df`` must be one row per one-second window in temporal order. Apply this to
    the whole trace before any split.
    """
    cols = count_columns or RAW_COUNT_COLUMNS
    out = df.copy()

    total = out[cols].sum(axis=1).astype(float)
    out["ops_total"] = total
    active = (total > 0).astype(int)
    out["is_active_window"] = active

    for w in trailing:
        roll = total.rolling(w, min_periods=1)
        out[f"ops_sum_{w}s"] = roll.sum()
        out[f"ops_mean_{w}s"] = roll.mean()
        out[f"ops_max_{w}s"] = roll.max()
        out[f"active_frac_{w}s"] = active.rolling(w, min_periods=1).mean()

    delta = total.diff().fillna(0.0)
    out["ops_delta"] = delta
    for w in trailing:
        out[f"ops_absdelta_mean_{w}s"] = (
            delta.abs().rolling(w, min_periods=1).mean())

    out["active_run_length"] = _run_length(active)

    baseline = total.rolling(baseline_window, min_periods=1).mean().replace(0, np.nan)
    short = out[f"ops_mean_{trailing[0]}s"]
    out["burst_ratio"] = ((short / baseline)
                          .replace([np.inf, -np.inf], np.nan)
                          .fillna(0.0).clip(0, 20))

    for w in trailing:
        mean = out[f"ops_mean_{w}s"]
        std = total.rolling(w, min_periods=1).std().fillna(0.0)
        out[f"ops_cv_{w}s"] = (std / mean.replace(0, np.nan)).fillna(0.0).clip(0, 10)

    return out


def feature_report(df):
    """Return the raw/temporal split as a small dict, for result manifests."""
    return {
        "raw_count_columns": list(RAW_COUNT_COLUMNS),
        "raw_count_count": len(RAW_COUNT_COLUMNS),
        "temporal_features": list(TEMPORAL_FEATURES),
        "temporal_feature_count": len(TEMPORAL_FEATURES),
        "model_feature_columns": list(MODEL_FEATURE_COLUMNS),
        "model_feature_count": len(MODEL_FEATURE_COLUMNS),
        "helper_columns_excluded": list(_HELPER_COLUMNS),
        "trailing_windows": list(TRAILING_WINDOWS),
        "baseline_window": BASELINE_WINDOW,
    }
