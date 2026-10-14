"""Phase 1 targeted tests.

Seven tests covering the three Phase 1 components:

1. temporal feature count is exactly 26 and matches the code
2. every temporal feature name carries a documented window width
3. the 3 raw counts are separate from, and never inside, the temporal features
4. hand-computed feature values are reproduced exactly
5. the emulator emits valid, correctly labelled windows
6. episode-aware splitting shares no attack episode across subsets
7. the compatibility repair makes xgboost usable with sklearn
"""

import numpy as np
import pandas as pd
import pytest

import assrd
from assrd import compat
from assrd.data import RAW_COUNT_COLUMNS as DATA_RAW_COUNTS
from assrd.data import build_trace, split_by_episode, trace_report
from assrd.features import (
    MODEL_FEATURE_COLUMNS,
    RAW_COUNT_COLUMNS,
    TEMPORAL_FEATURES,
    TRAILING_WINDOWS,
    add_temporal_features,
    feature_report,
    temporal_feature_names,
)


# ---------------------------------------------------------------------------
# 1. Feature count is exactly 26 and matches the code
# ---------------------------------------------------------------------------
def test_feature_count_is_26():
    assert len(TEMPORAL_FEATURES) == 26

    # The declared constant must equal what the code actually produces.
    df = pd.DataFrame({
        "create_count": [0, 1, 2, 3, 0, 0, 4, 0, 0, 0],
        "rename_count": [0] * 10,
        "delete_count": [0] * 10,
    })
    produced = add_temporal_features(df)
    produced_temporal = [c for c in produced.columns if c in TEMPORAL_FEATURES]
    assert set(produced_temporal) == set(TEMPORAL_FEATURES)
    assert len(produced_temporal) == 26

    # And the declared tuple must equal the derived one.
    assert TEMPORAL_FEATURES == temporal_feature_names(TRAILING_WINDOWS)

    # 4 widths x 6 families + 2 windowless features.
    assert 4 * 6 + 2 == 26


# ---------------------------------------------------------------------------
# 2. Every temporal feature name carries a documented window width
# ---------------------------------------------------------------------------
def test_feature_names_match_documented_windows():
    widths = {f"{w}s" for w in TRAILING_WINDOWS}
    windowless = {"active_run_length", "burst_ratio"}

    for name in TEMPORAL_FEATURES:
        suffix = name.rsplit("_", 1)[-1]
        if name in windowless:
            assert not suffix.endswith("s"), name
        else:
            assert suffix in widths, f"{name} has an undocumented width"

    # Each per-window family has exactly one feature per width.
    per_family = {
        "ops_sum": 0, "ops_mean": 0, "ops_max": 0,
        "active_frac": 0, "ops_absdelta_mean": 0, "ops_cv": 0,
    }
    for name in TEMPORAL_FEATURES:
        for family in per_family:
            if name.startswith(family + "_"):
                per_family[family] += 1
                break
    assert all(v == len(TRAILING_WINDOWS) for v in per_family.values()), per_family


# ---------------------------------------------------------------------------
# 3. Raw counts are separate from the temporal features
# ---------------------------------------------------------------------------
def test_raw_counts_separate_from_temporal():
    assert list(RAW_COUNT_COLUMNS) == ["create_count", "rename_count", "delete_count"]
    assert list(RAW_COUNT_COLUMNS) == list(DATA_RAW_COUNTS)

    for col in RAW_COUNT_COLUMNS:
        assert col not in TEMPORAL_FEATURES

    # 3 raw + 26 temporal = 29 model inputs.
    assert len(MODEL_FEATURE_COLUMNS) == 29
    assert list(MODEL_FEATURE_COLUMNS[:3]) == list(RAW_COUNT_COLUMNS)
    assert list(MODEL_FEATURE_COLUMNS[3:]) == list(TEMPORAL_FEATURES)

    # The three derivation helpers must not leak into the model inputs.
    for helper in ("ops_total", "is_active_window", "ops_delta"):
        assert helper not in MODEL_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# 4. Hand-computed feature values
# ---------------------------------------------------------------------------
def test_hand_computed_feature_values():
    # total per window: 0 1 2 3 0 0 4 0 0 0
    df = pd.DataFrame({
        "create_count": [0, 1, 2, 3, 0, 0, 4, 0, 0, 0],
        "rename_count": [0] * 10,
        "delete_count": [0] * 10,
    })
    out = add_temporal_features(df)

    assert list(out["ops_sum_3s"]) == [0, 1, 3, 6, 5, 3, 4, 4, 4, 0]
    assert list(out["ops_max_5s"]) == [0, 1, 2, 3, 3, 3, 4, 4, 4, 4]
    assert list(out["active_run_length"]) == [0, 1, 2, 3, 0, 0, 1, 0, 0, 0]
    # active windows are those with total > 0.
    assert list(out["is_active_window"]) == [0, 1, 1, 1, 0, 0, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# 5. Emulator emits valid, correctly labelled windows
# ---------------------------------------------------------------------------
def test_emulator_produces_valid_labelled_windows():
    df = build_trace(n_windows=6000, attack_ratio=0.05, seed=7)
    report = trace_report(df)

    for col in RAW_COUNT_COLUMNS:
        assert (df[col] >= 0).all()
        assert pd.api.types.is_integer_dtype(df[col])

    # Labels and episode ids are consistent both ways.
    assert (df.loc[df["label"] == 1, "episode_id"] >= 0).all()
    assert (df.loc[df["label"] == 0, "episode_id"] == -1).all()

    # Every attack window belongs to a family drawn from the profile set.
    assert df.loc[df["label"] == 1, "family"].notna().all()
    assert report["Episodes"] > 0
    assert report["Attack"] > 0
    assert report["Windows"] == 6000


# ---------------------------------------------------------------------------
# 6. Episode-aware split shares no attack episode
# ---------------------------------------------------------------------------
def test_split_by_episode_has_no_leakage():
    df = build_trace(n_windows=20000, attack_ratio=0.05, seed=42)
    train, val, test = split_by_episode(df, 0.1, 0.1, seed=42)

    def eps(part):
        return set(part.loc[part["episode_id"] >= 0, "episode_id"].tolist())

    tr, va, te = eps(train), eps(val), eps(test)
    assert tr.isdisjoint(te), "attack episodes shared between train and test"
    assert tr.isdisjoint(va), "attack episodes shared between train and validation"
    assert va.isdisjoint(te), "attack episodes shared between validation and test"

    # Every subset must contain attacks for a sensitivity figure to mean anything.
    for part in (train, val, test):
        assert int((part["label"] == 1).sum()) > 0

    # The split must be a partition of the whole trace.
    assert len(train) + len(val) + len(test) == len(df)

    # Deterministic for a fixed seed.
    train2, val2, test2 = split_by_episode(df, 0.1, 0.1, seed=42)
    assert len(train2) == len(train) and len(test2) == len(test)


# ---------------------------------------------------------------------------
# 7. Compatibility repair makes xgboost usable with sklearn
# ---------------------------------------------------------------------------
def test_compat_shim_makes_xgboost_classifier():
    import sklearn
    from sklearn.base import is_classifier
    from xgboost import XGBClassifier

    # After importing the package, sklearn must be able to classify the estimator.
    assert is_classifier(XGBClassifier()) is True

    diag = compat.diagnose()
    assert diag["sklearn_version"] == sklearn.__version__
    assert diag["is_classifier_ok"] is True

    # If this environment needs the repair, it must have been applied.
    if diag["needs_shim"]:
        assert diag["shim_applied"] is True

    # The threading backend must be usable for tuning (loky workers would not
    # carry the monkeypatch).
    rng = np.random.RandomState(0)
    X = rng.randn(200, 4)
    y = (X[:, 0] > 0).astype(int)
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    search = GridSearchCV(XGBClassifier(random_state=0, eval_metric="logloss"),
                          {"max_depth": [2, 3]}, cv=cv, scoring="recall", n_jobs=-1)
    with compat.threading_backend():
        search.fit(X, y)
    assert search.best_estimator_.predict_proba(X[:3]).shape == (3, 2)
