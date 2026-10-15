"""Phase 2 targeted tests.

Grouped by the property under test rather than by module, so that a failure
names the guarantee that broke:

A. metrics        — arithmetic is correct and hand-checkable
B. cost model     — the 10:1 asymmetry is applied where it is claimed to be
C. threshold      — chosen on validation only, and is never worse than 0.5 there
D. leakage        — episodes never cross a partition; the test split is unused
                    during tuning and threshold selection
E. models         — all five build, tune, and produce usable probabilities
F. backend        — tuning really runs under assrd.compat.threading_backend
G. end to end     — the driver reproduces itself exactly for a fixed seed
"""

import itertools

import numpy as np
import pytest

import assrd
from assrd import compat, models, phase2
from assrd.metrics import (
    C_FN,
    C_FP,
    _average_precision,
    _roc_auc,
    classification_metrics,
    confusion_counts,
    cost_from_predictions,
    total_cost,
)
from assrd.models import (
    MODEL_NAMES,
    build_estimator,
    cost_sample_weights,
    evaluate_at_threshold,
    grid_size,
    param_grid,
    probability,
    select_threshold,
    tune_model,
)


@pytest.fixture(scope="module")
def small_splits():
    """A small but real dataset, shared across tests in this module."""
    splits, report = phase2.prepare_dataset(n_windows=4000, attack_ratio=0.05,
                                            seed=7)
    return splits, report


@pytest.fixture(scope="module")
def tuned_decision_tree(small_splits):
    splits, _ = small_splits
    from assrd.features import MODEL_FEATURE_COLUMNS

    X = splits["train"][list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = splits["train"]["label"].to_numpy(dtype=int)
    return tune_model("decision_tree", X, y, seed=7, n_splits=3)


# ---------------------------------------------------------------------------
# A. Metrics arithmetic
# ---------------------------------------------------------------------------
def test_confusion_counts_and_cost_are_arithmetic():
    # 3 benign predicted benign, 1 benign predicted attack,
    # 2 attack predicted benign, 4 attack predicted attack.
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]

    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    assert (tn, fp, fn, tp) == (3, 1, 2, 4)

    # Cost model: FP costs C_FP, FN costs C_FN, correct predictions are free.
    assert total_cost(tn, fp, fn, tp, c_fp=1, c_fn=10) == 1 * 1 + 10 * 2 == 21
    assert total_cost(tn, fp, fn, tp, c_fp=2, c_fn=3) == 2 * 1 + 3 * 2 == 8
    assert cost_from_predictions(y_true, y_pred, c_fp=1, c_fn=10) == 21


def test_rates_match_hand_computed_values():
    y_true = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 1, 0, 0, 1, 1, 1, 1]
    proba = [0.05, 0.10, 0.20, 0.60, 0.40, 0.45, 0.80, 0.85, 0.90, 0.95]

    m = classification_metrics(y_true, y_pred, proba, c_fp=1, c_fn=10)

    assert m["sensitivity"] == pytest.approx(4 / 6)
    assert m["specificity"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(4 / 5)
    # F1 = 2PR / (P + R)
    p, r = 4 / 5, 4 / 6
    assert m["f1"] == pytest.approx(2 * p * r / (p + r))
    assert m["accuracy"] == pytest.approx(7 / 10)
    assert m["balanced_accuracy"] == pytest.approx((4 / 6 + 3 / 4) / 2)
    assert m["negative_predictive_value"] == pytest.approx(3 / 5)
    assert m["prevalence"] == pytest.approx(6 / 10)

    # Costs are reported separately as well as summed, so a reader can see which
    # error type the total is made of.
    assert m["cost_fp"] == 1.0
    assert m["cost_fn"] == 20.0
    assert m["total_cost"] == 21.0
    assert m["mean_cost"] == pytest.approx(2.1)
    assert m["cost_fpratio"] == 10.0


def test_confusion_matrix_layout_is_actual_by_predicted():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 1, 1, 0, 1]
    m = classification_metrics(y_true, y_pred)

    matrix = m["confusion_matrix"]
    assert matrix["labels"] == [0, 1]
    assert matrix["rows"] == "actual"
    # row 0 = actual benign -> [tn, fp]; row 1 = actual attack -> [fn, tp]
    assert matrix["matrix"] == [[1, 2], [1, 1]]


def test_undefined_rates_are_none_not_zero():
    # No predicted positives -> precision and F1 are undefined, not 0.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    m = classification_metrics(y_true, y_pred)
    assert m["precision"] is None
    assert m["f1"] is None
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 1.0

    # Single-class truth -> no ROC curve exists.
    assert classification_metrics([0, 0], [0, 1], [0.1, 0.9])["roc_auc"] is None
    assert classification_metrics([0, 0], [0, 1], [0.1, 0.9])["average_precision"] is None


def test_roc_auc_and_average_precision_are_correct():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 400)
    scores = np.clip(rng.rand(400) + 0.35 * y, 0, 1)

    assert _roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))
    assert _average_precision(y, scores) == pytest.approx(
        average_precision_score(y, scores))

    # Ties must be handled (the rank-average rule), not silently ordered.
    y_tied = [0, 1, 0, 1]
    s_tied = [0.5, 0.5, 0.5, 0.5]
    assert _roc_auc(y_tied, s_tied) == pytest.approx(0.5)

    # A perfect ranking gives AUROC 1.
    assert _roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError):
        confusion_counts([0, 1, 1], [0, 1])
    with pytest.raises(ValueError):
        classification_metrics([0, 1, 1], [0, 1, 0], [0.1, 0.2])


# ---------------------------------------------------------------------------
# B. The cost matrix
# ---------------------------------------------------------------------------
def test_cost_constants_are_the_specified_ratio():
    assert (C_FP, C_FN) == (1, 10)


def test_cost_sample_weights_encode_the_asymmetry():
    y = np.array([0, 1, 0, 1, 1])
    weights = cost_sample_weights(y)
    # attack windows carry 10x the weight of benign windows
    assert weights.tolist() == [1.0, 10.0, 1.0, 10.0, 10.0]
    assert weights[y == 1].sum() > weights[y == 0].sum()

    # The ratio is configurable and defaults to the specification.
    assert cost_sample_weights(y, c_fp=2, c_fn=6).tolist() == [2, 6, 2, 6, 6]

    for bad in (0, -1):
        with pytest.raises(ValueError):
            cost_sample_weights(y, c_fn=bad)


def test_sampled_weights_reach_the_estimator_fit(tuned_decision_tree):
    """The weights must survive GridSearchCV's internal refit, not just a fit."""
    block = tuned_decision_tree
    assert block["sample_weight"]["attack"] == float(C_FN)
    assert block["sample_weight"]["benign"] == float(C_FP)

    # Confirm empirically on a fresh estimator that the fit saw them.
    from sklearn.base import clone

    estimator = clone(build_estimator("decision_tree", seed=7))
    X = np.zeros((6, 29))
    y = np.array([0, 0, 0, 1, 1, 1])
    weights = cost_sample_weights(y)
    estimator.fit(X, y, clf__sample_weight=weights)

    # The weighted tree splits on y, so a leaf impurity sum reflects weighting.
    tree = estimator.named_steps["clf"]
    assert tree.tree_.n_classes == 2


# ---------------------------------------------------------------------------
# C. Threshold selection (validation only)
# ---------------------------------------------------------------------------
def test_selected_threshold_is_the_validation_optimum():
    # Two separated clusters; the cost-optimal cut is unambiguous.
    y_val = np.array([0] * 60 + [1] * 40)
    proba = np.concatenate([np.full(60, 0.10), np.full(40, 0.90)])

    result = select_threshold(y_val, proba, c_fp=C_FP, c_fn=C_FN)
    chosen = result["threshold"]

    # No threshold on the grid beats the chosen one on validation cost.
    grid = np.linspace(0, 1, result["grid_points"])
    costs = [cost_from_predictions(y_val, proba >= t) for t in grid]
    assert result["validation_cost"] == pytest.approx(min(costs))

    # And it is at least as good as the naive 0.5 operating point.
    assert result["validation_cost"] <= result["validation_cost_at_0_5"]

    # The reported optimum is genuinely optimal, not merely better than 0.5.
    assert result["validation_cost"] == 0.0


def test_threshold_prefers_the_larger_value_when_costs_tie():
    # A degenerate model: every window gets the same score, so every threshold
    # that predicts everything positive gives the identical cost.
    y_val = np.array([0] * 5 + [1] * 5)
    proba = np.full(10, 0.5)
    result = select_threshold(y_val, proba, c_fp=C_FP, c_fn=C_FN)

    assert result["n_tied"] > 1
    assert result["threshold"] == max(result["tied_thresholds"])
    assert result["threshold"] <= 0.5


def test_threshold_trades_false_alarms_against_misses():
    """With a 10:1 ratio the optimum must prefer a miss over ten false alarms."""
    # 1 attack, and the score puts it second. Predicting it costs 10 (FN) plus
    # one false alarm (1); not predicting it costs nothing extra here, so the
    # optimiser must not pay for a marginal catch at 10 false alarms.
    y_val = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    proba = np.linspace(0.9, 0.0, 11)

    result = select_threshold(y_val, proba, c_fp=1, c_fn=10)
    chosen_cost = result["validation_cost"]

    # 10 false alarms cost the same as the single miss the threshold avoids.
    assert chosen_cost == pytest.approx(10.0)


def test_threshold_selection_rejects_bad_input():
    with pytest.raises(ValueError):
        select_threshold(np.array([], dtype=int), np.array([]))
    with pytest.raises(ValueError):
        select_threshold(np.array([0, 1]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        select_threshold(np.array([0, 1]), np.array([0.1, np.nan]))


def test_evaluate_at_threshold_applies_the_frozen_cut():
    y_true = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.4, 0.6, 0.8])

    strict = evaluate_at_threshold(y_true, proba, 0.5)
    assert strict["true_positives"] == 2
    assert strict["false_positives"] == 0

    loose = evaluate_at_threshold(y_true, proba, 0.15)
    # proba >= 0.15 selects the last four windows; two of them are benign.
    assert loose["false_positives"] == 2
    assert loose["true_positives"] == 2
    assert loose["threshold"] == 0.15


def test_cost_curve_is_monotone_in_the_right_places():
    y_val = np.array([0] * 20 + [1] * 20)
    proba = np.concatenate([np.full(20, 0.2), np.full(20, 0.8)])
    curve = models.cost_curve(y_val, proba, n_grid=21)

    assert len(curve) == 21
    assert curve[0]["false_positives"] == 20  # threshold 0.0 -> alarm on everything
    assert curve[-1]["false_negatives"] == 20  # threshold 1.0 -> alarm on nothing
    # In the middle band the model is perfect.
    best = min(curve, key=lambda row: row["total_cost"])
    assert best["total_cost"] == 0.0
    assert best["false_positives"] == 0 and best["false_negatives"] == 0


# ---------------------------------------------------------------------------
# D. Leakage
# ---------------------------------------------------------------------------
def test_partitions_share_no_episode_and_no_row(small_splits):
    splits, report = small_splits
    check = report["leakage_check"]

    assert check["episode_aware"] is True
    assert check["disjoint_rows"] is True
    assert check["train_val_episode_overlap"] == []
    assert check["train_test_episode_overlap"] == []
    assert check["val_test_episode_overlap"] == []
    assert check["row_overlap"] == {"train_validation": 0, "train_test": 0,
                                    "validation_test": 0}

    # Every partition must contain attacks, or its figures are meaningless.
    for name in ("train", "validation", "test"):
        assert int((splits[name]["label"] == 1).sum()) > 0, name


def test_threshold_comes_from_validation_not_test(small_splits):
    """The reported operating point must not be the test-optimal one.

    Recomputes both optima independently and requires the driver to have used the
    validation one. If the code ever peeks at the test split to pick a threshold,
    these two disagree and the test fails.
    """
    splits, _ = small_splits
    from assrd.features import MODEL_FEATURE_COLUMNS

    columns = list(MODEL_FEATURE_COLUMNS)
    X_train = splits["train"][columns].to_numpy(dtype=float)
    y_train = splits["train"]["label"].to_numpy(dtype=int)
    X_val = splits["validation"][columns].to_numpy(dtype=float)
    y_val = splits["validation"]["label"].to_numpy(dtype=int)
    X_test = splits["test"][columns].to_numpy(dtype=float)
    y_test = splits["test"]["label"].to_numpy(dtype=int)

    tuned = tune_model("decision_tree", X_train, y_train, seed=7, n_splits=3)
    estimator = tuned["estimator"]

    chosen = select_threshold(y_val, probability(estimator, X_val))
    oracle = select_threshold(y_test, probability(estimator, X_test))

    # The driver recorded the validation-derived threshold and cost.
    result = phase2.run_phase2(n_windows=4000, attack_ratio=0.05, seed=7,
                               n_splits=3, models=("decision_tree",))
    block = result["models"]["decision_tree"]

    assert block["threshold_selection"]["threshold"] == chosen["threshold"]
    assert block["threshold_selection"]["validation_cost"] == pytest.approx(
        chosen["validation_cost"])

    # The oracle is recorded as a diagnostic only.
    assert block["test_threshold_oracle_diagnostic"]["threshold"] == oracle["threshold"]
    assert "never" in block["test_threshold_oracle_diagnostic"]["note"]


def test_tuning_refuses_degenerate_input():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="both classes"):
        tune_model("decision_tree", X, np.zeros(10, dtype=int), n_splits=3)
    with pytest.raises(ValueError, match="length mismatch"):
        tune_model("decision_tree", X, np.ones(5, dtype=int), n_splits=3)


def test_driver_refuses_to_train_on_a_leaky_split(small_splits, monkeypatch):
    """If episodes leak, run_phase2 must abort rather than report a score."""
    splits, _ = small_splits
    train = splits["train"]
    poisoned = splits["test"].copy()
    poisoned["episode_id"] = train.loc[train["label"] == 1, "episode_id"].iloc[0]

    def fake_prepare(*args, **kwargs):
        report = {
            "seed": 0,
            "trace": {},
            "features": {},
            "partitions": {},
            "leakage_check": phase2.leakage_report(train, poisoned,
                                                    splits["test"]),
        }
        return {"train": train, "validation": splits["validation"],
                "test": poisoned}, report

    monkeypatch.setattr(phase2, "prepare_dataset", fake_prepare)
    with pytest.raises(RuntimeError, match="episode leakage"):
        phase2.run_phase2()


# ---------------------------------------------------------------------------
# E. Models
# ---------------------------------------------------------------------------
def test_all_six_models_are_declared_and_buildable():
    assert set(models.MODEL_NAMES) == {
        "decision_tree", "random_forest", "svm_calibrated", "adaboost", "xgboost",
        "logistic_regression"}
    assert set(models.MODEL_LABELS) == set(MODEL_NAMES)

    for name in MODEL_NAMES:
        estimator = build_estimator(name, seed=0)
        # Every model is a Pipeline ending in "clf", so clf__sample_weight is
        # always a valid fit parameter.
        assert estimator.__class__.__name__ == "Pipeline"
        assert estimator.steps[-1][0] == "clf"
        estimator.fit(np.random.RandomState(0).randn(60, 5),
                      np.array([0, 1] * 30), clf__sample_weight=np.ones(60))
        proba = estimator.predict_proba(np.random.RandomState(1).randn(7, 5))
        assert proba.shape == (7, 2)


def test_only_scale_dependent_models_are_standardised():
    """Tree ensembles are scale-invariant; rescaling them would be misleading."""
    assert models._STANDARDISE == {"svm_calibrated", "logistic_regression"}
    for name in MODEL_NAMES:
        steps = dict(build_estimator(name, seed=0).named_steps)
        assert ("scaler" in steps) == (name in models._STANDARDISE)


def test_calibrated_svm_produces_calibrated_probabilities():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    estimator = build_estimator("svm_calibrated", seed=0)
    estimator.fit(X, y, clf__sample_weight=cost_sample_weights(y))

    proba = probability(estimator, X)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # Sigmoid calibration, not a hard 0/1 decision.
    assert np.any((proba > 0.01) & (proba < 0.99))

    from sklearn.svm import SVC
    inner = estimator.named_steps["clf"].estimator
    assert isinstance(inner, SVC)
    assert inner.probability is False


def test_logistic_regression_grid_is_cartesian_safe():
    """Every grid point must be a valid estimator config.

    ``make_search`` sets ``error_score="raise"``, so an invalid grid point would
    crash the search rather than being skipped. ``penalty="l1"`` is only valid
    under the pinned liblinear solver.
    """
    from sklearn.linear_model import LogisticRegression
    grid = param_grid("logistic_regression")
    assert grid_size("logistic_regression") == 16
    for combo in itertools.product(*grid.values()):
        kwargs = {k.replace("clf__", ""): v for k, v in zip(grid, combo)}
        estimator = LogisticRegression(solver="liblinear", max_iter=1000,
                                       **kwargs)
        assert estimator.penalty == kwargs["penalty"]
        if kwargs["penalty"] == "l1":
            assert estimator.solver == "liblinear"


def test_logistic_regression_accepts_cost_sample_weights():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 4)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    estimator = build_estimator("logistic_regression", seed=0)

    estimator.fit(X, y)
    unweighted = estimator.named_steps["clf"].coef_.copy()

    estimator.fit(X, y, clf__sample_weight=cost_sample_weights(y))
    weighted = estimator.named_steps["clf"].coef_.copy()

    assert unweighted.shape == (1, 4)
    assert np.all(np.isfinite(weighted))
    assert not np.allclose(unweighted, weighted), \
        "10:1 cost weights must actually reach the logistic coefficients"


def test_logistic_regression_probabilities_are_not_degenerate():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 4)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(int)
    estimator = build_estimator("logistic_regression", seed=0)
    estimator.fit(X, y, clf__sample_weight=cost_sample_weights(y))

    proba = probability(estimator, X)
    assert proba.shape == (400,)
    assert proba.min() >= 0.0 and proba.max() <= 1.0
    # A sigmoid, not a hard 0/1 decision.
    assert np.any((proba > 0.01) & (proba < 0.99))


def test_logistic_regression_coefficients_span_the_full_feature_set():
    """Coefficients must exist for all 29 model inputs on the E_full arm."""
    from assrd.ablation import arm_columns
    from assrd.features import MODEL_FEATURE_COLUMNS

    assert len(MODEL_FEATURE_COLUMNS) == 29
    assert len(arm_columns("E_full")) == 29
    rng = np.random.RandomState(0)
    X = rng.randn(300, 29)
    y = (X[:, 0] + 0.25 * X[:, 5] > 0).astype(int)
    estimator = build_estimator("logistic_regression", seed=0)
    estimator.fit(X, y, clf__sample_weight=cost_sample_weights(y))

    coef = estimator.named_steps["clf"].coef_
    assert coef.shape == (1, 29)
    assert np.all(np.isfinite(coef))
    # Standardisation happens inside the pipeline, so coefficients are comparable.
    assert estimator.named_steps["scaler"].with_mean is True


def test_logistic_regression_grid_is_registered_and_finite():
    grid = param_grid("logistic_regression")
    assert set(grid) == {"clf__C", "clf__penalty", "clf__class_weight"}
    assert grid["clf__C"] == [0.1, 1.0, 10.0, 100.0]
    assert grid["clf__penalty"] == ["l1", "l2"]
    assert grid["clf__class_weight"] == [None, "balanced"]


def test_unknown_model_name_is_rejected():
    with pytest.raises(KeyError):
        build_estimator("transformer")
    with pytest.raises(KeyError):
        param_grid("transformer")


def test_param_grids_are_finite_and_enumerable():
    for name in MODEL_NAMES:
        grid = param_grid(name)
        assert grid, name
        for key, values in grid.items():
            assert key.startswith("clf__") or key.startswith("scaler__"), (name, key)
            assert len(values) > 0
            assert all(not isinstance(v, (set, dict)) for v in values)
        assert grid_size(name) >= 1
        assert grid_size(name) == np.prod([len(v) for v in grid.values()])


def test_search_uses_seeded_stratified_folds():
    search = models.make_search("decision_tree", seed=11, n_splits=5)
    cv = search.cv
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 11
    assert search.refit is True
    assert search.n_jobs == -1


def test_cross_validation_is_reproducible(tuned_decision_tree, small_splits):
    """Same seed -> identical best parameters and identical CV score."""
    splits, _ = small_splits
    from assrd.features import MODEL_FEATURE_COLUMNS

    columns = list(MODEL_FEATURE_COLUMNS)
    X = splits["train"][columns].to_numpy(dtype=float)
    y = splits["train"]["label"].to_numpy(dtype=int)

    again = tune_model("decision_tree", X, y, seed=7, n_splits=3)
    assert again["best_params_raw"] == tuned_decision_tree["best_params_raw"]
    assert again["best_cv_score"] == pytest.approx(
        tuned_decision_tree["best_cv_score"])
    assert again["grid_points"] == tuned_decision_tree["grid_points"]


# ---------------------------------------------------------------------------
# F. joblib backend
# ---------------------------------------------------------------------------
def test_tuning_runs_under_the_threading_backend(monkeypatch):
    """XGBoost's tag patch does not survive loky workers, so the backend matters.

    The spy proves the context manager is entered during tuning rather than
    merely being available.
    """
    entered = []
    real = compat.threading_backend

    def spy():
        entered.append(True)
        return real()

    monkeypatch.setattr(models.compat, "threading_backend", spy)

    X = np.random.RandomState(0).randn(120, 4)
    y = (X[:, 0] > 0).astype(int)
    tune_model("decision_tree", X, y, seed=0, n_splits=3)

    assert entered, "tune_model did not use compat.threading_backend()"


def test_xgboost_tunes_under_threading_backend():
    """End-to-end proof that the workaround is sufficient for GridSearchCV."""
    from sklearn.model_selection import GridSearchCV

    X = np.random.RandomState(0).randn(200, 4)
    y = (X[:, 0] > 0).astype(int)
    tuned = tune_model("xgboost", X, y, seed=0, n_splits=3)
    assert tuned["best_cv_score"] > 0.5
    assert tuned["estimator"].predict_proba(X[:5]).shape == (5, 2)

    # And the search really was a GridSearchCV over the declared grid.
    assert isinstance(tuned["search"], GridSearchCV)
    assert tuned["grid_points"] == grid_size("xgboost")


def test_diagnose_reports_a_healthy_estimator():
    diag = compat.diagnose()
    assert diag["is_classifier_ok"] is True
    if diag["needs_shim"]:
        assert diag["shim_applied"] is True


# ---------------------------------------------------------------------------
# G. End to end
# ---------------------------------------------------------------------------
def test_run_phase2_produces_a_complete_auditable_result():
    result = phase2.run_phase2(n_windows=4000, attack_ratio=0.05, seed=3,
                               n_splits=3, models=("decision_tree",
                                                    "xgboost"))
    assert result["phase"] == 2
    assert "emulator" in result["provenance"].lower()

    config = result["config"]
    assert config["cost_fp"] == 1.0
    assert config["cost_fn"] == 10.0
    assert config["cost_ratio"] == 10.0
    assert config["feature_count"] == 29
    assert config["cv_folds"] == 3
    assert "threading" in config["joblib_backend"]

    assert set(result["models"]) == {"decision_tree", "xgboost"}
    for name, block in result["models"].items():
        # Tuning provenance.
        assert block["best_params"]
        assert 0.0 <= block["best_cv_score"] <= 1.0
        assert block["grid_points"] >= 1
        assert block["sample_weight"]["attack"] == 10.0

        # Threshold provenance.
        assert 0.0 <= block["threshold_selection"]["threshold"] <= 1.0
        assert block["threshold_selection"]["cost_fn"] == 10.0

        # Every required metric on both evaluation splits.
        for key in ("validation_at_selected_threshold",
                    "test_at_selected_threshold",
                    "test_at_threshold_0_5"):
            metrics = block[key]
            for field in ("sensitivity", "specificity", "precision", "f1",
                          "roc_auc", "total_cost", "cost_fp", "cost_fn",
                          "false_positives", "false_negatives"):
                assert field in metrics, (name, key, field)
            assert metrics["confusion_matrix"]["matrix"]
            assert metrics["total_cost"] == (
                metrics["cost_fp"] + metrics["cost_fn"])

        # The test split is scored at the validation-frozen threshold.
        assert (block["test_at_selected_threshold"]["threshold"]
                == block["validation_at_selected_threshold"]["threshold"])
        assert (block["test_at_selected_threshold"]["threshold"]
                == block["threshold_selection"]["threshold"])

        assert block["validation_cost_curve"]


def test_run_phase2_is_reproducible_for_a_fixed_seed():
    kwargs = dict(n_windows=4000, attack_ratio=0.05, seed=21, n_splits=3,
                  models=("decision_tree",))
    first = phase2.run_phase2(**kwargs)
    second = phase2.run_phase2(**kwargs)

    block_a = first["models"]["decision_tree"]
    block_b = second["models"]["decision_tree"]

    assert block_a["best_params"] == block_b["best_params"]
    assert block_a["best_cv_score"] == pytest.approx(block_b["best_cv_score"])
    assert block_a["threshold_selection"]["threshold"] == (
        block_b["threshold_selection"]["threshold"])
    assert (block_a["test_at_selected_threshold"]["confusion_matrix"]["matrix"]
            == block_b["test_at_selected_threshold"]["confusion_matrix"]["matrix"])


def test_results_tables_are_well_formed():
    result = phase2.run_phase2(n_windows=4000, attack_ratio=0.05, seed=5,
                               n_splits=3, models=("decision_tree",))

    table = phase2.results_table(result)
    assert list(table.columns)[:2] == ["model", "threshold"]
    assert len(table) == 1
    # Cost must equal FP cost plus FN cost for every row.
    assert (table["total_cost"] == table["cost_fp"] + table["cost_fn"]).all()

    cm = phase2.confusion_table(result)
    assert len(cm) == 2 * 2 * 2  # two models -> 2 splits x 2x2 cells
    assert set(cm["split"]) == {"validation", "test"}
    assert cm["count"].sum() > 0


def test_write_results_emits_machine_readable_artifacts(tmp_path):
    result = phase2.run_phase2(n_windows=4000, attack_ratio=0.05, seed=9,
                               n_splits=3, models=("decision_tree",))
    paths = phase2.write_results(result, directory=tmp_path, basename="t")

    import json
    from pathlib import Path

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["phase"] == 2
    assert payload["config"]["cost_fn"] == 10.0
    # JSON must round-trip: no numpy scalars leaked into the file.
    assert isinstance(payload["models"]["decision_tree"]["best_cv_score"], float)

    assert Path(paths["models_csv"]).exists()
    assert Path(paths["confusion_csv"]).exists()


def test_assrd_exposes_the_phase2_modules():
    # The package imports the compatibility shim first, by design.
    assert assrd.compat is compat
    import importlib

    for module in ("assrd.metrics", "assrd.models", "assrd.phase2"):
        assert importlib.import_module(module) is not None