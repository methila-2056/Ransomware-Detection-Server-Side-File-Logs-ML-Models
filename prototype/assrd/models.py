"""Phase 2 — the five cost-sensitive classifiers, tuning, and threshold choice.

Design contract
---------------
1. **Tuning sees the training split only.** :func:`tune_model` accepts one
   feature matrix and one label vector. The experiment driver is responsible for
   passing it the training partition; nothing in this module has access to the
   test partition.
2. **The threshold is chosen on validation data only.** :func:`select_threshold`
   is given the validation probabilities. The test split is scored once, at the
   already-frozen threshold.
3. **Costs enter twice, for two different reasons.**
   * During fitting, via per-sample weights (:func:`cost_sample_weights`): a
     missed attack is ten times a false alarm, so attack windows should dominate
     the loss the minimiser sees.
   * After fitting, via threshold selection: given the learned ranking, the
     operating point that minimises ``C_FP*FP + C_FN*FN`` on validation.
   Neither alone is sufficient. Weights alone leave the operating point at the
   model's arbitrary default; threshold selection alone cannot repair a model
   whose ranking was fitted to treat a false alarm as expensive as a miss.
4. **Tuning is scored on recall.** Under a 10:1 asymmetry, recall is the
   threshold-independent quantity that best tracks the objective. The search is
   deliberately biased away from false negatives; the cost-optimal threshold then
   pays back the resulting false alarms.

joblib backend
--------------
Every search runs inside :func:`assrd.compat.threading_backend`. ``loky``
workers are separate interpreters and would not carry the xgboost/sklearn tag
patch installed by ``assrd.compat``, so ``GridSearchCV`` over XGBoost would fail
or silently mis-tag the estimator. The context manager is applied uniformly to
all five models rather than conditionally, so that tuning behaves identically
regardless of which model is running. Estimator-internal ``n_jobs`` is pinned to
1 to avoid thread oversubscription inside joblib's own worker threads.
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from assrd import compat
from assrd.metrics import C_FN, C_FP, cost_from_predictions, total_cost
from assrd.metrics import confusion_counts

MODEL_NAMES = (
    "decision_tree",
    "random_forest",
    "svm_calibrated",
    "adaboost",
    "xgboost",
    "logistic_regression",
)

MODEL_LABELS = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm_calibrated": "Calibrated SVM (RBF, Platt scaling)",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression (L1/L2, standardised)",
}

#: Features whose scale matters. Tree ensembles are invariant to monotone
#: rescaling, so they are deliberately left unscaled. The distance/kernel-based
#: SVM and the L2-regularised logistic regression both define an objective that
#: is not scale-invariant, so those two are standardised.
_STANDARDISE = {"svm_calibrated", "logistic_regression"}


def cost_sample_weights(y, c_fp=C_FP, c_fn=C_FN):
    """Per-sample weights encoding the 10:1 error asymmetry.

    A window whose true label is attack is charged ``c_fn``; a benign window is
    charged ``c_fp``. Returned as float array aligned to ``y``.
    """
    y = np.asarray(y).astype(int)
    weights = np.where(y == 1, float(c_fn), float(c_fp)).astype(float)
    if not np.all(np.isfinite(weights)):
        raise ValueError("cost weights must be finite")
    if np.any(weights <= 0):
        raise ValueError("cost weights must be strictly positive")
    return weights


def build_estimator(name, *, seed=42, n_splits_inner=3):
    """Return the unfitted pipeline for one model.

    Every model is wrapped in a :class:`~sklearn.pipeline.Pipeline` so that
    ``clf__sample_weight`` is a valid fit parameter for all five, and so that
    scaling (where required) cannot be forgotten at a call site.
    """
    if name not in MODEL_NAMES:
        raise KeyError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")

    if name == "decision_tree":
        clf = DecisionTreeClassifier(random_state=seed)
    elif name == "random_forest":
        clf = RandomForestClassifier(random_state=seed, n_jobs=1)
    elif name == "svm_calibrated":
        # probability=False: the inner model never needs probabilities because
        # CalibratedClassifierCV fits its own sigmoid on the decision function.
        inner = SVC(kernel="rbf", probability=False, random_state=seed)
        clf = CalibratedClassifierCV(
            estimator=inner,
            method="sigmoid",
            cv=StratifiedKFold(n_splits=n_splits_inner, shuffle=True,
                               random_state=seed),
        )
    elif name == "adaboost":
        clf = AdaBoostClassifier(random_state=seed)
    elif name == "logistic_regression":
        # ``liblinear`` is pinned rather than searched because it supports every
        # penalty/class_weight combination in param_grid AND per-sample weights.
        # A grid crossing ``penalty="l1"`` with an lbfgs/saga solver would raise,
        # and ``make_search`` sets ``error_score="raise"``, so an invalid point
        # would crash the search instead of being skipped.
        clf = LogisticRegression(solver="liblinear", max_iter=1000,
                                 random_state=seed)
    else:  # xgboost
        from xgboost import XGBClassifier

        clf = XGBClassifier(random_state=seed, n_jobs=1, eval_metric="logloss",
                            verbosity=0)

    steps = []
    if name in _STANDARDISE:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def param_grid(name):
    """The exhaustive search space for one model.

    Sized so that a full Phase 2 run stays interactive while still covering the
    axes that matter (depth, capacity, regularisation, learning rate).
    """
    if name == "decision_tree":
        return {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [3, 5, 7, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__class_weight": [None, "balanced"],
        }
    if name == "random_forest":
        return {
            "clf__n_estimators": [50, 100, 150],
            "clf__max_depth": [5, 10, 15, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 3],
            "clf__class_weight": [None, "balanced_subsample"],
        }
    if name == "svm_calibrated":
        return {
            "clf__estimator__C": [0.1, 1.0, 10.0, 100.0],
            "clf__estimator__gamma": ["scale", "auto"],
            "clf__estimator__kernel": ["rbf"],
            "clf__estimator__class_weight": [None, "balanced"],
        }
    if name == "adaboost":
        return {
            "clf__n_estimators": [50, 100, 150, 200],
            "clf__learning_rate": [0.5, 0.8, 1.0, 1.5],
        }
    if name == "logistic_regression":
        # 4 x 2 x 2 = 16 points. Every combination is valid under the pinned
        # liblinear solver, so the Cartesian product is safe.
        return {
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__penalty": ["l1", "l2"],
            "clf__class_weight": [None, "balanced"],
        }
    if name == "xgboost":
        return {
            "clf__n_estimators": [50, 100, 150],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.1, 0.3, 0.5],
            "clf__subsample": [0.7, 0.9, 1.0],
            "clf__colsample_bytree": [0.7, 1.0],
            "clf__min_child_weight": [1, 5],
        }
    raise KeyError(name)


def grid_size(name):
    """Number of points in :func:`param_grid` (used in reports/tests)."""
    total = 1
    for values in param_grid(name).values():
        total *= len(values)
    return total


def make_search(name, *, seed=42, n_splits=5, scoring="recall", n_jobs=-1):
    """Build the (unfitted) :class:`GridSearchCV` for one model.

    The folds are :class:`StratifiedKFold` with a fixed shuffle seed, so the
    partition is identical on every run for a given ``seed``. Row-level folds are
    safe here only because the *callers* have already separated episodes; see
    ``assrd.data.split_by_episode``.
    """
    return GridSearchCV(
        estimator=build_estimator(name, seed=seed),
        param_grid=param_grid(name),
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed),
        scoring=scoring,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=False,
        error_score="raise",
    )


def tune_model(name, X, y, *, seed=42, n_splits=5, scoring="recall", n_jobs=-1,
               c_fp=C_FP, c_fn=C_FN, cost_weighted=True):
    """Tune one model on ``(X, y)`` — the training split, by contract.

    Returns a dict with the fitted search, the best parameters, the best CV
    score and the cost weights actually applied.

    ``cost_weighted=False`` disables the per-sample weights, which is the M3/M4
    half of the Phase 3 matrix: it isolates the effect of *weighting* from the
    effect of *threshold selection*. The default is unchanged.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")
    if np.unique(y).size < 2:
        raise ValueError("tuning requires both classes in the training split")

    weights = cost_sample_weights(y, c_fp, c_fn) if cost_weighted else None
    search = make_search(name, seed=seed, n_splits=n_splits, scoring=scoring,
                         n_jobs=n_jobs)

    fit_params = {"clf__sample_weight": weights} if weights is not None else {}

    # Required while the xgboost/sklearn tag patch is active; see module docstring.
    with compat.threading_backend():
        search.fit(X, y, **fit_params)

    sample_weight = (
        {"weighted": True, "attack": float(c_fn), "benign": float(c_fp),
         "mean": float(weights.mean())} if weights is not None
        else {"weighted": False, "attack": 1.0, "benign": 1.0, "mean": 1.0})
    return {
        "model": name,
        "search": search,
        "estimator": search.best_estimator_,
        "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
        "best_params_raw": dict(search.best_params_),
        "best_cv_score": float(search.best_score_),
        "cv_results_mean": _cv_summary(search),
        "grid_points": grid_size(name),
        "cv_folds": n_splits,
        "scoring": scoring,
        "cost_weighted": bool(cost_weighted),
        "sample_weight": sample_weight,
    }


def fit_fixed(name, X, y, params=None, *, seed=42, cost_weighted=True,
              c_fp=C_FP, c_fn=C_FN):
    """Fit one model at **fixed** hyperparameters — no search.

    Phase 3's ablation applies the parameters Phase 2 already selected, so that
    every arm differs only in its feature set. ``cost_weighted`` selects whether
    the 10:1 per-sample weights reach the estimator; the threshold half of the
    experiment is applied by the caller.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).astype(int)
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch: {X.shape[0]} vs {y.shape[0]}")

    estimator = build_estimator(name, seed=seed)
    if params:
        estimator.set_params(**params)

    weights = cost_sample_weights(y, c_fp, c_fn) if cost_weighted else None
    fit_params = {"clf__sample_weight": weights} if weights is not None else {}

    with compat.threading_backend():
        estimator.fit(X, y, **fit_params)

    sample_weight = (
        {"weighted": True, "attack": float(c_fn), "benign": float(c_fp),
         "mean": float(weights.mean())} if weights is not None
        else {"weighted": False, "attack": 1.0, "benign": 1.0, "mean": 1.0})
    return {
        "model": name,
        "estimator": estimator,
        "params": dict(params or {}),
        "cost_weighted": bool(cost_weighted),
        "sample_weight": sample_weight,
    }


def _cv_summary(search):
    """Mean/std of the CV score over the grid, for dispersion reporting."""
    scores = np.asarray(search.cv_results_["mean_test_score"], dtype=float)
    return {
        "grid_mean": float(np.mean(scores)),
        "grid_std": float(np.std(scores)),
        "grid_min": float(np.min(scores)),
        "grid_max": float(np.max(scores)),
        "candidates_scored": int(scores.size),
    }


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (str, bool, type(None))):
        return value
    return str(value)


def probability(estimator, X):
    """Attack-class probability for every row of ``X``."""
    proba = estimator.predict_proba(np.asarray(X, dtype=float))
    classes = list(getattr(estimator, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError(f"estimator has no attack class; classes={classes}")
    return np.asarray(proba[:, classes.index(1)], dtype=float)


def select_threshold(y_val, proba_val, *, c_fp=C_FP, c_fn=C_FN, n_grid=1001):
    """Pick the cost-minimising threshold on **validation** data.

    Sweeps a fixed grid of thresholds, scores each by
    ``C_FP*FP + C_FN*FN`` on ``y_val``, and returns the best one.

    Tie-breaking: when several thresholds achieve the same minimum cost, the
    **largest** is chosen. Among cost-equivalent operating points the higher
    threshold raises fewer alarms, which is the safer default for a detector
    that is expected to page a human. The choice is recorded in the result so a
    reader can see it was deliberate.

    Nothing here sees the test partition; that is enforced by the call site and
    covered by a test that asserts the returned threshold is the validation
    optimum, not the test optimum.
    """
    y_val = np.asarray(y_val).astype(int)
    proba_val = np.asarray(proba_val, dtype=float)
    if y_val.shape != proba_val.shape:
        raise ValueError(
            f"shape mismatch: y_val {y_val.shape} vs proba_val {proba_val.shape}")
    if y_val.size == 0:
        raise ValueError("cannot select a threshold from an empty validation split")
    if not np.all(np.isfinite(proba_val)):
        raise ValueError("non-finite probabilities in validation split")

    grid = np.linspace(0.0, 1.0, int(n_grid))
    costs = np.empty(grid.size, dtype=float)
    for i, threshold in enumerate(grid):
        costs[i] = cost_from_predictions(y_val, proba_val >= threshold,
                                         c_fp, c_fn)

    best_cost = costs.min()
    tied = np.flatnonzero(costs == best_cost)
    chosen_index = int(tied.max())  # largest threshold among the minimisers

    chosen = float(grid[chosen_index])
    return {
        "threshold": chosen,
        "validation_cost": float(best_cost),
        "tied_thresholds": [float(grid[i]) for i in tied],
        "n_tied": int(tied.size),
        "grid_points": int(grid.size),
        "cost_fp": float(c_fp),
        "cost_fn": float(c_fn),
        "validation_cost_at_0_5": float(
            cost_from_predictions(y_val, proba_val >= 0.5, c_fp, c_fn)),
    }


def evaluate_at_threshold(y_true, proba, threshold, *, c_fp=C_FP, c_fn=C_FN):
    """Metrics for one frozen threshold. This is the only test-time operation."""
    from assrd.metrics import classification_metrics

    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    out = classification_metrics(y_true, y_pred, proba, c_fp=c_fp, c_fn=c_fn)
    out["threshold"] = float(threshold)
    return out


def cost_curve(y_true, proba, *, c_fp=C_FP, c_fn=C_FN, n_grid=101):
    """Cost as a function of threshold — the evidence behind the choice."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    grid = np.linspace(0.0, 1.0, int(n_grid))
    rows = []
    for threshold in grid:
        tn, fp, fn, tp = confusion_counts(y_true, proba >= threshold)
        rows.append({
            "threshold": float(threshold),
            "false_positives": fp,
            "false_negatives": fn,
            "total_cost": total_cost(tn, fp, fn, tp, c_fp, c_fn),
        })
    return rows