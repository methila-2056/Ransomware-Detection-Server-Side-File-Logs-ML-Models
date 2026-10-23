"""Soft-voting ensemble over the six base classifiers.

The base paper selects a single best model -- XGBoost, on cost 175 -- and reports
one operating point per model (Sect. 3.6, Table 7). It does not evaluate whether
combining the classifiers would do better than the best of them.

That question is worth asking here because the six models do not fail alike. On
this project's data the Decision Tree reaches AUROC 0.98 while Random Forest, SVM,
AdaBoost, XGBoost and Logistic Regression all sit above 0.999, and their
confusion matrices differ:
the tree misses different windows than the boosted models. An average of
decorrelated-but-comparable classifiers is usually stronger than any one of them.

Two combination rules are provided and compared:

``soft``
    Mean of calibrated class probabilities. Uses confidence, so a model that is
    unsure contributes less than one that is certain.

``hard``
    Majority vote on predicted labels. Ignores confidence entirely.

Reported against the best single model on the same test split, so any gain is
attributable to the combination rather than to a different evaluation.

The paper's cost matrix is applied to the ensemble as well, so the comparison is
made in the paper's own currency.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ml.metrics import COST_MATRIX, binary_metrics, compute_cost

DEFAULT_WEIGHTS = None


def _fit_calibrator(y_true, proba):
    """Isotonic calibration of an over-confident probability column.

    Tree ensembles fitted on separable data routinely output probabilities near 0
    and 1. Averaging those raw values is close to averaging hard labels. Isotonic
    regression maps them back onto observed frequencies, which restores the
    confidence information that soft voting depends on.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(proba, y_true)
    return calibrator


def combine_soft(probas, weights=None):
    """Weighted mean of calibrated probabilities."""
    stack = np.asarray(probas, dtype=float)
    if stack.ndim != 2:
        raise ValueError("probas must be a (n_models, n_samples) array")

    if weights is None:
        weights = np.ones(stack.shape[0], dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        raise ValueError("weights must not sum to zero")

    return (stack * weights[:, None]).sum(axis=0) / weights.sum()


def combine_hard(predictions):
    """Majority vote over hard label predictions."""
    stack = np.asarray(predictions, dtype=int)
    if stack.ndim != 2:
        raise ValueError("predictions must be a (n_models, n_samples) array")
    return (stack.sum(axis=0) * 2 >= stack.shape[0]).astype(int)


class SoftVotingEnsemble:
    """Ensemble of fitted classifiers combined by calibrated probability mean.

    Fitting and predicting are separated so the ensemble can be evaluated on the
    same held-out split as its members without refitting.
    """

    def __init__(self, estimators=None, weights=None, calibrate=True):
        self.estimators = estimators or {}
        self.weights = weights or DEFAULT_WEIGHTS
        self.calibrate = calibrate
        self._calibrators = {}
        self.classes_ = None

    def fit(self, X, y):
        """Fit any unfitted members and the probability calibrators.

        Calibrators are fitted on the data given to ``fit``. Passing the training
        data here rather than the validation split keeps calibration from
        influencing model selection.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        fitted = {}
        probas = []
        names = []

        for name, model in self.estimators.items():
            fitted[name] = model
            proba = model.predict_proba(X)[:, 1]
            if self.calibrate:
                self._calibrators[name] = _fit_calibrator(y, proba)
                proba = self._calibrators[name].predict(proba)
            probas.append(proba)
            names.append(name)

        self.estimators = fitted
        self._fit_names = names
        self.classes_ = np.array([0, 1])
        self._stack = np.asarray(probas, dtype=float)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        probas = []
        for name, model in self.estimators.items():
            proba = model.predict_proba(X)[:, 1]
            if name in self._calibrators:
                proba = self._calibrators[name].predict(proba)
            probas.append(proba)
        combined = combine_soft(probas, self.weights)
        return np.column_stack([1.0 - combined, combined])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def evaluate_ensemble(y_test, member_probas, member_names, weights=None,
                     thresholds=(0.5,), best_single_name=None, verbose=False):
    """Evaluate soft and hard voting against the best single member.

    ``member_probas`` is a ``{name: probability array}`` mapping. Every member is
    scored on the same test split, so the ensemble's result and the single-model
    baseline are directly comparable.
    """
    names = list(member_names)
    stack = np.vstack([member_probas[n] for n in names])

    soft = combine_soft(stack, weights)
    hard = combine_hard([(member_probas[n] >= 0.5).astype(int) for n in names])

    report = {
        "Members": names,
        "Weights": list(weights) if weights is not None else None,
        "Rules": {},
    }

    for label, scores in (("soft", soft), ("hard", hard)):
        per_threshold = {}
        for thr in thresholds:
            pred = (scores >= thr).astype(int)
            m = binary_metrics(y_test, pred)
            per_threshold[str(thr)] = {
                "Accuracy": m["Accuracy"],
                "Sensitivity": m["Sensitivity"],
                "Specificity": m["Specificity"],
                "F1 Score": m["F1 Score"],
                "MCC": m["MCC"],
                "Total Cost": m["Total Cost"],
                "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            }
        report["Rules"][label] = per_threshold

    report["Best Single Member"] = {}
    for n in names:
        m = binary_metrics(y_test, (member_probas[n] >= 0.5).astype(int))
        report["Best Single Member"][n] = {
            "Accuracy": m["Accuracy"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Total Cost": m["Total Cost"],
        }

    if best_single_name and best_single_name in report["Best Single Member"]:
        best = report["Best Single Member"][best_single_name]
        for rule, per_threshold in report["Rules"].items():
            for thr, values in per_threshold.items():
                values["Cost vs Best Single"] = best["Total Cost"] - values["Total Cost"]
                values["Sensitivity vs Best Single"] = round(
                    values["Sensitivity"] - best["Sensitivity"], 6)

    if verbose:
        best_name = best_single_name or names[0]
        best = report["Best Single Member"].get(best_name, {})
        print(f"  best single: {best_name} "
              f"sens={best.get('Sensitivity', 0):.4f} "
              f"cost={best.get('Total Cost')}")
        for rule, per_threshold in report["Rules"].items():
            values = per_threshold["0.5"]
            print(f"  {rule:<5} sens={values['Sensitivity']:.4f} "
                  f"cost={values['Total Cost']} "
                  f"(vs single: sens {values['Sensitivity vs Best Single']:+.4f}, "
                  f"cost {values['Cost vs Best Single']:+d})")

    return report


def disagreement_analysis(y_test, member_probas, member_names):
    """Where the members disagree, and what the ensemble gains from it.

    If the members were making identical errors, combining them could not help.
    The number of windows on which exactly one member is correct is the headroom
    a combiner can recover.
    """
    names = list(member_names)
    preds = {n: (member_probas[n] >= 0.5).astype(int) for n in names}
    y = np.asarray(y_test)

    n = len(y)
    all_right = np.ones(n, dtype=bool)
    all_wrong = np.ones(n, dtype=bool)
    unanimous_alarm = np.ones(n, dtype=bool)
    unanimous_clear = np.ones(n, dtype=bool)

    for name in names:
        correct = preds[name] == y
        all_right &= correct
        all_wrong &= ~correct
        unanimous_alarm &= preds[name] == 1
        unanimous_clear &= preds[name] == 0

    solo_correct = np.zeros(n, dtype=int)
    for name in names:
        solo_correct += (preds[name] == y).astype(int)

    return {
        "Samples": n,
        "All Members Correct": int(all_right.sum()),
        "All Members Wrong": int(all_wrong.sum()),
        "Split Decisions": int(n - all_right.sum() - all_wrong.sum()),
        "Unanimous Alarm": int(unanimous_alarm.sum()),
        "Unanimous Clear": int(unanimous_clear.sum()),
        "Exactly One Member Correct": int((solo_correct == 1).sum()),
        "Unanimous Alarm But Benign": int((unanimous_alarm & (y == 0)).sum()),
        "Unanimous Clear But Attack": int((unanimous_clear & (y == 1)).sum()),
        "Recoverable By Combination": int((solo_correct == 1).sum()),
    }


def weight_search(y_val, member_probas, member_names, grid=None):
    """Search weight vectors on the validation split against the paper's cost.

    Uniform weights are rarely optimal: a weaker but differently-biased member can
    be worth more than its accuracy suggests. Searched on validation only, never
    on the test split.
    """
    names = list(member_names)
    stack = np.vstack([member_probas[n] for n in names])

    if grid is None:
        grid = [0.5, 1.0, 1.5, 2.0, 3.0]

    best = None
    trials = []

    # Uniform weight on subsets of members, plus uniform on all with a global
    # scale. Exhaustive search over the full grid is exponential in the members.
    subsets = []
    for mask in range(1, 1 << len(names)):
        subsets.append([n for i, n in enumerate(names) if mask & (1 << i)])

    for subset in subsets:
        idx = [names.index(n) for n in subset]
        sub_stack = stack[idx]
        if len(idx) == 1:
            continue
        scores = combine_soft(sub_stack)
        pred = (scores >= 0.5).astype(int)
        tp = int(((pred == 1) & (y_val == 1)).sum())
        tn = int(((pred == 0) & (y_val == 0)).sum())
        fp = int(((pred == 1) & (y_val == 0)).sum())
        fn = int(((pred == 0) & (y_val == 1)).sum())
        cost = compute_cost(tp, tn, fp, fn)
        entry = {"Members": subset, "Total Cost": int(cost)}
        trials.append(entry)
        if best is None or cost < best["Total Cost"]:
            best = entry

    return {
        "Best": best,
        "Trials Evaluated": len(trials),
        "Uniform All": next((t for t in trials
                             if len(t["Members"]) == len(names)), None),
        "Grid": grid,
    }