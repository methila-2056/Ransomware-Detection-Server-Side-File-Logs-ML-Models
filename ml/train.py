"""ML training pipeline for ransomware detection.

Based on: Aranyi, G., Miseta, T. & Szucs, V. (2026).
"Ransomware detection based on server-side file operation logs using machine learning."
Journal on Information Security, 2026, 8.

Methodology
-----------
- 6 models: Decision Tree, Random Forest, SVM, AdaBoost, XGBoost, and a
  standardised Logistic Regression added as a linear control
- Features: create_count, rename_count, delete_count per 1-second window
- Classes: 0 = NORMAL, 1 = RANSOMWARE
- Hyperparameter tuning via GridSearchCV, 5-fold stratified CV (paper Sect. 4.4)
- Split: 80-10-10 train/validation/test (paper Sect. 4.3)
- Cost-sensitive evaluation: C_TP=0, C_TN=0, C_FP=1, C_FN=10
- Best model selected by (total cost, latency, -sensitivity) with deterministic
  tie-breaking, since a generator can leave several models at cost 0.

Everything written to disk goes through ``output_dir``. Tests pass a temporary
directory so a test run never overwrites the recorded results.
"""

import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml import compat  # noqa: F401  installs xgboost/sklearn tag shim if needed
from ml.metrics import (
    COST_MATRIX,
    compute_cost,
    early_detection_metrics,
    measure_latency,
    pr_analysis,
    roc_analysis,
    separability_report,
    threshold_sweep,
)
from ml.streaming import causal_feature_columns, sequence_feature_columns
from ml.threshold import compare_to_paper_threshold, select_threshold

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
# Recorded evidence lives in a tracked directory so every figure quoted in the
# write-up resolves to a committed file.
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = ["create_count", "rename_count", "delete_count"]

CV_FOLDS = 5
TEST_FRACTION = 0.10
VALIDATION_FRACTION = 0.10
RANDOM_STATE = 42


def generate_normal_samples(n_samples=2000, seed=None):
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        profile = rng.choice(["secretary", "it_admin", "ceo", "mixed"])
        if profile == "secretary":
            create, rename, delete = rng.randint(2, 8), rng.randint(0, 3), rng.randint(0, 2)
        elif profile == "it_admin":
            create, rename, delete = rng.randint(1, 6), rng.randint(1, 5), rng.randint(1, 4)
        elif profile == "ceo":
            create, rename, delete = rng.randint(0, 3), rng.randint(0, 2), rng.randint(0, 1)
        else:
            create, rename, delete = rng.randint(1, 7), rng.randint(0, 4), rng.randint(0, 3)
        samples.append([create, rename, delete, 0])
    return samples


def generate_ransomware_samples(n_samples=2000, seed=None):
    rng = random.Random(seed)
    samples = []
    for _ in range(n_samples):
        family = rng.choice(["ryuk", "wannacry", "lockbit", "teslacrypt", "notpetya"])
        if family == "ryuk":
            create, rename, delete = rng.randint(15, 55), rng.randint(5, 25), rng.randint(5, 20)
        elif family == "wannacry":
            create, rename, delete = rng.randint(20, 60), rng.randint(10, 35), rng.randint(3, 15)
        elif family == "lockbit":
            create, rename, delete = rng.randint(25, 70), rng.randint(15, 40), rng.randint(8, 25)
        elif family == "teslacrypt":
            create, rename, delete = rng.randint(10, 40), rng.randint(8, 30), rng.randint(5, 20)
        else:
            create, rename, delete = rng.randint(30, 80), rng.randint(10, 30), rng.randint(10, 35)
        samples.append([create, rename, delete, 1, family])
    return samples


def generate_dataset(n_normal=2000, n_ransomware=2000, seed=None, output_dir=None,
                     save=True):
    """Build the training dataset and optionally persist it.

    ``output_dir`` overrides the destination directory. Tests pass a temporary
    path so the recorded dataset in ``data/`` is never overwritten.
    """
    if seed is not None:
        random.seed(seed)

    rng = random.Random(seed)
    normal = generate_normal_samples(n_normal, seed=rng.getrandbits(32))
    ransomware = generate_ransomware_samples(n_ransomware, seed=rng.getrandbits(32))
    all_samples = [row[:4] for row in ransomware] + normal
    rng.shuffle(all_samples)

    df = pd.DataFrame(all_samples, columns=FEATURE_COLUMNS + ["label"])

    if save:
        target = Path(output_dir) if output_dir else DATA_DIR
        target.mkdir(parents=True, exist_ok=True)
        csv_path = target / "training_dataset.csv"
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved: {csv_path} ({len(df)} samples)")
        print(f"  Normal: {n_normal}, Ransomware: {n_ransomware}")

    return df


#: Models whose fitted parameters depend on the scale of the inputs. Tree
#: ensembles are invariant to monotone rescaling of the features, so rescaling
#: them is unnecessary work; the kernel-based SVM and the linear Logistic
#: Regression are not. The experiment drivers import this so that every result
#: file agrees with the training pipeline on which models are standardised.
SCALED_MODELS = frozenset({"SVM", "Logistic Regression"})


def _models_config():
    return {
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "params": {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, 7, 10],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 3],
            },
            "use_scaled": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [50, 100, 150],
                "criterion": ["gini", "entropy"],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
            },
            "use_scaled": False,
        },
        "SVM": {
            # Calibrated so predict_proba is available for ROC/PR analysis. The
            # base paper reports no probability output for SVM, which is why it
            # cannot appear in a ROC or PR point there.
            "model": CalibratedClassifierCV(SVC(random_state=RANDOM_STATE), ensemble=False),
            "params": {
                "estimator__C": [0.1, 1, 10],
                "estimator__kernel": ["rbf", "linear"],
                "estimator__gamma": ["scale", "auto"],
            },
            "use_scaled": True,
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=RANDOM_STATE),
            "params": {
                "n_estimators": [50, 100, 150, 200],
                "learning_rate": [0.5, 0.8, 1.0, 1.5],
            },
            "use_scaled": False,
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            "params": {
                "n_estimators": [50, 100, 150],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.1, 0.3, 0.5],
                "subsample": [0.7, 0.9, 1.0],
                "gamma": [0, 1.0, 2.0],
            },
            "use_scaled": False,
        },
        # Linear control. The five paper-inspired models are all non-linear, so
        # without this baseline there is no way to tell whether the temporal
        # features carry information the trees exploit non-linearly, or merely
        # information at all. Standardised for the same reason as the SVM: the
        # decision boundary is a hyperplane through a weighted sum of inputs,
        # so the scale of each feature changes the fitted coefficients.
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"],
                "class_weight": [None, "balanced"],
            },
            "use_scaled": True,
        },
    }


def _evaluate(model, X_eval, y_eval, X_train_for_latency, repeats=20):
    """Full metric set for one fitted model on one evaluation split."""
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    tp = int(((y_pred == 1) & (y_eval == 1)).sum())
    tn = int(((y_pred == 0) & (y_eval == 0)).sum())
    fp = int(((y_pred == 1) & (y_eval == 0)).sum())
    fn = int(((y_pred == 0) & (y_eval == 1)).sum())

    accuracy = float((y_pred == y_eval).mean())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    f1 = float(2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) else 0.0
    # NPV and MCC complete the metric set. The base paper names NPV alongside
    # specificity and F1, and continuous integration asserts that every recorded
    # model result carries it, so its absence here was a broken contract rather
    # than a deliberate omission.
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = float((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0

    roc = roc_analysis(y_eval, y_proba)
    pr = pr_analysis(y_eval, y_proba)
    latency = measure_latency(model.predict, X_train_for_latency, repeats=repeats)
    early = early_detection_metrics(y_eval, y_pred)

    return {
        "Accuracy": round(accuracy, 6),
        "Precision": round(precision, 6),
        "Recall": round(sensitivity, 6),
        "Sensitivity": round(sensitivity, 6),
        "Specificity": round(specificity, 6),
        "NPV": round(npv, 6),
        "MCC": round(mcc, 6),
        "F1 Score": round(f1, 6),
        "ROC AUC": roc["AUC"],
        "Average Precision": pr["Average Precision"],
        **latency,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Total Cost": int(compute_cost(tp, tn, fp, fn)),
        "Confusion Matrix": [[tn, fp], [fn, tp]],
        **early,
        "ROC Data": roc,
        "PR Data": pr,
        "Threshold Sweep": threshold_sweep(y_eval, y_proba),
    }


def _metrics_at(model, X_eval, y_eval, threshold):
    """Metrics at an arbitrary decision threshold."""
    from ml.metrics import binary_metrics
    proba = model.predict_proba(X_eval)[:, 1]
    return binary_metrics(y_eval, (proba >= threshold).astype(int))


def _episode_detection(rows, proba, threshold):
    """First-detection offset within each attack episode of the test split.

    ``rows`` are the test-split rows in prediction order, each carrying
    ``label``, ``episode_id`` and ``window_index``.

    The paper's abstract claims early-phase detection but reports no number for
    it. Offsets here are seconds (windows) from episode onset to the first
    correctly classified window.
    """
    pred = (np.asarray(proba) >= threshold).astype(int)
    offsets = []
    detected = 0
    total = 0

    episodes = {}
    for (_, row), p in zip(rows.iterrows(), pred):
        if int(row["label"]) != 1:
            continue
        episodes.setdefault(int(row["episode_id"]), []).append(
            (int(row["window_index"]), int(p)))

    for ep, points in sorted(episodes.items()):
        points.sort()
        total += 1
        onset = points[0][0]
        hits = [w - onset for w, p in points if p == 1]
        if hits:
            detected += 1
            offsets.append(hits[0])

    return {
        "Episodes": total,
        "Episodes Detected": detected,
        "Episode Detection Rate": round(detected / total, 6) if total else 0.0,
        "Median Offset (s)": float(np.median(offsets)) if offsets else None,
        "Mean Offset (s)": round(float(np.mean(offsets)), 4) if offsets else None,
        "Max Offset (s)": int(max(offsets)) if offsets else None,
    }


def _select_best(results):
    """Pick the best model with an explicit, deterministic rule.

    The base paper selects on sensitivity and cost. Earlier revisions of this file
    used ``min(results, key=cost)``, which returns whichever cost-0 model happens
    to come first in dict order. Ordering by (cost, latency, -sensitivity) makes
    the choice reproducible and prefers the cheapest and fastest model when cost
    ties, which is the operational objective in Sect. 3.8.
    """
    return min(results, key=lambda r: (
        r["Total Cost"],
        r.get("Avg Latency (ms)", float("inf")),
        -r.get("Sensitivity", 0.0),
        r["Model"],
    ))


def train_models(df, output_dir=None, save_artifacts=True, tune=True,
                 cv_folds=CV_FOLDS, verbose=True, models_dir=None,
                 feature_set="sequence", split_positions=None):
    """Train, tune and evaluate all six models.

    When ``tune`` is true a baseline pass with default hyperparameters runs first
    so the before/after comparison of the paper's Table 7 is reproduced.

    ``feature_set`` selects the feature basis:
      ``counts``    the paper's three per-window counts, for reproduction
      ``sequence``  counts plus trailing temporal aggregates

    A validation-split threshold is selected against the paper's cost matrix and
    reported alongside the paper's fixed 0.5, rather than replacing it silently.

    ``split_positions`` is an optional ``(train, validation, test)`` triple of
    positional indices into ``df``. Callers that have already produced an
    episode-aware partition must pass it here: this function will otherwise fall
    back to a row-level stratified split, which can place two windows of the same
    attack episode on opposite sides of the boundary and silently discard the
    caller's leakage protection.

    ``output_dir`` receives the recorded evidence (CSV, JSON, manifest).
    ``models_dir`` receives the fitted estimators. They are separate because the
    application loads the deployed model from ``models/`` while the evidence
    belongs in the tracked results directory.
    """
    if feature_set == "sequence":
        if "ops_mean_5s" not in df.columns:
            raise ValueError(
                "feature_set='sequence' requires ml.streaming.add_sequence_features "
                "to have been applied to the frame before training. "
                "Call add_sequence_features(df) first, or pass feature_set='counts'."
            )
        feature_columns = sequence_feature_columns(df)
    elif feature_set == "causal":
        if not any(c.startswith("ops_sum_prev") for c in df.columns):
            raise ValueError(
                "feature_set='causal' requires ml.streaming.add_causal_features "
                "to have been applied to the frame before training. "
                "Call add_causal_features(df) first."
            )
        feature_columns = causal_feature_columns(df)
    else:
        feature_columns = list(FEATURE_COLUMNS)
        for required in ("label",):
            if required not in df.columns:
                raise ValueError(f"missing required column {required!r}")

    if "label" not in df.columns:
        raise ValueError("missing required column 'label'")

    X = df[feature_columns].values.astype(float)
    y = df["label"].values
    positions = np.arange(len(df))

    split_policy = "row-level stratified 80-10-10"

    if split_positions is not None:
        # Honour the caller's partition. Re-splitting here would put windows of
        # the same attack episode on both sides of the boundary, which is exactly
        # the leakage the caller split to prevent.
        train_pos, val_pos, test_pos = (
            np.asarray(p, dtype=int) for p in split_positions)
        if len(set(train_pos) & set(test_pos)):
            raise ValueError(
                "split_positions overlap between train and test")
        split_policy = "caller-supplied (episode-aware where available)"
    else:
        # 80-10-10, as in the paper's Sect. 4.3. The test split is held out and
        # never used for tuning or threshold selection. Indices are carried
        # alongside the feature matrices so per-episode reporting can recover the
        # trace metadata.
        holdout_pos, test_pos, _, _ = train_test_split(
            positions, y, test_size=TEST_FRACTION * 2,
            random_state=RANDOM_STATE, stratify=y
        )
        val_relative = VALIDATION_FRACTION / (1.0 - TEST_FRACTION * 2)
        train_pos, val_pos, _, _ = train_test_split(
            holdout_pos, y[holdout_pos], test_size=val_relative,
            random_state=RANDOM_STATE, stratify=y[holdout_pos],
        )

    # Confirm the partition does not leak attack episodes, whenever the trace
    # carries episode identifiers. Previously this check existed but was never
    # invoked, so a leaky split would have produced plausible-looking metrics.
    leakage = None
    if "episode_id" in df.columns:
        from ml.traces import leakage_check
        leakage = leakage_check(df, train_pos, test_pos)
        if verbose:
            print(f"Leakage check: {leakage}")
        if not leakage["Leakage Free"]:
            raise ValueError(
                f"attack episodes appear in both train and test: "
                f"{leakage['Shared Attack Episodes']} shared. "
                f"Pass episode-aware split_positions.")

    X_test = X[test_pos]
    X_train, X_val = X[train_pos], X[val_pos]
    y_train, y_val, y_test = y[train_pos], y[val_pos], y[test_pos]
    test_rows = df.iloc[test_pos]

    if verbose:
        print("Split sizes: train=%d val=%d test=%d (%s)"
              % (len(y_train), len(y_val), len(y_test), split_policy))
        print("Dataset separability:")
        # Reported on the three base counts specifically, not on whatever feature
        # set the model uses. A rolling 30s aggregate is not a per-second count
        # and labelling it "create_count" makes the report unintelligible.
        sep = separability_report(df[FEATURE_COLUMNS].values, y)
        print(f"  linearly separable by counts alone: "
              f"{sep['Perfectly Linearly Separable']}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    config = _models_config()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    baseline = []
    results = []

    for name, cfg in config.items():
        if verbose:
            print(f"\nTraining {name}...")

        def split_data(use_scaled):
            if use_scaled:
                return X_train_sc, X_val_sc, X_test_sc
            return X_train, X_val, X_test

        X_tr, X_va, X_te = split_data(cfg["use_scaled"])

        # Baseline: default hyperparameters, evaluated on the same held-out test
        # split so the two rows in the comparison table are comparable.
        t0 = time.perf_counter()
        if tune:
            default_model = type(cfg["model"])(random_state=RANDOM_STATE) \
                if "random_state" in cfg["model"].get_params() else type(cfg["model"])()
            default_model.fit(X_tr, y_train)
            base_eval = _evaluate(default_model, X_te, y_test, X_te, repeats=5)
            base_eval["Model"] = name
            base_eval["Train Time (s)"] = round(time.perf_counter() - t0, 2)
            base_eval["Best Params"] = "default"
            baseline.append(base_eval)

            search = GridSearchCV(
                cfg["model"], cfg["params"], cv=cv, scoring="recall",
                n_jobs=-1, refit=True,
            )
            with compat.thread_backend():
                search.fit(X_tr, y_train)
            best_estimator = search.best_estimator_
            best_params = search.best_params_
        else:
            best_estimator = type(cfg["model"])()
            best_estimator.fit(X_tr, y_train)
            best_params = "tuning disabled"

        train_time = time.perf_counter() - t0
        result = _evaluate(best_estimator, X_te, y_test, X_te)

        # Threshold chosen on the validation split against the paper's cost
        # matrix. Reported next to the paper's fixed 0.5 rather than instead of it.
        y_proba_val = best_estimator.predict_proba(X_va)[:, 1]
        selection = select_threshold(y_val, y_proba_val)
        result["Selected Threshold"] = selection["Threshold"]
        result["Threshold Reason"] = selection["Reason"]
        result["Threshold Comparison"] = compare_to_paper_threshold(selection)
        result["Metrics At Selected Threshold"] = _metrics_at(
            best_estimator, X_te, y_test, selection["Threshold"])

        # Early-phase detection measured per episode, using the trace's own
        # episode boundaries when present.
        if "episode_id" in test_rows.columns:
            result["Per Episode Detection"] = _episode_detection(
                test_rows, best_estimator.predict_proba(X_te)[:, 1],
                selection["Threshold"])

        result["Model"] = name
        result["Train Time (s)"] = round(train_time, 2)
        result["Best Params"] = str(best_params)
        results.append(result)

        if save_artifacts:
            target = Path(models_dir) if models_dir else MODELS_DIR
            target.mkdir(parents=True, exist_ok=True)
            joblib.dump(best_estimator, target / f"{name.lower().replace(' ', '_')}.pkl")

        if verbose:
            print(f"  Accuracy: {result['Accuracy']:.4f} "
                  f"Sensitivity: {result['Sensitivity']:.4f} "
                  f"Specificity: {result['Specificity']:.4f} "
                  f"F1: {result['F1 Score']:.4f}")
            print(f"  Cost: {result['Total Cost']}, "
                  f"Latency: {result['Avg Latency (ms)']:.3f} ms, "
                  f"Train: {result['Train Time (s)']:.2f}s")
            print(f"  Confusion: TP={result['TP']} TN={result['TN']} "
                  f"FP={result['FP']} FN={result['FN']}")

    best = _select_best(results)
    best_name = best["Model"]

    tied = [r["Model"] for r in results if r["Total Cost"] == best["Total Cost"]]
    if verbose and len(tied) > 1:
        print(f"  Note: {len(tied)} models tie at cost {best['Total Cost']} "
              f"({', '.join(tied)}); selected by lowest latency.")

    output_dir = Path(output_dir) if output_dir else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir) if models_dir else MODELS_DIR

    if save_artifacts:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, models_dir / "scaler.pkl")
        best_src = models_dir / f"{best_name.lower().replace(' ', '_')}.pkl"
        import shutil
        shutil.copy(best_src, models_dir / "best_model.pkl")

        info = {
            # Paths are recorded relative to the project root so the artefact is
            # portable; absolute paths made the file machine-specific and broke
            # any checkout on another host.
            "model_path": f"models/{best_src.name}",
            "model_name": best_name,
            "scaler_path": "models/scaler.pkl",
            "use_scaled": config[best_name]["use_scaled"],
            "selected_by": "min(total_cost, avg_latency, -sensitivity, model_name)",
            "tied_at_best_cost": tied,
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "feature_set": feature_set,
            "split_policy": split_policy,
            "leakage_check": leakage,
            "decision_threshold": next(
                (r.get("Selected Threshold") for r in results
                 if r["Model"] == best_name), 0.5),
            "paper_threshold": 0.5,
            "trained_utc": pd.Timestamp.utcnow().isoformat(),
            "sklearn_version": __import__("sklearn").__version__,
            "evidence_dir": f"{output_dir.name}/",
            "provenance": dataset_kind_label(),
        }
        with open(models_dir / "best_model_info.json", "w") as f:
            import json
            json.dump(info, f, indent=2)

    summary_cols = ["Model", "Accuracy", "Precision", "Recall", "Sensitivity",
                    "Specificity", "NPV", "MCC", "F1 Score", "ROC AUC",
                    "Average Precision", "Avg Latency (ms)", "Train Time (s)",
                    "Total Cost", "TP", "TN", "FP", "FN"]
    results_df = pd.DataFrame(results)
    results_df[summary_cols].to_csv(output_dir / "model_results.csv", index=False)

    if baseline:
        baseline_df = pd.DataFrame(baseline)
        baseline_df[["Model", "Accuracy", "Sensitivity", "Specificity",
                     "NPV", "MCC", "F1 Score", "ROC AUC", "Total Cost"]].to_csv(
            output_dir / "model_results_baseline.csv", index=False)

    # Separability is recorded next to the results so the effect of the paper's
    # fixed 0.5 threshold, and of a trivially separable dataset, is visible.
    # Reported over the three base counts plus the full feature basis, each with
    # its own names, so neither view is mislabelled.
    import json
    separability = {
        "base_counts": separability_report(
            df[FEATURE_COLUMNS].values, df["label"].values,
            feature_names=list(FEATURE_COLUMNS)),
        "full_feature_set": separability_report(
            X, y, feature_names=list(feature_columns)),
        "feature_set": feature_set,
        "feature_count": len(feature_columns),
        "split_policy": split_policy,
        "leakage_check": leakage,
        "test_window_count": int(len(y_test)),
    }
    with open(output_dir / "separability.json", "w") as f:
        json.dump(separability, f, indent=2)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"BEST MODEL: {best_name} (Cost: {best['Total Cost']}, "
              f"Sensitivity: {best['Sensitivity']:.4f})")
        print(f"{'=' * 70}")

    if save_artifacts:
        with open(output_dir / "model_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        with open(output_dir / "model_results_baseline.json", "w") as f:
            json.dump(baseline, f, indent=2, default=str)

    return results, best_name


def write_provenance(output_dir, df, results, best_name, seed, split_sizes,
                     dataset_kind, config):
    """Record how a result file was produced.

    Every number quoted in the write-up must be traceable to a run. This manifest
    records the seed, library versions, split sizes and selection rule for the run
    that produced the sibling result files, so a reader can tell whether a given
    CSV came from a full training run or from a test invocation.
    """
    import json
    import platform
    from datetime import datetime, timezone

    import sklearn
    import xgboost

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dataset": dataset_kind,
        "dataset_rows": int(len(df)),
        "dataset_class_balance": {
            "benign": int((df["label"] == 0).sum()),
            "attack": int((df["label"] == 1).sum()),
        },
        "split": {
            "train": split_sizes["train"],
            "validation": split_sizes["validation"],
            "test": split_sizes["test"],
            "policy": config.get("split_policy", "80-10-10 stratified"),
        },
        "feature_set": config.get("feature_set", "counts"),
        "cross_validation": {"folds": config["cv_folds"], "stratified": True,
                             "scoring": "recall"},
        "cost_matrix": COST_MATRIX,
        "selection_rule": "min(total_cost, avg_latency_ms, -sensitivity, model_name)",
        "threshold_policy": "reported at the paper's fixed 0.5 AND at a "
                            "validation-split threshold selected against the "
                            "paper's cost matrix",
        "best_model": best_name,
        "models_evaluated": [r["Model"] for r in results],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
            "tags_shim_applied": bool(compat.APPLIED),
        },
    }

    path = Path(output_dir) / "run_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def train_all_models(output_dir=None, n_normal=1500, n_ransomware=1500,
                     seed=RANDOM_STATE, dataset_kind="synthetic-generator"):
    random.seed(seed)
    np.random.seed(seed)

    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  RANSOMWARE DETECTION - ML MODEL TRAINING")
    print("  Based on: Aranyi et al. (2026)")
    print("  Cost Matrix: C_TP=0, C_TN=0, C_FP=1, C_FN=10")
    print(f"  Tuning: GridSearchCV ({CV_FOLDS}-fold stratified CV, maximize recall)")
    print("  Split: 80-10-10 train/validation/test")
    print(f"  Dataset: {dataset_kind}  Seed: {seed}")
    print(f"  Output:  {output_dir}")
    print("=" * 70)

    df = generate_dataset(n_normal=n_normal, n_ransomware=n_ransomware, seed=seed)
    results, best_name = train_models(df, output_dir=output_dir, models_dir=None)

    split_sizes = {"train": int(0.8 * len(df)), "validation": int(0.1 * len(df)),
                   "test": int(0.1 * len(df))}
    write_provenance(output_dir, df, results, best_name, seed, split_sizes,
                     dataset_kind, {"cv_folds": CV_FOLDS})

    print(f"\nAll models trained.")
    print(f"Best model: {best_name}")
    print(f"Results:    {output_dir}")
    return results


def train_on_trace(n_windows=20000, attack_ratio=0.05, seed=RANDOM_STATE,
                   variant="nozeroatt", output_dir=None, models_dir=None,
                   feature_set="sequence", test_fraction=0.1,
                   validation_fraction=0.1, tune=True, verbose=True):
    """Full evaluation on a generated temporal trace.

    Builds the trace, applies sequence features, splits by attack episode so no
    episode spans the train/test boundary, then trains and evaluates all six
    models. This is the entry point that produces the reported results.
    """
    from ml.streaming import add_sequence_features
    from ml.traces import (build_report, build_trace, group_aware_split,
                           variant_report)

    output_dir = Path(output_dir) if output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("  TRACE-BASED EVALUATION")
        print(f"  Dataset: {dataset_kind_label()}")
        print(f"  Windows: {n_windows}  Requested attack ratio: {attack_ratio}")
        print(f"  Split: episode-aware ({1-test_fraction-validation_fraction:.0%} "
              f"train / {validation_fraction:.0%} val / {test_fraction:.0%} test)")
        print(f"  Feature set: {feature_set}")
        print("=" * 70)

    raw = build_trace(n_windows=n_windows, attack_ratio=attack_ratio, seed=seed)
    report = build_report(raw, requested_ratio=attack_ratio)

    if feature_set == "sequence":
        raw = add_sequence_features(raw)
    elif feature_set == "causal":
        from ml.streaming import add_causal_features
        raw = add_causal_features(raw)

    from ml.traces import DATASET_VARIANTS
    variant_df = DATASET_VARIANTS[variant](raw)

    train_df, val_df, test_df = group_aware_split(
        variant_df, test_fraction, validation_fraction, seed=seed)

    # The trace pipeline already produced the 80-10-10 partition, so the frame
    # handed to train_models carries the partition explicitly and passes it on as
    # positional indices. Concatenating and letting train_models re-split would
    # undo the episode-aware boundary.
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    n_train, n_val = len(train_df), len(val_df)
    split_positions = (
        np.arange(0, n_train),
        np.arange(n_train, n_train + n_val),
        np.arange(n_train + n_val, n_train + n_val + len(test_df)),
    )
    partitions = {
        "train": n_train,
        "validation": n_val,
        "test": len(test_df),
    }

    if verbose:
        print(f"  episodes={report['Episodes']} "
              f"achieved_attack_ratio={report['Attack Fraction']:.4f} "
              f"stealth_windows={report['Stealth Attack Windows']}")
        print(f"  split: train={partitions['train']} "
              f"(attack {int((train_df['label']==1).sum())}) "
              f"val={partitions['validation']} "
              f"(attack {int((val_df['label']==1).sum())}) "
              f"test={partitions['test']} "
              f"(attack {int((test_df['label']==1).sum())})")
        print()

    results, best_name = train_models(
        combined, output_dir=output_dir, models_dir=models_dir,
        tune=tune, verbose=verbose, feature_set=feature_set,
        split_positions=split_positions)

    # Record the dataset-level evidence alongside the model metrics.
    import json
    with open(output_dir / "dataset_report.json", "w") as f:
        json.dump({"Trace": report, "Partitions": partitions,
                   "Variant": variant,
                   "Variants": variant_report(raw)}, f, indent=2)

    write_provenance(output_dir, combined, results, best_name, seed, partitions,
                     dataset_kind_label(), {"cv_folds": CV_FOLDS,
                                            "feature_set": feature_set,
                                            "split_policy": "episode-aware 80-10-10"})

    if verbose:
        print(f"\nResults written to {output_dir}")
    return results, best_name


def dataset_kind_label():
    """Provenance string recorded with every result file."""
    return ("behaviour-faithful emulator (ml.attack_sim); benign profiles "
            "transcribed from base paper Table 3; attack profiles modelled from "
            "published family characteristics, NOT captured from executed malware")


if __name__ == "__main__":
    train_on_trace()