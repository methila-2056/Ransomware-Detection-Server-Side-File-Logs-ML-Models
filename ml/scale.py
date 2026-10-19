"""Scale evaluation up to the base paper's dataset size.

The base paper reports 2.5 million log entries collected over 30 days. This
module trains and evaluates at that scale and records how performance varies
with dataset size, so "more data" is a measured curve rather than an assertion.

Practical constraint, measured not assumed
------------------------------------------
A single Decision Tree fits 1.15M training rows in roughly 17 seconds, but
AdaBoost with 100 boosted stages takes about 420 seconds on the same matrix.
Five models under 5-fold cross-validation at full scale would therefore take
several hours.

``budget`` exists for that reason. It caps the rows a slow model trains on while
leaving fast models at full scale, and the cap is recorded in the output. A model
trained on a subsample is labelled as such, because reporting its score as a
full-scale result would misstate what was computed. Sensitivity is the quantity
of interest and it is far more stable under subsampling than cost or accuracy,
which is why the full-scale headline numbers are checked against it separately.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ml.metrics import binary_metrics, roc_analysis
from ml.streaming import add_sequence_features, sequence_feature_columns
from ml.train import SCALED_MODELS
from ml.traces import DATASET_VARIANTS, build_trace, group_aware_split

# Models whose fit time grows badly with row count.
#
# Measured on 1.15M training rows x 26 features: Decision Tree ~20 s, XGBoost
# (hist, 100 trees) ~90 s, Random Forest (100 trees) several minutes, AdaBoost
# (100 stages) ~420 s, and CalibratedClassifierCV(SVC) worse still because the
# calibration wrapper fits three SVCs. The three below are therefore capped by
# ``budget`` while the two cheap models train on everything.
#
# The cap is recorded in the output as ``Subsampled``. A capped model's score is
# not a full-scale result and must not be reported as one.
SLOW_MODELS = {"AdaBoost", "Random Forest", "SVM"}


def _factory(name, seed):
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, criterion="entropy",
                                      random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=12,
                                       random_state=seed, n_jobs=-1)
    if name == "SVM":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import SVC
        return CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", random_state=seed),
                                      ensemble=False)
    if name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=1.0,
                                  random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             subsample=0.9, tree_method="hist",
                             random_state=seed, eval_metric="logloss",
                             n_jobs=-1)
    if name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    raise ValueError(f"unknown model {name}")


MODEL_NAMES = ("Decision Tree", "Random Forest", "SVM", "AdaBoost", "XGBoost",
               "Logistic Regression")


def prepare(n_windows, seed=42, attack_ratio=0.05, verbose=True):
    """Build, enrich, variant and split a trace at the requested size."""
    timings = {}

    started = time.perf_counter()
    df = build_trace(n_windows=n_windows, attack_ratio=attack_ratio, seed=seed)
    timings["build_seconds"] = round(time.perf_counter() - started, 2)
    timings["build_memory_mb"] = round(df.memory_usage(deep=True).sum() / 1e6, 1)

    started = time.perf_counter()
    enriched = add_sequence_features(df)
    timings["feature_seconds"] = round(time.perf_counter() - started, 2)
    timings["feature_memory_mb"] = round(
        enriched.memory_usage(deep=True).sum() / 1e6, 1)
    del df

    started = time.perf_counter()
    variant = DATASET_VARIANTS["nozeroatt"](enriched)
    del enriched
    train_df, val_df, test_df = group_aware_split(variant, 0.1, 0.1, seed=seed)
    timings["split_seconds"] = round(time.perf_counter() - started, 2)

    if verbose:
        print(f"  {n_windows:>9,} windows -> train={len(train_df):,} "
              f"val={len(val_df):,} test={len(test_df):,} "
              f"attack={(test_df['label'] == 1).sum():,}  "
              f"({sum(timings.values()):.1f}s)")

    return train_df, val_df, test_df, timings


def evaluate_scale(n_windows, seed=42, models=MODEL_NAMES, budget=None,
                   attack_ratio=0.05, verbose=True):
    """Train and score the models at one dataset size.

    ``budget`` caps training rows for the slow models only. When it binds, the
    returned record carries ``Trained On Rows`` so the result is not mistaken for
    a full-scale one.
    """
    train_df, val_df, test_df, timings = prepare(n_windows, seed, attack_ratio,
                                                verbose)

    features = sequence_feature_columns(train_df)
    scaler_mean, scaler_std = None, None

    Xtr_full = train_df[features].values.astype(np.float32)
    ytr_full = train_df["label"].values
    Xva = val_df[features].values.astype(np.float32)
    yva = val_df["label"].values
    Xte = test_df[features].values.astype(np.float32)
    yte = test_df["label"].values

    records = []
    for name in models:
        model = _factory(name, seed)
        needs_scaling = name in SCALED_MODELS
        trained_on = len(Xtr_full)

        if needs_scaling:
            scaler_mean = Xtr_full.mean(axis=0)
            scaler_std = Xtr_full.std(axis=0)
            scaler_std[scaler_std == 0] = 1.0

        started = time.perf_counter()
        Xtr, ytr = Xtr_full, ytr_full
        if budget and name in SLOW_MODELS and budget < len(Xtr_full):
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(Xtr_full), size=budget, replace=False))
            Xtr = Xtr_full[keep]
            ytr = ytr_full[keep]
            trained_on = budget

        def design(matrix):
            if needs_scaling:
                return (matrix - scaler_mean) / scaler_std
            return matrix

        model.fit(design(Xtr), ytr)
        fit_seconds = round(time.perf_counter() - started, 2)

        proba = model.predict_proba(design(Xte))[:, 1]
        m = binary_metrics(yte, (proba >= 0.5).astype(int))

        records.append({
            "Model": name,
            "Windows Requested": n_windows,
            "Train Rows": int(len(Xtr)),
            "Trained On Rows": int(trained_on),
            "Subsampled": bool(trained_on < len(Xtr_full)),
            "Test Rows": int(len(yte)),
            "Test Attack Windows": m["Support Attack"],
            "Fit Seconds": fit_seconds,
            "Accuracy": m["Accuracy"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "F1 Score": m["F1 Score"],
            "ROC AUC": roc_analysis(yte, proba)["AUC"],
            "Total Cost": m["Total Cost"],
            "Cost Per 1000 Windows": m["Cost Per 1000 Windows"],
            "Cost Per Attack Window": m["Cost Per Attack Window"],
        })

        if verbose:
            flag = " (subsampled)" if records[-1]["Subsampled"] else ""
            print(f"    {name:<16}{fit_seconds:>8.1f}s  "
                  f"sens={m['Sensitivity']:.4f} "
                  f"cost={m['Total Cost']:>6} "
                  f"cost/atk={m['Cost Per Attack Window']:.4f}{flag}")

    return {
        "Windows": n_windows,
        "Attack Ratio Requested": attack_ratio,
        "Timings": timings,
        "Budget": budget,
        "Feature Count": len(features),
        "Models": records,
    }


def run_scale_curve(sizes=(10_000, 100_000, 500_000, 2_500_000), seed=42,
                    budget=250_000, models=MODEL_NAMES, verbose=True,
                    output_path=None):
    """Evaluate at increasing dataset sizes and report the trend.

    The question is whether the paper's 91.87% sensitivity is a property of the
    method or of its dataset volume. A flat curve says more data does not help
    here; a rising one says the paper's figure is volume-limited.

    Progress is written to ``output_path`` after every size, so a long run leaves
    usable evidence behind even if it is interrupted. An earlier full-scale sweep
    ran past a 50-minute limit and produced nothing at all, because the results
    were only serialised at the end.
    """
    if verbose:
        print(f"scale curve: sizes={list(sizes)} budget={budget}")

    runs = []
    for position, size in enumerate(sizes):
        runs.append(evaluate_scale(size, seed=seed, models=models, budget=budget,
                                   verbose=verbose))
        if output_path:
            _write_partial(output_path, runs, sizes, budget, position + 1)
        if verbose:
            print(f"  [{position + 1}/{len(sizes)}] {size:,} complete")

    curves = {}
    for name in models:
        points = []
        for run in runs:
            for record in run["Models"]:
                if record["Model"] == name:
                    points.append({
                        "Windows": run["Windows"],
                        "Sensitivity": record["Sensitivity"],
                        "Specificity": record["Specificity"],
                        "Accuracy": record["Accuracy"],
                        "Cost Per Attack Window": record["Cost Per Attack Window"],
                        "Fit Seconds": record["Fit Seconds"],
                        "Subsampled": record["Subsampled"],
                    })
        curves[name] = points

    # Per-model delta across the largest two sizes: a flat slope means dataset
    # volume is no longer the binding constraint.
    trends = {}
    for name, points in curves.items():
        if len(points) >= 2:
            first, last = points[0], points[-1]
            delta = last["Sensitivity"] - first["Sensitivity"]
            trends[name] = {
                "Sensitivity At Smallest": first["Sensitivity"],
                "Sensitivity At Largest": last["Sensitivity"],
                "Delta": round(delta, 6),
                "Converged": abs(delta) < 0.01,
            }

    return {
        "Sizes": list(sizes),
        "Budget": budget,
        "Subsampled Models": sorted(SLOW_MODELS),
        "Base Paper": {
            "Sensitivity": 0.9187,
            "Attack Windows": 209,
            "Total Cost": 175,
            "Cost Per Attack Window": round(175 / 209, 4),
            "Log Entries": 2_500_000,
            "Note": "the paper's cost is a sum over its own test split and cannot "
                    "be compared with a different split size; cost per attack "
                    "window is the comparable quantity",
        },
        "Runs": runs,
        "Curves": curves,
        "Trends": trends,
    }


def _write_partial(path, runs, sizes, budget, completed):
    """Serialise whatever has finished so an interrupted run is not wasted."""
    import json

    payload = {
        "Sizes": list(sizes),
        "Budget": budget,
        "Completed Sizes": [r["Windows"] for r in runs],
        "Sizes Requested": len(sizes),
        "Sizes Complete": completed,
        "Partial": completed < len(sizes),
        "Runs": runs,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def run_full_scale(n_windows=2_500_000, seed=42, budget=None, verbose=True):
    """Single run at the base paper's dataset size."""
    return evaluate_scale(n_windows, seed=seed, budget=budget, verbose=verbose)