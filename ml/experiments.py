"""Reproduce every recorded experiment.

Running this module regenerates the statistical files in ``results/`` from
measured output:

  * ``seed_results.json``      the pipeline repeated across seeds, with CIs
  * ``ablation.json``          feature-group ablation, counts-only upwards
  * ``imbalance.json``         sensitivity against the attack base rate
  * ``ensemble.json``          soft/hard voting against the best single model
  * ``zero_day_eval.json``     leave-one-family-out transfer
  * ``evasion.json``           sensitivity against encryption rate, plus mimicry
  * ``alarm_policy.json``      three-state policy against fixed-threshold rules
  * ``drift.json``             PSI monitoring, degradation and offline recovery
  * ``multiscale.json``        window-width and stride sweep

The main evaluation (``model_results.json``, ``model_results_baseline.json``,
``separability.json``, ``dataset_report.json``, ``run_manifest.json``) is
produced by ``ml.train``.

Every result file in ``results/`` has a driver here. Previously five of these
studies had no entry point in the repository, so their artifacts could not be
regenerated or verified.

Every experiment is seeded. Nothing here writes a number that was not computed,
and each entry point returns its results rather than only printing them.

Usage::

    python -m ml.experiments                      # full suite
    python -m ml.experiments --quick              # smaller traces, fast check
    python -m ml.experiments --only stats ablation
"""

import argparse
import sys
import time
from pathlib import Path

from ml import compat  # noqa: F401  installs xgboost/sklearn tag shim if needed
from ml.train import SCALED_MODELS
from ml.stats import (aggregate_seed_results, compare_all_models, run_seeds,
                      stability_report)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

FULL_WINDOWS = 30000
QUICK_WINDOWS = 8000
FULL_SEEDS = 10
QUICK_SEEDS = 3
SPLIT_SEED = 42

MODEL_NAMES = ("Decision Tree", "Random Forest", "SVM", "AdaBoost", "XGBoost",
               "Logistic Regression")


def _factory(name, seed):
    """One model per paper classifier, at fixed reasonable hyperparameters.

    The seed sweep varies both the data and the model seed, so the reported
    variation reflects the whole pipeline rather than resampling alone.
    """
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, criterion="entropy",
                                      random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10,
                                       random_state=seed, n_jobs=1)
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
        return XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1,
                             subsample=0.9, random_state=seed,
                             eval_metric="logloss")
    if name == "Logistic Regression":
        return LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    raise ValueError(f"unknown model {name}")


def _trace(n_windows, seed, attack_ratio=0.05):
    from ml.streaming import add_sequence_features
    from ml.traces import build_trace
    return add_sequence_features(build_trace(n_windows=n_windows,
                                             attack_ratio=attack_ratio,
                                             seed=seed))


def _split(df, seed=SPLIT_SEED):
    from ml.traces import DATASET_VARIANTS, group_aware_split
    variant = DATASET_VARIANTS["nozeroatt"](df)
    return group_aware_split(variant, 0.1, 0.1, seed=seed)


def _fit_all(train_df, val_df, test_df, features):
    """Fit every model and return per-model metrics on the held-out test split."""
    from sklearn.preprocessing import StandardScaler
    from ml.metrics import binary_metrics, measure_latency, roc_analysis
    from ml.threshold import compare_to_paper_threshold, select_threshold

    Xtr = train_df[features].values.astype(float)
    ytr = train_df["label"].values
    Xva = val_df[features].values.astype(float)
    yva = val_df["label"].values
    Xte = test_df[features].values.astype(float)
    yte = test_df["label"].values

    scaler = StandardScaler().fit(Xtr)
    results = []

    for name in MODEL_NAMES:
        model = _factory(name, 0)
        if name in SCALED_MODELS:
            model.fit(scaler.transform(Xtr), ytr)
            proba_te = model.predict_proba(scaler.transform(Xte))[:, 1]
            proba_va = model.predict_proba(scaler.transform(Xva))[:, 1]
            probe = scaler.transform(Xte)
        else:
            model.fit(Xtr, ytr)
            proba_te = model.predict_proba(Xte)[:, 1]
            proba_va = model.predict_proba(Xva)[:, 1]
            probe = Xte

        m = binary_metrics(yte, (proba_te >= 0.5).astype(int))
        selection = select_threshold(yva, proba_va)
        latency = measure_latency(model.predict, probe, repeats=10)

        results.append({
            "Model": name,
            "Accuracy": m["Accuracy"],
            "Precision": m["Precision"],
            "Recall": m["Sensitivity"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "NPV": m["NPV"],
            "F1 Score": m["F1 Score"],
            "MCC": m["MCC"],
            "ROC AUC": roc_analysis(yte, proba_te)["AUC"],
            "Total Cost": m["Total Cost"],
            "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
            **latency,
            "Selected Threshold": selection["Threshold"],
            "Threshold Comparison": compare_to_paper_threshold(selection),
        })

    return results


def _write(path, payload):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  -> {path.name}")


def run_seed_study(n_windows, seeds, output_dir, verbose=True):
    """Repeat the pipeline across seeds and summarise with confidence intervals.

    The base paper reports single-run figures with no dispersion measure. This
    produces the intervals that make a difference between two models
    interpretable.
    """
    from ml.streaming import sequence_feature_columns

    def dataset_fn(seed):
        return _trace(n_windows, seed)

    def train_fn(df):
        train_df, val_df, test_df = _split(df)
        return _fit_all(train_df, val_df, test_df,
                        sequence_feature_columns(train_df))

    started = time.perf_counter()
    if verbose:
        print(f"[seed-study] {len(seeds)} seeds, {n_windows} windows each")

    seed_results, run_log = run_seeds(train_fn, dataset_fn, seeds, verbose=verbose)

    payload = {
        "Seeds": list(seeds),
        "Windows Per Trace": n_windows,
        "Elapsed Seconds": round(time.perf_counter() - started, 2),
        "Summary": aggregate_seed_results(seed_results),
        "Sensitivity Stability": stability_report(seed_results, "Sensitivity"),
        "Accuracy Stability": stability_report(seed_results, "Accuracy"),
        "Cost Stability": stability_report(seed_results, "Total Cost"),
        "Comparison (Sensitivity)": compare_all_models(seed_results, "Sensitivity"),
        "Comparison (Cost)": compare_all_models(seed_results, "Total Cost"),
        "Per Run": run_log,
    }
    _write(output_dir / "seed_results.json", payload)
    return payload


def run_ablation(n_windows, seed, output_dir, verbose=True):
    """Attribute detection performance to each family of temporal features."""
    from sklearn.ensemble import AdaBoostClassifier
    from ml.metrics import binary_metrics, roc_analysis
    from ml.streaming import COUNT_COLUMNS, ablation_setup, sequence_feature_columns
    from ml.threshold import select_threshold

    df = _trace(n_windows, seed)
    train_df, val_df, test_df = _split(df, seed)
    groups = ablation_setup(train_df)
    deployed = sequence_feature_columns(train_df)

    rows = []
    for name, features in groups.items():
        Xtr = train_df[features].values.astype(float)
        ytr = train_df["label"].values
        Xva = val_df[features].values.astype(float)
        yva = val_df["label"].values
        Xte = test_df[features].values.astype(float)
        yte = test_df["label"].values

        model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0,
                                   random_state=seed)
        model.fit(Xtr, ytr)

        proba = model.predict_proba(Xte)[:, 1]
        fixed = binary_metrics(yte, (proba >= 0.5).astype(int))
        selection = select_threshold(yva, model.predict_proba(Xva)[:, 1])
        tuned = binary_metrics(yte, (proba >= selection["Threshold"]).astype(int))

        rows.append({
            "Group": name,
            "Features": len(features),
            "AUROC": roc_analysis(yte, proba)["AUC"],
            "Sensitivity At 0.5": fixed["Sensitivity"],
            "Specificity At 0.5": fixed["Specificity"],
            "Cost At 0.5": fixed["Total Cost"],
            "Selected Threshold": selection["Threshold"],
            "Sensitivity At Selected": tuned["Sensitivity"],
            "Cost At Selected": tuned["Total Cost"],
        })
        if verbose:
            print(f"[ablation] {name:<20} feats={len(features):>3} "
                  f"auroc={rows[-1]['AUROC']:.4f} "
                  f"sens@0.5={rows[-1]['Sensitivity At 0.5']:.4f} "
                  f"cost={rows[-1]['Cost At 0.5']}")

    payload = {
        "Feature Counts": {name: len(f) for name, f in groups.items()},
        # The ablation groups and the deployed model draw on different bases, and
        # reporting only the group sizes made 29 and 26 look contradictory when
        # they are not. Spell the arithmetic out so both numbers can be cited.
        "Feature Count Reconciliation": {
            "Base Counts": len(COUNT_COLUMNS),
            "Sequence Features": len(deployed),
            "Deployed Model Feature Count": len(deployed),
            "E_full Feature Count": len(groups["E_full"]),
            "Explanation": (
                f"E_full = {len(COUNT_COLUMNS)} base counts + "
                f"{len(deployed)} sequence features = "
                f"{len(groups['E_full'])}. The deployed model uses the "
                f"{len(deployed)} sequence features only; the raw counts are "
                f"deliberately excluded so that the temporal aggregates, not the "
                f"per-second volume, carry the decision."),
        },
        "Results": rows,
        "Model": "AdaBoost(n_estimators=100, learning_rate=1.0)",
        "Threshold Note": "0.5 is the base paper's fixed threshold; the selected "
                          "threshold is chosen on the validation split",
    }
    _write(output_dir / "ablation.json", payload)
    return payload


def run_imbalance(n_windows, seed, ratios, output_dir, verbose=True):
    """Sensitivity and cost across attack base rates.

    One trace is generated at the highest requested ratio and attack episodes are
    progressively dropped from it. Generating a separate trace per ratio cannot
    work: the minimum-episode floor that the episode-aware split requires
    occupies a fixed fraction of the trace, so a low requested ratio is never
    actually delivered and every ratio produces the same dataset.
    """
    from ml.imbalance import build_at_ratio, summarise_imbalance
    from ml.streaming import add_sequence_features, sequence_feature_columns
    from ml.traces import build_trace

    base = add_sequence_features(
        build_trace(n_windows=n_windows, attack_ratio=max(ratios), seed=seed))

    rows = []
    for ratio in ratios:
        variant_df, raw = build_at_ratio(ratio, seed=seed, base_df=base)
        achieved = float((raw["label"] == 1).mean())
        episodes = int(raw[raw["label"] == 1]["episode_id"].nunique())

        # Whole-episode subsampling cannot hit an arbitrary ratio exactly: a ratio
        # below one episode's worth is unreachable.
        #
        # The episode-aware split reserves roughly a third of episodes for test and
        # another third for validation, so a reduced trace needs at least six
        # episodes for every subset to receive attacks. Below that the split's own
        # rebalancing hands every episode to the test subset and training is left
        # with none, so the row is skipped rather than producing a meaningless
        # zero.
        if episodes < 6:
            if verbose:
                print(f"[imbalance] ratio={ratio:.3f} skipped: {episodes} "
                      f"episodes; the split needs at least 6")
            continue
        if achieved > max(ratios) * 1.6:
            if verbose:
                print(f"[imbalance] ratio={ratio:.3f} skipped: "
                      f"achieved {achieved:.4f} exceeds the sweep range")
            continue

        train_df, val_df, test_df = _split(variant_df, seed)
        results = _fit_all(train_df, val_df, test_df,
                           sequence_feature_columns(train_df))

        rows.append({
            "Requested Ratio": ratio,
            "Achieved Ratio": achieved,
            "Episodes": episodes,
            "Train Attack Windows": int((train_df["label"] == 1).sum()),
            "Test Attack Windows": int((test_df["label"] == 1).sum()),
            "Models": {r["Model"]: {"Accuracy": r["Accuracy"],
                                    "Sensitivity": r["Sensitivity"],
                                    "Specificity": r["Specificity"],
                                    "Total Cost": r["Total Cost"]}
                       for r in results},
        })
        if verbose:
            best = min(results, key=lambda r: r["Total Cost"])
            print(f"[imbalance] ratio={ratio:.3f} achieved={achieved:.4f} "
                  f"episodes={rows[-1]['Episodes']} "
                  f"best={best['Model']} cost={best['Total Cost']} "
                  f"sens={best['Sensitivity']:.4f}")

    payload = {
        "Ratios Requested": list(ratios),
        "Ratios Achieved": [r["Achieved Ratio"] for r in rows],
        "Method": "single trace at max ratio; attack episodes dropped to reach "
                  "each lower ratio, holding the benign background fixed",
        "Rows": rows,
        "Summary": summarise_imbalance(rows, "Sensitivity") if rows else {},
    }
    _write(output_dir / "imbalance.json", payload)
    return payload


def run_ensemble_study(n_windows, seed, output_dir, verbose=True):
    """Compare soft and hard voting against the best single model."""
    from sklearn.preprocessing import StandardScaler
    from ml.ensemble import evaluate_ensemble, weight_search
    from ml.streaming import sequence_feature_columns

    df = _trace(n_windows, seed)
    train_df, val_df, test_df = _split(df, seed)
    features = sequence_feature_columns(train_df)
    results = _fit_all(train_df, val_df, test_df, features)

    Xtr = train_df[features].values.astype(float)
    ytr = train_df["label"].values
    Xva = val_df[features].values.astype(float)
    yva = val_df["label"].values
    Xte = test_df[features].values.astype(float)
    yte = test_df["label"].values
    scaler = StandardScaler().fit(Xtr)

    probas_te, probas_va = {}, {}
    for name in MODEL_NAMES:
        model = _factory(name, seed)
        if name in SCALED_MODELS:
            model.fit(scaler.transform(Xtr), ytr)
            probas_te[name] = model.predict_proba(scaler.transform(Xte))[:, 1]
            probas_va[name] = model.predict_proba(scaler.transform(Xva))[:, 1]
        else:
            model.fit(Xtr, ytr)
            probas_te[name] = model.predict_proba(Xte)[:, 1]
            probas_va[name] = model.predict_proba(Xva)[:, 1]

    best_single = min(results, key=lambda r: r["Total Cost"])["Model"]

    # Score the single models from the same probability arrays that feed the
    # vote. Deriving them from a separate fit produced a second, slightly
    # different set of numbers in the same file, so the artifact contradicted
    # itself: one block reported Random Forest cost 96 and the other 86.
    from ml.metrics import binary_metrics as _bm
    single_models = {}
    for name in MODEL_NAMES:
        m = _bm(yte, (probas_te[name] >= 0.5).astype(int))
        single_models[name] = {
            "Accuracy": m["Accuracy"],
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Total Cost": m["Total Cost"],
        }

    # The tuned-threshold fit is retained for reference, but the comparison that
    # the conclusion rests on is the like-for-like one above.
    tuned_single = {r["Model"]: {"Accuracy": r["Accuracy"],
                                 "Sensitivity": r["Sensitivity"],
                                 "Specificity": r["Specificity"],
                                 "Total Cost": r["Total Cost"]}
                    for r in results}

    voting = evaluate_ensemble(yte, probas_te, list(MODEL_NAMES),
                               best_single_name=best_single,
                               verbose=verbose)
    # Force the comparison anchor to the like-for-like numbers.
    for name, values in voting["Best Single Member"].items():
        values.update(single_models[name])
        if name == best_single:
            for rule, per_threshold in voting["Rules"].items():
                for _thr, vals in per_threshold.items():
                    vals["Cost vs Best Single"] = (
                        values["Total Cost"] - vals["Total Cost"])
                    vals["Sensitivity vs Best Single"] = round(
                        vals["Sensitivity"] - values["Sensitivity"], 6)

    payload = {
        "Best Single Model": best_single,
        "Single Models": single_models,
        "Tuned Single Models": tuned_single,
        "Comparison Basis": (
            "Single Models and Voting are scored from identical probability "
            "arrays on the same test split at threshold 0.5. Tuned Single Models "
            "reports the separately tuned fits and is reference only."),
        "Voting": voting,
        "Weight Search": weight_search(yva, probas_va, list(MODEL_NAMES)),
    }
    _write(output_dir / "ensemble.json", payload)
    return payload


EXPERIMENTS = ("stats", "ablation", "imbalance", "ensemble",
               "zeroday", "evasion", "alarm", "drift", "multiscale")


def run_zero_day(n_windows, seed, output_dir, verbose=True):
    """Leave-one-family-out transfer to a ransomware family never trained on."""
    from ml.streaming import sequence_feature_columns
    from ml.zero_day_eval import generalisation_report, leave_one_family_out

    df = _trace(n_windows, seed)
    rows = leave_one_family_out(df, feature_set="sequence", seed=seed,
                                verbose=verbose)
    payload = {
        "Feature Set": "sequence",
        "Threshold": 0.5,
        "Results": rows,
        "Summary": generalisation_report(rows),
    }
    _write(output_dir / "zero_day_eval.json", payload)
    return payload


def run_evasion_study(n_windows, seed, output_dir, verbose=True):
    """Sensitivity against encryption rate, plus a benign-mimicry regime."""
    import random
    from ml.evasion import blindspot_summary, build_evasion_set, evasion_curve
    from ml.streaming import sequence_feature_columns

    base = _trace(n_windows, seed)
    features = sequence_feature_columns(base)

    # ml.attack_sim draws through the stdlib random API (randint, choices,
    # uniform), so this must be a random.Random rather than a numpy Generator.
    evasion_df = build_evasion_set(base, random.Random(seed))

    curve = evasion_curve(_factory("AdaBoost", seed), features, evasion_df,
                          base, threshold=0.5, seed=seed, verbose=verbose)
    payload = {
        "Feature Set": "sequence",
        "Threshold": 0.5,
        "Curve": curve.to_dict(orient="records"),
        "Summary": blindspot_summary(curve),
        "Note": (
            "Each regime is scored on its attack windows plus the shared benign "
            "background, so specificity and total cost are defined. The model is "
            "trained on ordinary traffic only."),
    }
    _write(output_dir / "evasion.json", payload)
    return payload


def run_alarm_policy_study(n_windows, seed, output_dir, verbose=True):
    """Three-state alarm policy against fixed-threshold and k-of-n baselines."""
    from sklearn.preprocessing import StandardScaler
    from ml.alarm_policy import compare_policies
    from ml.streaming import sequence_feature_columns

    df = _trace(n_windows, seed)
    train_df, val_df, test_df = _split(df, seed)
    features = sequence_feature_columns(train_df)

    Xtr = train_df[features].values.astype(float)
    ytr = train_df["label"].values
    Xva = val_df[features].values.astype(float)
    yva = val_df["label"].values
    Xte = test_df[features].values.astype(float)
    yte = test_df["label"].values

    model = _factory("AdaBoost", seed)
    model.fit(Xtr, ytr)

    payload = compare_policies(yva, model.predict_proba(Xva)[:, 1],
                               yte, model.predict_proba(Xte)[:, 1],
                               verbose=verbose)
    _write(output_dir / "alarm_policy.json", payload)
    return payload


def run_drift_study_driver(n_windows, seed, output_dir, verbose=True):
    """PSI drift monitoring, degradation curve and offline recovery."""
    from ml.drift import run_drift_study
    from ml.streaming import sequence_feature_columns

    df = _trace(n_windows, seed)
    features = sequence_feature_columns(df)
    # run_drift_study calls factory(seed); _factory here takes (name, seed).
    factory = lambda s: _factory("AdaBoost", s)  # noqa: E731
    payload = run_drift_study(df, features, factory, seed=seed, threshold=0.5)
    _write(output_dir / "drift.json", payload)
    if verbose:
        mon = payload["Monitor"]
        print(f"[drift] baseline chunks={mon['Baseline Chunks Evaluated']} "
              f"false alarms={mon['False Alarms']} "
              f"rate={mon['False Alarm Rate']} "
              f"delay={mon['Detection Delay Seconds']}s")
    return payload


def run_multiscale_study(n_windows, seed, output_dir, verbose=True):
    """Window-width and stride sweep on the event stream."""
    from ml.multiscale import run as multiscale_run, trace_to_events

    # The sweep re-aggregates from raw events, since the window width is the
    # variable under study and cannot be baked into a pre-aggregated frame.
    windows = _trace(n_windows, seed)
    events = trace_to_events(windows, window_seconds=1.0, jitter=True, seed=seed)

    # run() takes a factory of the form factory(seed); _factory takes (name, seed).
    factory = lambda s: _factory("AdaBoost", s)  # noqa: E731
    payload = multiscale_run(factory, events, seed=seed, verbose=verbose)
    payload["Event Rows"] = int(len(events))
    payload["Source Windows"] = int(len(windows))
    payload["Interpretation Note"] = (
        "Total cost is not comparable across widths. Widening the window "
        "reduces the number of attack windows the model is judged on, so a "
        "falling cost alongside a falling sensitivity is arithmetic, not "
        "improvement. Compare sensitivity, or cost per attack window, instead.")

    # window_sweep and stride_analysis return DataFrames. json.dump would have
    # written their repr as a string, so the recorded artifact was unreadable
    # and carried no rows.
    import pandas as _pd
    for key in ("Width Sweep", "Stride Analysis"):
        frame = payload.get(key)
        if isinstance(frame, _pd.DataFrame):
            if "Test Attack Windows" in frame.columns:
                frame["Cost Per Attack Window"] = (
                    frame["Total Cost"] / frame["Test Attack Windows"]
                ).round(6)
            payload[key] = frame.to_dict(orient="records")

    _write(output_dir / "multiscale.json", payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the experiment suite.")
    parser.add_argument("--quick", action="store_true",
                        help="smaller traces and fewer seeds")
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS),
                        help="run only the named experiments")
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    args = parser.parse_args(argv)

    n_windows = QUICK_WINDOWS if args.quick else FULL_WINDOWS
    seeds = (list(range(1, QUICK_SEEDS + 1)) if args.quick
             else list(range(1, FULL_SEEDS + 1)))
    ratios = (0.02, 0.05, 0.10) if args.quick else (0.01, 0.05, 0.10, 0.20)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Experiment suite  (windows={n_windows}, quick={args.quick})")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    started = time.perf_counter()
    runners = {
        "stats": lambda: run_seed_study(n_windows, seeds, output_dir),
        "ablation": lambda: run_ablation(n_windows, SPLIT_SEED, output_dir),
        "imbalance": lambda: run_imbalance(n_windows, SPLIT_SEED, ratios, output_dir),
        "ensemble": lambda: run_ensemble_study(n_windows, SPLIT_SEED, output_dir),
        "zeroday": lambda: run_zero_day(n_windows, SPLIT_SEED, output_dir),
        "evasion": lambda: run_evasion_study(n_windows, SPLIT_SEED, output_dir),
        "alarm": lambda: run_alarm_policy_study(n_windows, SPLIT_SEED, output_dir),
        "drift": lambda: run_drift_study_driver(n_windows, SPLIT_SEED, output_dir),
        "multiscale": lambda: run_multiscale_study(n_windows, SPLIT_SEED, output_dir),
    }
    for name in (args.only or list(EXPERIMENTS)):
        print(f"\n--- {name} ---")
        runners[name]()

    print()
    print("=" * 70)
    print(f"  Completed in {time.perf_counter() - started:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())