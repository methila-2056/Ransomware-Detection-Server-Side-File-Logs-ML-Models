"""Phase 2 experiment driver.

Pipeline
--------
1. Build the emulated trace with ``assrd.data.build_trace`` (Phase 1).
2. Split it **by attack episode** with ``assrd.data.split_by_episode``.
3. Compute the 26 temporal features on the **whole trace in temporal order**
   before slicing, exactly as ``assrd.features`` documents.
4. Tune each of the six models on the **training** split only.
5. Choose the cost-minimising threshold on the **validation** split only.
6. Score the frozen model + threshold **once** on the test split.

Steps 4-6 are the leakage boundary. :func:`run_phase2` never passes the test
partition to :func:`assrd.models.tune_model` or to
:func:`assrd.models.select_threshold`; the tests assert this by checking that the
threshold chosen on validation is *not* the threshold that would have been chosen
on the test split.

Honesty note
------------
Attack traffic is emulated (see ``assrd.data``). Every figure this driver writes
describes the pipeline's behaviour on modelled behaviour and must not be read as
a field detection rate. The driver records that provenance in the output file
and makes no claim about reproducing any particular published result — it
reports only what it measured on the run that produced the file.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn
import xgboost

from assrd import compat
from assrd.data import build_trace, split_by_episode, trace_report
from assrd.features import MODEL_FEATURE_COLUMNS, RAW_COUNT_COLUMNS
from assrd.features import TEMPORAL_FEATURES, add_temporal_features, feature_report
from assrd.metrics import C_FN, C_FP, classification_metrics
from assrd.models import (
    MODEL_LABELS,
    MODEL_NAMES,
    cost_curve,
    evaluate_at_threshold,
    grid_size,
    probability,
    select_threshold,
    tune_model,
)

PROVENANCE = (
    "Attack windows are produced by a behavioural emulator, not by executed "
    "malware. Benign profiles are modelled on the base paper's Table 3; attack "
    "profiles on published family characteristics. Metrics describe pipeline "
    "behaviour on modelled behaviour and are not field detection rates."
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def prepare_dataset(n_windows=20000, attack_ratio=0.05, seed=42,
                    include_stealth=True, **trace_kwargs):
    """Trace -> whole-trace temporal features -> episode-aware split.

    Returns ``(splits, dataset_report)`` where ``splits`` is a dict of
    ``{"train": df, "validation": df, "test": df}``.
    """
    trace = build_trace(n_windows=n_windows, attack_ratio=attack_ratio,
                        seed=seed, include_stealth=include_stealth,
                        **trace_kwargs)
    enriched = add_temporal_features(trace)

    missing = [c for c in MODEL_FEATURE_COLUMNS if c not in enriched.columns]
    if missing:
        raise RuntimeError(f"feature columns missing after enrichment: {missing}")

    # Overwrite the index with the 1-D model input frame for every split.
    splits = split_by_episode(enriched, test_fraction=0.10,
                              validation_fraction=0.10, seed=seed)

    report = {
        "seed": seed,
        "trace": trace_report(trace),
        "features": feature_report(enriched),
        "partitions": {
            name: {
                "rows": int(len(part)),
                "attack_rows": int((part["label"] == 1).sum()),
                "attack_fraction": round(float((part["label"] == 1).mean()), 6),
                "attack_episodes": int(
                    part.loc[part["label"] == 1, "episode_id"].nunique()),
                "stealth_rows": int(part["is_stealth"].sum()),
            }
            for name, part in zip(("train", "validation", "test"), splits)
        },
    }
    report["leakage_check"] = leakage_report(*splits)
    return dict(zip(("train", "validation", "test"), splits)), report


def leakage_report(train, validation, test):
    """Verify the three partitions share no attack episode, and no row."""
    def episodes(part):
        return {int(e) for e in part.loc[part["label"] == 1, "episode_id"].tolist()}

    tr, va, te = episodes(train), episodes(validation), episodes(test)
    indices = {name: set(part["window_index"].tolist())
               for name, part in (("train", train), ("validation", validation),
                                  ("test", test))}
    return {
        "train_val_episode_overlap": sorted(tr & va),
        "train_test_episode_overlap": sorted(tr & te),
        "val_test_episode_overlap": sorted(va & te),
        "row_overlap": {
            "train_validation": len(indices["train"] & indices["validation"]),
            "train_test": len(indices["train"] & indices["test"]),
            "validation_test": len(indices["validation"] & indices["test"]),
        },
        "episode_aware": not (tr & va or tr & te or va & te),
        "disjoint_rows": not any(indices[a] & indices[b]
                                 for a, b in (("train", "validation"),
                                              ("train", "test"),
                                              ("validation", "test"))),
    }


def _xy(part, columns):
    X = part[list(columns)].to_numpy(dtype=float)
    y = part["label"].to_numpy(dtype=int)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("non-finite feature values in a partition")
    return X, y


def run_phase2(n_windows=20000, attack_ratio=0.05, seed=42, n_splits=5,
               scoring="recall", n_jobs=-1, c_fp=C_FP, c_fn=C_FN,
               feature_columns=MODEL_FEATURE_COLUMNS, models=MODEL_NAMES,
               include_stealth=True, threshold_grid=1001, cost_curve_points=101):
    """Run the full Phase 2 experiment and return the result dict.

    ``feature_columns`` defaults to the 29 model inputs (3 raw counts + 26
    temporal). The ablation against the 3 raw counts alone is Phase 3 work and is
    deliberately not run here.
    """
    feature_columns = tuple(feature_columns)
    splits, dataset = prepare_dataset(n_windows=n_windows,
                                      attack_ratio=attack_ratio, seed=seed,
                                      include_stealth=include_stealth)

    train, validation, test = splits["train"], splits["validation"], splits["test"]
    X_train, y_train = _xy(train, feature_columns)
    X_val, y_val = _xy(validation, feature_columns)
    X_test, y_test = _xy(test, feature_columns)

    if not dataset["leakage_check"]["episode_aware"]:
        raise RuntimeError("episode leakage detected; refusing to train")

    started = datetime.now(timezone.utc)
    per_model = {}
    for name in models:
        tuned = tune_model(name, X_train, y_train, seed=seed, n_splits=n_splits,
                           scoring=scoring, n_jobs=n_jobs, c_fp=c_fp, c_fn=c_fn)
        estimator = tuned["estimator"]

        proba_val = probability(estimator, X_val)
        proba_test = probability(estimator, X_test)

        # Threshold: validation only.
        chosen = select_threshold(y_val, proba_val, c_fp=c_fp, c_fn=c_fn,
                                  n_grid=threshold_grid)

        val_metrics = evaluate_at_threshold(y_val, proba_val,
                                            chosen["threshold"],
                                            c_fp=c_fp, c_fn=c_fn)
        test_metrics = evaluate_at_threshold(y_test, proba_test,
                                             chosen["threshold"],
                                             c_fp=c_fp, c_fn=c_fn)
        test_metrics_at_half = evaluate_at_threshold(y_test, proba_test, 0.5,
                                                     c_fp=c_fp, c_fn=c_fn)
        val_at_half = evaluate_at_threshold(y_val, proba_val, 0.5,
                                            c_fp=c_fp, c_fn=c_fn)

        # Diagnostics: what a test-optimal threshold *would* have been. Never
        # used for the reported operating point; recorded to expose the gap.
        oracle = select_threshold(y_test, proba_test, c_fp=c_fp, c_fn=c_fn,
                                  n_grid=threshold_grid)

        per_model[name] = {
            "label": MODEL_LABELS.get(name, name),
            "grid_points": tuned["grid_points"],
            "cv_folds": tuned["cv_folds"],
            "scoring": tuned["scoring"],
            "best_params": tuned["best_params"],
            "best_cv_score": tuned["best_cv_score"],
            "cv_dispersion": tuned["cv_results_mean"],
            "sample_weight": tuned["sample_weight"],
            "threshold_selection": chosen,
            "validation_at_selected_threshold": val_metrics,
            "validation_at_threshold_0_5": val_at_half,
            "test_at_selected_threshold": test_metrics,
            "test_at_threshold_0_5": test_metrics_at_half,
            "test_threshold_oracle_diagnostic": {
                "threshold": oracle["threshold"],
                "test_cost": oracle["validation_cost"],
                "note": ("diagnostic only; the reported operating point is the "
                         "validation-selected threshold and this value is never "
                         "used to score the model"),
            },
            "validation_cost_curve": cost_curve(y_val, proba_val, c_fp=c_fp,
                                               c_fn=c_fn, n_grid=cost_curve_points),
        }

    finished = datetime.now(timezone.utc)
    return {
        "phase": 2,
        "provenance": PROVENANCE,
        "generated_utc": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "config": {
            "n_windows": int(n_windows),
            "attack_ratio": float(attack_ratio),
            "seed": int(seed),
            "cv_folds": int(n_splits),
            "tuning_scoring": scoring,
            "n_jobs": n_jobs,
            "cost_fp": float(c_fp),
            "cost_fn": float(c_fn),
            "cost_ratio": (float(c_fn) / float(c_fp)) if c_fp else None,
            "include_stealth": bool(include_stealth),
            "threshold_grid_points": int(threshold_grid),
            "feature_columns": list(feature_columns),
            "feature_count": len(feature_columns),
            "models": list(models),
            "joblib_backend": "threading (assrd.compat.threading_backend)",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "compat": compat.diagnose(),
        },
        "dataset": dataset,
        "models": per_model,
    }


# ---------------------------------------------------------------------------
# Convenience: a flat table of the headline test figures.
# ---------------------------------------------------------------------------
def results_table(result):
    """Return a DataFrame of per-model headline metrics (test split)."""
    rows = []
    for name, block in result["models"].items():
        tuned = block["test_at_selected_threshold"]
        at_half = block["test_at_threshold_0_5"]
        rows.append({
            "model": block["label"],
            "threshold": block["threshold_selection"]["threshold"],
            "sensitivity": tuned["sensitivity"],
            "specificity": tuned["specificity"],
            "precision": tuned["precision"],
            "f1": tuned["f1"],
            "roc_auc": tuned["roc_auc"],
            "average_precision": tuned["average_precision"],
            "total_cost": tuned["total_cost"],
            "false_positives": tuned["false_positives"],
            "false_negatives": tuned["false_negatives"],
            "cost_fp": tuned["cost_fp"],
            "cost_fn": tuned["cost_fn"],
            "total_cost_at_0_5": at_half["total_cost"],
            "best_cv_score": block["best_cv_score"],
        })
    return pd.DataFrame(rows)


def confusion_table(result):
    """Long-form confusion matrix for every model, at the selected threshold."""
    rows = []
    for name, block in result["models"].items():
        for split in ("validation", "test"):
            m = block[f"{split}_at_selected_threshold"]
            for actual in (0, 1):
                for predicted in (0, 1):
                    rows.append({
                        "model": block["label"],
                        "split": split,
                        "actual": "benign" if actual == 0 else "attack",
                        "predicted": "benign" if predicted == 0 else "attack",
                        "count": m["confusion_matrix"]["matrix"][actual][predicted],
                    })
    return pd.DataFrame(rows)


def write_results(result, directory=RESULTS_DIR, basename="phase2_results"):
    """Write JSON + two CSVs. Returns the paths written."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / f"{basename}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")

    table_path = directory / f"{basename}_models.csv"
    results_table(result).to_csv(table_path, index=False)

    cm_path = directory / f"{basename}_confusion.csv"
    confusion_table(result).to_csv(cm_path, index=False)

    return {"json": str(json_path), "models_csv": str(table_path),
            "confusion_csv": str(cm_path)}


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="ASSRD Phase 2 experiment")
    parser.add_argument("--n-windows", type=int, default=20000)
    parser.add_argument("--attack-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--scoring", default="recall")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cost-fp", type=float, default=C_FP)
    parser.add_argument("--cost-fn", type=float, default=C_FN)
    parser.add_argument("--no-stealth", action="store_true")
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--basename", default="phase2_results")
    args = parser.parse_args(argv)

    result = run_phase2(
        n_windows=args.n_windows,
        attack_ratio=args.attack_ratio,
        seed=args.seed,
        n_splits=args.folds,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        c_fp=args.cost_fp,
        c_fn=args.cost_fn,
        models=tuple(args.models),
        include_stealth=not args.no_stealth,
    )
    paths = write_results(result, directory=args.output_dir,
                          basename=args.basename)

    table = results_table(result)
    print(table.to_string(index=False))
    print()
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return result


if __name__ == "__main__":
    main()