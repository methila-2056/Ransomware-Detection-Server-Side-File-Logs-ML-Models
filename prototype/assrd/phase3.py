"""Phase 3 experiment driver — feature ablation and robustness evaluation.

Pipeline
--------
1. Build the emulated trace (Phase 1) and the 26 temporal features.
2. Split it with the **constrained stratified allocator**
   (:mod:`assrd.splits`), not the Phase 2 random split. Attack episodes are
   placed so that every family of adequate size reaches validation and test and
   each evaluation split receives a minimum number of stealth episodes.
3. Run the pre-declared **ablation matrix**: six feature arms x five models x two
   training weightings. Each (arm, model, weighting) is fitted **once** at the
   hyperparameters Phase 2 already selected (:func:`assrd.ablation.fit_fixed`),
   then scored under two threshold rules:

       M1  cost-weighted     + validation-selected threshold   (primary)
       M2  cost-weighted     + fixed 0.5
       M3  unweighted        + validation-selected threshold
       M4  unweighted        + fixed 0.5                        (plain baseline)

4. Report robustness: per-family metrics, stealth vs non-stealth metrics, and a
   bootstrap of the threshold **resampled by independent episode**, so a wide
   interval can expose a threshold plateau.

Diagnostics isolation
---------------------
The test-optimal ("oracle") threshold is computed for every fitted model, but it
lives **only** under ``result["diagnostics"]["test_oracle"]``. It is never an
input to model selection, feature-set choice, hyperparameter choice or threshold
choice, and it never appears in the primary ablation table or CSV. A test-label
dependent upper bound is not deployment performance; :func:`assert_oracle_isolated`
enforces the separation structurally.

Honesty note
------------
Attack traffic is emulated (see :mod:`assrd.data`). No field detection rate is
claimed and no published result is reproduced; every figure describes the
pipeline's behaviour on modelled behaviour. Undefined rates are ``None``/blank,
never 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import sklearn
import xgboost

from assrd import compat
from assrd.ablation import (
    ABLATION_ARMS,
    NESTED_ARMS,
    arm_columns,
    fit_fixed,
    load_phase2_params,
    robustness_rows,
    threshold_stability,
    validate_arms,
)
from assrd.data import FAMILY_NAMES, build_trace, trace_report
from assrd.features import add_temporal_features, feature_report
from assrd.metrics import C_FN, C_FP
from assrd.models import (
    MODEL_LABELS,
    MODEL_NAMES,
    cost_curve,
    evaluate_at_threshold,
    probability,
    select_threshold,
    tune_model,
)
from assrd.splits import (
    SplitConstraintError,
    bootstrap_groups,
    manifest_table,
    split_manifest,
    stratified_split,
)

PROVENANCE = (
    "Attack windows are produced by a behavioural emulator, not by executed "
    "malware. Benign profiles are modelled on the base paper's Table 3; attack "
    "profiles on published family characteristics. Metrics describe pipeline "
    "behaviour on modelled behaviour and are not field detection rates."
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

PHASE2_ARTIFACTS = (
    "phase2_results.json",
    "phase2_results_models.csv",
    "phase2_results_confusion.csv",
)

#: Modes of the pre-declared matrix. ``cost_weighted`` is the training half;
#: ``threshold_rule`` is the operating-point half.
MODE_SPECS = {
    "M1": {"cost_weighted": True, "threshold_rule": "validation",
           "label": "cost-weighted + validation-selected threshold (primary)"},
    "M2": {"cost_weighted": True, "threshold_rule": "0.5",
           "label": "cost-weighted + fixed 0.5 threshold"},
    "M3": {"cost_weighted": False, "threshold_rule": "validation",
           "label": "unweighted + validation-selected threshold"},
    "M4": {"cost_weighted": False, "threshold_rule": "0.5",
           "label": "unweighted + fixed 0.5 threshold (plain baseline)"},
}
PRIMARY_MODE = "M1"

#: Comparisons fixed *before* any result is seen. C1 and C5 are the two claims
#: the abstract makes; C2-C4 decompose how cost sensitivity acts.
PRIMARY_COMPARISONS = (
    {"id": "C1", "metric": "total_cost",
     "question": "Do the 26 temporal features add over the 3 raw counts?",
     "lhs": {"arm": "E_full", "mode": "M1"},
     "rhs": {"arm": "A_raw_only", "mode": "M1"}},
    {"id": "C2", "metric": "total_cost",
     "question": "Does the cost-optimal threshold beat a fixed 0.5?",
     "lhs": {"mode": "M1"}, "rhs": {"mode": "M2"}, "within": "arm"},
    {"id": "C3", "metric": "total_cost",
     "question": "Do cost weights matter beyond the threshold?",
     "lhs": {"mode": "M1"}, "rhs": {"mode": "M3"}, "within": "arm"},
    {"id": "C4", "metric": "total_cost",
     "question": "Does cost modelling help end to end?",
     "lhs": {"mode": "M1"}, "rhs": {"mode": "M4"}, "within": "arm"},
    {"id": "C5", "metric": "total_cost",
     "question": "Do the raw counts add once the aggregates are present?",
     "lhs": {"arm": "E_full", "mode": "M1"},
     "rhs": {"arm": "T_temporal_only", "mode": "M1"}},
)

ORACLE_NOTE = (
    "diagnostic only. The test-optimal threshold is a test-label-dependent upper "
    "bound used to expose the generalisation gap. It is never an input to model, "
    "feature-set, hyperparameter or threshold selection and is NOT achievable "
    "deployment performance."
)

LIMITATIONS = [
    "Attack traffic is emulated, not executed malware; results are emulated-family "
    "behaviour and must not be read as field detection rates.",
    "Hyperparameters are carried over from Phase 2, which selected them on a "
    "different trace. That transfer assumption is documented; --retune-full probes it.",
    "Some (family, stealth) cells hold too few independent episodes to support a "
    "comparison. They are reported with a support flag and no claim is drawn from "
    "an 'insufficient' or 'absent' cell.",
    "Stealth robustness is measured against the emulator's rate-limited generator. "
    "It bounds behaviour on modelled evasion, not real evasion.",
    "Precision is prevalence-bound (~5% attack). High sensitivity is paid for in "
    "false alarms; that reflects class balance, not a defect.",
]

REQUIRED_MANIFEST_SPLIT_KEYS = (
    "rows", "benign_rows", "attack_rows", "attack_fraction", "attack_episodes",
    "benign_episodes", "benign_managed_blocks", "attack_episodes_by_family",
    "attack_episodes_stealth", "attack_episodes_non_stealth",
    "attack_rows_by_family", "attack_rows_stealth", "attack_rows_non_stealth",
    "cells_family_stealth", "support_flags",
)

QUICK = {"n_windows": 20000, "min_episode": 8, "max_episode": 16,
         "stability_bootstrap": 30}
FULL = {"n_windows": 60000, "min_episode": 12, "max_episode": 24,
        "stability_bootstrap": 100}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(*, n_windows, min_episode, max_episode, attack_ratio, seed,
                    include_stealth=True, stealth_share=0.25,
                    test_fraction=0.20, validation_fraction=0.20,
                    min_stealth_episodes=6, min_stealth_train=2):
    """Trace -> temporal features -> stratified split -> explicit manifest."""
    trace = build_trace(n_windows=n_windows, attack_ratio=attack_ratio, seed=seed,
                        include_stealth=include_stealth, stealth_share=stealth_share,
                        min_episode=min_episode, max_episode=max_episode)
    enriched = add_temporal_features(trace)

    train, validation, test, alloc = stratified_split(
        enriched, test_fraction=test_fraction,
        validation_fraction=validation_fraction, seed=seed,
        min_stealth_episodes=min_stealth_episodes,
        min_stealth_train=min_stealth_train)

    manifest = split_manifest(train, validation, test,
                              block_size=alloc["benign_block_size"], alloc=alloc)
    report = {
        "seed": seed,
        "trace": trace_report(trace),
        "features": feature_report(enriched),
        "allocation": alloc,
        "split_manifest": manifest,
    }
    splits = {"train": train, "validation": validation, "test": test}
    return splits, report


def _xy(part, columns):
    X = part[list(columns)].to_numpy(dtype=float)
    y = part["label"].to_numpy(dtype=int)
    if not np.all(np.isfinite(X)):
        raise RuntimeError("non-finite feature values in a partition")
    return X, y


# ---------------------------------------------------------------------------
# Diagnostics isolation
# ---------------------------------------------------------------------------
def _iter_keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _iter_keys(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_keys(item)


def assert_oracle_isolated(result):
    """Fail if any oracle-sourced value leaked into the primary block."""
    primary = result.get("primary", {})
    leaked = sorted({k for k in _iter_keys(primary) if "oracle" in str(k).lower()})
    if leaked:
        raise AssertionError(
            f"test-oracle data leaked into primary results: {leaked}")
    if "test_oracle" not in result.get("diagnostics", {}):
        raise AssertionError("diagnostics block is missing its test_oracle entry")
    return True


def _oracle_isolation_static():
    """Exercise the checker itself: accept a clean result, reject a poisoned one."""
    good = {"primary": {"a|b|M1": {"total_cost": 1.0}},
            "diagnostics": {"test_oracle": {}}}
    bad = {"primary": {"test_oracle": {"threshold": 0.5}},
           "diagnostics": {"test_oracle": {}}}
    assert_oracle_isolated(good)
    try:
        assert_oracle_isolated(bad)
    except AssertionError:
        return True, "checker rejects oracle keys under primary"
    return False, "checker failed to reject a poisoned primary block"


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
def run_phase3(*, n_windows=60000, min_episode=12, max_episode=24,
               attack_ratio=0.05, seed=42, include_stealth=True,
               stealth_share=0.25, test_fraction=0.20, validation_fraction=0.20,
               min_stealth_episodes=6, min_stealth_train=2,
               arms=tuple(ABLATION_ARMS), models=MODEL_NAMES,
               modes=tuple(MODE_SPECS), params=None, c_fp=C_FP, c_fn=C_FN,
               threshold_grid=1001, stability_bootstrap=100, stability_grid=201,
               cost_curve_points=101, retune_full=False,
               family_names=FAMILY_NAMES):
    """Run the Phase 3 ablation + robustness experiment; return the result dict."""
    validate_arms()
    arms = tuple(arms)
    models = tuple(models)
    modes = tuple(modes)
    for mode in modes:
        if mode not in MODE_SPECS:
            raise KeyError(f"unknown mode {mode!r}; expected {tuple(MODE_SPECS)}")

    splits, dataset = prepare_dataset(
        n_windows=n_windows, min_episode=min_episode, max_episode=max_episode,
        attack_ratio=attack_ratio, seed=seed, include_stealth=include_stealth,
        stealth_share=stealth_share, test_fraction=test_fraction,
        validation_fraction=validation_fraction,
        min_stealth_episodes=min_stealth_episodes,
        min_stealth_train=min_stealth_train)

    leakage = dataset["split_manifest"]["leakage"]
    if not (leakage["episode_aware"] and leakage["disjoint_rows"]):
        raise RuntimeError("episode leakage detected; refusing to train")

    if params is None:
        params, params_source = load_phase2_params()
    else:
        params_source = {"path": "caller-supplied", "found": True,
                         "models_from_file": sorted(params)}

    train, validation, test = splits["train"], splits["validation"], splits["test"]
    y_val = validation["label"].to_numpy(dtype=int)
    y_test = test["label"].to_numpy(dtype=int)
    meta = test[["episode_id", "family", "is_stealth"]].reset_index(drop=True)
    block_size = dataset["allocation"]["benign_block_size"]
    val_groups = bootstrap_groups(validation, block_size)

    started = datetime.now(timezone.utc)
    primary = {}
    diagnostics = {"test_oracle": {}, "note": ORACLE_NOTE,
                   "achievable_deployment_performance": False}
    stability_records = []
    curve_records = []
    robustness_flat = []

    def process(arm, model, weighted, estimator, columns):
        """Score one fitted estimator under every applicable mode."""
        columns = list(columns)
        X_val = validation[columns].to_numpy(dtype=float)
        X_test = test[columns].to_numpy(dtype=float)
        proba_val = probability(estimator, X_val)
        proba_test = probability(estimator, X_test)

        selected = select_threshold(y_val, proba_val, c_fp=c_fp, c_fn=c_fn,
                                    n_grid=threshold_grid)
        oracle = select_threshold(y_test, proba_test, c_fp=c_fp, c_fn=c_fn,
                                  n_grid=threshold_grid)
        weight_key = "weighted" if weighted else "unweighted"
        diagnostics["test_oracle"][f"{arm}|{model}|{weight_key}"] = {
            "arm": arm, "model": model, "weighting": weight_key,
            "threshold": float(oracle["threshold"]),
            "test_oracle_cost": float(oracle["validation_cost"]),
            "test_oracle_cost_at_selected": float(evaluate_at_threshold(
                y_test, proba_test, selected["threshold"], c_fp=c_fp,
                c_fn=c_fn)["total_cost"]),
            "note": ORACLE_NOTE,
        }

        for mode in modes:
            spec = MODE_SPECS[mode]
            if spec["cost_weighted"] != weighted:
                continue
            threshold = (selected["threshold"]
                         if spec["threshold_rule"] == "validation" else 0.5)
            test_metrics = evaluate_at_threshold(y_test, proba_test, threshold,
                                                 c_fp=c_fp, c_fn=c_fn)
            val_metrics = evaluate_at_threshold(y_val, proba_val, threshold,
                                                c_fp=c_fp, c_fn=c_fn)
            record = {
                "arm": arm, "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "mode": mode, "mode_label": spec["label"],
                "cost_weighted": bool(weighted),
                "threshold_rule": spec["threshold_rule"],
                "threshold": float(threshold),
                "n_features": int(len(columns)),
                "validation_selected_threshold": float(selected["threshold"]),
                "validation_cost_at_selected": float(selected["validation_cost"]),
                "validation_cost_at_0_5": float(selected["validation_cost_at_0_5"]),
                "n_tied_thresholds": int(selected["n_tied"]),
                "validation": val_metrics,
                "test": test_metrics,
            }
            y_pred_test = (proba_test >= threshold).astype(int)
            rows = robustness_rows(meta, y_test, y_pred_test, proba_test,
                                   family_names=family_names, c_fp=c_fp, c_fn=c_fn)
            record["per_family"] = [r for r in rows if r["scope"] == "family"]
            record["stealth_breakdown"] = [r for r in rows if r["scope"] == "stealth"]
            primary[f"{arm}|{model}|{mode}"] = record
            for row in rows:
                robustness_flat.append({"arm": arm, "model": model, "mode": mode,
                                        **{k: row[k] for k in (
                                            "scope", "group", "support_episodes",
                                            "support_flag", "support_attack_windows",
                                            "sensitivity", "specificity",
                                            "precision", "f1", "roc_auc",
                                            "average_precision", "total_cost",
                                            "false_positives", "false_negatives")}})

        stability = threshold_stability(y_val, proba_val, val_groups,
                                        n_boot=stability_bootstrap, seed=seed,
                                        c_fp=c_fp, c_fn=c_fn, n_grid=stability_grid)
        stability_records.append({"arm": arm, "model": model,
                                  "weighting": weight_key, **stability})
        for point in cost_curve(y_val, proba_val, c_fp=c_fp, c_fn=c_fn,
                                n_grid=cost_curve_points):
            curve_records.append({"arm": arm, "model": model,
                                  "weighting": weight_key, **point})

    for arm in arms:
        columns = list(arm_columns(arm))
        X_train, y_train = _xy(train, columns)
        for model in models:
            for weighted in (True, False):
                fitted = fit_fixed(model, X_train, y_train, params.get(model),
                                   seed=seed, cost_weighted=weighted,
                                   c_fp=c_fp, c_fn=c_fn)
                process(arm, model, weighted, fitted["estimator"], columns)

    if retune_full:
        columns = list(arm_columns("E_full"))
        X_train, y_train = _xy(train, columns)
        for model in models:
            for weighted in (True, False):
                tuned = tune_model(model, X_train, y_train, seed=seed,
                                   n_splits=5, scoring="recall",
                                   c_fp=c_fp, c_fn=c_fn, cost_weighted=weighted)
                process("E_full_retuned", model, weighted, tuned["estimator"],
                        columns)

    finished = datetime.now(timezone.utc)
    result = {
        "phase": 3,
        "provenance": PROVENANCE,
        "generated_utc": finished.isoformat(),
        "duration_seconds": round((finished - started).total_seconds(), 3),
        "config": {
            "n_windows": int(n_windows),
            "min_episode": int(min_episode),
            "max_episode": int(max_episode),
            "attack_ratio": float(attack_ratio),
            "seed": int(seed),
            "include_stealth": bool(include_stealth),
            "stealth_share": float(stealth_share),
            "test_fraction": float(test_fraction),
            "validation_fraction": float(validation_fraction),
            "min_stealth_episodes": int(min_stealth_episodes),
            "min_stealth_train": int(min_stealth_train),
            "arms": list(arms),
            "models": list(models),
            "modes": list(modes),
            "cost_fp": float(c_fp),
            "cost_fn": float(c_fn),
            "cost_ratio": (float(c_fn) / float(c_fp)) if c_fp else None,
            "threshold_grid_points": int(threshold_grid),
            "stability_bootstrap": int(stability_bootstrap),
            "stability_grid_points": int(stability_grid),
            "retune_full": bool(retune_full),
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
        "params_source": params_source,
        "matrix": {
            "arms": {a: len(arm_columns(a)) for a in arms},
            "nested_arms": list(NESTED_ARMS),
            "models": list(models),
            "modes": {m: MODE_SPECS[m] for m in modes},
            "n_primary_cells": len(primary),
        },
        "comparisons_declared": list(PRIMARY_COMPARISONS),
        "comparisons": compute_comparisons(primary),
        "primary": primary,
        "diagnostics": diagnostics,
        "robustness": {"threshold_stability": stability_records},
        "limitations": list(LIMITATIONS),
    }
    result["oracle_isolated"] = assert_oracle_isolated(result)
    return result


def compute_comparisons(primary):
    """Realised deltas for the comparisons that were declared in advance."""
    out = []
    for spec in PRIMARY_COMPARISONS:
        rows = []
        if "within" in spec:
            for arm in NESTED_ARMS + ("T_temporal_only", "E_full_retuned"):
                for model in MODEL_NAMES:
                    lhs = primary.get(f"{arm}|{model}|{spec['lhs']['mode']}")
                    rhs = primary.get(f"{arm}|{model}|{spec['rhs']['mode']}")
                    rows.append(_comparison_row(spec, model, arm, lhs, rhs))
        else:
            for model in MODEL_NAMES:
                lhs = primary.get(
                    f"{spec['lhs']['arm']}|{model}|{spec['lhs']['mode']}")
                rhs = primary.get(
                    f"{spec['rhs']['arm']}|{model}|{spec['rhs']['mode']}")
                rows.append(_comparison_row(spec, model, spec["lhs"]["arm"],
                                            lhs, rhs))
        out.append({"id": spec["id"], "question": spec["question"],
                    "metric": spec["metric"], "rows": [r for r in rows if r]})
    return out


def _comparison_row(spec, model, arm, lhs, rhs):
    if lhs is None or rhs is None:
        return None
    metric = spec["metric"]
    lhs_value = lhs["test"].get(metric)
    rhs_value = rhs["test"].get(metric)
    delta = (None if lhs_value is None or rhs_value is None
             else float(lhs_value) - float(rhs_value))
    return {"model": model, "arm": arm,
            "lhs": f"{spec['lhs'].get('arm', arm)}|{spec['lhs']['mode']}",
            "rhs": f"{spec['rhs'].get('arm', arm)}|{spec['rhs']['mode']}",
            "lhs_value": lhs_value, "rhs_value": rhs_value, "delta": delta,
            # Negative delta means lhs costs less, i.e. lhs is better on this metric.
            "lhs_better": (None if delta is None else bool(delta < 0))}


# ---------------------------------------------------------------------------
# Preflight self-check
# ---------------------------------------------------------------------------
def _artifact_hashes(directory=RESULTS_DIR):
    hashes = {}
    for name in PHASE2_ARTIFACTS:
        path = Path(directory) / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            hashes[name] = None
    return hashes


def validate_manifest_schema(manifest):
    """Structural check that the manifest carries every promised field."""
    missing = []
    for key in ("family_names", "splits", "allocation", "shortfalls", "leakage"):
        if key not in manifest:
            missing.append(key)
    for split, block in manifest.get("splits", {}).items():
        for key in REQUIRED_MANIFEST_SPLIT_KEYS:
            if key not in block:
                missing.append(f"{split}.{key}")
        flags = block.get("support_flags", {})
        if "families" not in flags or "family_stealth_cells" not in flags:
            missing.append(f"{split}.support_flags")
    return (not missing), {"missing": missing}


def _composition_ok(manifest, *, min_stealth_episodes):
    problems = []
    splits = manifest["splits"]
    for name, block in splits.items():
        if block["attack_rows"] <= 0:
            problems.append(f"{name}: no attack windows")
        if block["benign_rows"] <= 0:
            problems.append(f"{name}: no benign windows")
    val, test = splits["validation"], splits["test"]
    if val["attack_episodes_stealth"] < min_stealth_episodes:
        problems.append(f"validation stealth {val['attack_episodes_stealth']} "
                        f"< {min_stealth_episodes}")
    guaranteed = manifest["allocation"].get("targets", {}).get(
        "guaranteed_families", [])
    for fam in guaranteed:
        if val["attack_episodes_by_family"].get(fam, 0) == 0:
            problems.append(f"validation missing family {fam}")
        if test["attack_episodes_by_family"].get(fam, 0) == 0:
            problems.append(f"test missing family {fam}")
    return (not problems), {"problems": problems, "guaranteed_families": guaranteed}


def self_check(*, n_windows, min_episode, max_episode, seed=42,
               min_stealth_episodes=6, min_stealth_train=2,
               time_budget_s=600.0, memory_budget_mb=4096.0):
    """Preflight only. Builds and validates the dataset; **never fits a model**."""
    checks = []
    before = _artifact_hashes()
    checks.append(("G0_phase2_artifacts_present",
                   all(v is not None for v in before.values()),
                   {"artifacts": before}))

    started = time.perf_counter()
    tracemalloc.start()
    try:
        splits, dataset = prepare_dataset(
            n_windows=n_windows, min_episode=min_episode,
            max_episode=max_episode, attack_ratio=0.05, seed=seed,
            min_stealth_episodes=min_stealth_episodes,
            min_stealth_train=min_stealth_train)
        manifest = dataset["split_manifest"]

        leakage = manifest["leakage"]
        checks.append(("G1_leakage",
                       bool(leakage["episode_aware"] and leakage["disjoint_rows"]),
                       leakage))

        comp_ok, comp_detail = _composition_ok(
            manifest, min_stealth_episodes=min_stealth_episodes)
        checks.append(("G2_composition", comp_ok, comp_detail))

        splits2, dataset2 = prepare_dataset(
            n_windows=n_windows, min_episode=min_episode,
            max_episode=max_episode, attack_ratio=0.05, seed=seed,
            min_stealth_episodes=min_stealth_episodes,
            min_stealth_train=min_stealth_train)
        reproducible = (dataset["split_manifest"] ==
                        dataset2["split_manifest"])
        checks.append(("G3_reproducibility", bool(reproducible),
                       {"identical_manifest_on_rerun": bool(reproducible)}))

        schema_ok, schema_detail = validate_manifest_schema(manifest)
        checks.append(("G4_schema", schema_ok, schema_detail))

        oracle_ok, oracle_detail = _oracle_isolation_static()
        checks.append(("G5_oracle_isolation", oracle_ok,
                       {"detail": oracle_detail}))
    except SplitConstraintError as exc:
        checks.append(("allocator_constraints", False,
                       {"error": str(exc), "shortfall": exc.shortfall}))
    finally:
        elapsed = time.perf_counter() - started
        peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)
    checks.append(("G6_feasibility",
                   elapsed <= time_budget_s and peak_mb <= memory_budget_mb,
                   {"elapsed_seconds": round(elapsed, 3),
                    "peak_memory_mb": round(peak_mb, 2),
                    "time_budget_s": time_budget_s,
                    "memory_budget_mb": memory_budget_mb}))

    after = _artifact_hashes()
    checks.append(("G7_phase2_preserved", before == after,
                   {"before": before, "after": after}))

    passed = all(ok for _, ok, _ in checks)
    return {"checks": [{"name": n, "passed": bool(ok), "detail": d}
                       for n, ok, d in checks],
            "passed": bool(passed),
            "fits_run": 0}


def assert_not_phase2_basename(basename):
    if str(basename).lower().startswith("phase2"):
        raise ValueError(
            f"refusing to write to basename {basename!r}; it would collide with a "
            f"Phase 2 artefact")
    return str(basename)


# ---------------------------------------------------------------------------
# Result tables and writers
# ---------------------------------------------------------------------------
def ablation_table(result):
    """One row per primary cell (arm x model x mode), primary metrics only."""
    rows = []
    for key, block in result["primary"].items():
        test = block["test"]
        rows.append({
            "arm": block["arm"], "model": block["model"], "mode": block["mode"],
            "cost_weighted": block["cost_weighted"],
            "threshold_rule": block["threshold_rule"],
            "n_features": block["n_features"],
            "threshold": block["threshold"],
            "sensitivity": test["sensitivity"],
            "specificity": test["specificity"],
            "precision": test["precision"],
            "f1": test["f1"],
            "roc_auc": test["roc_auc"],
            "average_precision": test["average_precision"],
            "false_positives": test["false_positives"],
            "false_negatives": test["false_negatives"],
            "cost_fp": test["cost_fp"],
            "cost_fn": test["cost_fn"],
            "total_cost": test["total_cost"],
            "support_attack": test["support_attack"],
            "support_benign": test["support_benign"],
        })
    return pd.DataFrame(sorted(rows, key=lambda r: (r["arm"], r["model"],
                                                    r["mode"])))


def diagnostics_table(result):
    """Separate table for the test-oracle upper bound. Never merged with primary."""
    rows = []
    for key, block in result["diagnostics"]["test_oracle"].items():
        rows.append({
            "arm": block["arm"], "model": block["model"],
            "weighting": block["weighting"],
            "oracle_threshold": block["threshold"],
            "oracle_test_cost": block["test_oracle_cost"],
            "test_cost_at_selected_threshold": block["test_oracle_cost_at_selected"],
            "is_diagnostic": True,
            "achievable_deployment_performance": False,
        })
    return pd.DataFrame(rows)


def per_family_table(result):
    """Per-family and stealth/non-stealth metrics for every primary cell."""
    rows = []
    for block in result["primary"].values():
        for scope, entries in (("family", block["per_family"]),
                               ("stealth", block["stealth_breakdown"])):
            for m in entries:
                rows.append({
                    "arm": block["arm"], "model": block["model"],
                    "mode": block["mode"], "scope": m["scope"],
                    "group": m["group"],
                    "support_episodes": m["support_episodes"],
                    "support_flag": m["support_flag"],
                    "support_attack_windows": m["support_attack_windows"],
                    "sensitivity": m["sensitivity"],
                    "specificity": m["specificity"],
                    "precision": m["precision"], "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                    "average_precision": m["average_precision"],
                    "total_cost": m["total_cost"],
                    "false_positives": m["false_positives"],
                    "false_negatives": m["false_negatives"],
                })
    return pd.DataFrame(rows)


def threshold_table(result):
    """Bootstrap threshold stability per (arm, model, weighting)."""
    return pd.DataFrame(result["robustness"]["threshold_stability"])


def write_results(result, directory=RESULTS_DIR, basename="phase3_results"):
    """Write JSON + CSVs. Refuses any basename that could shadow Phase 2 output."""
    basename = assert_not_phase2_basename(basename)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    json_path = directory / f"{basename}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")

    paths = {"json": str(json_path)}
    manifest = result["dataset"]["split_manifest"]
    split_path = directory / f"{basename}_splits.csv"
    manifest_table(manifest).to_csv(split_path, index=False)
    paths["splits_csv"] = str(split_path)

    ablation_path = directory / f"{basename}_ablation.csv"
    ablation_table(result).to_csv(ablation_path, index=False)
    paths["ablation_csv"] = str(ablation_path)

    family_path = directory / f"{basename}_per_family.csv"
    per_family_table(result).to_csv(family_path, index=False)
    paths["per_family_csv"] = str(family_path)

    threshold_path = directory / f"{basename}_thresholds.csv"
    threshold_table(result).to_csv(threshold_path, index=False)
    paths["thresholds_csv"] = str(threshold_path)

    diagnostics_path = directory / f"{basename}_diagnostics.csv"
    diagnostics_table(result).to_csv(diagnostics_path, index=False)
    paths["diagnostics_csv"] = str(diagnostics_path)
    return paths


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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="ASSRD Phase 3 — feature ablation and robustness evaluation")
    parser.add_argument("--quick", action="store_true",
                        help="small trace (%d windows)" % QUICK["n_windows"])
    parser.add_argument("--self-check", action="store_true",
                        help="run preflight checks only and exit; never fits a model")
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--min-episode", type=int, default=None)
    parser.add_argument("--max-episode", type=int, default=None)
    parser.add_argument("--attack-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--min-stealth-episodes", type=int, default=6)
    parser.add_argument("--min-stealth-train", type=int, default=2)
    parser.add_argument("--no-stealth", action="store_true")
    parser.add_argument("--arms", nargs="*", default=list(ABLATION_ARMS))
    parser.add_argument("--models", nargs="*", default=list(MODEL_NAMES))
    parser.add_argument("--modes", nargs="*", default=list(MODE_SPECS))
    parser.add_argument("--threshold-grid", type=int, default=1001)
    parser.add_argument("--stability-bootstrap", type=int, default=None)
    parser.add_argument("--stability-grid", type=int, default=201)
    parser.add_argument("--cost-curve-points", type=int, default=101)
    parser.add_argument("--retune-full", action="store_true")
    parser.add_argument("--time-budget-s", type=float, default=600.0)
    parser.add_argument("--memory-budget-mb", type=float, default=4096.0)
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    parser.add_argument("--basename", default="phase3_results")
    return parser


def _resolve(args):
    profile = QUICK if args.quick else FULL
    config = {
        "n_windows": args.n_windows if args.n_windows is not None
        else profile["n_windows"],
        "min_episode": args.min_episode if args.min_episode is not None
        else profile["min_episode"],
        "max_episode": args.max_episode if args.max_episode is not None
        else profile["max_episode"],
        "stability_bootstrap": (args.stability_bootstrap
                                if args.stability_bootstrap is not None
                                else profile["stability_bootstrap"]),
    }
    return config


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = _resolve(args)

    if args.self_check:
        outcome = self_check(
            n_windows=config["n_windows"], min_episode=config["min_episode"],
            max_episode=config["max_episode"], seed=args.seed,
            min_stealth_episodes=args.min_stealth_episodes,
            min_stealth_train=args.min_stealth_train,
            time_budget_s=args.time_budget_s,
            memory_budget_mb=args.memory_budget_mb)
        for check in outcome["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            print(f"[{mark}] {check['name']}")
            if not check["passed"]:
                print(f"       {json.dumps(check['detail'], default=str)}")
        print(f"\npreflight {'PASSED' if outcome['passed'] else 'FAILED'} "
              f"(no models fitted)")
        return 0 if outcome["passed"] else 1

    assert_not_phase2_basename(args.basename)
    result = run_phase3(
        n_windows=config["n_windows"], min_episode=config["min_episode"],
        max_episode=config["max_episode"], attack_ratio=args.attack_ratio,
        seed=args.seed, include_stealth=not args.no_stealth,
        test_fraction=args.test_fraction,
        validation_fraction=args.validation_fraction,
        min_stealth_episodes=args.min_stealth_episodes,
        min_stealth_train=args.min_stealth_train,
        arms=tuple(args.arms), models=tuple(args.models),
        modes=tuple(args.modes), threshold_grid=args.threshold_grid,
        stability_bootstrap=config["stability_bootstrap"],
        stability_grid=args.stability_grid,
        cost_curve_points=args.cost_curve_points, retune_full=args.retune_full)
    paths = write_results(result, directory=args.output_dir,
                          basename=args.basename)

    table = ablation_table(result)
    print(table.to_string(index=False))
    print()
    print("split composition:")
    print(manifest_table(result["dataset"]["split_manifest"]).to_string(index=False))
    print()
    for kind, path in paths.items():
        print(f"{kind}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
