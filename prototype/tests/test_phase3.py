"""Phase 3 targeted tests.

Grouped by the guarantee under test, so a failure names the promise that broke:

A. feature arms   — the ablation arms are strictly nested and correctly sized
B. allocator      — constraints are met or the allocator fails loudly
C. composition    — the manifest reports episodes, families and stealth honestly
D. matrix         — M1..M4 are distinct and the fixed-0.5 modes are exactly 0.5
E. leakage        — the primary threshold comes from validation, never from test
F. oracle         — the test-oracle is quarantined and never in the primary path
G. robustness     — per-family / stealth support is explicit; undefined is null
H. reproducibility, schema, no-grid-search
I. artifacts      — writers separate tables and never touch the Phase 2 files
J. preflight      — --self-check passes, fits nothing, and preserves Phase 2
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import assrd.phase3 as p3
from assrd.ablation import (
    ABLATION_ARMS,
    NESTED_ARMS,
    arm_columns,
    coerce_param_value,
    load_phase2_params,
)
from assrd.features import MODEL_FEATURE_COLUMNS, RAW_COUNT_COLUMNS
from assrd.features import TEMPORAL_FEATURES
from assrd.models import fit_fixed
from assrd.splits import (
    SplitConstraintError,
    assign_episodes,
    support_flag,
)

TINY = dict(n_windows=3000, min_episode=6, max_episode=12, seed=42,
            min_stealth_episodes=1, min_stealth_train=0, attack_ratio=0.05)
TINY_RUN = dict(TINY, arms=("A_raw_only", "E_full"),
                models=("decision_tree",),
                modes=("M1", "M2", "M3", "M4"),
                stability_bootstrap=3, stability_grid=51, threshold_grid=101)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


@pytest.fixture(scope="module")
def tiny_result():
    return p3.run_phase3(**TINY_RUN)


def _phase2_hashes():
    out = {}
    for name in p3.PHASE2_ARTIFACTS:
        path = RESULTS_DIR / name
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return out


# ---------------------------------------------------------------------------
# A. Feature arms
# ---------------------------------------------------------------------------
def test_arms_are_strictly_nested_and_sized():
    ordered = [set(arm_columns(a)) for a in NESTED_ARMS]
    for smaller, larger in zip(ordered, ordered[1:]):
        assert smaller < larger, "arms must be strictly nested"

    assert len(arm_columns("A_raw_only")) == 3
    assert len(arm_columns("B_trailing_level")) == 15
    assert len(arm_columns("C_plus_activity")) == 19
    assert len(arm_columns("D_plus_volatility")) == 27
    assert len(arm_columns("E_full")) == 29
    assert len(arm_columns("T_temporal_only")) == 26


def test_arm_membership_is_exact():
    assert set(arm_columns("A_raw_only")) == set(RAW_COUNT_COLUMNS)
    assert set(arm_columns("E_full")) == set(MODEL_FEATURE_COLUMNS)
    assert set(arm_columns("T_temporal_only")) == set(TEMPORAL_FEATURES)
    # The temporal-only arm deliberately excludes the raw counts.
    assert not (set(arm_columns("T_temporal_only")) & set(RAW_COUNT_COLUMNS))


def test_unknown_arm_is_rejected():
    with pytest.raises(KeyError):
        arm_columns("Z_not_an_arm")


def test_phase2_param_coercion_recovers_types():
    assert coerce_param_value("3") == 3
    assert coerce_param_value("1.0") == 1.0
    assert coerce_param_value("0.5") == 0.5
    assert coerce_param_value("balanced") == "balanced"
    assert coerce_param_value("None") is None
    assert coerce_param_value(7) == 7

    params, source = load_phase2_params()
    assert source["found"] is True
    for name in ("decision_tree", "random_forest", "svm_calibrated",
                 "adaboost", "xgboost"):
        assert name in params and params[name]
    assert params["decision_tree"]["clf__max_depth"] == 3


# ---------------------------------------------------------------------------
# B. Allocator constraints (safeguard 1)
# ---------------------------------------------------------------------------
def test_allocator_satisfies_declared_constraints():
    from assrd.data import build_trace

    trace = build_trace(n_windows=20000, min_episode=8, max_episode=16,
                        attack_ratio=0.05, seed=42)
    assign, report = assign_episodes(trace, test_fraction=0.20,
                                     validation_fraction=0.20, seed=42,
                                     min_stealth_episodes=6, min_stealth_train=2)
    assert report["satisfied"] is True
    assert report["shortfall"] == []
    assert report["realised"]["stealth_episodes_by_split"]["validation"] >= 6
    assert report["realised"]["stealth_episodes_by_split"]["test"] >= 6
    for split in ("validation", "test"):
        for fam in report["targets"]["guaranteed_families"]:
            assert report["realised"]["family_episodes_by_split"][split][fam] > 0
    assert len(assign) == trace[trace["label"] == 1]["episode_id"].nunique()


def test_allocator_fails_loudly_on_impossible_stealth_target():
    from assrd.data import build_trace

    trace = build_trace(n_windows=3000, min_episode=6, max_episode=12,
                        attack_ratio=0.05, seed=42)
    with pytest.raises(SplitConstraintError) as excinfo:
        assign_episodes(trace, min_stealth_episodes=1000, min_stealth_train=0)
    shortfall = excinfo.value.shortfall
    assert "stealth_episodes" in shortfall
    assert shortfall["stealth_episodes"]["satisfied"] is False


def test_allocator_fails_loudly_when_train_reserve_impossible():
    from assrd.data import build_trace

    trace = build_trace(n_windows=3000, min_episode=6, max_episode=12,
                        attack_ratio=0.05, seed=42)
    with pytest.raises(SplitConstraintError) as excinfo:
        assign_episodes(trace, test_fraction=0.6, validation_fraction=0.6,
                        seed=42, min_stealth_episodes=1, min_stealth_train=0)
    assert "training_reserve" in excinfo.value.shortfall


# ---------------------------------------------------------------------------
# C. Composition manifest
# ---------------------------------------------------------------------------
def test_manifest_reports_episodes_families_and_stealth(tiny_result):
    manifest = tiny_result["dataset"]["split_manifest"]
    assert manifest["leakage"]["episode_aware"] is True
    assert manifest["leakage"]["disjoint_rows"] is True

    splits, _ = p3.prepare_dataset(**TINY)
    for name, block in manifest["splits"].items():
        frame = splits[name]
        assert block["rows"] == len(frame)
        assert block["attack_rows"] == int((frame["label"] == 1).sum())
        assert block["attack_episodes"] == int(
            frame.loc[frame["label"] == 1, "episode_id"].nunique())
        # every family reported, explicit zeros included
        assert set(block["attack_episodes_by_family"]) == set(
            manifest["family_names"])
        # benign is not an episode; it is reported as blocks
        assert block["benign_episodes"] == 0
        assert "benign" in block["benign_episodes_note"].lower()
        assert block["benign_managed_blocks"] >= 0
        assert "support_flags" in block


def test_manifest_cells_and_flags_are_consistent(tiny_result):
    manifest = tiny_result["dataset"]["split_manifest"]
    for block in manifest["splits"].values():
        cells = block["cells_family_stealth"]
        assert cells["of"] == len(manifest["family_names"]) * 2
        assert cells["present"] <= cells["of"]
        # a cell flagged absent must have zero episodes
        for key, flag in block["support_flags"]["family_stealth_cells"].items():
            if flag == "absent":
                assert cells["counts"][key] == 0


def test_shortfalls_are_reported_not_hidden(tiny_result):
    manifest = tiny_result["dataset"]["split_manifest"]
    guarded = manifest["allocation"]["targets"]["guaranteed_families"]
    for fam in guarded:
        assert manifest["splits"]["validation"][
            "attack_episodes_by_family"][fam] > 0
        assert manifest["splits"]["test"]["attack_episodes_by_family"][fam] > 0
    # every shortfall row carries an explicit flag
    for row in manifest["shortfalls"]:
        assert row["flag"] in {"absent", "insufficient", "thin", "adequate"}


def test_support_flag_boundaries():
    assert support_flag(0) == "absent"
    assert support_flag(1) == "insufficient"
    assert support_flag(2) == "thin"
    assert support_flag(4) == "thin"
    assert support_flag(5) == "adequate"
    assert support_flag(99) == "adequate"


# ---------------------------------------------------------------------------
# D. The experiment matrix
# ---------------------------------------------------------------------------
def test_matrix_contains_all_four_modes(tiny_result):
    modes = {block["mode"] for block in tiny_result["primary"].values()}
    assert modes == {"M1", "M2", "M3", "M4"}
    for block in tiny_result["primary"].values():
        spec = p3.MODE_SPECS[block["mode"]]
        assert block["cost_weighted"] == spec["cost_weighted"]
        assert block["threshold_rule"] == spec["threshold_rule"]


def test_fixed_half_modes_use_exactly_half(tiny_result):
    for block in tiny_result["primary"].values():
        if block["threshold_rule"] == "0.5":
            assert block["threshold"] == 0.5
        else:
            assert block["threshold"] == block["validation_selected_threshold"]


def test_unweighted_fit_passes_no_sample_weight(monkeypatch):
    """cost_weighted=False must not build cost weights at all."""
    called = {"n": 0}
    import assrd.models as models

    real = models.cost_sample_weights

    def spy(y, *args, **kwargs):
        called["n"] += 1
        return real(y, *args, **kwargs)

    monkeypatch.setattr(models, "cost_sample_weights", spy)

    X = np.random.RandomState(0).randn(80, 4)
    y = (X[:, 0] > 0).astype(int)

    unweighted = fit_fixed("decision_tree", X, y, cost_weighted=False)
    assert unweighted["sample_weight"]["weighted"] is False
    assert called["n"] == 0

    weighted = fit_fixed("decision_tree", X, y, cost_weighted=True)
    assert weighted["sample_weight"]["weighted"] is True
    assert called["n"] == 1


def test_comparisons_are_declared_in_advance(tiny_result):
    ids = {c["id"] for c in tiny_result["comparisons"]}
    assert {"C1", "C2", "C3", "C4", "C5"} <= ids
    for comparison in tiny_result["comparisons"]:
        assert comparison["question"]
        assert comparison["metric"] == "total_cost"


# ---------------------------------------------------------------------------
# E. Leakage: the primary threshold comes from validation only
# ---------------------------------------------------------------------------
def test_primary_threshold_is_validation_selected(tiny_result):
    for block in tiny_result["primary"].values():
        if block["threshold_rule"] == "validation":
            assert block["threshold"] == block["validation_selected_threshold"]


def test_select_threshold_sees_only_validation_and_test_arrays(monkeypatch):
    calls = []
    real = p3.select_threshold

    def spy(y_true, proba, **kwargs):
        calls.append(np.array(y_true, copy=True))
        return real(y_true, proba, **kwargs)

    # Silence the bootstrap's own threshold selections so this test counts only
    # the driver's per-fit calls.
    import assrd.ablation as ablation

    monkeypatch.setattr(ablation, "select_threshold",
                        lambda *a, **k: {"threshold": 0.5})
    monkeypatch.setattr(p3, "select_threshold", spy)
    result = p3.run_phase3(**TINY_RUN)

    splits, _ = p3.prepare_dataset(**TINY)
    y_val = splits["validation"]["label"].to_numpy(dtype=int)
    y_test = splits["test"]["label"].to_numpy(dtype=int)

    n_val = sum(1 for c in calls if np.array_equal(c, y_val))
    n_test = sum(1 for c in calls if np.array_equal(c, y_test))
    n_other = len(calls) - n_val - n_test

    # one fit per (arm, model, weighting) -> one validation pick and one oracle
    expected = len(TINY_RUN["arms"]) * len(TINY_RUN["models"]) * 2
    assert n_val == expected
    assert n_test == expected
    assert n_other == 0, "select_threshold saw an array that is neither split"

    for block in result["primary"].values():
        if block["threshold_rule"] == "validation":
            assert block["threshold"] == block["validation_selected_threshold"]


# ---------------------------------------------------------------------------
# F. Oracle isolation
# ---------------------------------------------------------------------------
def test_oracle_never_appears_in_primary(tiny_result):
    assert "oracle" not in json.dumps(tiny_result["primary"]).lower()
    assert p3.assert_oracle_isolated(tiny_result) is True


def test_oracle_is_recorded_under_diagnostics(tiny_result):
    oracle = tiny_result["diagnostics"]["test_oracle"]
    assert oracle
    assert tiny_result["diagnostics"]["achievable_deployment_performance"] is False
    for block in oracle.values():
        assert 0.0 <= block["threshold"] <= 1.0
        assert "not achievable deployment performance" in block["note"].lower()


def test_oracle_checker_rejects_a_poisoned_primary():
    poisoned = {"primary": {"test_oracle": {}}, "diagnostics": {"test_oracle": {}}}
    with pytest.raises(AssertionError):
        p3.assert_oracle_isolated(poisoned)


def test_diagnostics_table_is_separate(tiny_result):
    ablation = p3.ablation_table(tiny_result)
    assert not any("oracle" in col.lower() for col in ablation.columns)

    diagnostics = p3.diagnostics_table(tiny_result)
    assert set(diagnostics["is_diagnostic"]) == {True}
    assert set(diagnostics["achievable_deployment_performance"]) == {False}


# ---------------------------------------------------------------------------
# G. Robustness reporting
# ---------------------------------------------------------------------------
def test_per_family_support_is_explicit(tiny_result):
    table = p3.per_family_table(tiny_result)
    assert set(table["scope"]) == {"family", "stealth"}
    for _, row in table.iterrows():
        assert row["support_flag"] in {"absent", "insufficient", "thin", "adequate"}
        assert row["support_attack_windows"] >= 0
    assert set(table.loc[table["scope"] == "stealth", "group"]) == {
        "stealth", "non_stealth"}


def test_undefined_subgroup_metrics_are_none_not_zero():
    from assrd.ablation import subgroup_eval
    import pandas as pd

    # meta says every attack is ryuk; asking for wannacry yields an empty group.
    meta = pd.DataFrame({
        "episode_id": [0, 0, 0, 0],
        "family": ["ryuk", "ryuk", None, None],
        "is_stealth": [False, False, False, False],
    })
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    proba = np.array([0.9, 0.4, 0.1, 0.2])

    absent = subgroup_eval(meta, y_true, y_pred, proba, scope="family",
                           value="wannacry")
    assert absent["support_episodes"] == 0
    assert absent["support_flag"] == "absent"
    assert absent["sensitivity"] is None
    assert absent["precision"] is None
    assert absent["roc_auc"] is None


def test_changing_family_metrics_use_benign_as_the_negative_class():
    from assrd.ablation import subgroup_eval
    import pandas as pd

    meta = pd.DataFrame({
        "episode_id": [0, 0, 1, 1],
        "family": ["ryuk", "ryuk", "ryuk", "ryuk"],
        "is_stealth": [False, False, False, False],
    })
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    proba = np.array([0.9, 0.4, 0.1, 0.2])
    row = subgroup_eval(meta, y_true, y_pred, proba, scope="family", value="ryuk")
    assert row["support_attack_windows"] == 2
    assert row["support_episodes"] == 1
    assert row["sensitivity"] == pytest.approx(0.5)


def test_threshold_stability_is_reported(tiny_result):
    records = tiny_result["robustness"]["threshold_stability"]
    assert records
    for record in records:
        assert "base_threshold" in record
        assert record["n_boot"] >= 1
        assert record["min"] <= record["median"] <= record["max"]


# ---------------------------------------------------------------------------
# H. Reproducibility, schema, no grid search
# ---------------------------------------------------------------------------
def test_run_is_reproducible_for_a_fixed_seed():
    first = p3.run_phase3(**TINY_RUN)
    second = p3.run_phase3(**TINY_RUN)
    assert set(first["primary"]) == set(second["primary"])
    for key in first["primary"]:
        a = first["primary"][key]["test"]
        b = second["primary"][key]["test"]
        assert a["total_cost"] == b["total_cost"]
        assert a["false_negatives"] == b["false_negatives"]
        assert first["primary"][key]["threshold"] == second["primary"][key]["threshold"]


def test_result_schema_is_complete(tiny_result):
    for key in ("phase", "provenance", "generated_utc", "config", "environment",
                "dataset", "matrix", "comparisons_declared", "primary",
                "diagnostics", "robustness", "limitations"):
        assert key in tiny_result, key
    assert tiny_result["phase"] == 3
    assert "emulator" in tiny_result["provenance"].lower()
    assert tiny_result["config"]["cost_fn"] == 10.0
    assert tiny_result["config"]["cost_fp"] == 1.0


def test_ablation_path_constructs_no_grid_search(monkeypatch):
    import assrd.models as models

    counter = {"n": 0}
    real = models.make_search

    def spy(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(models, "make_search", spy)
    p3.run_phase3(**TINY_RUN)
    assert counter["n"] == 0, "the ablation path must not build a GridSearchCV"


def test_retune_full_does_use_grid_search(monkeypatch):
    import assrd.models as models

    counter = {"n": 0}
    real = models.make_search

    def spy(*args, **kwargs):
        counter["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(models, "make_search", spy)
    p3.run_phase3(**dict(TINY_RUN, retune_full=True, arms=("E_full",)))
    assert counter["n"] > 0, "--retune-full must actually re-tune"


# ---------------------------------------------------------------------------
# I. Artifacts
# ---------------------------------------------------------------------------
def test_write_results_emits_separated_artifacts(tiny_result, tmp_path):
    paths = p3.write_results(tiny_result, directory=tmp_path,
                             basename="phase3_results")
    for key in ("json", "splits_csv", "ablation_csv", "per_family_csv",
                "thresholds_csv", "diagnostics_csv"):
        assert Path(paths[key]).exists(), key

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["phase"] == 3
    assert isinstance(payload["primary"], dict)

    ablation_text = Path(paths["ablation_csv"]).read_text(encoding="utf-8")
    assert "oracle" not in ablation_text.lower()
    diagnostics_text = Path(paths["diagnostics_csv"]).read_text(encoding="utf-8")
    assert "is_diagnostic" in diagnostics_text


def test_write_results_refuses_a_phase2_basename(tiny_result, tmp_path):
    with pytest.raises(ValueError):
        p3.write_results(tiny_result, directory=tmp_path,
                         basename="phase2_results")


# ---------------------------------------------------------------------------
# J. Preflight self-check (safeguard 2)
# ---------------------------------------------------------------------------
def test_self_check_passes_and_fits_nothing():
    before = _phase2_hashes()
    outcome = p3.self_check(n_windows=TINY["n_windows"],
                            min_episode=TINY["min_episode"],
                            max_episode=TINY["max_episode"],
                            min_stealth_episodes=TINY["min_stealth_episodes"],
                            min_stealth_train=TINY["min_stealth_train"])
    after = _phase2_hashes()

    assert outcome["passed"] is True
    assert outcome["fits_run"] == 0
    names = {c["name"] for c in outcome["checks"]}
    assert {"G1_leakage", "G2_composition", "G3_reproducibility", "G4_schema",
            "G5_oracle_isolation", "G6_feasibility",
            "G7_phase2_preserved"} <= names
    # Phase 2 artifacts must be byte-for-byte unchanged.
    assert before == after


def test_self_check_fails_on_impossible_constraints():
    outcome = p3.self_check(n_windows=TINY["n_windows"],
                            min_episode=TINY["min_episode"],
                            max_episode=TINY["max_episode"],
                            min_stealth_episodes=1000, min_stealth_train=0)
    assert outcome["passed"] is False
    failed = {c["name"] for c in outcome["checks"] if not c["passed"]}
    assert "allocator_constraints" in failed


def test_cli_self_check_runs_preflight_only():
    exit_code = p3.main(["--quick", "--self-check"])
    assert exit_code in (0, 1)
