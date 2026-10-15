"""Safety tests for the six-model re-baselining orchestrator.

Every test runs against a **sandbox** created in ``tmp_path``. The module's
path constants are redirected with ``monkeypatch``, so no test can read, write
or delete the real ``prototype/results`` tree, and the nine Tier A artifacts
under test are throwaway files created by the test itself.

Nothing here executes a reset, a rollback or a PRF against real data.
"""

import hashlib
import json
import os

import pytest

from assrd import rebaseline as rb


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect every path constant at a throwaway tree."""
    results = tmp_path / "results"
    run_dir = results / "rebaseline_6model"
    quick_dir = run_dir / "quick"
    forensics = results / "_rebaseline_forensics"
    for directory in (run_dir, quick_dir, forensics):
        directory.mkdir(parents=True)

    tier_a = {}
    for index, name in enumerate(sorted(rb.TIER_A)):
        # TIER_A keys are prototype-relative (e.g. "results/phase2_..."), and
        # PROTOTYPE_DIR is redirected at tmp_path, so resolve against tmp_path.
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"tier-a-{index}\n", encoding="utf-8")
        tier_a[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    monkeypatch.setattr(rb, "RESULTS_DIR", results)
    monkeypatch.setattr(rb, "RUN_DIR", run_dir)
    monkeypatch.setattr(rb, "QUICK_DIR", quick_dir)
    monkeypatch.setattr(rb, "FORENSICS_PARENT", forensics)
    monkeypatch.setattr(rb, "LOCK_PATH", forensics / "lock")
    monkeypatch.setattr(rb, "PROTOTYPE_DIR", tmp_path)
    monkeypatch.setattr(rb, "TIER_A", tier_a)
    monkeypatch.setattr(rb, "_head", lambda: "test-head")
    monkeypatch.setattr(rb, "_porcelain", lambda: {})
    monkeypatch.setattr(rb, "EXPECTED_HEAD", "test-head")
    return {"root": tmp_path, "results": results, "run": run_dir,
            "quick": quick_dir, "forensics": forensics, "tier_a": tier_a}


def _seed_tier_b(sandbox):
    for name in rb.TIER_B_NAMES:
        (sandbox["run"] / name).write_text(f"b:{name}\n", encoding="utf-8")
    for name in rb.TIER_D_NAMES:
        (sandbox["quick"] / name).write_text(f"d:{name}\n", encoding="utf-8")


def _write_m1(sandbox):
    lines = ["# sandbox baseline", "# git HEAD: test-head"]
    for rel, digest in sorted(sandbox["tier_a"].items()):
        lines.append(f"{digest}  {rel}")
    (sandbox["run"] / rb.M1_NAME).write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")


def _write_m3(sandbox):
    digests = {n: rb._sha256(sandbox["quick"] / n) for n in rb.TIER_D_NAMES}
    payload = {"schema": "assrd.rebaseline/quick-manifest/1",
               "authoritative": False,
               "ledger": {"stages": ["quick_phase3"], "complete": True,
                          "claims": list(rb.TIER_D_NAMES), "digests": digests}}
    (sandbox["quick"] / rb.M3_NAME).write_text(rb._canonical(payload),
                                               encoding="utf-8")


def _write_m2(sandbox, generation=2, tier_c=False):
    stages = {"baseline": {"complete": True, "claims": [rb.M1_NAME]},
              "phase2": {"complete": True, "claims": list(rb.TIER_B_NAMES),
                         "digests": {n: rb._sha256(sandbox["run"] / n)
                                     for n in rb.TIER_B_NAMES}}}
    if generation >= 2:
        stages["quick_phase3"] = {
            "complete": True, "claimed_by": f"quick/{rb.M3_NAME}",
            "manifest_digest": rb._sha256(sandbox["quick"] / rb.M3_NAME)}
    if generation >= 3:
        stages["phase3_full"] = {"complete": True,
                                 "claims": list(rb.TIER_C_NAMES),
                                 "digests": {n: rb._sha256(sandbox["run"] / n)
                                             for n in rb.TIER_C_NAMES}}
    docs = {rb.M1_NAME: rb._sha256(sandbox["run"] / rb.M1_NAME),
            f"quick/{rb.M3_NAME}": rb._sha256(sandbox["quick"] / rb.M3_NAME)}
    payload = {"schema": "assrd.rebaseline/manifest/1",
               "generation": generation, "created_utc": rb._now(),
               "head": "test-head", "documents": docs, "stages": stages,
               "run_complete": generation >= 3}
    (sandbox["run"] / rb.M2_NAME).write_text(rb._canonical(payload),
                                             encoding="utf-8")


def _full_sandbox(sandbox, generation=3, tier_c=True):
    _seed_tier_b(sandbox)
    _write_m1(sandbox)
    _write_m3(sandbox)
    if tier_c:
        for name in rb.TIER_C_NAMES:
            (sandbox["run"] / name).write_text(f"c:{name}\n", encoding="utf-8")
    _write_m2(sandbox, generation=generation, tier_c=tier_c)


# ---------------------------------------------------------------------------
# A. Lock: exclusivity, ownership, never auto-remove
# ---------------------------------------------------------------------------
def test_lock_is_created_with_identity_and_holder(sandbox):
    lock = rb.acquire_lock("unit")
    try:
        data = json.loads(rb.LOCK_PATH.read_text(encoding="utf-8"))
        assert data["pid"] == os.getpid()
        assert data["command"] == "unit"
        assert data["nonce"] == lock.nonce
        assert data["host"]
    finally:
        rb.release_lock(lock)


def test_second_acquisition_fails_closed_and_keeps_the_lock(sandbox):
    first = rb.acquire_lock("first")
    original = rb.LOCK_PATH.read_text(encoding="utf-8")
    with pytest.raises(rb.GateError, match="lock already held"):
        rb.acquire_lock("second")
    # The incumbent's lock is untouched.
    assert rb.LOCK_PATH.read_text(encoding="utf-8") == original
    rb.release_lock(first)
    assert not rb.LOCK_PATH.exists()


def test_release_refuses_when_nonce_does_not_match(sandbox):
    lock = rb.acquire_lock("unit")
    data = json.loads(rb.LOCK_PATH.read_text(encoding="utf-8"))
    data["nonce"] = "someone-else"
    rb.LOCK_PATH.write_text(rb._canonical(data), encoding="utf-8")
    with pytest.raises(rb.GateError, match="ownership could not be verified"):
        rb.release_lock(lock)
    # A lock we cannot prove we own is left exactly where it is.
    assert rb.LOCK_PATH.exists()


def test_release_refuses_when_lock_vanished(sandbox):
    lock = rb.acquire_lock("unit")
    rb.LOCK_PATH.unlink()
    with pytest.raises(rb.GateError, match="ownership cannot be verified"):
        rb.release_lock(lock)


def test_release_deletes_the_verified_path(sandbox):
    lock = rb.acquire_lock("unit")
    rb.release_lock(lock)
    assert lock.released is True
    assert not rb.LOCK_PATH.exists()


def test_stale_lock_is_never_auto_removed(sandbox):
    rb.LOCK_PATH.write_text(rb._canonical(
        {"pid": 999999, "host": "other-host", "utc": "old", "command": "crashed",
         "nonce": "dead"}), encoding="utf-8")
    with pytest.raises(rb.GateError):
        rb.acquire_lock("new")
    assert rb.LOCK_PATH.exists(), "a possibly live lock must survive a refusal"


def test_unreadable_lock_is_never_treated_as_stale(sandbox):
    rb.LOCK_PATH.write_text("not json at all", encoding="utf-8")
    with pytest.raises(rb.GateError):
        rb.acquire_lock("new")
    assert rb.LOCK_PATH.read_text(encoding="utf-8") == "not json at all"


def test_context_manager_releases_after_success(sandbox):
    with rb.locked("unit"):
        assert rb.LOCK_PATH.exists()
    assert not rb.LOCK_PATH.exists()


def test_context_manager_releases_after_failure_and_reraises(sandbox):
    with pytest.raises(ValueError):
        with rb.locked("unit"):
            raise ValueError("boom")
    assert not rb.LOCK_PATH.exists()


def test_lock_requires_pre_existing_forensics_parent(sandbox, monkeypatch):
    monkeypatch.setattr(rb, "FORENSICS_PARENT", sandbox["forensics"] / "absent")
    with pytest.raises(rb.GateError, match="must be"):
        rb.acquire_lock("unit")


# ---------------------------------------------------------------------------
# B. Snapshot: unique, sealed, Tier A never copied
# ---------------------------------------------------------------------------
def test_snapshot_is_unique_and_sealed(sandbox):
    _full_sandbox(sandbox)
    before = set(os.listdir(sandbox["forensics"]))
    target, sealed = rb.snapshot({"trigger": "unit"})
    assert target.is_dir()
    assert target.name.startswith("snapshot-")
    assert not (before & {target.name})

    names = {p.name for p in target.iterdir()}
    assert rb.SIDECAR_NAME not in names
    assert "MANIFEST.json" in names and "SHA256SUMS.txt" in names
    # every sealed file is present in the returned digest map
    for name in names:
        assert sealed[name] == rb._sha256(target / name)


def test_snapshot_never_copies_or_touches_tier_a(sandbox):
    _full_sandbox(sandbox)
    before = {rel: rb._sha256(sandbox["root"] / rel)
              for rel in sandbox["tier_a"]}
    target, _ = rb.snapshot({"trigger": "unit"})
    after = {rel: rb._sha256(sandbox["root"] / rel)
             for rel in sandbox["tier_a"]}
    assert before == after
    # Tier A content must not appear anywhere inside the snapshot.
    blob = b"".join(p.read_bytes() for p in target.iterdir())
    for rel in sandbox["tier_a"]:
        assert (sandbox["root"] / rel).read_bytes() not in blob


def test_snapshot_refuses_unexpected_forensics_entry(sandbox):
    stray = sandbox["forensics"] / "not-a-snapshot.txt"
    stray.write_text("stray", encoding="utf-8")
    _full_sandbox(sandbox)
    with pytest.raises(rb.GateError, match="unexpected entry"):
        rb.snapshot({"trigger": "unit"})
    assert stray.exists()


def test_snapshot_refuses_symlinked_run_directory(sandbox):
    _full_sandbox(sandbox)
    real = sandbox["root"] / "elsewhere"
    real.mkdir()
    link = sandbox["root"] / "linked_run"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform/account")
    monkey = rb.RUN_DIR
    try:
        rb.RUN_DIR = link          # direct rebind for this test only
        with pytest.raises(rb.GateError, match="symlink"):
            rb.snapshot({"trigger": "unit"})
    finally:
        rb.RUN_DIR = monkey


def test_snapshot_refuses_missing_forensics_parent(sandbox, monkeypatch):
    monkeypatch.setattr(rb, "FORENSICS_PARENT", sandbox["forensics"] / "gone")
    _full_sandbox(sandbox)
    with pytest.raises(rb.GateError, match="missing or not a real directory"):
        rb.snapshot({"trigger": "unit"})


# ---------------------------------------------------------------------------
# C. Classification and Tier A protection
# ---------------------------------------------------------------------------
def test_tier_a_tampering_is_detected(sandbox):
    _full_sandbox(sandbox)
    assert rb.verify_tier_a() == []
    target = sandbox["root"] / next(iter(sandbox["tier_a"]))
    target.write_text("tampered", encoding="utf-8")
    findings = rb.verify_tier_a()
    assert any("MODIFIED" in message for message in findings)
    assert any(cls == "B" for cls, _ in rb.classify())


def test_head_mismatch_is_class_e(sandbox, monkeypatch):
    _full_sandbox(sandbox)
    monkeypatch.setattr(rb, "_head", lambda: "deadbeef")
    assert any(cls == "E" for cls, _ in rb.classify())


def test_tier_c_present_without_claim_is_class_a(sandbox):
    _full_sandbox(sandbox, generation=2, tier_c=False)
    for name in rb.TIER_C_NAMES:
        (sandbox["run"] / name).write_text("c\n", encoding="utf-8")
    assert any(cls == "A" for cls, _ in rb.classify())


def test_tier_c_claimed_at_generation_three_is_not_residue(sandbox):
    _full_sandbox(sandbox, generation=3, tier_c=True)
    assert not any(cls == "A" and "leftover full-run" in message
                   for cls, message in rb.classify())


def test_self_hash_in_manifest_is_class_d(sandbox):
    _full_sandbox(sandbox)
    payload = json.loads((sandbox["run"] / rb.M2_NAME).read_text(encoding="utf-8"))
    payload["documents"][rb.M2_NAME] = "deadbeef"
    (sandbox["run"] / rb.M2_NAME).write_text(rb._canonical(payload),
                                             encoding="utf-8")
    assert any(cls == "D" and "self-hash" in message
               for cls, message in rb.classify())


# ---------------------------------------------------------------------------
# D. Porcelain: evidence is surfaced, never hidden, never an artifact set
# ---------------------------------------------------------------------------
def test_forensic_paths_are_evidence_not_unexpected(sandbox, monkeypatch):
    monkeypatch.setattr(rb, "_porcelain", lambda: {
        rb.EVIDENCE_PREFIX + "snapshot-20260101T000000Z-abcd/lock": "??"})
    findings = rb.check_porcelain("source")
    assert [cls for cls, _ in findings] == ["evidence"]


def test_unrelated_paths_are_class_e(sandbox, monkeypatch):
    monkeypatch.setattr(rb, "_porcelain", lambda: {"prototype/evil.py": "??"})
    findings = rb.check_porcelain("source")
    assert findings and findings[0][0] == "E"


def test_expected_sets_do_not_include_evidence_or_artifacts(sandbox):
    for stage, expected in rb.STAGE_SETS.items():
        assert not any(p.startswith(rb.EVIDENCE_PREFIX) for p in expected), stage
    # The approved source allowlist holds only reviewed files. The audit added
    # its own test module; the documentation pass added README.md.
    assert set(rb.APPROVED_SOURCE) == {
        "prototype/assrd/models.py", "prototype/assrd/phase2.py",
        "prototype/tests/test_phase2.py", "prototype/README.md",
        "prototype/assrd/rebaseline.py", "prototype/tests/test_rebaseline.py"}
    # No result artifact and no evidence path is ever in an expected set.
    assert not any("rebaseline_6model" in p for p in rb.APPROVED_SOURCE)
    assert rb.STAGE_SETS["source"] == set(rb.APPROVED_SOURCE)
    assert len(rb.STAGE_FULL) == len(rb.STAGE_SOURCE) + 3 + 6 + 6 + 3


# ---------------------------------------------------------------------------
# E. Reset gating
# ---------------------------------------------------------------------------
def test_reset_refuses_when_tier_a_is_broken(sandbox):
    _full_sandbox(sandbox)
    victim = sandbox["root"] / next(iter(sandbox["tier_a"]))
    victim.write_text("tampered", encoding="utf-8")
    with pytest.raises(rb.GateError):
        rb.reset_new_dir("unit")
    assert sandbox["run"].exists(), "nothing may be deleted"


def test_reset_refuses_on_non_a_findings(sandbox, monkeypatch):
    _full_sandbox(sandbox)
    monkeypatch.setattr(rb, "classify", lambda: [("C", "bad config")])
    with pytest.raises(rb.GateError, match="highest finding class is C"):
        rb.reset_new_dir("unit")
    assert sandbox["run"].exists()


def test_reset_happy_path_snapshots_then_deletes(sandbox):
    _full_sandbox(sandbox)
    rb.reset_new_dir("unit")
    assert not sandbox["run"].exists()
    snapshots = [p for p in sandbox["forensics"].iterdir()
                 if p.name.startswith("snapshot-")]
    assert len(snapshots) == 1
    assert (sandbox["forensics"] / rb.SIDECAR_NAME).exists()
    assert rb.verify_tier_a() == []


def test_reset_post_failure_is_terminal_prf(sandbox, monkeypatch):
    _full_sandbox(sandbox)
    # Pre-reset checks run while the run directory still exists and pass; the
    # post-reset check runs after deletion and fails.
    monkeypatch.setattr(rb, "verify_tier_a",
                        lambda: [] if sandbox["run"].exists()
                        else ["Tier A MODIFIED (simulated)"])
    with pytest.raises(rb.PostResetFailure):
        rb.reset_new_dir("unit")
    # evidence survives, nothing was recreated
    assert [p for p in sandbox["forensics"].iterdir()
            if p.name.startswith("snapshot-")]
    assert not sandbox["run"].exists()


# ---------------------------------------------------------------------------
# F. PRF reporting never writes into the sealed snapshot
# ---------------------------------------------------------------------------
def test_report_prf_does_not_modify_the_snapshot(sandbox):
    _full_sandbox(sandbox)
    rb.reset_new_dir("unit")
    snapshot_dir = next(p for p in sandbox["forensics"].iterdir()
                        if p.name.startswith("snapshot-"))
    before = {p.name: rb._sha256(p) for p in snapshot_dir.iterdir()}
    rb._report_prf("simulated failure", snapshot_dir, "unit")
    after = {p.name: rb._sha256(p) for p in snapshot_dir.iterdir()}
    assert before == after


# ---------------------------------------------------------------------------
# G. Rollback
# ---------------------------------------------------------------------------
def test_rollback_refuses_generation_one(sandbox):
    _full_sandbox(sandbox, generation=1, tier_c=False)
    with pytest.raises(rb.GateError, match="not resumable"):
        rb.rollback_w3()
    assert (sandbox["run"] / rb.M2_NAME).exists()


def test_rollback_refuses_class_e(sandbox, monkeypatch):
    _full_sandbox(sandbox)
    monkeypatch.setattr(rb, "classify", lambda: [("E", "unexpected file")])
    with pytest.raises(rb.GateError, match="class E"):
        rb.rollback_w3()


def test_rollback_restores_generation_two_and_removes_tier_c(sandbox):
    _full_sandbox(sandbox, generation=3, tier_c=True)
    tier_b_before = {n: rb._sha256(sandbox["run"] / n)
                     for n in rb.TIER_B_NAMES}
    rb.rollback_w3()
    payload = json.loads((sandbox["run"] / rb.M2_NAME).read_text(encoding="utf-8"))
    assert payload["generation"] == 2
    assert payload["run_complete"] is False
    assert "phase3_full" not in payload["stages"]
    for name in rb.TIER_C_NAMES:
        assert not (sandbox["run"] / name).exists()
    assert {n: rb._sha256(sandbox["run"] / n)
            for n in rb.TIER_B_NAMES} == tier_b_before
    assert (sandbox["quick"] / rb.M3_NAME).exists()


def test_rollback_resumes_from_partial_tier_c(sandbox):
    _full_sandbox(sandbox, generation=3, tier_c=True)
    for name in rb.TIER_C_NAMES[:3]:
        (sandbox["run"] / name).unlink()
    rb.rollback_w3()
    payload = json.loads((sandbox["run"] / rb.M2_NAME).read_text(encoding="utf-8"))
    assert payload["generation"] == 2
    for name in rb.TIER_C_NAMES:
        assert not (sandbox["run"] / name).exists()


def test_rollback_is_idempotent_on_a_second_call(sandbox):
    _full_sandbox(sandbox, generation=3, tier_c=True)
    rb.rollback_w3()
    rb.rollback_w3()
    payload = json.loads((sandbox["run"] / rb.M2_NAME).read_text(encoding="utf-8"))
    assert payload["generation"] == 2


def test_rollback_verifies_tier_a_before_mutating(sandbox):
    _full_sandbox(sandbox, generation=3, tier_c=True)
    victim = sandbox["root"] / next(iter(sandbox["tier_a"]))
    victim.write_text("tampered", encoding="utf-8")
    with pytest.raises(rb.GateError, match="Tier A is not intact"):
        rb.rollback_w3()
    # Tier C must still be present: nothing was deleted.
    assert (sandbox["run"] / rb.TIER_C_NAMES[0]).exists()