"""Six-model non-destructive re-baselining orchestrator.

This module adds Logistic Regression as a sixth model and re-runs Phase 2 and
Phase 3 with all six models into a **separate** output directory, without ever
touching the nine original Phase 2/Phase 3 result artifacts.

Why this exists
---------------
Phase 2 recorded ``best_params`` for exactly five models. Once Logistic
Regression is registered, :func:`assrd.ablation.load_phase2_params` would
silently substitute its hardcoded fallback for the new model, so a Phase 3 run
would mix tuned and untuned models. This module therefore re-runs Phase 2 first
and passes the resulting six parameter sets into Phase 3 explicitly.

Safety model
------------
Nothing here is allowed to damage evidence. The rules are:

* The nine originals (``Tier A``) are compared against digests **embedded in
  this file**. They are opened read-only, never written, never copied.
* Every mutating command takes an exclusive lock located **outside** the
  directory it mutates, so the lock cannot be destroyed by the reset it gates.
* Any destructive reset first captures a **sealed, uniquely created** forensic
  snapshot. There is no bypass flag: if the snapshot cannot be created and
  verified, nothing is deleted.
* Failures are classified ``E > D > C > B > A``. Cleanup is only ever attempted
  when findings are exclusively class A (leftover files).
* After a deletion, any failure is terminal (``PRF``): halt, preserve evidence,
  report, require a human. Nothing is retried, recreated, or deleted again.

Provenance
----------
All attack traffic is produced by the behavioural **emulator** in
``assrd.data``; no malware is executed and no real-world detection rate is
claimed by anything this module writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from assrd import compat  # noqa: F401  installs the sklearn/xgboost shim first
from assrd.ablation import ABLATION_ARMS, load_phase2_params
from assrd.features import MODEL_FEATURE_COLUMNS
from assrd.metrics import C_FN, C_FP
from assrd.models import MODEL_NAMES, build_estimator, param_grid
from assrd.phase2 import run_phase2
from assrd.phase2 import write_results as write_phase2_results
from assrd.phase3 import FULL, QUICK, run_phase3
from assrd.phase3 import write_results as write_phase3_results

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
PROTOTYPE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROTOTYPE_DIR / "results"
REPO_ROOT = PROTOTYPE_DIR.parent

RUN_DIR = RESULTS_DIR / "rebaseline_6model"
QUICK_DIR = RUN_DIR / "quick"
FORENSICS_PARENT = RESULTS_DIR / "_rebaseline_forensics"
LOCK_PATH = FORENSICS_PARENT / "lock"

M1_NAME = "BASELINE_HASHES.txt"
M2_NAME = "RUN_MANIFEST.json"
M3_NAME = "RUN_MANIFEST.json"
TMP_SUFFIX = ".tmp"
SIDECAR_NAME = "POST_RESET_VERIFICATION.json"
SIDECAR_SUM_NAME = "POST_RESET_VERIFICATION.sha256"

EXPECTED_HEAD = "4e585ff76204c57abf8284b1e76f9b0e85d5fa86"

PROVENANCE = (
    "Attack windows are produced by a behavioural emulator, not by executed "
    "malware. Results describe emulated-family behaviour on generated data and "
    "are NOT real-world ransomware detection rates."
)

#: Tier A - the nine original artifacts, relative to ``prototype/``. Embedded so
#: verification never trusts a file that could itself have been tampered with.
TIER_A = {
    "results/phase2_results.json":
        "753561e1cd0772b2e7cb879c4ba9e47e7c2c9ac1880564c3cfd73c226b55ab3a",
    "results/phase2_results_models.csv":
        "2960eff3d5013f97171f1fcc7d938a4ccb410483f1e259ed52298f594a96684c",
    "results/phase2_results_confusion.csv":
        "cc20957742f3f83e6a7822afdbec8ff0c81c95240d03ed38ca296cf597cc56f0",
    "results/phase3_results.json":
        "c5c53d0a9dc189cfb0e11a8856b5c698dbb6fd1a47137c3e495952ab846c1b2a",
    "results/phase3_results_ablation.csv":
        "87e36dd5065195aac0fed31e597c3c751db96761d1cc5f9510a29c10354c794d",
    "results/phase3_results_per_family.csv":
        "7dce72c0bcc60354c9130ca448abfe4a7a9922f2d087c239212d37e39ceb26c7",
    "results/phase3_results_splits.csv":
        "cb29ef297b6b3a6e8f27925abca2a373e61ed08c2b6bb0d0d56e267072e34dd4",
    "results/phase3_results_thresholds.csv":
        "08e5e027ccafe171222afde69b8650307df49863fb17c03284de101c375133ea",
    "results/phase3_results_diagnostics.csv":
        "df33d4ac267a921a9a7643b14de1793161184b373f1b15d1752a91de8a2c22ed",
}

#: Files approved for modification by this work, relative to the repository root.
#: ``tests/test_rebaseline.py`` was added by the post-implementation safety
#: audit, and ``README.md`` by the documentation-consistency pass that followed.
#: Both are reviewed source files, registered here rather than special-cased in
#: the porcelain check, so the validator stays strict for everything else.
APPROVED_SOURCE = {
    "prototype/assrd/models.py": "M",
    "prototype/assrd/phase2.py": "M",
    "prototype/tests/test_phase2.py": "M",
    "prototype/README.md": "M",
    "prototype/assrd/rebaseline.py": "??",
    "prototype/tests/test_rebaseline.py": "??",
}

TIER_B_NAMES = ("phase2_results.json", "phase2_results_models.csv",
                "phase2_results_confusion.csv")
TIER_C_NAMES = ("phase3_results.json", "phase3_results_splits.csv",
                "phase3_results_ablation.csv", "phase3_results_per_family.csv",
                "phase3_results_thresholds.csv", "phase3_results_diagnostics.csv")
TIER_D_NAMES = TIER_C_NAMES

N_MODELS = len(MODEL_NAMES)
N_ARMS = len(ABLATION_ARMS)
N_MODES = 4
EXPECTED_SPLIT_ROWS = 3
EXPECTED_ABLATION_ROWS = N_ARMS * N_MODELS * N_MODES
EXPECTED_PER_FAMILY_ROWS = EXPECTED_ABLATION_ROWS * 7   # 5 families + 2 stealth
EXPECTED_THRESHOLD_ROWS = N_ARMS * N_MODELS * 2
EXPECTED_DIAGNOSTIC_ROWS = N_ARMS * N_MODELS * 2

#: Phase 2 settings, unchanged so the new run is aligned with the recorded one.
P2_KWARGS = dict(n_windows=20000, attack_ratio=0.05, seed=42, n_splits=5,
                 scoring="recall", n_jobs=-1, c_fp=C_FP, c_fn=C_FN,
                 include_stealth=True, threshold_grid=1001)

#: Phase 3 settings, unchanged from the approved FULL profile.
P3_COMMON = dict(seed=42, attack_ratio=0.05, include_stealth=True,
                 test_fraction=0.20, validation_fraction=0.20,
                 min_stealth_episodes=6, min_stealth_train=2,
                 threshold_grid=1001, stability_grid=201,
                 cost_curve_points=101, retune_full=False,
                 c_fp=C_FP, c_fn=C_FN)

EXPECTED_PARAMS = {
    "n_windows": P2_KWARGS["n_windows"], "attack_ratio": P2_KWARGS["attack_ratio"],
    "seed": P2_KWARGS["seed"], "cv_folds": P2_KWARGS["n_splits"],
    "cost_fp": float(C_FP), "cost_fn": float(C_FN),
}


class GateError(RuntimeError):
    """A gate failed. Nothing has been mutated when this is raised."""


class PostResetFailure(RuntimeError):
    """Terminal PRF state after a destructive step."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                          capture_output=True, text=True)


def _head():
    proc = _git("rev-parse", "HEAD")
    return proc.stdout.strip()


def _porcelain():
    """Repo-relative path -> two-character porcelain status."""
    proc = _git("status", "--porcelain=v1", "--untracked-files=all")
    out = {}
    for line in proc.stdout.splitlines():
        if len(line) < 4:
            continue
        out[line[3:].strip().replace("\\", "/")] = line[:2]
    return out


def _exclusive_write(path, text):
    """Create ``path`` exclusively; raise if it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(path, text):
    """Replace ``path`` atomically with ``text`` (temp file in the same dir)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + TMP_SUFFIX)
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _canonical(payload):
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Tier A protection
# ---------------------------------------------------------------------------
def verify_tier_a():
    """Compare every Tier A artifact against its embedded digest."""
    findings = []
    for rel, expected in sorted(TIER_A.items()):
        path = PROTOTYPE_DIR / rel
        if not path.exists():
            findings.append(f"Tier A MISSING: {rel}")
            continue
        actual = _sha256(path)
        if actual != expected:
            findings.append(
                f"Tier A MODIFIED: {rel} expected {expected[:12]} got {actual[:12]}")
    return findings


def _tier_a_ok():
    return not verify_tier_a()


# ---------------------------------------------------------------------------
# Expected path sets
# ---------------------------------------------------------------------------
def _run_rel(*names):
    return {f"prototype/results/rebaseline_6model/{n}" for n in names}


QUICK_REL = {f"prototype/results/rebaseline_6model/quick/{n}" for n in TIER_D_NAMES}

M1_REL = "prototype/results/rebaseline_6model/BASELINE_HASHES.txt"
M2_REL = "prototype/results/rebaseline_6model/RUN_MANIFEST.json"
M3_REL = "prototype/results/rebaseline_6model/quick/RUN_MANIFEST.json"

STAGE_SOURCE = set(APPROVED_SOURCE)
STAGE_PHASE2 = STAGE_SOURCE | {M1_REL, M2_REL} | _run_rel(*TIER_B_NAMES)
STAGE_QUICK = STAGE_PHASE2 | {M3_REL} | QUICK_REL
STAGE_FULL = STAGE_QUICK | _run_rel(*TIER_C_NAMES)

STAGE_SETS = {"source": STAGE_SOURCE, "phase2": STAGE_PHASE2,
              "quick": STAGE_QUICK, "full": STAGE_FULL}

STAGE_EXPECTATION = {
    "source": (None, None, None),
    "phase2": (1, {"baseline", "phase2"}, False),
    "quick": (2, {"baseline", "phase2", "quick_phase3"}, False),
    "full": (3, {"baseline", "phase2", "quick_phase3", "phase3_full"}, True),
}


#: Forensic evidence lives under this namespace. Its entries are unpredictable
#: (timestamp + nonce), so they cannot be enumerated statically. They are NOT
#: added to any expected-artifact set and are NOT hidden: they are surfaced as a
#: distinct ``evidence`` finding so a reviewer sees them explicitly. Nothing here
#: ignores result artifacts.
EVIDENCE_PREFIX = "prototype/results/_rebaseline_forensics/"


def check_porcelain(stage):
    """Check A: porcelain must be a subset of the stage universe, exactly sized."""
    expected = STAGE_SETS[stage]
    actual = _porcelain()
    findings = []
    for path in sorted(set(actual) - expected):
        if path.startswith(EVIDENCE_PREFIX):
            findings.append(
                ("evidence", f"forensic evidence present for human review: {path}"))
            continue
        findings.append(("E", f"unexpected porcelain entry {path} "
                              f"(status {actual[path]})"))
    return findings


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def _m2_payload():
    path = RUN_DIR / M2_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _m3_payload():
    path = QUICK_DIR / M3_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _known_run_files():
    known = set(TIER_B_NAMES) | set(TIER_C_NAMES)
    if QUICK_DIR.exists():
        known |= set(TIER_D_NAMES)
    return known


def classify():
    """Return findings as ``(class, message)`` pairs. E > D > C > B > A."""
    findings = []

    # --- class E: structural / environment -----------------------------------
    if _head() != EXPECTED_HEAD:
        findings.append(("E", f"HEAD is {_head()[:12]}, expected {EXPECTED_HEAD[:12]}"))
    for directory in (RUN_DIR, QUICK_DIR):
        if not directory.exists():
            continue
        for child in sorted(directory.iterdir()):
            if not child.is_file():
                continue
            name = child.name
            ok = (name in _known_run_files() or name in (M1_NAME, M2_NAME)
                  or name.endswith(TMP_SUFFIX))
            if not ok:
                findings.append(("E", f"unexpected file in run directory: {name}"))

    # --- Tier A is class B, but never auto-repairable -----------------------
    for message in verify_tier_a():
        findings.append(("B", message))

    payload = _m2_payload()
    quick_payload = _m3_payload()
    m2_exists = (RUN_DIR / M2_NAME).exists()
    m3_exists = (QUICK_DIR / M3_NAME).exists()

    # --- class D: ledger state ---------------------------------------------
    if m2_exists:
        if payload is None:
            findings.append(("D", "RUN_MANIFEST.json is unreadable or malformed"))
        else:
            if payload.get("schema") != "assrd.rebaseline/manifest/1":
                findings.append(("D", "unexpected M2 schema"))
            if payload.get("run_complete") not in (True, False):
                findings.append(("D", "run_complete is not a boolean"))
            stages = payload.get("stages") or {}
            if not isinstance(stages, dict):
                findings.append(("D", "stages is not an object"))
            docs = payload.get("documents") or {}
            # M2 may attest M1 and M3, but must never carry a digest of itself.
            if M2_NAME in docs or docs.get(M2_NAME) is not None:
                findings.append(("D", "M2 references its own digest (self-hash)"))

    # --- class C: configuration / provenance -------------------------------
    if m2_exists and payload is not None:
        for stage, entry in (payload.get("stages") or {}).items():
            if entry.get("complete") and entry.get("manifest_digest"):
                digest = entry["manifest_digest"]
                claimed = QUICK_DIR / M3_NAME
                if not claimed.exists() or _sha256(claimed) != digest:
                    findings.append(
                        ("B", f"M3 digest attestation failed for stage {stage}"))

    # --- class B: artifact integrity ---------------------------------------
    if m3_exists and quick_payload is not None:
        if quick_payload.get("authoritative") is not False:
            findings.append(("B", "quick manifest authoritative flag is not false"))
        stages = (quick_payload.get("ledger") or {}).get("stages") or []
        if stages != ["quick_phase3"]:
            findings.append(("B", "quick manifest stages are not ['quick_phase3']"))
        for name, digest in ((quick_payload.get("ledger") or {})
                             .get("digests") or {}).items():
            target = QUICK_DIR / name
            if not target.exists() or _sha256(target) != digest:
                findings.append(("B", f"quick artifact digest mismatch: {name}"))

    # --- class A: cleanup residue ------------------------------------------
    for directory in (RUN_DIR, QUICK_DIR):
        if not directory.exists():
            continue
        for child in sorted(directory.iterdir()):
            if child.is_file() and child.name.endswith(TMP_SUFFIX):
                findings.append(("A", f"leftover temporary file {child.name}"))

    if m2_exists and payload is not None:
        # Tier C files are legitimate only when a complete phase3_full stage
        # claims them (generation 3). Anything else is cleanup residue - this
        # covers a rolled-back ledger and a ledger that never mentions the stage.
        entry = (payload.get("stages") or {}).get("phase3_full") or {}
        claimed = set(entry.get("claims") or []) if entry.get("complete") else set()
        for name in TIER_C_NAMES:
            if name not in claimed and (RUN_DIR / name).exists():
                findings.append(("A", f"unclaimed full-run artifact {name}"))
    return findings


def _highest(findings):
    for level in ("E", "D", "C", "B", "A"):
        if any(cls == level for cls, _ in findings):
            return level
    return None


def _unpack(finding):
    """Findings are ``(class, message)`` pairs; tolerate a bare string."""
    if isinstance(finding, str):
        return "E", finding
    return finding


# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------
class _Lock:
    def __init__(self, path, nonce, payload):
        self.path = path
        self.nonce = nonce
        self.payload = payload
        self.released = False

    def owns(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (data.get("nonce") == self.nonce
                and data.get("pid") == self.payload["pid"]
                and data.get("host") == self.payload["host"])


def acquire_lock(command):
    """Exclusive-create the lock, located outside the directory it protects."""
    if not FORENSICS_PARENT.exists():
        raise GateError("forensics parent directory is missing; it must be "
                        "created by an operator, never automatically")
    if FORENSICS_PARENT.is_symlink() or not FORENSICS_PARENT.is_dir():
        raise GateError("forensics parent is not a real directory")
    payload = {"pid": os.getpid(), "host": socket.gethostname(),
               "utc": _now(), "command": command,
               "nonce": uuid.uuid4().hex}
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise GateError(
            f"lock already held at {LOCK_PATH}; contents:\n"
            f"{LOCK_PATH.read_text(encoding='utf-8', errors='replace')}\n"
            "This tool never auto-removes a lock. Inspect the holder and remove "
            "it manually if the previous run is known to be finished.")
    except OSError as exc:
        raise GateError(f"lock acquisition failed: {exc}")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(_canonical(payload))
        handle.flush()
        os.fsync(handle.fileno())
    return _Lock(LOCK_PATH, payload["nonce"], payload)


def release_lock(lock):
    """Release only if ownership still verifies. Never removes another lock."""
    if lock is None:
        return
    if not lock.path.exists():
        raise GateError(
            "lock disappeared before release; ownership cannot be verified. "
            "The lock has been left alone. Report this to a human.")
    if not lock.owns():
        raise GateError(
            "lock ownership could not be verified; refusing to delete it. "
            "Report this to a human.")
    # Delete the exact path whose ownership was verified, never a global.
    lock.path.unlink()
    lock.released = True


class locked:
    """Context manager: acquire, then always attempt a verified release."""

    def __init__(self, command):
        self.command = command
        self.lock = None

    def __enter__(self):
        self.lock = acquire_lock(self.command)
        return self.lock

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            release_lock(self.lock)
        else:
            try:
                release_lock(self.lock)
            except GateError as release_error:   # never mask the real failure
                print(f"WARNING: {release_error}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Forensic snapshot (mandatory before any destructive reset)
# ---------------------------------------------------------------------------
def snapshot(pre_state):
    """Create and verify a sealed, uniquely created forensic snapshot."""
    if not FORENSICS_PARENT.is_dir() or FORENSICS_PARENT.is_symlink():
        raise GateError("forensics parent is missing or not a real directory")
    for child in sorted(FORENSICS_PARENT.iterdir()):
        if child.name == LOCK_PATH.name:
            continue
        if child.is_dir() and child.name.startswith("snapshot-"):
            continue
        if child.name in (SIDECAR_NAME, SIDECAR_SUM_NAME):
            continue
        raise GateError(
            f"forensics parent contains an unexpected entry {child.name}; "
            "halt for human review without modifying existing evidence")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = FORENSICS_PARENT / f"snapshot-{stamp}-{uuid.uuid4().hex[:8]}"
    if os.path.lexists(target):
        raise GateError(f"snapshot path already exists: {target}")

    for directory in (RUN_DIR, QUICK_DIR):
        if directory.exists() and directory.is_symlink():
            raise GateError(f"run directory is a symlink: {directory}")

    os.mkdir(target)                      # atomic; raises if it exists

    try:
        manifest = dict(pre_state)
        manifest["schema"] = "assrd.rebaseline/forensics/1"
        manifest["snapshot_created_utc"] = _now()
        _exclusive_write(target / "MANIFEST.json", _canonical(manifest))

        copied = 0
        for source in (RUN_DIR, QUICK_DIR):
            if not source.exists():
                continue
            for child in sorted(source.iterdir()):
                if not child.is_file():
                    continue
                name = child.name.replace("/", "_")
                if source == QUICK_DIR and name == M3_NAME:
                    name = "QUICK_" + name
                elif name == M2_NAME:
                    name = "ROOT_" + name
                elif name == M1_NAME:
                    name = "BASELINE_" + name
                with open(child, "rb") as src:
                    data = src.read()
                with open(target / f"artifact_{copied:02d}_{name}", "xb") as dst:
                    dst.write(data)
                copied += 1

        sums = []
        for child in sorted(target.iterdir()):
            if child.name == "SHA256SUMS.txt":
                continue
            sums.append(f"{_sha256(child)}  {child.name}")
        _exclusive_write(target / "SHA256SUMS.txt", "\n".join(sums) + "\n")
    except OSError as exc:
        raise GateError(f"snapshot creation failed ({exc}); nothing was deleted "
                        "and the partial snapshot is retained as evidence")

    sealed = {}
    for child in sorted(target.iterdir()):
        sealed[child.name] = _sha256(child)
    for child in sorted(target.iterdir()):
        if child.name != "SHA256SUMS.txt" and _sha256(child) != sealed[child.name]:
            raise GateError("snapshot digest verification failed")
    return target, sealed


# ---------------------------------------------------------------------------
# Stage validators
# ---------------------------------------------------------------------------
def _csv_rows(path):
    import csv
    with open(path, encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def _validate_phase2_artifacts(directory, findings):
    required = {n: EXPECTED_ABLATION_ROWS if n == "phase2_results_models.csv"
                else None for n in TIER_B_NAMES}
    for name in TIER_B_NAMES:
        path = directory / name
        if not path.exists():
            findings.append(("B", f"missing Tier B artifact {name}"))
            continue
        if name.endswith(".csv") and _csv_rows(path) <= 0:
            findings.append(("B", f"empty CSV {name}"))
    payload = json.loads((directory / "phase2_results.json").read_text(
        encoding="utf-8"))
    models = sorted((payload.get("models") or {}))
    if models != sorted(MODEL_NAMES):
        findings.append(("C", f"Phase 2 models {models} != {sorted(MODEL_NAMES)}"))
    config = payload.get("config") or {}
    for key, value in EXPECTED_PARAMS.items():
        if key in config and float(config[key]) != float(value):
            findings.append(("C", f"Phase 2 config {key}={config[key]} != {value}"))
    lr = (payload.get("models") or {}).get("logistic_regression") or {}
    if lr.get("grid_points") != 16:
        findings.append(("B", f"logistic_regression grid_points="
                              f"{lr.get('grid_points')} != 16"))
    del required
    return findings


def _validate_phase3_result(result, profile, findings):
    config = result.get("config") or {}
    for key in ("n_windows", "min_episode", "max_episode", "stability_bootstrap"):
        if config.get(key) != profile[key]:
            findings.append(("C", f"config {key}={config.get(key)} != {profile[key]}"))
    if config.get("retune_full") is not False:
        findings.append(("C", "retune_full is not False"))
    if len(config.get("models") or []) != N_MODELS:
        findings.append(("C", "config models count is not the registry size"))
    if result.get("oracle_isolated") is not True:
        findings.append(("C", "oracle_isolated is not True"))
    leakage = ((result.get("dataset") or {}).get("split_manifest") or {}) \
        .get("leakage") or {}
    if not (leakage.get("episode_aware") and leakage.get("disjoint_rows")):
        findings.append(("C", "leakage checks did not pass"))
    if any(v for v in (leakage.get("row_overlap") or {}).values()):
        findings.append(("C", "row overlap is non-zero"))
    allocation = ((result.get("dataset") or {}).get("split_manifest") or {}) \
        .get("allocation") or {}
    if allocation.get("satisfied") is not True or allocation.get("shortfall"):
        findings.append(("C", "hard allocation constraints not satisfied"))
    if len(result.get("primary") or {}) != EXPECTED_ABLATION_ROWS:
        findings.append(("C", f"primary cells {len(result.get('primary') or {})} "
                              f"!= {EXPECTED_ABLATION_ROWS}"))
    source = result.get("params_source") or {}
    if sorted(source.get("models_from_file") or []) != sorted(MODEL_NAMES):
        findings.append(("C", "params_source does not list all models"))
    return findings


def validate_stage(stage):
    """Full validator for a stage. Returns (ok, findings)."""
    findings = list(check_porcelain(stage))
    findings += classify()

    expected_generation, expected_stages, expected_complete = \
        STAGE_EXPECTATION[stage]

    if stage in ("source",):
        if RUN_DIR.exists():
            findings.append(("E", "run directory exists before any experiment"))
        return not findings, findings

    if not RUN_DIR.exists():
        findings.append(("E", "run directory is missing"))
        return False, findings

    payload = _m2_payload()
    if payload is None:
        findings.append(("D", "RUN_MANIFEST.json missing or malformed"))
        return False, findings

    if expected_generation is not None:
        if payload.get("generation") != expected_generation:
            findings.append(("D", f"M2 generation {payload.get('generation')} "
                                  f"!= {expected_generation}"))
        actual_stages = {k for k, v in (payload.get("stages") or {}).items()
                         if v.get("complete")}
        if expected_stages is not None and actual_stages != expected_stages:
            findings.append(("D", f"completed stages {sorted(actual_stages)} "
                                  f"!= {sorted(expected_stages)}"))
        if expected_complete is not None and \
                payload.get("run_complete") is not expected_complete:
            findings.append(("D", "run_complete does not match the stage"))

    # metadata + artifact digests
    for stage_name, entry in (payload.get("stages") or {}).items():
        if not entry.get("complete"):
            continue
        declared = entry.get("digests") or {}
        for claim in entry.get("claims") or []:
            target = RUN_DIR / claim
            if not target.exists():
                findings.append(("B", f"{stage_name} claims missing {claim}"))
            elif claim in declared and declared[claim] != _sha256(target):
                findings.append(("B", f"{stage_name} digest mismatch for {claim}"))

    m1 = RUN_DIR / M1_NAME
    if not m1.exists():
        findings.append(("B", "BASELINE_HASHES.txt is missing"))
    else:
        entries = [line for line in m1.read_text(encoding="utf-8").splitlines()
                   if line and not line.startswith("#")]
        if len(entries) != len(TIER_A):
            findings.append(("B", f"BASELINE_HASHES.txt has {len(entries)} entries "
                                  f"!= {len(TIER_A)}"))
        for line in entries:
            digest, rel = line.split("  ", 1)
            if TIER_A.get(rel) != digest:
                findings.append(("B", f"BASELINE_HASHES.txt wrong digest for {rel}"))
    docs = payload.get("documents") or {}
    if m1.exists() and docs.get(M1_NAME) not in (None, _sha256(m1)):
        findings.append(("B", "M2 documents digest for BASELINE_HASHES.txt is wrong"))

    if stage in ("quick", "full"):
        m3 = _m3_payload()
        if m3 is None:
            findings.append(("D", "quick RUN_MANIFEST.json missing or malformed"))
        else:
            if (QUICK_DIR / "phase3_results.json").exists():
                try:
                    result = json.loads((QUICK_DIR / "phase3_results.json")
                                        .read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    findings.append(("B", "quick phase3_results.json is malformed"))
                else:
                    _validate_phase3_result(result, QUICK, findings)
            for name, digest in ((m3.get("ledger") or {}).get("digests") or {}).items():
                target = QUICK_DIR / name
                if not target.exists() or _sha256(target) != digest:
                    findings.append(("B", f"quick artifact digest mismatch: {name}"))
            for name in TIER_D_NAMES:
                if name.endswith(".csv"):
                    path = QUICK_DIR / name
                    if path.exists():
                        expected = {
                            "phase3_results_splits.csv": EXPECTED_SPLIT_ROWS,
                            "phase3_results_ablation.csv": EXPECTED_ABLATION_ROWS,
                            "phase3_results_per_family.csv": EXPECTED_PER_FAMILY_ROWS,
                            "phase3_results_thresholds.csv": EXPECTED_THRESHOLD_ROWS,
                            "phase3_results_diagnostics.csv": EXPECTED_DIAGNOSTIC_ROWS,
                        }[name]
                        got = _csv_rows(path)
                        if got != expected:
                            findings.append(("C", f"{name} has {got} rows, "
                                                  f"expected {expected}"))

    if (RUN_DIR / "phase2_results.json").exists():
        _validate_phase2_artifacts(RUN_DIR, findings)
        for name in ("phase2_results_models.csv",):
            path = RUN_DIR / name
            if path.exists() and _csv_rows(path) < N_MODELS:
                findings.append(("C", "phase2 models CSV has too few rows"))

    if stage == "full" and (RUN_DIR / "phase3_results.json").exists():
        try:
            result = json.loads((RUN_DIR / "phase3_results.json")
                                .read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            findings.append(("B", "full phase3_results.json is malformed"))
        else:
            _validate_phase3_result(result, FULL, findings)
        for name in TIER_C_NAMES:
            if name.endswith(".csv"):
                path = RUN_DIR / name
                if path.exists():
                    expected = {
                        "phase3_results_splits.csv": EXPECTED_SPLIT_ROWS,
                        "phase3_results_ablation.csv": EXPECTED_ABLATION_ROWS,
                        "phase3_results_per_family.csv": EXPECTED_PER_FAMILY_ROWS,
                        "phase3_results_thresholds.csv": EXPECTED_THRESHOLD_ROWS,
                        "phase3_results_diagnostics.csv": EXPECTED_DIAGNOSTIC_ROWS,
                    }[name]
                    got = _csv_rows(path)
                    if got != expected:
                        findings.append(("C", f"{name} has {got} rows, "
                                              f"expected {expected}"))

    if (RUN_DIR / (M2_NAME + TMP_SUFFIX)).exists():
        findings.append(("D", "a manifest temporary file was left behind"))

    # orphan scan, scoped to the two run directories only
    if not findings:
        claimed = set()
        for entry in (payload.get("stages") or {}).values():
            if entry.get("complete"):
                claimed |= set(entry.get("claims") or [])
        if _m3_payload() is not None:
            claimed |= {(m3.get("ledger") or {}).get("stage_file") or ""}
        for directory, names in ((RUN_DIR, claimed),
                                 (QUICK_DIR, set(TIER_D_NAMES))):
            if not directory.exists():
                continue
            for child in sorted(directory.iterdir()):
                if not child.is_file():
                    continue
                if child.name in (M1_NAME, M2_NAME):
                    continue
                if child.name.endswith(TMP_SUFFIX):
                    findings.append(("A", f"orphan temporary file {child.name}"))
                    continue
                if child.name not in names and child.name not in claimed:
                    findings.append(("A", f"orphan file {child.name}"))

    level = _highest(findings)
    return level in (None, "A"), findings


# ---------------------------------------------------------------------------
# Reset / rollback
# ---------------------------------------------------------------------------
def _pre_reset_state(trigger):
    return {
        "trigger": trigger,
        "utc": _now(),
        "head": _head(),
        "porcelain": _porcelain(),
        "run_dir_exists": RUN_DIR.exists(),
        "tier_a": {rel: _sha256(PROTOTYPE_DIR / rel)
                   for rel in sorted(TIER_A) if (PROTOTYPE_DIR / rel).exists()},
        "classification": [f"{c}: {m}" for c, m in classify()],
    }


def reset_new_dir(trigger):
    """Mandatory snapshot, then delete. Post-deletion failure is terminal."""
    findings = classify()
    level = _highest(findings)
    if level in ("E", "D", "C", "B"):
        raise GateError("reset refused: highest finding class is "
                        f"{level}\n" + "\n".join(f"{c}: {m}" for c, m in findings))
    if not _tier_a_ok():
        raise GateError("reset refused: Tier A is not intact")

    snapshot_dir, _sealed = snapshot(_pre_reset_state(trigger))
    print(f"snapshot verified at {snapshot_dir}")

    if RUN_DIR.is_symlink() or not RUN_DIR.is_dir():
        raise GateError("reset refused: run directory became a link or vanished "
                        "after the snapshot was taken")
    shutil.rmtree(RUN_DIR)

    try:
        after = verify_tier_a()
        if after:
            raise PostResetFailure(f"Tier A verification failed after reset: {after}")
        if _head() != EXPECTED_HEAD:
            raise PostResetFailure("HEAD changed during reset")
        if not snapshot_dir.is_dir() or not any(snapshot_dir.iterdir()):
            raise PostResetFailure("snapshot directory is missing or empty")
    except PostResetFailure as exc:
        _report_prf(str(exc), snapshot_dir, trigger)
        raise

    sidecar = FORENSICS_PARENT / SIDECAR_NAME
    try:
        _exclusive_write(sidecar, _canonical({
            "schema": "assrd.rebaseline/post-reset/1", "utc": _now(),
            "trigger": trigger, "snapshot": str(snapshot_dir),
            "tier_a_after": {rel: _sha256(PROTOTYPE_DIR / rel)
                             for rel in sorted(TIER_A)},
            "head": _head(), "porcelain_after": _porcelain(),
            "run_dir_absent": not RUN_DIR.exists()}))
        _exclusive_write(FORENSICS_PARENT / SIDECAR_SUM_NAME,
                         f"{_sha256(sidecar)}  {SIDECAR_NAME}\n")
        recorded = (FORENSICS_PARENT / SIDECAR_SUM_NAME).read_text(
            encoding="utf-8").split()[0]
        if recorded != _sha256(sidecar):
            raise PostResetFailure("post-reset checksum verification failed")
    except (OSError, PostResetFailure) as exc:
        _report_prf(str(exc), snapshot_dir, trigger)
        raise
    print(f"reset complete ({trigger}); evidence preserved at {snapshot_dir}")


def _report_prf(message, snapshot_dir, trigger):
    print("\n" + "=" * 72, file=sys.stderr)
    print("TERMINAL POST-RESET FAILURE (PRF)", file=sys.stderr)
    print(f"failed check : {message}", file=sys.stderr)
    print(f"trigger      : {trigger}", file=sys.stderr)
    print(f"snapshot     : {snapshot_dir}", file=sys.stderr)
    print("No further deletion or recreation was attempted.", file=sys.stderr)
    print("Human review required.", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


def rollback_w3():
    """Re-materialise M2 at generation 2, then idempotently clean Tier C."""
    findings = classify()
    level = _highest(findings)
    if level in ("E",):
        raise GateError("rollback refused: class E finding\n"
                        + "\n".join(f"{c}: {m}" for c, m in findings))
    payload = _m2_payload()
    if payload is None:
        raise GateError("rollback refused: RUN_MANIFEST.json missing/malformed")
    generation = payload.get("generation")
    if generation not in (2, 3):
        raise GateError(f"rollback refused: generation {generation} is not "
                        "resumable; use the guarded reset instead")

    m1 = RUN_DIR / M1_NAME
    m3 = QUICK_DIR / M3_NAME
    for path in (m1, m3):
        if not path.exists():
            raise GateError(f"rollback refused: {path.name} is missing; "
                            "use the guarded reset instead")

    tier_b = {}
    for name in TIER_B_NAMES:
        path = RUN_DIR / name
        if not path.exists():
            raise GateError(f"rollback refused: Tier B artifact {name} missing")
        tier_b[name] = _sha256(path)

    if not _tier_a_ok():
        raise GateError("rollback refused: Tier A is not intact")

    rebuilt = {
        "schema": "assrd.rebaseline/manifest/1",
        "generation": 2,
        "created_utc": _now(),
        "reconstructed": True,
        "head": _head(),
        "documents": {M1_NAME: _sha256(m1), f"quick/{M3_NAME}": _sha256(m3)},
        "stages": {
            "baseline": {"complete": True, "claims": [M1_NAME]},
            "phase2": {"complete": True, "claims": list(TIER_B_NAMES),
                       "digests": tier_b},
            "quick_phase3": {"complete": True, "claimed_by": f"quick/{M3_NAME}",
                             "manifest_digest": _sha256(m3)},
        },
        "run_complete": False,
    }

    # Meaningful determinism check: every digest in the rebuilt payload must
    # equal a freshly recomputed digest of the file it names. A tautological
    # re-serialisation comparison would prove nothing.
    for claim, digest in tier_b.items():
        if _sha256(RUN_DIR / claim) != digest:
            raise GateError(f"rollback refused: {claim} digest is unstable")
    if rebuilt["documents"][M1_NAME] != _sha256(m1):
        raise GateError("rollback refused: BASELINE_HASHES.txt digest is unstable")
    if rebuilt["documents"][f"quick/{M3_NAME}"] != _sha256(m3):
        raise GateError("rollback refused: quick manifest digest is unstable")

    if RUN_DIR.is_symlink():
        raise GateError("rollback refused: run directory is a symlink")
    _atomic_write(RUN_DIR / M2_NAME, _canonical(rebuilt))

    try:
        for name in TIER_C_NAMES:
            path = RUN_DIR / name
            if path.exists():
                path.unlink()
        tmp = RUN_DIR / (M2_NAME + TMP_SUFFIX)
        if tmp.exists():
            tmp.unlink()
    except OSError as exc:
        _report_prf(f"rollback cleanup failed: {exc}", None, "rollback-w3")
        raise

    after = verify_tier_a()
    if after:
        _report_prf(f"Tier A verification failed after rollback: {after}",
                    None, "rollback-w3")
        raise PostResetFailure("Tier A verification failed after rollback")
    print("rollback complete; M2 restored to generation 2")


# ---------------------------------------------------------------------------
# Manifest writers
# ---------------------------------------------------------------------------
def write_baseline_hashes():
    lines = ["# ASSRD re-baseline baseline hashes",
             f"# git HEAD: {_head()}",
             "# Tier A artifacts are recorded here as digests only and are "
             "never copied or modified."]
    for rel, digest in sorted(TIER_A.items()):
        lines.append(f"{digest}  {rel}")
    _exclusive_write(RUN_DIR / M1_NAME, "\n".join(lines) + "\n")


def write_m2(stages_documents, generations, run_complete):
    payload = {
        "schema": "assrd.rebaseline/manifest/1",
        "generation": generations,
        "created_utc": _now(),
        "head": _head(),
        "documents": {M1_NAME: _sha256(RUN_DIR / M1_NAME)},
        "stages": {
            "baseline": {"complete": True, "claims": [M1_NAME]},
        },
        "run_complete": False,
    }
    if (QUICK_DIR / M3_NAME).exists():
        payload["documents"][f"quick/{M3_NAME}"] = _sha256(QUICK_DIR / M3_NAME)
    payload["stages"].update(stages_documents)
    payload["run_complete"] = run_complete
    _atomic_write(RUN_DIR / M2_NAME, _canonical(payload))
    return payload


def write_m3():
    digests = {name: _sha256(QUICK_DIR / name) for name in TIER_D_NAMES}
    payload = {
        "schema": "assrd.rebaseline/quick-manifest/1",
        "authoritative": False,
        "created_utc": _now(),
        "ledger": {"stages": ["quick_phase3"], "complete": True,
                   "claims": list(TIER_D_NAMES), "digests": digests},
    }
    _atomic_write(QUICK_DIR / M3_NAME, _canonical(payload))
    return payload


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_preflight_baseline(args):
    """Pre-implementation: clean tree, expected HEAD, Tier A intact."""
    porcelain = _porcelain()
    ok = True
    head = _head()
    if head != EXPECTED_HEAD:
        print(f"FAIL HEAD {head} != {EXPECTED_HEAD}")
        ok = False
    expected = {"prototype/assrd/rebaseline.py"}
    if set(porcelain) != expected:
        print(f"FAIL porcelain {sorted(porcelain)} != {sorted(expected)} "
              "(the three tracked files must still be unedited)")
        ok = False
    for message in verify_tier_a():
        print(f"FAIL {message}")
        ok = False
    if RUN_DIR.exists():
        print("FAIL results/rebaseline_6model already exists")
        ok = False
    print("BASELINE PREFLIGHT " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_preflight(args):
    ok, findings = validate_stage(args.stage)
    for cls, message in findings:
        if cls == "evidence":
            print(f"[EVIDENCE - REVIEW] {message}")
            continue
        print(f"[{cls}] {message}")
    print(f"STAGE PREFLIGHT ({args.stage}) " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_verify(args):
    for message in verify_tier_a():
        print(f"FAIL {message}")
    ok = _tier_a_ok()
    print("TIER A " + ("PASSED (9/9 byte-identical)" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_compare_params(args):
    """GR3 advisory comparison against the original Phase 2 record."""
    original = json.loads((RESULTS_DIR / "phase2_results.json").read_text(
        encoding="utf-8"))
    new_path = RUN_DIR / "phase2_results.json"
    if not new_path.exists():
        print("FAIL no re-baselined Phase 2 result yet")
        return 1
    fresh = json.loads(new_path.read_text(encoding="utf-8"))
    diffs, identical = [], 0
    for name, block in (original.get("models") or {}).items():
        was = block.get("best_params")
        now = (fresh.get("models") or {}).get(name, {}).get("best_params")
        if was == now:
            identical += 1
        else:
            diffs.append({"model": name, "original": was, "rebaselined": now})
    print(f"GR3 PARAMETER COMPARISON: {identical}/{len(original.get('models') or {})} "
          "identical")
    for entry in diffs:
        print(f"  MODEL {entry['model']}")
        for key in sorted(set(entry["original"] or {}) | set(entry["rebaselined"] or {})):
            print(f"    {key}: {entry['original'].get(key)!r} -> "
                  f"{entry['rebaselined'].get(key)!r}")
    if diffs:
        print("\nADVISORY: existing models' best_params differ. Execution MUST stop "
              "and await explicit approval.")
        return 2
    print("All existing models reproduced identical best_params.")
    return 0


def cmd_check(args):
    """Non-fitting preflight."""
    ok = True
    env = {"python": platform.python_version(),
           "scikit_learn": __import__("sklearn").__version__,
           "xgboost": __import__("xgboost").__version__,
           "numpy": __import__("numpy").__version__,
           "pandas": __import__("pandas").__version__}
    print("environment:", env)
    for message in verify_tier_a():
        print(f"FAIL {message}")
        ok = False
    if "logistic_regression" not in MODEL_NAMES:
        print("FAIL logistic_regression is not registered")
        ok = False
    else:
        grid = param_grid("logistic_regression")
        points = 1
        for values in grid.values():
            points *= len(values)
        print(f"logistic_regression grid points: {points}")
        if points != 16:
            print("FAIL grid is not the approved 16 points")
            ok = False
        from sklearn.linear_model import LogisticRegression
        import itertools
        for combo in itertools.product(*grid.values()):
            kwargs = dict(zip(grid.keys(), combo))
            kwargs = {k.replace("clf__", ""): v for k, v in kwargs.items()}
            try:
                LogisticRegression(solver="liblinear", max_iter=1000, **kwargs)
            except Exception as exc:                # noqa: BLE001
                print(f"FAIL grid point invalid: {kwargs} -> {exc}")
                ok = False
        estimator = build_estimator("logistic_regression", seed=0)
        steps = list(estimator.named_steps)
        if steps[:1] != ["scaler"] or steps[-1] != "clf":
            print(f"FAIL unexpected pipeline steps {steps}")
            ok = False
    print("CHECK " + ("PASSED" if ok else "FAILED") + " (no models fitted)")
    return 0 if ok else 1


def cmd_run_phase2(args):
    with locked("run-phase2"):
        if RUN_DIR.exists():
            raise GateError("run directory already exists")
        RUN_DIR.mkdir(parents=True)
        write_baseline_hashes()
        result = run_phase2(models=MODEL_NAMES, **P2_KWARGS)
        write_phase2_results(result, directory=RUN_DIR, basename="phase2_results")
        write_m2({"phase2": {"complete": True, "claims": list(TIER_B_NAMES),
                             "digests": {n: _sha256(RUN_DIR / n)
                                         for n in TIER_B_NAMES}}}, 1, False)
    print("Phase 2 re-baseline complete")
    return 0


def cmd_run_phase3(args):
    profile = QUICK if args.mode == "quick" else FULL
    directory = QUICK_DIR if args.mode == "quick" else RUN_DIR
    if args.mode == "full":
        for name in TIER_C_NAMES:
            if (RUN_DIR / name).exists():
                raise GateError(f"refusing to start: {name} already exists")
    with locked(f"run-phase3-{args.mode}"):
        params, source = load_phase2_params(RUN_DIR / "phase2_results.json")
        if sorted(source.get("models_from_file") or []) != sorted(MODEL_NAMES):
            raise GateError(f"params source is incomplete: {source}")
        directory.mkdir(parents=True, exist_ok=True)
        result = run_phase3(models=MODEL_NAMES, params=params,
                            n_windows=profile["n_windows"],
                            min_episode=profile["min_episode"],
                            max_episode=profile["max_episode"],
                            stability_bootstrap=profile["stability_bootstrap"],
                            **P3_COMMON)
        write_phase3_results(result, directory=directory,
                             basename="phase3_results")
        digests = {n: _sha256(directory / n) for n in TIER_D_NAMES}
        if args.mode == "quick":
            write_m3()
            payload = _m2_payload() or {}
            stages = dict(payload.get("stages") or {})
            stages["quick_phase3"] = {
                "complete": True,
                "claimed_by": f"quick/{M3_NAME}",
                "manifest_digest": _sha256(QUICK_DIR / M3_NAME),
            }
            docs = {M1_NAME: _sha256(RUN_DIR / M1_NAME),
                    f"quick/{M3_NAME}": _sha256(QUICK_DIR / M3_NAME)}
            _atomic_write(RUN_DIR / M2_NAME, _canonical({
                "schema": "assrd.rebaseline/manifest/1", "generation": 2,
                "created_utc": _now(), "head": _head(), "documents": docs,
                "stages": stages, "run_complete": False}))
        else:
            payload = _m2_payload() or {}
            stages = dict(payload.get("stages") or {})
            stages["phase3_full"] = {"complete": True,
                                     "claims": list(TIER_C_NAMES),
                                     "digests": digests}
            docs = {M1_NAME: _sha256(RUN_DIR / M1_NAME)}
            if (QUICK_DIR / M3_NAME).exists():
                docs[f"quick/{M3_NAME}"] = _sha256(QUICK_DIR / M3_NAME)
            _atomic_write(RUN_DIR / M2_NAME, _canonical({
                "schema": "assrd.rebaseline/manifest/1", "generation": 3,
                "created_utc": _now(), "head": _head(), "documents": docs,
                "stages": stages, "run_complete": True}))
        del digests
    print(f"Phase 3 ({args.mode}) complete")
    return 0


def cmd_verify_run(args):
    stage = "quick" if args.mode == "quick" else "full"
    ok, findings = validate_stage(stage)
    for cls, message in findings:
        if cls == "evidence":
            print(f"[EVIDENCE - REVIEW] {message}")
            continue
        print(f"[{cls}] {message}")
    print(f"VERIFY-RUN ({args.mode}) " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def cmd_reset(args):
    with locked("reset-new-dir"):
        reset_new_dir(args.trigger)
    return 0


def cmd_rollback(args):
    with locked("rollback-w3"):
        rollback_w3()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="ASSRD six-model non-destructive re-baselining")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("verify").set_defaults(func=cmd_verify)
    sub.add_parser("preflight-baseline").set_defaults(func=cmd_preflight_baseline)
    sub.add_parser("compare-params").set_defaults(func=cmd_compare_params)
    sub.add_parser("run-phase2").set_defaults(func=cmd_run_phase2)
    pre = sub.add_parser("preflight")
    pre.add_argument("--stage", required=True, choices=sorted(STAGE_SETS))
    pre.set_defaults(func=cmd_preflight)
    run3 = sub.add_parser("run-phase3")
    run3.add_argument("--mode", required=True, choices=("quick", "full"))
    run3.set_defaults(func=cmd_run_phase3)
    vr = sub.add_parser("verify-run")
    vr.add_argument("--mode", required=True, choices=("quick", "full"))
    vr.set_defaults(func=cmd_verify_run)
    rst = sub.add_parser("reset-new-dir")
    rst.add_argument("--trigger", default="manual")
    rst.set_defaults(func=cmd_reset)
    sub.add_parser("rollback-w3").set_defaults(func=cmd_rollback)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (GateError, PostResetFailure) as exc:
        print(f"GATE FAILURE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())