"""Phase 3 — constrained stratified episode allocation and composition manifest.

Why this exists
---------------
Phase 2 assigned attack episodes to partitions at random
(``assrd.data.split_by_episode``). On the seed-42 24-episode trace that placed
two episodes in validation and *zero* stealth windows there — measured, and
recorded as a limitation in ``prototype/README.md``. A per-family or
stealth-conditioned claim cannot be computed from two episodes.

This module replaces *random* episode assignment with a **constrained,
stratified allocator** and makes the resulting composition auditable. It does not
modify ``split_by_episode``; the Phase 2 path is untouched.

What is guaranteed, and what is not
-----------------------------------
* **family** — every family with at least ``min_family_episodes`` independent
  episodes is placed in *both* validation and test.
* **stealth** — each evaluation split receives at least ``min_stealth_episodes``
  stealth episodes, drawn from the stealth pool while leaving
  ``min_stealth_train`` of it in training.
* **(family, stealth) cells — NOT guaranteed.** There are only finitely many
  independent episodes. A cell holding four episodes can supply validation,
  test and training only by splitting 1/1/2. Cells are therefore *reported* with
  an explicit support flag; a thin cell is never silently guaranteed, and a
  missing one is never silently dropped.

If a constraint cannot be met, :func:`assign_episodes` raises
:class:`SplitConstraintError` carrying a machine-readable ``shortfall`` rather
than emitting a split that quietly violates it.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from assrd.data import FAMILY_NAMES

#: A family needs at least this many independent episodes before its presence in
#: validation and test can be guaranteed.
MIN_FAMILY_EPISODES = 3

#: Support-flag boundaries, fixed here before any result is seen. ``adequate``
#: means a comparative claim may be drawn; ``insufficient`` and ``absent`` mean
#: it may not.
SUPPORT_ADEQUATE = 5
SUPPORT_THIN = 2

SPLIT_NAMES = ("train", "validation", "test")


class SplitConstraintError(ValueError):
    """Raised when the allocator cannot satisfy a declared constraint.

    ``shortfall`` is a dict of ``{constraint: {...target, available, satisfied}}``
    so a caller (or a test) can see exactly which requirement failed and by how
    much, instead of inferring it from a generic error message.
    """

    def __init__(self, message, shortfall):
        super().__init__(message)
        self.shortfall = dict(shortfall)


def support_flag(n_episodes):
    """Map an independent-episode count to a declared support flag."""
    n = int(n_episodes)
    if n <= 0:
        return "absent"
    if n == 1:
        return "insufficient"
    if n < SUPPORT_ADEQUATE:
        return "thin"
    return "adequate"


def episode_frame(df):
    """One row per attack episode: family, stealth flag and window count."""
    attack = df[df["label"] == 1]
    frame = attack.groupby("episode_id").agg(
        family=("family", "first"),
        is_stealth=("is_stealth", "first"),
        n_windows=("label", "size"),
    )
    frame.index = frame.index.astype(int)
    return frame


def _shuffled(candidates, rng):
    pool = sorted(candidates)
    rng.shuffle(pool)
    return pool


def _select_episodes(candidates, need, coverage_families, stealth_min, meta, rng):
    """Greedily choose ``need`` episodes from ``candidates`` (mutated in place).

    Priority order — coverage beats proportion beats randomness:

    1. one episode of every ``coverage_families`` member, preferring a stealth
       episode when the stealth quota is still short;
    2. stealth episodes until the quota is met, spreading across families;
    3. proportional fill, always taking from the least-represented family.
    """
    selected = []

    def stealth_count():
        return sum(1 for e in selected if bool(meta.loc[e, "is_stealth"]))

    def family_count():
        counts = {}
        for e in selected:
            fam = meta.loc[e, "family"]
            counts[fam] = counts.get(fam, 0) + 1
        return counts

    # Phase 1 — family coverage.
    for fam in coverage_families:
        if len(selected) >= need:
            break
        if any(meta.loc[e, "family"] == fam for e in selected):
            continue
        pool = _shuffled((e for e in candidates if meta.loc[e, "family"] == fam), rng)
        if not pool:
            continue
        stealth_pool = [e for e in pool if bool(meta.loc[e, "is_stealth"])]
        prefer = stealth_pool if (stealth_pool and stealth_count() < stealth_min) else pool
        chosen = prefer[0]
        selected.append(chosen)
        candidates.discard(chosen)

    # Phase 2 — stealth quota.
    while stealth_count() < stealth_min and len(selected) < need:
        pool = _shuffled((e for e in candidates if bool(meta.loc[e, "is_stealth"])), rng)
        if not pool:
            break
        counts = family_count()
        pool.sort(key=lambda e: counts.get(meta.loc[e, "family"], 0))
        chosen = pool[0]
        selected.append(chosen)
        candidates.discard(chosen)

    # Phase 3 — proportional fill towards the least-represented family.
    while len(selected) < need and candidates:
        pool = _shuffled(candidates, rng)
        counts = family_count()
        pool.sort(key=lambda e: counts.get(meta.loc[e, "family"], 0))
        chosen = pool[0]
        selected.append(chosen)
        candidates.discard(chosen)

    return selected


def assign_episodes(df, *, test_fraction=0.20, validation_fraction=0.20, seed=42,
                    min_stealth_episodes=6, min_stealth_train=2,
                    min_family_episodes=MIN_FAMILY_EPISODES):
    """Allocate each attack episode to ``train``, ``validation`` or ``test``.

    Returns ``(assign, report)`` where ``assign`` maps ``episode_id -> split`` and
    ``report`` records every constraint that was checked, the target and the
    available supply. Raises :class:`SplitConstraintError` if any constraint is
    unsatisfiable, with the failing constraints in ``.shortfall``.
    """
    meta = episode_frame(df)
    ids = sorted(int(e) for e in meta.index)
    n_episodes = len(ids)
    if n_episodes == 0:
        raise SplitConstraintError("no attack episodes to allocate", {
            "episodes": {"target": 1, "available": 0, "satisfied": False}})

    n_test = max(1, int(round(n_episodes * test_fraction)))
    n_val = max(1, int(round(n_episodes * validation_fraction)))
    min_train = n_episodes - n_test - n_val

    family_counts = meta["family"].value_counts().to_dict()
    coverage_families = sorted(f for f, c in family_counts.items()
                               if c >= min_family_episodes)
    n_stealth = int(meta["is_stealth"].sum())
    stealth_needed = 2 * int(min_stealth_episodes) + int(min_stealth_train)

    constraints = {
        "total_episodes": {"target": n_test + n_val + 1, "available": n_episodes},
        "training_reserve": {"target": 1, "available": min_train},
        "family_coverage_slots": {"target": len(coverage_families),
                                  "available": min(n_val, n_test)},
        "stealth_episodes": {"target": stealth_needed, "available": n_stealth},
    }
    shortfall = {}
    for key, row in constraints.items():
        satisfied = row["available"] >= row["target"]
        row["satisfied"] = bool(satisfied)
        if not satisfied:
            shortfall[key] = dict(row)

    if shortfall:
        detail = "; ".join(
            f"{k}: need {v['target']}, have {v['available']}"
            for k, v in shortfall.items())
        raise SplitConstraintError(
            f"cannot satisfy split constraints ({detail}). Increase the trace "
            f"size/episode count or relax min_stealth_episodes/min_stealth_train.",
            shortfall)

    rng = random.Random(seed)
    candidates = set(ids)
    test_sel = _select_episodes(candidates, n_test, coverage_families,
                                min_stealth_episodes, meta, rng)
    val_sel = _select_episodes(candidates, n_val, coverage_families,
                               min_stealth_episodes, meta, rng)
    train_sel = sorted(candidates)

    assign = {e: "test" for e in test_sel}
    assign.update({e: "validation" for e in val_sel})
    for e in train_sel:
        assign.setdefault(e, "train")

    report = _allocation_report(meta, assign, constraints, n_val, n_test,
                                min_stealth_episodes, min_stealth_train,
                                coverage_families)
    if report["shortfall"]:
        raise SplitConstraintError(
            "allocator produced a split that violates its constraints: "
            + "; ".join(report["shortfall"]), report["shortfall"])
    return assign, report


def _allocation_report(meta, assign, constraints, n_val, n_test,
                       min_stealth_episodes, min_stealth_train, coverage_families):
    """Post-hoc verification that the realised allocation honours every promise."""
    by_split = {}
    for split in SPLIT_NAMES:
        ids = [e for e, s in assign.items() if s == split]
        by_split[split] = ids

    family_by_split = {
        split: {f: sum(1 for e in ids if meta.loc[e, "family"] == f)
                for f in FAMILY_NAMES}
        for split, ids in by_split.items()}
    stealth_by_split = {
        split: sum(1 for e in ids if bool(meta.loc[e, "is_stealth"]))
        for split, ids in by_split.items()}

    shortfall = []
    for split, need in (("validation", n_val), ("test", n_test)):
        if len(by_split[split]) < need:
            shortfall.append(f"{split} episodes {len(by_split[split])} < {need}")
        if stealth_by_split[split] < min_stealth_episodes:
            shortfall.append(
                f"{split} stealth {stealth_by_split[split]} < {min_stealth_episodes}")

    for split in ("validation", "test"):
        missing = [f for f in coverage_families if family_by_split[split][f] == 0]
        if missing:
            shortfall.append(f"{split} missing families {missing}")

    if stealth_by_split["train"] < min_stealth_train:
        shortfall.append(
            f"train stealth {stealth_by_split['train']} < {min_stealth_train}")
    if len(by_split["train"]) < 1:
        shortfall.append("train has no episodes")

    return {
        "seed_used_for_selection": True,
        "targets": {
            "validation_episodes": n_val,
            "test_episodes": n_test,
            "min_stealth_episodes_per_eval_split": min_stealth_episodes,
            "min_stealth_train": min_stealth_train,
            "guaranteed_families": coverage_families,
        },
        "constraints": constraints,
        "realised": {
            "episodes_by_split": {s: len(ids) for s, ids in by_split.items()},
            "stealth_episodes_by_split": stealth_by_split,
            "family_episodes_by_split": family_by_split,
        },
        "shortfall": shortfall,
        "satisfied": not shortfall,
    }


def stratification_shortfalls(train, validation, test, *, min_family_episodes):
    """List every stratum that is thin or absent, so it is reported, not hidden.

    Returns a list of ``{scope, split, group, episodes, flag}`` rows covering each
    family and each (family, stealth) cell. This is intentionally *observational*:
    cells are not required to appear everywhere, only to be disclosed.
    """
    rows = []
    for split, part in (("train", train), ("validation", validation), ("test", test)):
        attack = part[part["label"] == 1]
        eps = attack.groupby("episode_id")
        for fam in FAMILY_NAMES:
            fam_eps = eps.first()
            n = int((fam_eps["family"] == fam).sum())
            if n < SUPPORT_ADEQUATE:
                rows.append({"scope": "family", "split": split, "group": fam,
                             "episodes": n, "flag": support_flag(n)})
        for fam in FAMILY_NAMES:
            for stealth, label in ((True, "stealth"), (False, "non_stealth")):
                n = int(((eps.family.first() == fam)
                         & (eps.is_stealth.first() == stealth)).sum())
                if n < SUPPORT_ADEQUATE:
                    rows.append({"scope": "family_stealth", "split": split,
                                 "group": f"{fam}|{label}", "episodes": n,
                                 "flag": support_flag(n)})
    return rows


def benign_block_size(n_rows):
    """Contiguous-block size used to group benign windows during a split."""
    return max(1, int(n_rows) // 200)


def stratified_split(df, *, test_fraction=0.20, validation_fraction=0.20, seed=42,
                     min_stealth_episodes=6, min_stealth_train=2,
                     min_family_episodes=MIN_FAMILY_EPISODES):
    """Split ``df`` into ``(train, validation, test, report)``.

    Attack episodes come from :func:`assign_episodes`; benign windows are grouped
    into contiguous blocks and assigned with the same fractions, so adjacent
    benign windows do not straddle a boundary. ``report`` carries the allocation
    constraints and the realised composition.
    """
    assign, alloc = assign_episodes(
        df, test_fraction=test_fraction, validation_fraction=validation_fraction,
        seed=seed, min_stealth_episodes=min_stealth_episodes,
        min_stealth_train=min_stealth_train, min_family_episodes=min_family_episodes)

    rng = random.Random(seed + 1)
    episode_of = df["episode_id"].to_numpy()
    index_of = df["window_index"].to_numpy()
    block_size = benign_block_size(len(df))

    benign_keys = sorted({int(k) for k in index_of[episode_of < 0] // block_size})
    rng.shuffle(benign_keys)
    n_test_blocks = max(1, int(round(len(benign_keys) * test_fraction)))
    n_val_blocks = max(1, int(round(len(benign_keys) * validation_fraction)))
    test_blocks = set(benign_keys[:n_test_blocks])
    val_blocks = set(benign_keys[n_test_blocks:n_test_blocks + n_val_blocks])

    row_blocks = index_of // block_size
    is_attack = episode_of >= 0
    split_of_episode = np.array([assign.get(int(e), "train") for e in episode_of])

    is_test = np.where(is_attack, split_of_episode == "test",
                       np.isin(row_blocks, list(test_blocks)))
    is_val = np.where(is_attack, split_of_episode == "validation",
                      np.isin(row_blocks, list(val_blocks)))
    is_test = np.asarray(is_test, dtype=bool)
    is_val = np.asarray(is_val, dtype=bool)
    is_train = ~(is_test | is_val)

    frames = (
        df[is_train].reset_index(drop=True),
        df[is_val].reset_index(drop=True),
        df[is_test].reset_index(drop=True),
    )
    for name, part in zip(SPLIT_NAMES, frames):
        if int((part["label"] == 1).sum()) == 0:
            raise SplitConstraintError(
                f"degenerate split: {name} has no attack windows",
                {f"{name}_attack_windows": {"target": 1, "available": 0,
                                            "satisfied": False}})

    alloc["benign_block_size"] = block_size
    alloc["benign_blocks"] = {
        "total": len(benign_keys),
        "test": len(test_blocks),
        "validation": len(val_blocks),
    }
    return frames[0], frames[1], frames[2], alloc


def leakage_check(train, validation, test):
    """Assert the three partitions share no attack episode and no window row."""
    def episodes(part):
        return {int(e) for e in part.loc[part["label"] == 1, "episode_id"].tolist()}

    tr, va, te = episodes(train), episodes(validation), episodes(test)
    indices = {name: set(part["window_index"].tolist())
               for name, part in zip(SPLIT_NAMES, (train, validation, test))}
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


def bootstrap_groups(part, block_size):
    """Independent resampling units: attack episodes and benign blocks."""
    episode_of = part["episode_id"].to_numpy()
    index_of = part["window_index"].to_numpy()
    return np.where(episode_of >= 0, episode_of, -(index_of // block_size) - 1)


def _split_block(part, *, block_size, family_names):
    """Explicit per-split episode accounting for the manifest."""
    attack = part[part["label"] == 1]
    benign = part[part["label"] == 0]
    eps = attack.groupby("episode_id")
    fam_eps = eps.family.first() if len(attack) else pd.Series(dtype=object)
    stealth_eps = eps.is_stealth.first() if len(attack) else pd.Series(dtype=bool)

    family_episode_counts = {f: 0 for f in family_names}
    for fam, n in fam_eps.value_counts().items():
        family_episode_counts[fam] = int(n)

    family_row_counts = {f: 0 for f in family_names}
    for fam, n in attack["family"].value_counts().items():
        family_row_counts[fam] = int(n)

    cells = {}
    for fam in family_names:
        for stealth, label in ((True, "stealth"), (False, "non_stealth")):
            cells[f"{fam}|{label}"] = int(
                ((fam_eps == fam) & (stealth_eps == stealth)).sum())

    n_attack = int(len(attack))
    n_stealth_rows = int(attack["is_stealth"].sum())
    benign_blocks = int(len(set((benign["window_index"].to_numpy()
                                 // block_size).tolist()))) if len(benign) else 0

    return {
        "rows": int(len(part)),
        "benign_rows": int(len(benign)),
        "attack_rows": n_attack,
        "attack_fraction": round(float(n_attack / len(part)), 6) if len(part) else None,
        "attack_episodes": int(attack["episode_id"].nunique()),
        "benign_episodes": 0,
        "benign_episodes_note": (
            "not applicable: benign windows carry episode_id == -1 and are not "
            "episodes; see benign_managed_blocks"),
        "benign_managed_blocks": benign_blocks,
        "attack_episodes_by_family": family_episode_counts,
        "attack_episodes_stealth": int(stealth_eps.sum()),
        "attack_episodes_non_stealth": int(len(stealth_eps) - stealth_eps.sum()),
        "attack_rows_by_family": family_row_counts,
        "attack_rows_stealth": n_stealth_rows,
        "attack_rows_non_stealth": int(n_attack - n_stealth_rows),
        "cells_family_stealth": {
            "counts": cells,
            "present": sum(1 for v in cells.values() if v > 0),
            "of": len(cells),
        },
        "support_flags": {
            "families": {f: support_flag(n) for f, n in family_episode_counts.items()},
            "family_stealth_cells": {k: support_flag(v) for k, v in cells.items()},
        },
    }


def split_manifest(train, validation, test, *, block_size, alloc=None,
                   family_names=FAMILY_NAMES):
    """Build the explicit episode composition manifest for all three splits."""
    blocks = {name: _split_block(part, block_size=block_size,
                                 family_names=family_names)
              for name, part in zip(SPLIT_NAMES, (train, validation, test))}
    return {
        "family_names": list(family_names),
        "splits": blocks,
        "allocation": alloc or {},
        "shortfalls": stratification_shortfalls(
            train, validation, test,
            min_family_episodes=MIN_FAMILY_EPISODES),
        "leakage": leakage_check(train, validation, test),
    }


def manifest_table(manifest):
    """Flatten the manifest to one row per split, for ``phase3_splits.csv``."""
    rows = []
    for name, block in manifest["splits"].items():
        row = {"split": name}
        for key in ("rows", "benign_rows", "attack_rows", "attack_fraction",
                    "attack_episodes", "benign_episodes",
                    "benign_managed_blocks", "attack_episodes_stealth",
                    "attack_episodes_non_stealth", "attack_rows_stealth",
                    "attack_rows_non_stealth"):
            row[key] = block[key]
        row["cells_present_of"] = (f"{block['cells_family_stealth']['present']}"
                                   f"/{block['cells_family_stealth']['of']}")
        for fam in manifest["family_names"]:
            row[f"attack_episodes_{fam}"] = block["attack_episodes_by_family"][fam]
            row[f"attack_rows_{fam}"] = block["attack_rows_by_family"][fam]
        rows.append(row)
    return pd.DataFrame(rows)
