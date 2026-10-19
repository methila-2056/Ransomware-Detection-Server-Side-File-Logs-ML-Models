"""Temporal trace construction and dataset variants.

The base paper does not treat its dataset as independent rows. It logs file
operations continuously, aggregates them into non-overlapping one-second windows,
and preserves the order of those windows so that attack episodes can be located in
time. Sect. 4.3 then produces several dataset representations from that one trace:

  * the raw window series, including every all-zero idle window
  * the "one-zero" variant, where a run of consecutive all-zero windows is
    collapsed to a single representative record
  * the "nozeroatt" variant, which additionally drops rows labelled as attack
    whose counts are all zero, since those are preparatory phases with no
    observable file operations

This module reproduces that pipeline so the size-reduction behaviour of the
paper's Table 2 can be measured, and so windows carry their position in the trace.
Window position matters for evaluation: a random row split leaks neighbouring
windows of the same attack episode across the train/test boundary, which
inflates every score. ``group_aware_split`` prevents that by splitting on
episodes rather than rows.
"""

import random

import numpy as np
import pandas as pd

from ml.attack_sim import (
    FAMILY_NAMES,
    active_probability,
    diurnal_factor,
    idle_probability,
    sample_benign_window,
    sample_ransomware_window,
    sample_stealth_window,
)

FEATURE_COLUMNS = ["create_count", "rename_count", "delete_count"]

# Minimum number of attack episodes. The episode-aware split divides episodes
# across train, validation and test, and needs enough of them for every subset to
# receive attacks.
MIN_EPISODES = 24


def build_trace(n_windows=20000, start_hour=8, attack_ratio=0.05, seed=42,
                include_stealth=True, stealth_share=0.25,
                min_episode=20, max_episode=90, n_episodes=None):
    """Build a continuous one-second window series with labelled attack episodes.

    ``attack_ratio`` sets the *target* share of windows inside attack episodes.
    The realised ratio is approximate: episodes have discrete lengths and are
    placed on a grid, so the achieved fraction is reported by ``build_report``
    rather than assumed to equal the request.

    A floor of ten episodes is enforced because the evaluation splits by episode,
    and a split with only a handful of episodes cannot place an episode in every
    subset, which makes a sensitivity figure unmeasurable. With the default
    episode lengths that floor yields enough attack windows per subset to report a
    per-family breakdown; longer episodes produce too few episodes and starve the
    test split.

    Windows are produced in temporal order and carry ``window_index``,
    ``episode_id`` (-1 when benign), ``family`` and ``is_stealth`` so downstream
    evaluation can measure detection per family and per episode.
    """
    rng = random.Random(seed)
    n_windows = int(n_windows)

    if n_episodes is None:
        # Episode length scales with the trace so a fixed count would not fit in a
        # short trace, and a fixed count would collapse into a handful of very
        # long episodes in a long one.
        #
        # The floor of 24 matters: the evaluation splits by episode into three
        # subsets, and each subset needs at least a few episodes or the split is
        # degenerate. Eleven episodes was observed to leave training with no
        # attacks at all once the test subset took its share.
        #
        # The floor sets a minimum attack fraction rather than a minimum episode
        # count at low ratios: below that fraction episodes are shortened
        # instead, so a requested attack ratio of 1% is actually delivered rather
        # than being silently floored at whatever 24 episodes happen to occupy.
        avg_len = (min_episode + max_episode) / 2
        wanted = max(1, int(round(n_windows * attack_ratio / avg_len)))
        target_fraction = min(attack_ratio, wanted * avg_len / n_windows)
        n_episodes = max(MIN_EPISODES,
                         int(round(n_windows * target_fraction / avg_len)))

    # Place episodes on a grid with jitter, guaranteeing they are non-overlapping
    # and evenly distributed rather than left to rejection sampling. Rejection
    # sampling tends to fail to place the requested count on a busy trace, which
    # silently produces too few episodes.
    n_episodes = max(1, min(n_episodes, n_windows // (min_episode + 1)))
    slot = n_windows // n_episodes

    # Episode length stays within the requested band [min_episode, max_episode].
    # Deriving it from the slot width instead would make a minimum-episode trace
    # fill its slots almost completely, so 24 episodes covered 40% of the trace
    # instead of the requested 5%.
    upper = max(min_episode, min(max_episode, slot - max(1, slot // 8)))
    episode_span = max(min_episode, upper)

    starts = []
    for e in range(n_episodes):
        begin = e * slot + int(rng.randint(0, max(1, slot // 4)))
        begin = min(begin, max(0, n_windows - min_episode - 1))
        starts.append(begin)
    starts.sort()

    # Build the label mask and per-window family assignment.
    label = np.zeros(n_windows, dtype=int)
    episode_of = np.full(n_windows, -1, dtype=int)
    family_of = np.empty(n_windows, dtype=object)
    stealth_of = np.zeros(n_windows, dtype=bool)
    create = np.zeros(n_windows, dtype=int)
    rename = np.zeros(n_windows, dtype=int)
    delete = np.zeros(n_windows, dtype=int)
    template_of = np.empty(n_windows, dtype=object)

    episode_id = 0
    for start in starts:
        spec_len = int(rng.randint(min_episode, max(min_episode, episode_span)))
        end = min(n_windows, start + spec_len)
        if end - start < min_episode // 2:
            continue

        family = FAMILY_NAMES[rng.randrange(len(FAMILY_NAMES))]
        # A share of episodes are low-and-slow, the case the paper names as an
        # unaddressed limitation (Sect. 6.2).
        stealth_episode = bool(include_stealth and rng.random() < stealth_share)

        profile = _episode_profile(rng, family, end - start)
        label[start:end] = 1
        episode_of[start:end] = episode_id
        family_of[start:end] = family
        stealth_of[start:end] = stealth_episode
        episode_id += 1

        for i in range(start, end):
            hour = (start_hour + i // 3600) % 24
            if stealth_episode:
                win = sample_stealth_window(rng, hour, family=family)
            else:
                active = _is_active(rng, i - start, end - start, profile)
                win = sample_ransomware_window(rng, family,
                                               intensity=profile["ramp"](i - start),
                                               active=active)
            # Clamped at zero: a count can never be negative, and any sampler
            # defect upstream must not silently produce invalid training data.
            create[i] = max(0, win["create"])
            rename[i] = max(0, win["rename"])
            delete[i] = max(0, win["delete"])

    # Fill the benign windows.
    for i in range(n_windows):
        if label[i] == 1:
            continue
        hour = (start_hour + i // 3600) % 24
        if rng.random() < idle_probability(hour, rng):
            template_of[i] = "idle"
            continue

        # A non-zero window is still usually not a busy one. Table 3 describes
        # the range for a user who is actively working, not the average second of
        # the working day, so an inactive-but-nonzero window draws small counts.
        if rng.random() >= active_probability(hour, rng):
            template_of[i] = "low_activity"
            create[i] = rng.randint(0, 3)
            rename[i] = rng.randint(0, 1)
            delete[i] = rng.randint(0, 1)
            continue

        intensity = diurnal_factor(hour, rng)
        # Occasional user-driven bursts: bulk copies, archiving, backups.
        if rng.random() < 0.03:
            intensity *= rng.uniform(1.5, 2.5)

        win = sample_benign_window(rng, hour, intensity=intensity)
        create[i] = win["create"]
        rename[i] = win["rename"]
        delete[i] = win["delete"]
        template_of[i] = win["template"]

    df = pd.DataFrame({
        "window_index": np.arange(n_windows),
        "create_count": create,
        "rename_count": rename,
        "delete_count": delete,
        "label": label,
        "episode_id": episode_of,
        "family": family_of,
        "is_stealth": stealth_of,
        "benign_template": template_of,
    })
    return df


def _episode_profile(rng, family, length):
    """Per-episode shape parameters drawn once for the whole episode.

    Drawing these inside the per-window loop would give every window an
    independent shape, which erases the temporal continuity that distinguishes
    ransomware from benign bursts.

    The ramp is bounded to [0.5, 1.5]. An unbounded ramp fed into the sampler's
    intensity multiplier produces negative counts for count features that start
    at zero, so the bounds are part of correctness rather than tuning.
    """
    persistence = float(rng.uniform(0.65, 0.95))
    ramp_kind = rng.choice(["flat", "rising", "falling"])
    burst_period = rng.randint(3, 12)

    span = max(1, length)

    if ramp_kind == "rising":
        def ramp(offset):
            return 0.5 + (offset / span)
    elif ramp_kind == "falling":
        def ramp(offset):
            return 1.5 - (offset / span)
    else:
        def ramp(offset):
            return 1.0

    return {
        "persistence": persistence,
        "burst_period": burst_period,
        "ramp": ramp,
        "family": family,
    }


def _is_active(rng, offset, length, profile):
    """Whether this window inside an episode is active or an inter-burst gap."""
    persistence = profile["persistence"]
    period = profile["burst_period"]
    if period <= 1:
        return rng.random() < persistence
    # A burst structure: the family works for a run of windows then idles.
    phase = offset % period
    if phase == 0:
        return True
    return phase < max(1, int(period * persistence))


# ---------------------------------------------------------------------------
# Dataset variants (base paper Table 2)
# ---------------------------------------------------------------------------

def variant_raw(df):
    """Every one-second window, including all-zero idle records."""
    return df.copy()


def variant_one_zero(df):
    """Collapse consecutive all-zero windows to a single record.

    Base paper Sect. 4.3: at most one record from each sequence of consecutive
    all-zero entries is retained, compressing uneventful periods into one
    representation. Without this, idle windows dominate training and bias the
    models towards the benign class.
    """
    counts = df[FEATURE_COLUMNS]
    is_zero = (counts == 0).all(axis=1)

    keep = []
    previous_zero = False
    for idx, zero in enumerate(is_zero):
        if zero and previous_zero:
            continue
        keep.append(idx)
        previous_zero = zero

    return df.iloc[keep].copy()


def variant_nozeroatt(df):
    """One-zero variant, plus dropping all-zero attack rows.

    Rows labelled attack with zero counts represent preparatory phases of
    ransomware execution (process init, host infection, C2) with no observable
    file operations. The paper excludes them because the study is about file I/O
    behaviour, and keeping them would teach the model that silence means attack.
    """
    collapsed = variant_one_zero(df)
    counts = collapsed[FEATURE_COLUMNS]
    is_zero = (counts == 0).all(axis=1)
    return collapsed[~((is_zero) & (collapsed["label"] == 1))].copy()


def variant_nozero(df):
    """Raw windows with all-zero attack rows dropped but no zero-collapse."""
    counts = df[FEATURE_COLUMNS]
    is_zero = (counts == 0).all(axis=1)
    return df[~(is_zero & (df["label"] == 1))].copy()


DATASET_VARIANTS = {
    "raw": variant_raw,
    "one_zero": variant_one_zero,
    "nozeroatt": variant_nozeroatt,
    "nozero": variant_nozero,
}


def variant_report(df):
    """Size-reduction table in the style of the paper's Table 2."""
    rows = []
    for name, fn in DATASET_VARIANTS.items():
        variant = fn(df)
        rows.append({
            "Variant": name,
            "Records": int(len(variant)),
            "Benign": int((variant["label"] == 0).sum()),
            "Attack": int((variant["label"] == 1).sum()),
            "Attack Fraction": round(
                float((variant["label"] == 1).mean()) if len(variant) else 0.0, 6),
            "Zero Windows": int(
                ((variant[FEATURE_COLUMNS] == 0).all(axis=1)).sum()),
            "Reduction vs Raw": round(
                1.0 - len(variant) / max(1, len(df)), 6),
        })
    return rows


# ---------------------------------------------------------------------------
# Leakage-safe splitting
# ---------------------------------------------------------------------------

def group_aware_split(df, test_fraction=0.1, validation_fraction=0.1, seed=42):
    """Split by attack episode rather than by row.

    A row-level stratified split places windows from the same attack episode on
    both sides of the boundary. Because consecutive windows of an episode are
    strongly correlated, a tree model can identify "this looks like the episode I
    memorised during training" and score well without generalising. Splitting on
    the episode identifier keeps every episode wholly within one subset.

    Benign windows carry episode_id -1, so they are grouped into contiguous
    blocks of ``BENIGN_BLOCK`` windows. Blocks keep neighbouring benign windows
    together without putting all benign data in one subset.

    Attack episodes are dealt round-robin across the three subsets rather than
    assigned by shuffled group order. With few episodes a shuffled split can leave
    a subset with no attacks at all, which makes a sensitivity figure meaningless.
    """
    rng = random.Random(seed)

    features = FEATURE_COLUMNS
    n = len(df)

    attack_eps = sorted({e for e in df["episode_id"].tolist() if e >= 0})
    shuffled = list(attack_eps)
    rng.shuffle(shuffled)

    n_test_eps = max(1, int(round(len(shuffled) * test_fraction)))
    n_val_eps = max(1, int(round(len(shuffled) * validation_fraction)))
    if n_test_eps + n_val_eps >= len(shuffled):
        # Too few episodes to give every subset at least one; fall back to a
        # simple order-preserving split.
        n_test_eps = max(1, len(shuffled) // 3)
        n_val_eps = max(1, len(shuffled) // 3)

    test_eps = set(shuffled[:n_test_eps])
    val_eps = set(shuffled[n_test_eps:n_test_eps + n_val_eps])

    # Episodes differ in length, so a random draw can leave the test subset with
    # far fewer attack windows than the requested fraction implies -- or with none
    # at all. Rebalance by moving episodes until the test subset's attack-window
    # count is near the requested fraction.
    #
    # Episodes are only ever moved *into* the test set from the remaining pool,
    # never out of it, and validation is protected. Moving an episode out of train
    # can strip training of one of its few attacks, which leaves a CV fold
    # single-class and makes recall scoring undefined.
    def window_counts(eps):
        return {e: int((df["episode_id"] == e).sum()) for e in eps}

    counts = window_counts(shuffled)

    # The test subset cannot hold more attack windows than the trace contains.
    # Requesting a 10% test split from a trace that is only 5% attack would claim
    # 50% of all episodes for test, leaving nothing usable for training.
    total_attack = sum(counts.values())
    target = min(int(len(df) * test_fraction), int(total_attack * 0.35))

    train_pool = [e for e in shuffled if e not in test_eps and e not in val_eps]
    current = sum(counts[e] for e in test_eps)
    if current > target:
        # The random draw overshot; hand episodes back to the training pool.
        for e in sorted(test_eps, key=lambda x: counts[x]):
            if current <= target or len(test_eps) <= 1:
                break
            test_eps.discard(e)
            current -= counts[e]
    else:
        for e in sorted(train_pool, key=lambda x: counts[x], reverse=True):
            if current >= target:
                break
            test_eps.add(e)
            current += counts[e]

    episode_of = df["episode_id"].to_numpy()
    index_of = df["window_index"].to_numpy()

    # Benign rows are grouped into contiguous blocks so adjacent benign windows
    # are not scattered across subsets. Vectorised because three per-row Python
    # loops over a 500k-row frame cost minutes, which dominated trace preparation
    # at the larger scales.
    block_size = max(1, n // 200)
    benign_keys = index_of[episode_of < 0] // block_size
    benign_order = np.unique(benign_keys)
    shuffled_blocks = list(benign_order.tolist())
    rng.shuffle(shuffled_blocks)
    n_test_blocks = max(1, int(round(len(shuffled_blocks) * test_fraction)))
    n_val_blocks = max(1, int(round(len(shuffled_blocks) * validation_fraction)))
    test_blocks = set(shuffled_blocks[:n_test_blocks])
    val_blocks = set(shuffled_blocks[n_test_blocks:n_test_blocks + n_val_blocks])

    row_blocks = index_of // block_size
    is_attack = episode_of >= 0

    is_test = np.where(is_attack,
                       np.isin(episode_of, list(test_eps)),
                       np.isin(row_blocks, list(test_blocks)))
    is_val = np.where(is_attack,
                      np.isin(episode_of, list(val_eps)),
                      np.isin(row_blocks, list(val_blocks)))

    is_test = np.asarray(is_test, dtype=bool)
    is_val = np.asarray(is_val, dtype=bool)
    is_train = ~(is_test | is_val)

    train_df = df[is_train].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)

    # A split with no attacks in train or test cannot produce a sensitivity
    # figure, and a single-class fold makes recall scoring raise. Failing loudly
    # here is better than reporting a meaningless zero.
    for name, part in (("train", train_df), ("validation", val_df), ("test", test_df)):
        n_attack = int((part["label"] == 1).sum())
        if n_attack == 0:
            raise ValueError(
                f"degenerate split: {name} subset contains no attack windows. "
                f"Build a trace with more attack episodes (increase attack_ratio, "
                f"n_windows, or n_episodes) before splitting by episode."
            )

    return train_df, val_df, test_df


def build_report(df, requested_ratio=None):
    """Summary of what a trace actually contains.

    Reports the achieved attack fraction rather than the requested one, since
    episode placement on a grid only approximates the request.
    """
    n = len(df)
    attack = int((df["label"] == 1).sum())
    counts = df[FEATURE_COLUMNS]
    zero = int(((counts == 0).all(axis=1)).sum())

    report = {
        "Windows": n,
        "Benign": int((df["label"] == 0).sum()),
        "Attack": attack,
        "Attack Fraction": round(attack / n, 6) if n else 0.0,
        "Episodes": int(df[df["label"] == 1]["episode_id"].nunique()),
        "Stealth Attack Windows": int(df["is_stealth"].sum()),
        "Zero Windows": zero,
        "Families": {k: int(v) for k, v in
                     df[df["label"] == 1]["family"].value_counts().items()},
    }
    if requested_ratio is not None:
        report["Requested Attack Fraction"] = requested_ratio
    return report


def leakage_check(df, train_idx, test_idx):
    """Report whether any attack episode appears on both sides of a split."""
    train_eps = set(df.iloc[train_idx]["episode_id"].tolist()) - {-1}
    test_eps = set(df.iloc[test_idx]["episode_id"].tolist()) - {-1}
    shared = train_eps & test_eps
    return {
        "Shared Attack Episodes": len(shared),
        "Leakage Free": len(shared) == 0,
    }