"""Behavioural emulator and leakage-safe dataset splitting.

Provenance and honesty
----------------------
Attack traffic here comes from a **behaviour-faithful emulator**, not from
executed malware. Benign profiles are modelled on the base paper's Table 3
(per-role create/rename/delete ranges, split by working shift); attack profiles
are modelled from published family characteristics. No ransomware sample is
executed anywhere in this prototype, and none should be.

This is the project's principal limitation and it is recorded in every result
file the prototype writes. All transfer results are therefore *emulated-family*
transfer, never real-world generalisation.

Why per-window counts alone are not enough
------------------------------------------
Benign roles reach tens of creates per second, and the attack count ranges
overlap that envelope by construction. A single attack window can therefore look
exactly like a busy user. What distinguishes the two is *persistence*: an attack
sustains elevated operation counts across consecutive windows, whereas a user's
burst is a spike followed by quiet. That is why the features in ``assrd.features``
exist, and it is a property of the data, not of any one classifier.

Splitting
---------
Windows are split by **attack episode**, never by row. Consecutive windows of one
episode are strongly correlated, so a row-level split places near-duplicates on
both sides of the boundary and inflates every score. ``split_by_episode`` keeps
each episode wholly inside one subset and is asserted by the Phase 1 tests to
share zero episode IDs across train/validation/test.
"""

import random

import numpy as np
import pandas as pd

RAW_COUNT_COLUMNS = ["create_count", "rename_count", "delete_count"]

MIN_EPISODES = 24

# ---------------------------------------------------------------------------
# Benign profiles (modelled on base paper Table 3)
# ---------------------------------------------------------------------------
# per-role per-second count ranges for an *actively working* user, by shift.
BENIGN_TEMPLATES = {
    "users_at_work":     {"shift": "day",   "create": (30, 60), "rename": (10, 30), "delete": (5, 20)},
    "secretary_at_work": {"shift": "day",   "create": (30, 60), "rename": (10, 20), "delete": (5, 20)},
    "it_admin_at_work":  {"shift": "day",   "create": (30, 50), "rename": (10, 30), "delete": (10, 20)},
    "ceo_at_work":       {"shift": "day",   "create": (0, 10),  "rename": (0, 5),   "delete": (0, 5)},
    "users_at_remote":   {"shift": "night", "create": (20, 60), "rename": (10, 20), "delete": (5, 10)},
    "secretary_remote":  {"shift": "night", "create": (20, 30), "rename": (5, 10),  "delete": (5, 10)},
    "it_admin_remote":   {"shift": "night", "create": (0, 20),  "rename": (10, 20), "delete": (0, 20)},
    "ceo_remote":        {"shift": "night", "create": (0, 5),   "rename": (0, 2),   "delete": (0, 2)},
}

DAY_TEMPLATES = [n for n, s in BENIGN_TEMPLATES.items() if s["shift"] == "day"]
NIGHT_TEMPLATES = [n for n, s in BENIGN_TEMPLATES.items() if s["shift"] == "night"]

DAY_WEIGHTS = {"users_at_work": 0.30, "secretary_at_work": 0.28,
               "it_admin_at_work": 0.32, "ceo_at_work": 0.10}
NIGHT_WEIGHTS = {"users_at_remote": 0.20, "secretary_remote": 0.22,
                 "it_admin_remote": 0.46, "ceo_remote": 0.12}

# ---------------------------------------------------------------------------
# Ransomware family profiles (modelled from published characteristics)
# ---------------------------------------------------------------------------
# intensity   : multiplier on the sampled window counts
# jitter      : per-window multiplicative noise (low => flat envelope)
# persistence : probability a window stays active once the episode has started
# profile     : 'burst' alternates active/idle, 'sustained' stays active,
#               'ramp' rises across the episode
RANSOMWARE_FAMILIES = {
    "ryuk":       {"create": (15, 55), "rename": (5, 25),  "delete": (5, 20),
                   "intensity": 1.0,  "jitter": 0.35, "persistence": 0.93, "profile": "sustained"},
    "wannacry":   {"create": (20, 60), "rename": (10, 35), "delete": (3, 15),
                   "intensity": 1.15, "jitter": 0.45, "persistence": 0.88, "profile": "burst"},
    "lockbit":    {"create": (25, 70), "rename": (15, 40), "delete": (8, 25),
                   "intensity": 1.30, "jitter": 0.30, "persistence": 0.95, "profile": "sustained"},
    "teslacrypt": {"create": (10, 40), "rename": (8, 30),  "delete": (5, 20),
                   "intensity": 0.90, "jitter": 0.55, "persistence": 0.82, "profile": "burst"},
    "notpetya":   {"create": (30, 80), "rename": (10, 30), "delete": (10, 35),
                   "intensity": 1.45, "jitter": 0.60, "persistence": 0.72, "profile": "ramp"},
}

FAMILY_NAMES = sorted(RANSOMWARE_FAMILIES)


def _int_range(rng, low, high, scale=1.0):
    low = max(0.0, low * scale)
    high = max(low, high * scale)
    return int(round(rng.uniform(low, high)))


def sample_benign_window(rng, hour_of_day, intensity=1.0):
    """One second of benign file activity.

    ``hour_of_day`` picks the day/night shift templates; ``intensity`` scales the
    counts to model workload variation.
    """
    day_shift = 8 <= hour_of_day < 16
    if day_shift:
        template = rng.choices(DAY_TEMPLATES,
                               weights=[DAY_WEIGHTS[t] for t in DAY_TEMPLATES], k=1)[0]
    else:
        template = rng.choices(NIGHT_TEMPLATES,
                               weights=[NIGHT_WEIGHTS[t] for t in NIGHT_TEMPLATES], k=1)[0]
    spec = BENIGN_TEMPLATES[template]
    return {
        "create": _int_range(rng, *spec["create"], scale=intensity),
        "rename": _int_range(rng, *spec["rename"], scale=intensity),
        "delete": _int_range(rng, *spec["delete"], scale=intensity),
        "template": template,
    }


def sample_ransomware_window(rng, family, intensity=1.0, active=True):
    """One second of ransomware-like file activity.

    ``active=False`` models an idle window *inside* an episode (a partially swept
    file tree keeps generating operations, so idle is not silence).
    """
    spec = RANSOMWARE_FAMILIES[family]
    if not active:
        return {
            "create": int(rng.uniform(0, spec["create"][0] * 0.25 * intensity)),
            "rename": int(rng.uniform(0, spec["rename"][0] * 0.25 * intensity)),
            "delete": int(rng.uniform(0, spec["delete"][0] * 0.25 * intensity)),
            "family": family, "active": False,
        }
    scale = max(0.05, intensity * spec["intensity"]
                * (1.0 + rng.uniform(-spec["jitter"], spec["jitter"])))
    return {
        "create": _int_range(rng, *spec["create"], scale=scale),
        "rename": _int_range(rng, *spec["rename"], scale=scale),
        "delete": _int_range(rng, *spec["delete"], scale=scale),
        "family": family, "active": True,
    }


def sample_stealth_window(rng, hour_of_day, family="lockbit", intensity=1.0):
    """One second of a rate-limited ("low-and-slow") attack.

    Counts are drawn from the benign envelope (so a count-only model cannot see
    them), the total is suppressed, and the rename/delete share is nudged towards
    ransomware behaviour. This is the case the abstract names as the principal
    limitation; whether it actually defeats a model is measured, not assumed.
    """
    benign = sample_benign_window(rng, hour_of_day, intensity=intensity)
    total = benign["create"] + benign["rename"] + benign["delete"]
    if total == 0:
        return {"create": 0, "rename": 0, "delete": 0,
                "family": family, "stealth": True}

    bump_rename = rng.uniform(0.10, 0.20)
    bump_delete = rng.uniform(0.06, 0.14)
    total_bump = bump_rename + bump_delete
    if total_bump > 0.9:
        bump_rename *= 0.9 / total_bump
        bump_delete *= 0.9 / total_bump

    rename_share = min(0.9, (benign["rename"] + 1) / total + bump_rename)
    delete_share = min(0.9, (benign["delete"] + 1) / total + bump_delete)
    suppressed = rng.uniform(0.35, 0.7)

    create = int(total * (1.0 - rename_share - delete_share) * suppressed)
    rename = int(total * rename_share * suppressed)
    delete = int(total * delete_share * suppressed)

    if 8 <= hour_of_day < 16:
        create, rename, delete = min(create, 60), min(rename, 30), min(delete, 20)
    else:
        spec = RANSOMWARE_FAMILIES[family]
        create = min(create, spec["create"][1])
        rename = min(rename, spec["rename"][1])
        delete = min(delete, spec["delete"][1])

    return {"create": max(0, create), "rename": max(0, rename),
            "delete": max(0, delete), "family": family, "stealth": True}


def diurnal_factor(hour_of_day, rng=None):
    """Workload multiplier by hour (office hours busiest, small hours quietest)."""
    if 8 <= hour_of_day < 16:
        base = 1.0
    elif 6 <= hour_of_day < 8 or 16 <= hour_of_day < 20:
        base = 0.6
    elif 20 <= hour_of_day < 23:
        base = 0.35
    else:
        base = 0.15
    if rng is not None:
        base *= rng.uniform(0.75, 1.25)
    return base


def idle_probability(hour_of_day, rng=None):
    """Probability a benign window carries no file activity at all."""
    if 8 <= hour_of_day < 16:
        p = 0.45
    elif 6 <= hour_of_day < 8 or 16 <= hour_of_day < 20:
        p = 0.55
    else:
        p = 0.80
    if rng is not None:
        p = min(0.98, max(0.0, p * rng.uniform(0.85, 1.15)))
    return p


def active_probability(hour_of_day, rng=None):
    """Probability a non-idle benign window is an *actively worked* window.

    Table 3 describes a user who is doing something; it does not mean every second
    of the working day is busy. This keeps benign activity rare, which is what
    makes "attack sustains activity on an otherwise quiet server" the real signal.
    """
    if 8 <= hour_of_day < 16:
        p = 0.06
    elif 6 <= hour_of_day < 8 or 16 <= hour_of_day < 20:
        p = 0.03
    else:
        p = 0.008
    if rng is not None:
        p = min(0.95, max(0.0, p * rng.uniform(0.7, 1.3)))
    return p


def _episode_profile(rng, family, length):
    """Per-episode shape parameters drawn once (so continuity is preserved)."""
    ramp_kind = rng.choice(["flat", "rising", "falling"])
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
        "persistence": float(rng.uniform(0.65, 0.95)),
        "burst_period": rng.randint(3, 12),
        "ramp": ramp,
        "family": family,
    }


def _is_active(rng, offset, profile):
    persistence = profile["persistence"]
    period = profile["burst_period"]
    if period <= 1:
        return rng.random() < persistence
    phase = offset % period
    if phase == 0:
        return True
    return phase < max(1, int(period * persistence))


def build_trace(n_windows=20000, start_hour=8, attack_ratio=0.05, seed=42,
                include_stealth=True, stealth_share=0.25,
                min_episode=20, max_episode=90, n_episodes=None):
    """Build a continuous one-second window series with labelled attack episodes.

    Returns a DataFrame (temporal order) with the three counts plus
    ``window_index``, ``label`` (0/1), ``episode_id`` (-1 for benign),
    ``family``, ``is_stealth`` and ``benign_template``.

    ``attack_ratio`` is a target; the realised fraction is reported by
    :func:`trace_report` rather than assumed to equal the request.
    """
    rng = random.Random(seed)
    n_windows = int(n_windows)

    avg_len = (min_episode + max_episode) / 2
    if n_episodes is None:
        wanted = max(1, int(round(n_windows * attack_ratio / avg_len)))
        target_fraction = min(attack_ratio, wanted * avg_len / n_windows)
        n_episodes = max(MIN_EPISODES,
                         int(round(n_windows * target_fraction / avg_len)))
    n_episodes = max(1, min(n_episodes, n_windows // (min_episode + 1)))
    slot = n_windows // n_episodes

    upper = max(min_episode, min(max_episode, slot - max(1, slot // 8)))
    episode_span = max(min_episode, upper)

    starts = []
    for e in range(n_episodes):
        begin = e * slot + int(rng.randint(0, max(1, slot // 4)))
        begin = min(begin, max(0, n_windows - min_episode - 1))
        starts.append(begin)
    starts.sort()

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
                active = _is_active(rng, i - start, profile)
                win = sample_ransomware_window(rng, family,
                                               intensity=profile["ramp"](i - start),
                                               active=active)
            create[i] = max(0, win["create"])
            rename[i] = max(0, win["rename"])
            delete[i] = max(0, win["delete"])

    for i in range(n_windows):
        if label[i] == 1:
            continue
        hour = (start_hour + i // 3600) % 24
        if rng.random() < idle_probability(hour, rng):
            template_of[i] = "idle"
            continue
        if rng.random() >= active_probability(hour, rng):
            template_of[i] = "low_activity"
            create[i] = rng.randint(0, 3)
            rename[i] = rng.randint(0, 1)
            delete[i] = rng.randint(0, 1)
            continue
        intensity = diurnal_factor(hour, rng)
        if rng.random() < 0.03:
            intensity *= rng.uniform(1.5, 2.5)
        win = sample_benign_window(rng, hour, intensity=intensity)
        create[i], rename[i], delete[i] = win["create"], win["rename"], win["delete"]
        template_of[i] = win["template"]

    return pd.DataFrame({
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


def trace_report(df):
    """Summary of what a trace actually contains (achieved, not requested)."""
    n = len(df)
    counts = df[RAW_COUNT_COLUMNS]
    return {
        "Windows": int(n),
        "Benign": int((df["label"] == 0).sum()),
        "Attack": int((df["label"] == 1).sum()),
        "Attack Fraction": round(float((df["label"] == 1).mean()), 6) if n else 0.0,
        "Episodes": int(df[df["label"] == 1]["episode_id"].nunique()),
        "Stealth Windows": int(df["is_stealth"].sum()),
        "Zero Windows": int((counts == 0).all(axis=1).sum()),
        "Families": {k: int(v) for k, v in
                     df[df["label"] == 1]["family"].value_counts().items()},
    }


def split_by_episode(df, test_fraction=0.10, validation_fraction=0.10, seed=42):
    """Split by attack episode; return (train_df, val_df, test_df).

    Attack episodes go wholly into one subset. Benign windows carry ``episode_id
    == -1`` and are grouped into contiguous blocks of neighbouring windows so
    adjacent benign windows are not scattered across subsets.

    Raises ``ValueError`` if any subset ends up with no attack windows, because a
    sensitivity figure from such a split is meaningless.
    """
    rng = random.Random(seed)
    n = len(df)

    attack_eps = sorted({int(e) for e in df["episode_id"].tolist() if e >= 0})
    shuffled = list(attack_eps)
    rng.shuffle(shuffled)

    n_test_eps = max(1, int(round(len(shuffled) * test_fraction)))
    n_val_eps = max(1, int(round(len(shuffled) * validation_fraction)))
    if n_test_eps + n_val_eps >= len(shuffled):
        n_test_eps = max(1, len(shuffled) // 3)
        n_val_eps = max(1, len(shuffled) // 3)

    test_eps = set(shuffled[:n_test_eps])
    val_eps = set(shuffled[n_test_eps:n_test_eps + n_val_eps])

    episode_of = df["episode_id"].to_numpy()
    index_of = df["window_index"].to_numpy()

    block_size = max(1, n // 200)
    benign_keys = index_of[episode_of < 0] // block_size
    benign_order = list(np.unique(benign_keys).tolist())
    rng.shuffle(benign_order)
    n_test_blocks = max(1, int(round(len(benign_order) * test_fraction)))
    n_val_blocks = max(1, int(round(len(benign_order) * validation_fraction)))
    test_blocks = set(benign_order[:n_test_blocks])
    val_blocks = set(benign_order[n_test_blocks:n_test_blocks + n_val_blocks])

    row_blocks = index_of // block_size
    is_attack = episode_of >= 0

    is_test = np.where(is_attack, np.isin(episode_of, list(test_eps)),
                       np.isin(row_blocks, list(test_blocks)))
    is_val = np.where(is_attack, np.isin(episode_of, list(val_eps)),
                      np.isin(row_blocks, list(val_blocks)))
    is_test = np.asarray(is_test, dtype=bool)
    is_val = np.asarray(is_val, dtype=bool)
    is_train = ~(is_test | is_val)

    train_df = df[is_train].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)

    for name, part in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if int((part["label"] == 1).sum()) == 0:
            raise ValueError(
                f"degenerate split: {name} subset contains no attack windows; "
                f"build a trace with more episodes (increase attack_ratio, "
                f"n_windows or n_episodes).")

    return train_df, val_df, test_df
