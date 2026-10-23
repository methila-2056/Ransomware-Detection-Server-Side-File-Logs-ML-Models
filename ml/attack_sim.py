"""Behavioural profiles for benign activity and ransomware attacks.

Provenance
----------
Benign profiles are transcribed from the base paper's Table 3 ("Job templates and
associated file operation activity ranges"), which gives per-second count ranges
for eight user roles split by working-hours and remote-work shifts.

The earlier generator in ``ml.train`` used invented ranges of roughly 0-8 creates
per second for benign activity and 10-80 for attacks. Those ranges do not overlap:
the benign maximum sat below the attack minimum on the create feature, so the two
classes were separable by a single threshold and every model scored 1.000. Using
the paper's real Table 3 values, benign activity reaches 30-60 creates per second
and the attack ranges sit inside that envelope, which is the source of the
overlap the paper's own models must contend with.

Attack profiles are derived from published family characteristics and from the
operation sequences described in the paper (Sect. 3.3: the read-encrypt-delete
cycle expressed as create/rename/delete counts). They are modelled, not captured
from executing malware, and are labelled as such wherever results are reported.

The distinguishing signal is temporal, not just per-window magnitude. Benign
activity follows a diurnal cycle with sporadic user-driven bursts; ransomware
sustains elevated rates across consecutive windows. Per-window count ranges
therefore overlap substantially between classes, and the separation has to come
from the shape of the sequence.
"""

import math

# Benign job templates, transcribed from base paper Table 3.
# Fields: (create_low, create_high, rename_low, rename_high, delete_low, delete_high)
BENIGN_TEMPLATES = {
    "users_at_work":      {"hours": "day",   "create": (30, 60), "rename": (10, 30), "delete": (5, 20)},
    "users_at_remote":    {"hours": "night", "create": (20, 60), "rename": (10, 20), "delete": (5, 10)},
    "secretary_at_work":  {"hours": "day",   "create": (30, 60), "rename": (10, 20), "delete": (5, 20)},
    "secretary_remote":   {"hours": "night", "create": (20, 30), "rename": (5, 10),  "delete": (5, 10)},
    "it_admin_at_work":   {"hours": "day",   "create": (30, 50), "rename": (10, 30), "delete": (10, 20)},
    "it_admin_remote":    {"hours": "night", "create": (0, 20),  "rename": (10, 20), "delete": (0, 20)},
    "ceo_at_work":        {"hours": "day",   "create": (0, 10),  "rename": (0, 5),   "delete": (0, 5)},
    "ceo_remote":         {"hours": "night", "create": (0, 5),   "rename": (0, 2),   "delete": (0, 2)},
}

DAY_TEMPLATES = [name for name, spec in BENIGN_TEMPLATES.items() if spec["hours"] == "day"]
NIGHT_TEMPLATES = [name for name, spec in BENIGN_TEMPLATES.items() if spec["hours"] == "night"]

# Relative probability of each template being active in a given window. Office
# roles are present on the day shift and remote variants at night; the CEO is
# rarely active. Normalised within each shift.
DAY_WEIGHTS = {
    "users_at_work": 0.30,
    "secretary_at_work": 0.28,
    "it_admin_at_work": 0.32,
    "ceo_at_work": 0.10,
}
NIGHT_WEIGHTS = {
    "users_at_remote": 0.20,
    "secretary_remote": 0.22,
    "it_admin_remote": 0.46,
    "ceo_remote": 0.12,
}

# Ransomware family profiles.
#
# Each entry describes the per-second count envelope during an active attack, a
# shape descriptor, and how persistently the attack sustains activity.
#
#   intensity  -- multiplier applied to the sampled window counts
#   jitter     -- multiplicative noise per window; low means a very flat envelope
#   persistence-- probability a window stays active once an attack has started
#   profile    -- 'burst' alternates active/idle, 'sustained' stays active,
#                 'ramp' increases intensity across the episode
#
# Ranges overlap the benign envelope by construction, because the paper's benign
# templates reach 30-60 creates per second.
RANSOMWARE_FAMILIES = {
    "ryuk": {
        "create": (15, 55), "rename": (5, 25), "delete": (5, 20),
        "intensity": 1.0, "jitter": 0.35, "persistence": 0.93, "profile": "sustained",
        "note": "targeted deployment, moderate sustained throughput",
    },
    "wannacry": {
        "create": (20, 60), "rename": (10, 35), "delete": (3, 15),
        "intensity": 1.15, "jitter": 0.45, "persistence": 0.88, "profile": "burst",
        "note": "worm-like propagation, intermittent bursts with idle gaps",
    },
    "lockbit": {
        "create": (25, 70), "rename": (15, 40), "delete": (8, 25),
        "intensity": 1.3, "jitter": 0.30, "persistence": 0.95, "profile": "sustained",
        "note": "high-throughput affiliate payload, flattest envelope",
    },
    "teslacrypt": {
        "create": (10, 40), "rename": (8, 30), "delete": (5, 20),
        "intensity": 0.9, "jitter": 0.55, "persistence": 0.82, "profile": "burst",
        "note": "older, slower family with irregular pacing",
    },
    "notpetya": {
        "create": (30, 80), "rename": (10, 30), "delete": (10, 35),
        "intensity": 1.45, "jitter": 0.60, "persistence": 0.72, "profile": "ramp",
        "note": "fastest spread with a rising intensity ramp",
    },
}

FAMILY_NAMES = sorted(RANSOMWARE_FAMILIES)


def _lerp(low, high, t):
    return low + (high - low) * t


def _int_range(rng, low, high, scale=1.0):
    """Sample an integer from a range after applying an intensity scale.

    Clamped at zero so a low-intensity profile cannot produce negative counts.
    """
    low = max(0.0, low * scale)
    high = max(low, high * scale)
    return int(round(rng.uniform(low, high)))


def sample_benign_window(rng, hour_of_day, intensity=1.0):
    """Sample one second of benign file activity.

    ``hour_of_day`` selects the day or night shift templates, reproducing the
    diurnal structure of the paper's four user profiles (Sect. 4.2). ``intensity``
    scales the counts to model workload variation.
    """
    day_shift = 8 <= hour_of_day < 16

    if day_shift:
        template = rng.choices(DAY_TEMPLATES, weights=[DAY_WEIGHTS[t] for t in DAY_TEMPLATES],
                               k=1)[0]
    else:
        template = rng.choices(NIGHT_TEMPLATES, weights=[NIGHT_WEIGHTS[t] for t in NIGHT_TEMPLATES],
                               k=1)[0]

    spec = BENIGN_TEMPLATES[template]
    return {
        "create": _int_range(rng, *spec["create"], scale=intensity),
        "rename": _int_range(rng, *spec["rename"], scale=intensity),
        "delete": _int_range(rng, *spec["delete"], scale=intensity),
        "template": template,
    }


def sample_ransomware_window(rng, family, intensity=1.0, active=True):
    """Sample one second of ransomware-like file activity.

    ``active`` False models an idle window inside an episode, which the worm-like
    families produce between propagation bursts.
    """
    spec = RANSOMWARE_FAMILIES[family]
    if not active:
        # Idle windows inside an attack still see residual activity, not silence:
        # a partially-swept file tree continues to generate operations.
        return {
            "create": int(rng.uniform(0, spec["create"][0] * 0.25 * intensity)),
            "rename": int(rng.uniform(0, spec["rename"][0] * 0.25 * intensity)),
            "delete": int(rng.uniform(0, spec["delete"][0] * 0.25 * intensity)),
            "family": family,
            "active": False,
        }

    scale = intensity * spec["intensity"] * (1.0 + rng.uniform(-spec["jitter"],
                                                                spec["jitter"]))
    return {
        "create": _int_range(rng, *spec["create"], scale=max(0.05, scale)),
        "rename": _int_range(rng, *spec["rename"], scale=max(0.05, scale)),
        "delete": _int_range(rng, *spec["delete"], scale=max(0.05, scale)),
        "family": family,
        "active": True,
    }


def sample_stealth_window(rng, hour_of_day, family="lockbit", intensity=1.0):
    """Sample a rate-limited "low-and-slow" attack window.

    The base paper names this as an unaddressed limitation (Sect. 6.2):
    count-based features are vulnerable to rate-limited encryption that stays
    inside the benign envelope. This generator models exactly that case: counts
    are drawn from the benign templates but with the rename/delete ratio shifted
    towards ransomware behaviour, and the absolute rate suppressed.

    The intent is to produce attacks that a count-only model genuinely struggles
    with, so the resulting failure is a measured blind spot rather than an
    assumption. Whether this defeats the model is an empirical question and is not
    assumed here.
    """
    benign = sample_benign_window(rng, hour_of_day, intensity=intensity)
    spec = RANSOMWARE_FAMILIES[family]

    total = benign["create"] + benign["rename"] + benign["delete"]
    if total == 0:
        return {"create": 0, "rename": 0, "delete": 0,
                "family": family, "stealth": True}

    # Shares are clamped so the three components sum to at most 1. Without this
    # the two independent bumps can together exceed 1, which yields a negative
    # create count.
    bump_rename = rng.uniform(0.10, 0.20)
    bump_delete = rng.uniform(0.06, 0.14)
    total_bump = bump_rename + bump_delete
    if total_bump > 0.9:
        bump_rename *= 0.9 / total_bump
        bump_delete *= 0.9 / total_bump

    rename_share = min(0.9, (benign["rename"] + 1) / total + bump_rename)
    delete_share = min(0.9, (benign["delete"] + 1) / total + bump_delete)

    # Suppress the absolute rate so the window stays inside the benign envelope
    # observed at the same hour rather than exceeding it.
    suppressed = rng.uniform(0.35, 0.7)
    create = int(total * (1.0 - rename_share - delete_share) * suppressed)
    rename = int(total * rename_share * suppressed)
    delete = int(total * delete_share * suppressed)

    # Final clamp against the benign envelope. A stealth attack louder than
    # legitimate activity is not stealthy, so the ceiling is the benign maximum
    # for the active shift rather than the family's own ceiling.
    if 8 <= hour_of_day < 16:
        create = min(create, 60)
        rename = min(rename, 30)
        delete = min(delete, 20)
    else:
        create = min(create, spec["create"][1])
        rename = min(rename, spec["rename"][1])
        delete = min(delete, spec["delete"][1])

    return {
        "create": max(0, create),
        "rename": max(0, rename),
        "delete": max(0, delete),
        "family": family,
        "stealth": True,
    }


def diurnal_factor(hour_of_day, rng=None):
    """Workload multiplier for an hour of the day.

    Office hours carry the heaviest load; early morning and late night are quiet.
    A mild random component is applied when ``rng`` is supplied.
    """
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
    """Probability that a window carries no file activity at all.

    Real file servers spend most of their time idle. The base paper's own sample
    of the processed dataset (Table 4) shows benign windows as almost entirely
    (0,0,0) or single-digit counts, with attack windows reaching (19,17,0) and
    (0,0,21). That is the real shape of the data: the discriminating signal is
    that an attack sustains activity while the server is otherwise quiet.

    This function returns only the probability of a fully zero window. Benign
    activity below is additionally governed by a burst model, since a file server
    is idle most of the time even during working hours.
    """
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
    """Probability that a non-idle window is actively worked rather than idle-but-nonzero.

    Table 3 gives the count ranges for a user who is doing something. It does not
    imply every second of the working day is busy.

    The base paper's own processed sample (Table 4) shows benign windows almost
    entirely (0,0,0), (1,0,0), (0,0,1), (0,0,2) -- a maximum of two operations
    per second -- while attack windows run to (19,17,0) and (0,0,21). Table 3's
    30-60 creates per second describes an active editing or copy session, which
    occupies only a small minority of seconds on a shared file server.

    This probability is what makes benign activity rare rather than continuous,
    matching the observed distribution rather than the template maxima.
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


def describe_profiles():
    """Return the profile table for reporting alongside results."""
    return {
        "benign_templates": {
            name: {
                "shift": spec["hours"],
                "create": list(spec["create"]),
                "rename": list(spec["rename"]),
                "delete": list(spec["delete"]),
            }
            for name, spec in BENIGN_TEMPLATES.items()
        },
        "ransomware_families": {
            name: {
                "create": list(spec["create"]),
                "rename": list(spec["rename"]),
                "delete": list(spec["delete"]),
                "profile": spec["profile"],
                "persistence": spec["persistence"],
                "intensity": spec["intensity"],
                "note": spec["note"],
            }
            for name, spec in RANSOMWARE_FAMILIES.items()
        },
        "source": {
            "benign": "base paper Table 3, transcribed",
            "attack": "modelled from published family characteristics; not captured "
                      "from executed malware",
        },
    }