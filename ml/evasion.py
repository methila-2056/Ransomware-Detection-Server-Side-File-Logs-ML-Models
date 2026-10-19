"""Adversarial evaluation: rate-limited and evasive attack traffic.

The base paper names this as its principal unaddressed limitation. Sect. 6.2
lists "dependent on availability of a sufficiently representative training
dataset" as the main limitation, and the surrounding discussion concedes that
count-based features are the weak point. Sect. 6.3 adds that precise data-loss
quantification is impossible under one-second aggregation, and that reducing the
window or adding predictive execution control would be future work.

Nothing in the paper measures how far a count-based detector degrades when an
attacker deliberately slows down. This module does that.

The evaluation is a detection-rate curve rather than a single number: the attack
rate is varied continuously from full speed down to a slow trickle, and the
detector's sensitivity is measured at each rate. The question is where the curve
collapses, and whether a rate-limited attack becomes indistinguishable from
benign activity.

Two regimes are reported:

``opportunistic``
    Encryption limited to a fixed files-per-second budget, which is what a
    throttled ransomware binary or an intentionally patient operator produces.

``mimicry``
    Counts drawn to match the local benign distribution, so the attack is not
    merely slower but statistically shaped like normal work. This is the harder
    case and the one that matters for the paper's claim that the method is
    "insensitive to obfuscation".

A defensible claim either way is only available if the measurement exists, so
the module reports the blind spot rather than assuming it.
"""

import numpy as np
import pandas as pd

from ml.attack_sim import (active_probability, diurnal_factor,
                            idle_probability, sample_benign_window)
from ml.metrics import binary_metrics
from ml.streaming import add_causal_features, add_sequence_features, \
    causal_feature_columns, sequence_feature_columns
from ml.traces import DATASET_VARIANTS, group_aware_split

COUNT_COLUMNS = ["create_count", "rename_count", "delete_count"]


def rate_limited_episode(rng, n_windows, files_per_second, start_hour=9,
                         family="lockbit", active_fraction=0.9):
    """Build one attack episode whose absolute rate is capped.

    ``files_per_second`` is the ceiling on total operations per second. Windows
    beyond it draw zero, so the attack's volume per second is bounded by
    construction rather than merely reduced on average.
    """
    rows = []
    for i in range(n_windows):
        hour = (start_hour + i // 3600) % 24
        if rng.random() > active_fraction:
            rows.append({"create_count": 0, "rename_count": 0,
                         "delete_count": 0, "label": 1, "episode_id": 0,
                         "family": family, "is_stealth": True,
                         "benign_template": "rate_limited",
                         "window_index": i})
            continue

        # Distribute the budget across the three operations in the proportion
        # ransomware actually uses: a new encrypted file plus a rename of the
        # original plus eventual unlink of the staged copy.
        budget = int(files_per_second)
        create = int(budget * 0.45)
        rename = int(budget * 0.30)
        delete = budget - create - rename

        jitter = rng.uniform(0.7, 1.0)
        rows.append({
            "create_count": int(create * jitter),
            "rename_count": int(rename * jitter),
            "delete_count": int(delete * jitter),
            "label": 1, "episode_id": 0, "family": family,
            "is_stealth": True, "benign_template": "rate_limited",
            "window_index": i,
        })
    return pd.DataFrame(rows)


def mimicry_episode(rng, n_windows, start_hour=9, family="teslacrypt",
                    target_quantile=0.75):
    """Build an attack episode whose counts follow the benign distribution.

    Each window draws benign counts for its hour and then perturbs the
    rename/delete components upwards, which is the only behavioural trace
    available to a count-based detector. The result is an attack with the same
    marginal distribution as normal work but a different operation mix.
    """
    rows = []
    for i in range(n_windows):
        hour = (start_hour + i // 3600) % 24

        if rng.random() < idle_probability(hour, rng) * 0.5:
            rows.append({"create_count": 0, "rename_count": 0,
                         "delete_count": 0, "label": 1, "episode_id": 0,
                         "family": family, "is_stealth": True,
                         "benign_template": "mimicry",
                         "window_index": i})
            continue

        benign = sample_benign_window(rng, hour,
                                      intensity=diurnal_factor(hour, rng))
        total = (benign["create"] + benign["rename"] + benign["delete"]) or 1

        # Shift the mix towards the rename-heavy profile ransomware leaves,
        # while keeping the total volume inside the benign envelope.
        shift = rng.uniform(0.10, 0.22) * target_quantile
        rename_share = min(0.85, (benign["rename"] + 1) / total + shift)
        delete_share = min(0.85, (benign["delete"] + 1) / total + shift * 0.7)

        create = max(0, int(total * (1.0 - rename_share - delete_share)))
        rename = max(0, int(total * rename_share))
        delete = max(0, int(total * delete_share))

        rows.append({
            "create_count": create, "rename_count": rename,
            "delete_count": delete, "label": 1, "episode_id": 0,
            "family": family, "is_stealth": True,
            "benign_template": "mimicry", "window_index": i,
        })
    return pd.DataFrame(rows)


def build_evasion_set(base_df, rng, rates=(60, 40, 25, 15, 10, 6, 4, 2, 1),
                      episode_length=180, include_mimicry=True,
                      start_hour=9, seed=7):
    """Assemble a trace of rate-limited episodes at several speeds."""
    episodes = []
    for rate in rates:
        for rep in range(2):
            ep = rate_limited_episode(rng, episode_length, rate,
                                      start_hour=start_hour)
            # Each speed gets its own regime label and episode id, so the curve
            # can be read per rate and group-aware splitting keeps whole speeds
            # on one side of the boundary.
            ep["episode_id"] = len(episodes)
            ep["regime"] = f"rate_limited_{rate}"
            ep["window_index"] = np.arange(len(ep))
            episodes.append(ep)

    if include_mimicry:
        for rep in range(2):
            ep = mimicry_episode(rng, episode_length, start_hour=start_hour)
            ep["episode_id"] = len(episodes)
            ep["regime"] = "mimicry"
            ep["window_index"] = np.arange(len(ep))
            episodes.append(ep)

    # Benign periods are placed between episodes, not sampled around them. The
    # previous loop walked the attack frame and skipped every row, which meant
    # that frame contained only attacks, so the skip condition was true for every
    # iteration and the benign filler came out empty. An evasion trace with no
    # benign windows cannot produce a specificity or a meaningful cost.
    def benign_window(i):
        hour = (start_hour + i // 3600) % 24
        if rng.random() < idle_probability(hour, rng):
            return {"create_count": 0, "rename_count": 0,
                    "delete_count": 0, "label": 0,
                    "episode_id": -1, "family": None,
                    "is_stealth": False,
                    "benign_template": "idle",
                    "regime": "benign"}
        if rng.random() >= active_probability(hour, rng):
            return {"create_count": rng.randint(0, 3),
                    "rename_count": rng.randint(0, 1),
                    "delete_count": rng.randint(0, 1),
                    "label": 0, "episode_id": -1, "family": None,
                    "is_stealth": False,
                    "benign_template": "low_activity",
                    "regime": "benign"}
        win = sample_benign_window(rng, hour,
                                   intensity=diurnal_factor(hour, rng))
        return {"create_count": win["create"],
                "rename_count": win["rename"],
                "delete_count": win["delete"],
                "label": 0, "episode_id": -1, "family": None,
                "is_stealth": False,
                "benign_template": win["template"],
                "regime": "benign"}

    # Lay the episodes out end to end with a benign run before each one, so each
    # attack follows ordinary traffic rather than another attack.
    rows = []
    cursor = 0
    for ep in episodes:
        gap = rng.randint(60, 240)
        for k in range(gap):
            rows.append(benign_window(cursor + k))
        rows.extend(ep.to_dict(orient="records"))
        cursor += gap + len(ep)
    for k in range(rng.randint(60, 240)):
        rows.append(benign_window(cursor + k))

    combined = pd.DataFrame(rows)
    combined["window_index"] = np.arange(len(combined))
    combined["episode_id"] = combined["episode_id"].fillna(-1).astype(int)
    combined["label"] = combined["label"].astype(int)
    return combined.reset_index(drop=True)


def evasion_curve(model, feature_columns, evasion_df, base_df, scaler=None,
                  threshold=0.5, seed=42, verbose=True):
    """Sensitivity as a function of attack rate.

    The model is trained on ``base_df`` -- ordinary traffic only -- and then
    applied unchanged to each rate. No evasion data is used for fitting, so a
    collapse is a genuine generalisation failure rather than a training artefact.

    ``base_df`` is expected to already carry the sequence features, since the
    same construction must be present on both the training and evasion frames.
    """
    # Split the enriched frame, not the raw one: the feature columns must already
    # exist on the subset that training reads from.
    enriched_base = add_sequence_features(base_df)
    train_df, _, _ = group_aware_split(
        DATASET_VARIANTS["nozeroatt"](enriched_base), 0.1, 0.1, seed=seed)

    X_train = train_df[feature_columns].values.astype(float)
    y_train = train_df["label"].values
    if scaler is not None:
        X_train = scaler.transform(X_train)
    model.fit(X_train, y_train)

    enriched = add_sequence_features(evasion_df)
    # The benign background is scored alongside every rate, so each row reports
    # that rate's sensitivity together with the specificity it was achieved at.
    benign_block = enriched[(enriched["label"] == 0)]

    rows = []
    for regime, group in enriched[enriched["label"] == 1].groupby("regime"):
        if regime == "benign":
            continue

        # Score each rate against the benign background, not against itself.
        # Scoring attack windows alone leaves no negatives, so specificity is
        # undefined and the cost figure silently becomes FN-only. Pairing the two
        # makes sensitivity, specificity and cost mutually comparable.
        block = pd.concat([group, benign_block], ignore_index=True)

        X = block[feature_columns].values.astype(float)
        if scaler is not None:
            X = scaler.transform(X)
        y = block["label"].values
        pred = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
        m = binary_metrics(y, pred)

        ops = float(group[COUNT_COLUMNS].sum(axis=1).mean())
        declared = regime.rsplit("_", 1)[-1]
        rate = float(declared) if declared.isdigit() else None

        rows.append({
            "Regime": regime,
            "Declared Files Per Second": rate,
            "Observed Ops Per Second": round(ops, 2),
            "Attack Windows": int((group["label"] == 1).sum()),
            "Benign Windows": int((block["label"] == 0).sum()),
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Total Cost": m["Total Cost"],
            "FP": m["FP"], "FN": m["FN"], "TP": m["TP"], "TN": m["TN"],
        })

    frame = pd.DataFrame(rows)
    # Order by observed volume, not by regime name: "rate_limited_1",
    # "rate_limited_10", "rate_limited_15" sort lexically into a sequence that
    # reads as if the rates were non-monotonic.
    frame = frame.sort_values(
        ["Observed Ops Per Second"], ascending=False).reset_index(drop=True)

    if verbose:
        print(f"{'regime':<20}{'declared':>9}{'obs ops/s':>11}"
              f"{'windows':>9}{'sens':>9}{'spec':>9}{'cost':>9}")
        for _, r in frame.iterrows():
            declared = "-" if r["Declared Files Per Second"] is None \
                else f"{r['Declared Files Per Second']:.0f}"
            print(f"{r['Regime']:<20}{declared:>9}{r['Observed Ops Per Second']:>11.1f}"
                  f"{r['Attack Windows']:>9}{r['Sensitivity']:>9.4f}"
                  f"{r['Specificity']:>9.4f}{r['Total Cost']:>9}")

    return frame


def blindspot_summary(frame, paper_reference_sensitivity=0.9187):
    """Where the detector stops working, stated as a number.

    The paper reports 91.87% sensitivity on unthrottled attack traffic and makes
    no claim about throttled traffic. This summary states the rate at which the
    detector's sensitivity falls below that reference, which is the point where
    the paper's reported performance no longer applies.
    """
    limited = frame[frame["Regime"].str.startswith("rate_limited")].copy()
    mimic = frame[frame["Regime"] == "mimicry"]

    summary = {
        "Paper Reference Sensitivity": paper_reference_sensitivity,
        "Rate Limited": {},
        "Mimicry": {},
    }

    if not limited.empty:
        limited = limited.sort_values("Declared Files Per Second",
                                      ascending=False)
        below = limited[limited["Sensitivity"] < paper_reference_sensitivity]
        weakest = limited.loc[limited["Sensitivity"].idxmin()]

        summary["Rate Limited"]["Fastest Rate Tested"] = float(
            limited["Declared Files Per Second"].max())
        summary["Rate Limited"]["Sensitivity At Fastest Rate"] = round(
            float(limited.iloc[0]["Sensitivity"]), 6)
        summary["Rate Limited"]["Weakest Sensitivity"] = round(
            float(weakest["Sensitivity"]), 6)
        summary["Rate Limited"]["Rate At Weakest"] = float(
            weakest["Declared Files Per Second"])

        # The highest rate at which the paper's reported sensitivity no longer
        # holds. Below this rate an attacker's throttling defeats the detector
        # to the point where the paper's reported performance is not available.
        summary["Rate Limited"]["Highest Rate Below Paper Reference"] = (
            float(below["Declared Files Per Second"].max())
            if not below.empty else None)
        summary["Rate Limited"]["Sensitivity At That Rate"] = (
            round(float(below.loc[below["Declared Files Per Second"].idxmax(),
                                  "Sensitivity"]), 6)
            if not below.empty else None)
        summary["Rate Limited"]["Rates Tested"] = [
            float(v) for v in limited["Declared Files Per Second"].tolist()]

    if not mimic.empty:
        summary["Mimicry"]["Sensitivity"] = round(
            float(mimic["Sensitivity"].iloc[0]), 6)
        summary["Mimicry"]["Cost"] = int(mimic["Total Cost"].iloc[0])
        summary["Mimicry"]["Below Paper Reference"] = bool(
            mimic["Sensitivity"].iloc[0] < paper_reference_sensitivity)

    return summary


def run(base_df, model_factory, feature_set="sequence", seed=42, scaler=None,
        threshold=0.5, rates=(60, 40, 25, 15, 10, 6, 4, 2, 1), verbose=True):
    """Full evasion study: train on ordinary traffic, attack at varying rates."""
    import random

    if feature_set == "sequence":
        base = add_sequence_features(base_df)
    elif feature_set == "causal":
        base = add_causal_features(base_df)
    else:
        base = base_df.copy()

    feature_columns = (sequence_feature_columns(base)
                       if feature_set == "sequence"
                       else causal_feature_columns(base)
                       if feature_set == "causal"
                       else list(COUNT_COLUMNS))

    rng = random.Random(seed)
    evasion = build_evasion_set(base_df, rng, rates=rates, seed=seed)

    model = model_factory(seed)
    frame = evasion_curve(model, feature_columns, evasion, base_df,
                          scaler=scaler, threshold=threshold, seed=seed,
                          verbose=verbose)
    summary = blindspot_summary(frame)

    return {
        "Feature Set": feature_set,
        "Threshold": threshold,
        "Curve": frame.to_dict(orient="records"),
        "Summary": summary,
    }