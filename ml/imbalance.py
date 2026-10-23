"""Class-imbalance sweep.

The base paper states that its data is "naturally imbalanced" (Sect. 1.1) and
that the framework is "naturally robust to imbalanced data streams" (Table 9), but
it reports a single dataset and never states the ratio it observed. Robustness to
imbalance is therefore asserted rather than measured.

This module trains the model family at several attack fractions so that claim can
be checked. Each ratio is a separate trace, because subsampling a balanced trace
keeps the temporal structure of the surviving episodes intact while changing the
class prior -- which is the realistic way a deployment sees a rarer attack.
"""

import time

import numpy as np
import pandas as pd

from ml.stats import bootstrap_ci
from ml.traces import DATASET_VARIANTS, build_report, build_trace, group_aware_split

DEFAULT_RATIOS = (0.01, 0.05, 0.20)


def build_at_ratio(ratio, n_windows=20000, seed=42, variant="nozeroatt",
                   add_features=True, base_df=None, **trace_kwargs):
    """Build a trace at a requested attack fraction and apply a dataset variant.

    Two routes are available and they answer different questions:

    fresh
        Generate a new trace with ``attack_ratio`` requested. This is a genuinely
        different workload at each ratio.

    subsampled
        Reuse ``base_df`` and drop attack episodes until the fraction is close to
        ``ratio``. Episode lengths and the benign background are held fixed, so
        the only variable is how much attack traffic is present.

    Subsampling is used by the sweep because a freshly generated trace cannot
    deliver a low attack ratio: the episode-aware split requires a minimum number
    of episodes, and that floor occupies a fixed fraction of a short trace
    regardless of what was requested. Generating at four ratios and getting four
    identical datasets measures nothing.

    ``add_features`` applies the sequence feature construction, which the
    evaluation pipeline requires.
    """
    import pandas as pd

    if base_df is not None:
        df = _subsample_attack(base_df, ratio, rng_seed=seed)
    else:
        df = build_trace(n_windows=n_windows, attack_ratio=ratio, seed=seed,
                         **trace_kwargs)

    if add_features:
        from ml.streaming import add_sequence_features
        df = add_sequence_features(df)

    variant_fn = DATASET_VARIANTS[variant]
    return variant_fn(df), df


def _subsample_attack(df, ratio, rng_seed=42):
    """Drop whole attack episodes until the attack fraction reaches ``ratio``."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(rng_seed)

    attack = df[df["label"] == 1]
    benign = df[df["label"] == 0]
    if len(attack) == 0 or len(benign) == 0:
        return df.copy()

    target_attack = int(round(ratio * len(df)))
    if target_attack >= len(attack):
        return df.copy()

    # Keep whole episodes so the episode-aware split remains meaningful.
    episodes = sorted(attack["episode_id"].unique())
    lengths = attack.groupby("episode_id").size().to_dict()
    rng.shuffle(episodes)

    kept, running = [], 0
    for ep in episodes:
        if running >= target_attack:
            break
        kept.append(ep)
        running += lengths[ep]

    if not kept:
        return df.copy()

    keep_ids = set(kept)
    mask = (df["label"] == 0) | (df["episode_id"].isin(keep_ids))
    reduced = df[mask].copy()
    reduced["window_index"] = np.arange(len(reduced))
    return reduced


def sweep(n_windows=20000, ratios=DEFAULT_RATIOS, seed=42, variant="nozeroatt",
          train_fn=None, metrics_fn=None, n_seeds=1, verbose=True,
          **trace_kwargs):
    """Train and evaluate across attack fractions.

    ``train_fn(train_df, val_df, test_df)`` must return per-model results and
    ``metrics_fn`` is unused when results already carry metrics. Kept as plain
    callables so this module does not depend on ``ml.train``'s internals.
    """
    rows = []
    for ratio in ratios:
        started = time.perf_counter()
        variant_df, raw_df = build_at_ratio(ratio, n_windows=n_windows, seed=seed,
                                            variant=variant, **trace_kwargs)
        report = build_report(raw_df, requested_ratio=ratio)

        train_df, val_df, test_df = group_aware_split(variant_df, 0.1, 0.1, seed=seed)

        entry = {
            "Requested Ratio": ratio,
            "Achieved Ratio": report["Attack Fraction"],
            "Train Windows": int(len(train_df)),
            "Val Windows": int(len(val_df)),
            "Test Windows": int(len(test_df)),
            "Train Attack Count": int((train_df["label"] == 1).sum()),
            "Test Attack Count": int((test_df["label"] == 1).sum()),
            "Test Episodes": int(test_df[test_df["label"] == 1]["episode_id"].nunique()),
            "Elapsed (s)": None,
            "Models": {},
        }

        if train_fn is not None:
            results = train_fn(train_df, val_df, test_df)
            entry["Models"] = {
                r["Model"]: {
                    "Accuracy": r.get("Accuracy"),
                    "Sensitivity": r.get("Sensitivity"),
                    "Specificity": r.get("Specificity"),
                    "F1 Score": r.get("F1 Score"),
                    "Total Cost": r.get("Total Cost"),
                }
                for r in results
            }

        entry["Elapsed (s)"] = round(time.perf_counter() - started, 2)
        rows.append(entry)

        if verbose:
            print(f"ratio={ratio:.2f} achieved={report['Attack Fraction']:.4f} "
                  f"train_attack={entry['Train Attack Count']} "
                  f"test_attack={entry['Test Attack Count']} "
                  f"({entry['Elapsed (s)']}s)")

    return rows


def sensitivity_table(rows, model="XGBoost"):
    """Pivot a sweep into a compact ratio-versus-sensitivity table."""
    records = []
    for row in rows:
        stats = row["Models"].get(model, {})
        if not stats:
            continue
        records.append({
            "Model": model,
            "Requested Ratio": row["Requested Ratio"],
            "Achieved Ratio": row["Achieved Ratio"],
            "Accuracy": stats.get("Accuracy"),
            "Sensitivity": stats.get("Sensitivity"),
            "Specificity": stats.get("Specificity"),
            "Total Cost": stats.get("Total Cost"),
        })
    return pd.DataFrame(records)


def summarise_imbalance(rows, metric="Sensitivity"):
    """Report the spread of a metric across ratios.

    A framework that is genuinely robust to imbalance shows a metric that holds
    steady as the attack fraction shrinks. A metric that degrades sharply is
    evidence against the robustness claim, and the degradation is reported rather
    than hidden.
    """
    per_model = {}
    for row in rows:
        for model, stats in row["Models"].items():
            per_model.setdefault(model, []).append(stats.get(metric))

    summary = {}
    for model, values in per_model.items():
        clean = [v for v in values if v is not None]
        entry = {"metric": metric,
                 "by_ratio": {r["Requested Ratio"]: r["Models"].get(model, {}).get(metric)
                              for r in rows}}
        # bootstrap_ci returns a dict; it must be merged, not unpacked inside a
        # dict literal, which is not valid syntax.
        if clean:
            entry.update(bootstrap_ci(clean))
        summary[model] = entry
    return summary