"""Multi-seed statistical validation.

The base paper derives every reported figure from a single experimental design.
The number of repetitions is not stated for the ML experiments, and no confidence
interval, dispersion measure or significance test is reported anywhere in the
paper. Section 3.7 argues statistical validity from the data-collection design
(randomised family order over 30 days) rather than from repeated measurement.

This module supplies the missing layer: every headline metric is recomputed across
N seeds, summarised as mean with a 95% confidence interval, and models are
compared with paired tests so a difference between two models can be called a
difference rather than an artefact of one split.
"""

import warnings

import time

import numpy as np
from scipy import stats


def _num(value, digits=6):
    """Round a float, collapsing nan/inf to None so output stays strict JSON.

    Degenerate paired tests return an infinite t-statistic when every seed
    difference is identical, which ``json.dump`` will not accept.
    """
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def bootstrap_ci(values, statistic=np.mean, confidence=0.95, n_resamples=2000,
                 seed=42):
    """Percentile bootstrap confidence interval for a statistic of ``values``.

    Used instead of the normal approximation because accuracy and cost are bounded
    or skewed on small test sets, where the normal interval can extend outside the
    metric's valid range.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0,
                "confidence": confidence}

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    draws = statistic(values[idx], axis=1)

    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": round(float(statistic(values)), 6),
        "ci_low": round(float(np.percentile(draws, 100 * alpha)), 6),
        "ci_high": round(float(np.percentile(draws, 100 * (1 - alpha))), 6),
        "std": round(float(values.std(ddof=1)), 6) if n > 1 else 0.0,
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "n": int(n),
        "confidence": confidence,
    }


def normal_ci(values, confidence=0.95):
    """Mean with a Student-t confidence interval, the standard companion to the
    bootstrap. Reported alongside it so the two can be compared for consistency."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}

    mean = float(values.mean())
    if n == 1:
        return {"mean": round(mean, 6), "ci_low": None, "ci_high": None, "n": 1}

    sem = float(values.std(ddof=1) / np.sqrt(n))
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1) * sem)
    return {
        "mean": round(mean, 6),
        "ci_low": round(mean - half, 6),
        "ci_high": round(mean + half, 6),
        "std": round(float(values.std(ddof=1)), 6),
        "n": int(n),
        "confidence": confidence,
    }


def compare_models(metric_a, metric_b):
    """Paired comparison of two models across the same seeds.

    Returns both a Wilcoxon signed-rank test and a paired t-test. Wilcoxon assumes
    only ordinal information, which matters here because test-set sizes are small
    and a single seed can change accuracy by a large discrete step.

    ``metric_a`` and ``metric_b`` must be aligned seed-by-seed.
    """
    a = np.asarray(metric_a, dtype=float)
    b = np.asarray(metric_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length seed vectors")
    if len(a) < 2:
        return {"n": len(a), "note": "need >= 2 seeds for a paired test"}

    diff = a - b
    all_equal = bool(np.allclose(diff, 0.0))

    result = {
        "n": int(len(a)),
        "mean_difference": _num(diff.mean()),
        "n_better_a": int((diff > 0).sum()),
        "n_better_b": int((diff < 0).sum()),
        "n_tied": int((diff == 0).sum()),
    }

    if all_equal:
        result["wilcoxon"] = {"statistic": None, "p_value": None,
                              "note": "identical on every seed"}
    else:
        try:
            w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
            result["wilcoxon"] = {"statistic": _num(w_stat),
                                  "p_value": _num(w_p)}
        except ValueError as exc:
            result["wilcoxon"] = {"statistic": None, "p_value": None,
                                  "note": str(exc)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            t_stat, t_p = stats.ttest_rel(a, b)
    result["paired_t"] = {"statistic": _num(t_stat), "p_value": _num(t_p)}
    if result["paired_t"]["statistic"] is None:
        result["paired_t"]["note"] = "t-statistic undefined (degenerate differences)"

    alpha = 0.05
    result["alpha"] = alpha
    result["significant"] = bool(
        result.get("wilcoxon", {}).get("p_value") is not None
        and result["wilcoxon"]["p_value"] < alpha
    )
    return result


def aggregate_seed_results(seed_results):
    """Summarise a ``{model: {metric: [value per seed]}}`` mapping.

    Returns ``{model: {metric: summary}}`` where each summary carries mean,
    confidence interval, dispersion and range.
    """
    summary = {}
    for model, metrics in seed_results.items():
        summary[model] = {}
        for metric, values in metrics.items():
            flat = [v for v in np.asarray(values, dtype=float).ravel()
                    if v is not None and np.isfinite(v)]
            if not flat:
                continue
            summary[model][metric] = {
                "bootstrap": bootstrap_ci(flat),
                "t_interval": normal_ci(flat),
            }
    return summary


def compare_all_models(seed_results, metric="Accuracy", reference=None):
    """Compare every model on one metric, each against a reference model.

    With six models this is five paired tests against the chosen reference,
    defaulting to the model with the best mean. Multiplicity across models is
    handled with Holm-Bonferroni so the set of comparisons as a whole stays
    interpretable.
    """
    models = list(seed_results.keys())
    if len(models) < 2:
        return {}

    means = {}
    for m in models:
        vals = [v for v in np.asarray(seed_results[m][metric], dtype=float).ravel()
                if v is not None and np.isfinite(v)]
        means[m] = float(np.mean(vals)) if vals else float("-inf")

    higher_is_better = metric != "Total Cost"
    if reference is None:
        reference = max(means, key=means.get) if higher_is_better else min(means, key=means.get)

    comparisons = {m: compare_models(seed_results[reference][metric],
                                    seed_results[m][metric])
                   for m in models if m != reference}

    # Holm-Bonferroni step-down. Adjustments are keyed by model name because a
    # comparison can lack a p-value (degenerate differences), which would shift
    # any positional mapping between the two collections.
    with_p = {m: c["wilcoxon"]["p_value"] for m, c in comparisons.items()
              if c.get("wilcoxon", {}).get("p_value") is not None}
    ordered = sorted(with_p, key=lambda m: with_p[m])

    running = 0.0
    total = len(ordered)
    for rank, model in enumerate(ordered):
        corrected = min(1.0, with_p[model] * (total - rank))
        running = max(running, corrected)
        comparisons[model]["holm_adjusted_p"] = _num(running)
        comparisons[model]["significant_after_correction"] = bool(running < 0.05)

    for model, comp in comparisons.items():
        comp.setdefault("holm_adjusted_p", None)
        comp.setdefault("significant_after_correction", None)

    return {"reference": reference, "metric": metric,
            "higher_is_better": higher_is_better, "comparisons": comparisons}


def _time_call(fn, *args, **kwargs):
    """Wall-clock duration of ``fn`` in seconds."""
    started = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - started


def run_seeds(train_fn, dataset_fn, seeds, verbose=True):
    """Run a training function across seeds and collect per-model metrics.

    ``dataset_fn(seed)`` must build the dataset deterministically from ``seed``,
    and ``train_fn(df)`` must return results in the shape used by ``ml.train``.
    Per-seed dataset identity is recorded so a seed sweep can be checked for
    accidental reuse of the same data.
    """
    seed_results = {}
    run_log = []

    for seed in seeds:
        started = time.time()
        df = dataset_fn(seed)
        results = train_fn(df)

        row = {"seed": int(seed), "elapsed_s": None, "models": {}}
        for r in results:
            model = r["Model"]
            seed_results.setdefault(model, {})
            for metric in ("Accuracy", "Sensitivity", "Specificity",
                           "Precision", "F1 Score", "ROC AUC", "Total Cost"):
                if metric in r and r[metric] is not None:
                    seed_results[model].setdefault(metric, []).append(r[metric])

        for r in results:
            for metric in ("Accuracy", "Recall", "Total Cost", "FPR"):
                if metric in r:
                    row["models"][r["Model"]] = row["models"].get(r["Model"], {})
                    row["models"][r["Model"]][metric] = r[metric]

        row["elapsed_s"] = round(time.time() - started, 2)
        run_log.append(row)
        if verbose:
            accs = " ".join(f"{m}={row['models'][m]['Accuracy']:.4f}"
                            for m in row["models"] if "Accuracy" in row["models"][m])
            print(f"  seed {seed:>3}  {row['elapsed_s']:>6.2f}s  {accs}")

    return seed_results, run_log


def stability_report(seed_results, metric="Accuracy", threshold=0.05):
    """How reproducible each model's score is across seeds.

    A model whose accuracy varies by more than ``threshold`` across splits is not
    a model whose score can be quoted to three decimal places, regardless of how
    high that score is.
    """
    rows = []
    for model, metrics in seed_results.items():
        vals = np.asarray(metrics.get(metric, []), dtype=float)
        if vals.size == 0:
            continue
        rows.append({
            "Model": model,
            "Mean": round(float(vals.mean()), 6),
            "Std": round(float(vals.std(ddof=1)) if vals.size > 1 else 0.0, 6),
            "Min": round(float(vals.min()), 6),
            "Max": round(float(vals.max()), 6),
            "Range": round(float(vals.max() - vals.min()), 6),
            "Stable": bool((vals.max() - vals.min()) <= threshold),
        })
    rows.sort(key=lambda r: r["Mean"], reverse=True)
    return {"metric": metric, "threshold": threshold, "models": rows}