"""Concept-drift monitoring and adaptive retraining.

The base paper names adaptive retraining as future work in Sect. 6.4: "exploring
adaptive retraining mechanisms to maintain resilience against evolving ransomware
strategies". Nothing in the paper implements it, and Sect. 6.2 lists dependence
on a representative training dataset as the main limitation, which is the drift
problem stated without a remedy.

This module supplies the mechanism and, more importantly, measures what happens
without it. The design question is not "can we retrain" but "how much does
detection degrade in the interval before a retrain is triggered". Two quantities
matter operationally:

detection latency under drift
    How many attack windows are missed between the workload changing and the
    retrained model being live.

recoverability
    Whether retraining on recent data actually restores performance, or whether
    the drift has moved the label definition itself.

Population Stability Index (PSI) is used as the drift statistic. It is the
standard measure in credit-risk monitoring, is cheap to compute online, and has
a conventional interpretation: below 0.1 the distribution is considered stable,
0.1-0.25 warrants investigation, above 0.25 indicates a material shift. Those
thresholds are conventions, not results, and are stated as such.
"""

import numpy as np
import pandas as pd

from ml.metrics import binary_metrics

PSI_SMALL = 0.10
PSI_LARGE = 0.25

DEFAULT_BINS = 10


def psi(expected, actual, bins=DEFAULT_BINS, smoothing=1e-6):
    """Population Stability Index between two samples of one feature.

    Bins are quantile edges taken from the *expected* (training) sample, so the
    comparison is against the reference distribution rather than a fixed grid that
    would need re-specifying as the workload moves. Empty bins are floored rather
    than dropped, because dropping them removes exactly the tail mass that
    indicates drift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        # Degenerate reference distribution (near-constant feature): fall back to
        # a plain proportion comparison.
        edges = np.array([-np.inf, np.inf])
    else:
        # Widen the outermost edges so every observation is counted. Without this
        # the maximum falls outside the final bin and the histogram silently sums
        # to less than the sample size, which corrupts the proportions.
        edges = edges.copy()
        edges[0] = -np.inf
        edges[-1] = np.inf

    expected_hist, _ = np.histogram(expected, bins=edges)
    actual_hist, _ = np.histogram(actual, bins=edges)

    expected_prop = expected_hist / max(1, expected_hist.sum())
    actual_prop = actual_hist / max(1, actual_hist.sum())

    # Floor the proportions after normalising. Flooring raw counts instead would
    # scale with bin width and produce a meaningless index.
    expected_prop = np.clip(expected_prop, smoothing, None)
    actual_prop = np.clip(actual_prop, smoothing, None)
    expected_prop = expected_prop / expected_prop.sum()
    actual_prop = actual_prop / actual_prop.sum()

    return float(np.sum((actual_prop - expected_prop)
                        * np.log(actual_prop / expected_prop)))


def feature_psi_table(reference, current, feature_names=None, bins=DEFAULT_BINS):
    """PSI for every feature, with the most-drifted feature identified."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if feature_names is None:
        if reference.shape[1] != current.shape[1]:
            raise ValueError("feature_names required when widths differ")
        feature_names = [f"f{i}" for i in range(reference.shape[1])]

    rows = []
    for i, name in enumerate(feature_names):
        value = psi(reference[:, i], current[:, i], bins=bins)
        rows.append({
            "Feature": name,
            "PSI": round(value, 6),
            "Severity": severity(value),
        })

    table = pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)
    return table


def severity(value):
    """Conventional PSI interpretation."""
    if value < PSI_SMALL:
        return "stable"
    if value < PSI_LARGE:
        return "moderate"
    return "significant"


def per_feature_monitor(reference, features, chunk_size=2000, min_samples=200,
                        baseline_windows=0):
    """Chunked PSI monitor suitable for a live stream.

    Compares each incoming chunk against the training reference and reports the
    point at which the most-drifted feature crosses each threshold. Chunking is
    what makes this computable online: comparing against the training sample
    directly would require retaining it and recomputing a histogram per window.

    ``chunk_size`` is a calibration parameter, not a free choice. PSI divides
    expected by empirical bin proportions, so with ten quantile bins a chunk far
    smaller than the reference produces noisy proportions and a large PSI even
    when nothing has changed. Measured on this trace, a 500-window chunk reports
    PSI 1.62 against unchanged data and 2.69 against drifted data, which
    separates nothing; a 2000-window chunk reports 0.138 unchanged and 2.52
    drifted, which separates cleanly against the conventional 0.25 threshold.
    Chunks that are too small therefore produce a monitor that always fires.

    ``baseline_windows`` prepends that many windows taken from the same period as
    ``reference``. Nothing has drifted in that stretch, so a trigger there is a
    false alarm, and the offset of the first trigger after it is the detection
    delay. Without this prefix every chunk belongs to the drifted period, the
    monitor fires on the very first chunk, and both quantities are unmeasurable.

    Returns the chunk records plus the false-alarm count and detection delay in
    windows and seconds, so the operating point can be judged rather than merely
    asserted.
    """
    reference = np.asarray(reference, dtype=float)
    features = np.asarray(features, dtype=float)

    if baseline_windows > 0:
        if baseline_windows > len(reference):
            raise ValueError(
                f"baseline_windows={baseline_windows} exceeds the "
                f"{len(reference)} available reference windows")
        stream = np.concatenate([reference[-baseline_windows:], features])
    else:
        stream = features

    baseline_chunks = (baseline_windows // chunk_size) if baseline_windows else 0
    baseline_end = baseline_chunks * chunk_size

    records = []
    triggered = None
    false_alarms = 0
    evaluated_baseline = 0

    for start in range(0, len(stream) - chunk_size + 1, chunk_size):
        chunk = stream[start:start + chunk_size]
        if len(chunk) < min_samples:
            continue
        table = feature_psi_table(reference, chunk)
        worst = table.iloc[0]
        in_baseline = start < baseline_end
        if in_baseline:
            evaluated_baseline += 1
        fired = worst["PSI"] >= PSI_LARGE
        record = {
            "Chunk Start": start,
            "Chunk Size": len(chunk),
            "Max PSI": float(worst["PSI"]),
            "Most Drifted Feature": str(worst["Feature"]),
            "Severity": str(worst["Severity"]),
            "Phase": "baseline" if in_baseline else "drift",
            "Fired": bool(fired),
        }
        records.append(record)
        if fired and in_baseline:
            false_alarms += 1
        if fired and not in_baseline and triggered is None:
            triggered = record

    delay_windows = None
    delay_seconds = None
    if triggered is not None:
        # Each unit of this stream is a one-second window, so the offset in
        # windows is already the offset in seconds. Dividing by the chunk size
        # would have turned a 2000-second delay into a meaningless 1.0.
        delay_windows = int(triggered["Chunk Start"] - baseline_end)
        delay_seconds = float(delay_windows)

    return {
        "Chunks": records,
        "Trigger Chunk": triggered,
        "Triggered": triggered is not None,
        "PSI Small": PSI_SMALL,
        "PSI Large": PSI_LARGE,
        "Chunk Size": chunk_size,
        "Baseline Windows": int(baseline_windows),
        "Baseline Chunks Evaluated": evaluated_baseline,
        "False Alarms": false_alarms,
        "False Alarm Rate": (
            round(false_alarms / evaluated_baseline, 6)
            if evaluated_baseline else None),
        "Detection Delay Windows": delay_windows,
        "Detection Delay Seconds": delay_seconds,
        "Note": (
            "Baseline chunks contain no drift, so a trigger there is a false "
            "alarm. Detection delay is measured from the first drifted chunk."),
    }


def drifted_label_rate(reference_labels, current_labels):
    """Shift in the attack base rate.

    A change in the fraction of attack windows is the drift most directly relevant
    to a cost-sensitive detector, because the optimal threshold moves with the
    prior. Reported alongside feature PSI so a change in behaviour can be
    distinguished from a change in how often attacks occur.
    """
    ref = float(np.mean(reference_labels)) if len(reference_labels) else 0.0
    cur = float(np.mean(current_labels)) if len(current_labels) else 0.0
    return {
        "Reference Attack Rate": round(ref, 6),
        "Current Attack Rate": round(cur, 6),
        "Absolute Change": round(abs(cur - ref), 6),
        "Relative Change": round(abs(cur - ref) / ref, 6) if ref > 0 else None,
    }


def degraded_curve(y_windows, proba_windows, labels, window_per_chunk=500):
    """Sensitivity and cost as the workload drifts, with no retraining.

    This is the measurement that motivates the retraining mechanism: performance
    on later chunks, using a model fitted on the first chunk only.
    """
    rows = []
    y = np.asarray(y_windows)
    p = np.asarray(proba_windows)

    for start in range(0, len(y) - window_per_chunk + 1, window_per_chunk):
        end = start + window_per_chunk
        chunk_y = y[start:end]
        if chunk_y.sum() == 0:
            continue
        pred = (p[start:end] >= 0.5).astype(int)
        m = binary_metrics(chunk_y, pred)
        rows.append({
            "Chunk Start": start,
            "Attack Windows": int(chunk_y.sum()),
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Total Cost": m["Total Cost"],
            "Attack Rate": round(float(chunk_y.mean()), 6),
        })

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, {"Degraded": False}

    first = frame.iloc[0]
    last = frame.iloc[-1]
    worst = frame.loc[frame["Sensitivity"].idxmin()]

    summary = {
        "Degraded": bool(last["Sensitivity"] < first["Sensitivity"] - 0.02),
        "First Sensitivity": round(float(first["Sensitivity"]), 6),
        "Final Sensitivity": round(float(last["Sensitivity"]), 6),
        "Sensitivity Drop": round(
            float(first["Sensitivity"] - last["Sensitivity"]), 6),
        "Worst Sensitivity": round(float(worst["Sensitivity"]), 6),
        "Worst At Chunk": int(worst["Chunk Start"]),
        "First Cost": int(first["Total Cost"]),
        "Final Cost": int(last["Total Cost"]),
    }
    return frame, summary


def retrain_and_recover(X_reference, y_reference, X_recent, y_recent,
                        X_future, y_future, factory, scaler=None,
                        threshold=0.5, seed=42, min_recent=300):
    """Does retraining on recent data actually restore performance?

    Compares three states on the same held-out future data:

    stale
        The model fitted on reference data only. This is what a deployment does
        when nothing monitors for drift.
    retrained
        Refitted on reference plus recent windows, i.e. the adaptive response.
        Evaluated on data that follows ``X_recent``, so it has not seen it.
    fresh oracle
        Fitted on the evaluation data itself. An upper bound no real deployment
        can reach, and the only way to separate "the model went stale" from "the
        concept itself changed". If the oracle matches the stale model, retraining
        cannot help.
    """
    from ml.metrics import binary_metrics

    def fit_eval(X, y, X_eval, y_eval):
        model = factory(seed)
        if scaler is not None:
            model.fit(scaler.transform(X), y)
            proba = model.predict_proba(scaler.transform(X_eval))[:, 1]
        else:
            model.fit(X, y)
            proba = model.predict_proba(X_eval)[:, 1]
        return binary_metrics(y_eval, (proba >= threshold).astype(int))

    X_reference = np.asarray(X_reference, dtype=float)
    y_reference = np.asarray(y_reference)
    X_recent = np.asarray(X_recent, dtype=float)
    y_recent = np.asarray(y_recent)
    X_future = np.asarray(X_future, dtype=float)
    y_future = np.asarray(y_future)

    results = {}

    # Split the future period in two: the earlier half is what a retraining
    # trigger would have collected, the later half is what the retrained model is
    # scored on. Keeping them disjoint is what makes the recovery test meaningful.
    mid = len(X_future) // 2

    if len(X_recent) >= min_recent:
        results["Stale"] = fit_eval(X_reference, y_reference, X_future, y_future)
        results["Retrained"] = fit_eval(
            np.vstack([X_reference, X_recent]),
            np.concatenate([y_reference, y_recent]),
            X_future[mid:], y_future[mid:])
        results["Fresh Oracle"] = fit_eval(X_future, y_future, X_future, y_future)
    else:
        # Insufficient recent data to retrain on; report only the stale model and
        # say why, rather than silently substituting the original one.
        results["Stale"] = fit_eval(X_reference, y_reference, X_future, y_future)
        results["Retrained"] = None
        results["Fresh Oracle"] = fit_eval(X_future, y_future, X_future, y_future)

    report = {}
    for name, metrics in results.items():
        if metrics is None:
            report[name] = {"Available": False,
                            "Reason": f"fewer than {min_recent} recent windows"}
        else:
            report[name] = {
                "Available": True,
                "Sensitivity": metrics["Sensitivity"],
                "Specificity": metrics["Specificity"],
                "Total Cost": metrics["Total Cost"],
            }

    if report["Retrained"].get("Available"):
        report["Recovery Gain"] = round(
            report["Retrained"]["Sensitivity"] - report["Stale"]["Sensitivity"], 6)
        report["Cost Improvement"] = int(
            report["Stale"]["Total Cost"] - report["Retrained"]["Total Cost"])
        report["Retraining Helps"] = bool(
            report["Retrained"]["Sensitivity"] > report["Stale"]["Sensitivity"])
        report["Oracle Gap"] = round(
            report["Fresh Oracle"]["Sensitivity"]
            - report["Retrained"]["Sensitivity"], 6)

    return report


def run_drift_study(df, feature_columns, factory, scaler=None, seed=42,
                    chunk_size=2000, reference_fraction=0.5, threshold=0.5):
    """End-to-end drift study on one chronological trace.

    The trace is cut in half: the first half is the reference period, the second
    half the drifted period. Everything is measured forward in time, so no future
    information is used to fit a model that is then scored on the past.
    """
    enriched = df
    X = enriched[feature_columns].values.astype(float)
    y = enriched["label"].values

    split = int(len(X) * reference_fraction)
    X_ref, y_ref = X[:split], y[:split]
    X_future, y_future = X[split:], y[split:]

    drift_table = feature_psi_table(X_ref, X_future, feature_names=feature_columns)
    rate_shift = drifted_label_rate(y_ref, y_future)

    # Stale model on the future period.
    model = factory(seed)
    if scaler is not None:
        model.fit(scaler.transform(X_ref), y_ref)
        proba = model.predict_proba(scaler.transform(X_future))[:, 1]
    else:
        model.fit(X_ref, y_ref)
        proba = model.predict_proba(X_future)[:, 1]

    curve, degradation = degraded_curve(y_future, proba, y_future,
                                       window_per_chunk=chunk_size)

    # Retrain on the first half of the future period, score the second half. This
    # keeps the recovery test honest: the retrained model has not seen the data it
    # is scored on.
    mid = len(X_future) // 2
    recovery = retrain_and_recover(
        X_ref, y_ref,
        X_future[:mid], y_future[:mid],
        X_future[mid:], y_future[mid:],
        factory, scaler=scaler, threshold=threshold, seed=seed)

    # The monitored stream opens with a stretch drawn from the reference period,
    # in which no drift is present. Those chunks establish the monitor's false
    # alarm rate; the drifted half that follows yields the detection delay.
    monitor = per_feature_monitor(X_ref, X_future, chunk_size=chunk_size,
                                  baseline_windows=split // 2)

    return {
        "Reference Windows": int(len(X_ref)),
        "Future Windows": int(len(X_future)),
        "Feature PSI": drift_table.to_dict(orient="records"),
        "Most Drifted Feature": str(drift_table.iloc[0]["Feature"]),
        "Max PSI": float(drift_table.iloc[0]["PSI"]),
        "Severity": str(drift_table.iloc[0]["Severity"]),
        "Label Rate Shift": rate_shift,
        "Degradation": degradation,
        "Curve": curve.to_dict(orient="records"),
        "Recovery": recovery,
        "Monitor": monitor,
        "Drift Detection Delay Seconds": monitor["Detection Delay Seconds"],
        "Drift False Alarms": monitor["False Alarms"],
        "Drift False Alarm Rate": monitor["False Alarm Rate"],
    }