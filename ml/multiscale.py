"""Multi-scale window aggregation and the window-size ablation.

The base paper fixes a one-second aggregation window and states in Sect. 3.2 that
this "does not reflect system latency limitations but represents a practical design
decision". Sect. 6.3 then names shortening the window as future work, and Sect.
1 defines "quasi real-time" as millisecond-scale events aggregated into one-second
windows.

Two directions follow from that, and neither is evaluated in the paper:

shorter windows
    Sub-second aggregation trades statistical stability for latency. A one-second
    count is a sum of many operations and therefore stable; a 200ms count is
    noisier but detects sooner. Where the crossover lies is an empirical question.

overlapping windows
    The paper uses non-overlapping windows. Sliding windows with a stride shorter
    than their width observe every event in every window, which reduces the
    chance of an operation falling on a boundary. They multiply the sample count,
    so the cost of that redundancy needs measuring too.

This module generates windows at several durations and strides, and provides the
window-size ablation the paper's Sect. 6.3 lists as future work.
"""

import numpy as np
import pandas as pd

COUNT_COLUMNS = ["create_count", "rename_count", "delete_count"]

_OPERATIONS = (("create", "create_count"),
               ("rename", "rename_count"),
               ("delete", "delete_count"))


def trace_to_events(df, window_seconds=1.0, start=0.0, jitter=False,
                    seed=42):
    """Expand a window-level trace into one row per file operation.

    ``resample_windows`` and the sweep take an *event* frame, because the window
    width is the variable under study and cannot be pre-baked into the input. The
    rest of the pipeline works on aggregated windows, so a study of window width
    needs a way back down to events. This provides it.

    Each window ``i`` contributes exactly ``create_count + rename_count +
    delete_count`` events, spread uniformly across its span, so re-aggregating at
    the original width reproduces the original counts. Labels, episode ids and
    family travel with the events.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        base = start + i * window_seconds
        for op, col in _OPERATIONS:
            n = int(getattr(row, col))
            for k in range(n):
                offset = ((k + 1) / (n + 1)) * window_seconds if jitter else 0.0
                if jitter:
                    offset = float(rng.uniform(0.0, window_seconds))
                rows.append({
                    "timestamp": base + offset,
                    "operation": op,
                    "label": int(getattr(row, "label")),
                    "episode_id": int(getattr(row, "episode_id", -1)),
                    "family": getattr(row, "family", None),
                })
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["timestamp", "operation", "label",
                                      "episode_id", "family"])
    return frame.sort_values("timestamp").reset_index(drop=True)


def resample_windows(df, width_seconds=1, stride_seconds=1,
                     count_columns=None, label_mode="any"):
    """Aggregate an event-level frame into sliding windows.

    ``df`` must have one row per file *operation* with a timestamp in
    ``timestamp``, not one row per window. Aggregation happens here rather than
    upstream because the window width is the variable under study.

    ``label_mode`` decides how a window's label is derived when several events
    fall inside it:

    ``any``     label 1 if any event is an attack event. Appropriate for detection,
                where acting on a partially-observed attack is correct.
    ``majority`` label 1 if most events are attack events. Appropriate for
                attribution studies that want the window to describe its whole span.
    """
    cols = count_columns or COUNT_COLUMNS
    if len(cols) != 3:
        raise ValueError(f"expected create/rename/delete columns, got {cols}")

    frame = df.copy()
    if "timestamp" in frame.columns:
        # ml.ingest emits epoch *seconds*. Without an explicit unit, pandas
        # treats a bare number as nanoseconds, which compressed a 2000-second
        # trace into a single window and made the whole sweep meaningless.
        ts = pd.to_datetime(frame["timestamp"], unit="s")
    elif "window_index" in frame.columns:
        # Already windowed at one second; synthesise a timestamp from the index.
        ts = pd.to_datetime(frame["window_index"], unit="s", origin="2024-01-01")
    else:
        raise ValueError("frame needs a 'timestamp' or 'window_index' column")

    frame["_ts"] = ts
    frame = frame.sort_values("_ts")

    start = frame["_ts"].min()
    frame["_offset"] = (frame["_ts"] - start).dt.total_seconds()

    width = int(width_seconds)
    stride = int(stride_seconds)
    if stride <= 0:
        raise ValueError("stride_seconds must be positive")

    # Window index of each event, then aggregate by that index.
    frame["_win"] = (frame["_offset"] // stride).astype(int)

    rows = []
    total_span = int(frame["_offset"].max() // stride) + 1

    for w in range(total_span):
        lo = w * stride
        hi = lo + width
        # A window is complete only if its span is fully covered by data.
        if lo + width > frame["_offset"].max() + 1:
            continue
        chunk = frame[(frame["_offset"] >= lo) & (frame["_offset"] < hi)]
        if len(chunk) == 0:
            continue

        create = int((chunk["operation"] == "create").sum()) if "operation" in chunk \
            else int(chunk[cols[0]].sum())
        rename = int((chunk["operation"] == "rename").sum()) if "operation" in chunk \
            else int(chunk[cols[1]].sum())
        delete = int((chunk["operation"] == "delete").sum()) if "operation" in chunk \
            else int(chunk[cols[2]].sum())

        if "label" in chunk.columns:
            attack_events = int((chunk["label"] == 1).sum())
            if label_mode == "majority":
                label = int(attack_events * 2 > len(chunk))
            else:
                label = int(attack_events > 0)
        else:
            label = 0

        row = {
            "window_index": w,
            "window_start_s": lo,
            "create_count": create,
            "rename_count": rename,
            "delete_count": delete,
            "label": label,
            "event_count": len(chunk),
        }

        # Carry episode identity through the resampling so that the episode-aware
        # split can still keep whole attacks on one side of the boundary. Without
        # it a re-aggregated window loses the ability to be grouped, and the
        # evaluation silently falls back to a row split that leaks.
        if "episode_id" in chunk.columns:
            eps = chunk.loc[chunk["label"] == 1, "episode_id"] \
                if "label" in chunk.columns else chunk["episode_id"]
            row["episode_id"] = int(eps.iloc[0]) if len(eps) else -1
        if "family" in chunk.columns:
            fams = chunk.loc[chunk["label"] == 1, "family"] \
                if "label" in chunk.columns else chunk["family"]
            row["family"] = fams.iloc[0] if len(fams) else None
        if "is_stealth" in chunk.columns:
            rows_stealth = chunk.loc[chunk["label"] == 1, "is_stealth"] \
                if "label" in chunk.columns else chunk["is_stealth"]
            row["is_stealth"] = bool(rows_stealth.iloc[0]) if len(rows_stealth) else False

        rows.append(row)

    return pd.DataFrame(rows)


def window_sweep(model_factory, event_df, widths=(1, 2, 5, 10, 30),
                 strides=None, seed=42, threshold=0.5, scaler=None,
                 label_mode="any", verbose=True):
    """Train and evaluate at several aggregation widths.

    Reports sensitivity, detection latency and window count for each width, so the
    latency-versus-stability trade-off is visible rather than summarised into one
    preferred setting.
    """
    from ml.metrics import binary_metrics
    from ml.streaming import add_sequence_features, sequence_feature_columns
    from ml.traces import DATASET_VARIANTS, group_aware_split

    rows = []

    for width in widths:
        stride = strides.get(width) if strides else (1 if width <= 2 else width // 2)
        windows = resample_windows(event_df, width_seconds=width,
                                   stride_seconds=stride,
                                   label_mode=label_mode)
        if len(windows) < 200:
            continue

        enriched = add_sequence_features(windows)
        feats = sequence_feature_columns(enriched)

        try:
            train_df, _, test_df = group_aware_split(
                DATASET_VARIANTS["nozeroatt"](enriched), 0.15, 0.0, seed=seed)
        except ValueError as exc:
            if verbose:
                print(f"  width {width}s: cannot split ({exc})")
            continue

        if (train_df["label"] == 1).sum() < 20 or (test_df["label"] == 1).sum() < 20:
            if verbose:
                print(f"  width {width}s: too few attack windows")
            continue

        Xtr = train_df[feats].values.astype(float)
        ytr = train_df["label"].values
        Xte = test_df[feats].values.astype(float)
        yte = test_df["label"].values

        model = model_factory(seed)
        if scaler is not None:
            model.fit(scaler.transform(Xtr), ytr)
            proba = model.predict_proba(scaler.transform(Xte))[:, 1]
        else:
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]

        m = binary_metrics(yte, (proba >= threshold).astype(int))

        # Detection latency in seconds: distance from episode onset to the first
        # correctly flagged window.
        latency = None
        if "window_index" in test_df.columns and "episode_id" in test_df.columns:
            pred = (proba >= threshold).astype(int)
            offsets = []
            atk = test_df[test_df["label"] == 1]
            for _, group in atk.groupby("episode_id"):
                group = group.sort_values("window_start_s")
                onset = group["window_start_s"].iloc[0]
                hits = group[pred[group.index] == 1]["window_start_s"]
                if len(hits):
                    offsets.append(float(hits.iloc[0] - onset))
            if offsets:
                latency = round(float(np.median(offsets)), 3)

        rows.append({
            "Width Seconds": width,
            "Stride Seconds": stride,
            "Windows": int(len(windows)),
            "Window Inflation vs 1s": round(len(windows) / max(1, len(event_df)), 3),
            "Test Attack Windows": int((yte == 1).sum()),
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Total Cost": m["Total Cost"],
            "Median Detection Latency (s)": latency,
        })

        if verbose:
            print(f"  width={width:>3}s stride={stride:>2}s "
                  f"windows={len(windows):>6} sens={m['Sensitivity']:.4f} "
                  f"cost={m['Total Cost']:<6} latency={latency}")

    frame = pd.DataFrame(rows)
    return frame


def stride_analysis(model_factory, event_df, width=5, strides=(1, 2, 3, 5),
                    seed=42, threshold=0.5, scaler=None, verbose=True):
    """Vary the stride at a fixed width.

    Stride 1 with width 5 makes every window observe five seconds of activity,
    overlapping its neighbour by four. Stride equal to width is the paper's
    non-overlapping scheme. The comparison shows what overlap buys and what it
    costs in redundant rows.
    """
    from ml.metrics import binary_metrics
    from ml.streaming import add_sequence_features, sequence_feature_columns
    from ml.traces import DATASET_VARIANTS, group_aware_split

    rows = []
    for stride in strides:
        windows = resample_windows(event_df, width_seconds=width,
                                   stride_seconds=stride)
        if len(windows) < 200:
            continue

        enriched = add_sequence_features(windows)
        feats = sequence_feature_columns(enriched)
        try:
            train_df, _, test_df = group_aware_split(
                DATASET_VARIANTS["nozeroatt"](enriched), 0.15, 0.0, seed=seed)
        except ValueError:
            continue

        if (train_df["label"] == 1).sum() < 20 or (test_df["label"] == 1).sum() < 20:
            continue

        model = model_factory(seed)
        Xtr = train_df[feats].values.astype(float)
        ytr = train_df["label"].values
        Xte = test_df[feats].values.astype(float)
        yte = test_df["label"].values
        if scaler is not None:
            model.fit(scaler.transform(Xtr), ytr)
            proba = model.predict_proba(scaler.transform(Xte))[:, 1]
        else:
            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]

        m = binary_metrics(yte, (proba >= threshold).astype(int))

        rows.append({
            "Width Seconds": width,
            "Stride Seconds": stride,
            "Overlap Fraction": round(1 - stride / width, 3),
            "Windows": int(len(windows)),
            "Sensitivity": m["Sensitivity"],
            "Specificity": m["Specificity"],
            "Total Cost": m["Total Cost"],
        })
        if verbose:
            print(f"  width={width}s stride={stride} "
                  f"overlap={1 - stride/width:.0%} sens={m['Sensitivity']:.4f} "
                  f"cost={m['Total Cost']}")

    return pd.DataFrame(rows)


def run(model_factory, event_df, widths=(1, 2, 5, 10, 30), seed=42,
        threshold=0.5, scaler=None, verbose=True):
    """Window-width sweep plus the stride comparison at the widest setting."""
    sweep = window_sweep(model_factory, event_df, widths=widths, seed=seed,
                         threshold=threshold, scaler=scaler, verbose=verbose)
    stride = stride_analysis(model_factory, event_df, width=5, seed=seed,
                             threshold=threshold, scaler=scaler,
                             verbose=verbose)
    return {"Width Sweep": sweep, "Stride Analysis": stride}