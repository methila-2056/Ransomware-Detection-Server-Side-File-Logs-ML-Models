"""Throughput and latency benchmark for the live monitoring path.

The base paper states that its framework "exhibits runtime efficiency" and
"prediction latency remains within the millisecond range" (Sect. 6.1), and in
Sect. 6.3 argues its latency "remains within the quasi real-time range even when
extrapolated to scenarios involving several hundred clients". That last claim is
an extrapolation, not a measurement.

This benchmark measures the three quantities that decide whether the claim holds:

1. **Ingest** -- events per second through the parsers and window aggregator.
2. **Feature + inference** -- latency of building the temporal feature vector
   from a rolling history and classifying one window, per model.
3. **Steady-state throughput** -- how many one-second windows per second the
   pipeline can serve end to end, which is what bounds the number of shares it
   can watch at once.

Run directly::

    python benchmarks/ingest_bench.py --events 500000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from ml.ingest import events_to_windows, parse_samba  # noqa: E402
from ml.streaming import add_sequence_features, sequence_feature_columns  # noqa: E402


def synthetic_stream(n_events, ops_per_second=50):
    """Generate a Samba full_audit stream at a chosen event rate."""
    lines = []
    for i in range(n_events):
        op = ("create_file", "renameat", "unlinkat", "open")[i % 4]
        second = i // max(1, ops_per_second)
        lines.append(
            f"Feb  1 {8 + second // 3600:02d}:{second // 60 % 60:02d}:"
            f"{second % 60:02d} z1 smbd_audit[1]: "
            f"user{i % 20}|10.0.0.1|Data|{op}|ok|/srv/share/f{i}.docx")
    return lines


def bench_ingest(n_events, ops_per_second):
    lines = synthetic_stream(n_events, ops_per_second)

    started = time.perf_counter()
    events, report = parse_samba(lines)
    parse_seconds = time.perf_counter() - started

    started = time.perf_counter()
    windows = events_to_windows(events, window_seconds=1.0)
    aggregate_seconds = time.perf_counter() - started

    total = parse_seconds + aggregate_seconds
    return {
        "Events Requested": n_events,
        "Events Parsed": report.parsed,
        "Windows": int(len(windows)),
        "Parse Seconds": round(parse_seconds, 3),
        "Aggregate Seconds": round(aggregate_seconds, 3),
        "Total Seconds": round(total, 3),
        "Events Per Second": round(n_events / total) if total else 0,
        "Windows Per Second": round(len(windows) / total) if total else 0,
    }


def bench_features(n_windows):
    """Time the temporal feature build, which runs every monitoring tick."""
    rng = np.random.default_rng(0)
    frame = {
        "create_count": rng.integers(0, 60, n_windows),
        "rename_count": rng.integers(0, 40, n_windows),
        "delete_count": rng.integers(0, 30, n_windows),
    }
    import pandas as pd
    df = pd.DataFrame(frame)

    started = time.perf_counter()
    enriched = add_sequence_features(df)
    elapsed = time.perf_counter() - started

    return {
        "Windows": n_windows,
        "Feature Columns": len(sequence_feature_columns(enriched)),
        "Milliseconds": round(elapsed * 1000, 3),
        "Windows Per Second": round(n_windows / elapsed) if elapsed else 0,
    }


def bench_inference(n_windows=2000, repeats=200):
    """Time one classification per model over a realistic rolling history."""
    import joblib

    from ml.streaming import add_sequence_features

    models_dir = BASE_DIR / "models"
    info_path = models_dir / "best_model_info.json"
    if not info_path.exists():
        return {"Available": False,
                "Reason": "no trained model; run python -m ml.train first"}

    with open(info_path) as f:
        info = json.load(f)
    columns = info.get("feature_columns")
    if not columns:
        return {"Available": False,
                "Reason": "best_model_info.json has no feature_columns"}

    scaler = None
    if info.get("use_scaled"):
        scaler = joblib.load(str(models_dir / "scaler.pkl"))

    # A rolling history long enough that every trailing feature is populated.
    rng = np.random.default_rng(1)
    history = {
        "create_count": rng.integers(0, 60, n_windows),
        "rename_count": rng.integers(0, 40, n_windows),
        "delete_count": rng.integers(0, 30, n_windows),
    }
    import pandas as pd
    enriched = add_sequence_features(pd.DataFrame(history))
    probe = enriched[columns].tail(1).values.astype(float)
    if scaler is not None:
        probe = scaler.transform(probe)

    results = {}
    for name in ("decision_tree", "random_forest", "svm", "adaboost", "xgboost",
                 "logistic_regression"):
        path = models_dir / f"{name}.pkl"
        if not path.exists():
            continue
        model = joblib.load(str(path))

        for _ in range(10):
            model.predict(probe)

        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            model.predict(probe)
            samples.append((time.perf_counter() - started) * 1000)

        arr = np.asarray(samples)
        results[name] = {
            "Mean ms": round(float(arr.mean()), 4),
            "P50 ms": round(float(np.median(arr)), 4),
            "P95 ms": round(float(np.percentile(arr, 95)), 4),
            "P99 ms": round(float(np.percentile(arr, 99)), 4),
            "Max ms": round(float(arr.max()), 4),
            "Windows Per Second": round(1000.0 / arr.mean()) if arr.mean() else 0,
        }

    return {"Available": True, "Repeats": repeats, "Models": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=200_000)
    parser.add_argument("--rate", type=int, default=50,
                        help="events per second in the synthetic stream")
    parser.add_argument("--output", default=str(BASE_DIR / "results"
                                               / "benchmarks.json"))
    args = parser.parse_args(argv)

    print("=" * 72)
    print("  INGEST, FEATURE AND INFERENCE BENCHMARK")
    print("=" * 72)

    print(f"\n[1/3] ingest: {args.events:,} events")
    ingest = bench_ingest(args.events, args.rate)
    print(f"      {ingest['Events Parsed']:,} parsed -> {ingest['Windows']:,} "
          f"windows in {ingest['Total Seconds']}s")
    print(f"      {ingest['Events Per Second']:,} events/s")

    print("\n[2/3] feature construction")
    features = bench_features(10_000)
    print(f"      {features['Feature Columns']} columns from "
          f"{features['Windows']:,} windows in {features['Milliseconds']}ms")
    print(f"      {features['Windows Per Second']:,} windows/s")

    print("\n[3/3] single-window inference latency")
    inference = bench_inference()
    if not inference.get("Available"):
        print(f"      skipped: {inference.get('Reason')}")
    else:
        print(f"      {'model':<16}{'mean':>9}{'p50':>9}{'p95':>9}"
              f"{'p99':>9}{'win/s':>10}")
        for name, stats in inference["Models"].items():
            print(f"      {name:<16}{stats['Mean ms']:>9.4f}{stats['P50 ms']:>9.4f}"
                  f"{stats['P95 ms']:>9.4f}{stats['P99 ms']:>9.4f}"
                  f"{stats['Windows Per Second']:>10,}")

    payload = {
        "Ingest": ingest,
        "Features": features,
        "Inference": inference,
        "Interpretation": {
            "Monitoring Headroom": "One tick per second per share. A model's "
                                   "p99 latency must sit far below 1000 ms for "
                                   "many shares to be served by one process.",
            "Paper Comparison": "The paper reports prediction latencies of "
                                "0.00019-0.01208 s (Table 7 context, Sect. 5.3) "
                                "and extrapolates to several hundred clients "
                                "without measuring it (Sect. 6.3). The p99 "
                                "figures here bound that claim directly.",
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()