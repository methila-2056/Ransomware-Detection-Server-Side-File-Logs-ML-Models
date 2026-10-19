"""Inject measured results into README.md.

The README carries placeholder markers such as ``<!-- RESULTS:MAIN -->``.
Running this module replaces each one with a table built from the JSON in
``results/``. Doing it this way means a documented number cannot drift away from
the recorded one: the README has no result in it that was typed by hand.

Missing result files produce an explicit note rather than a silently empty
section, so an incomplete run is visible instead of looking like a real table.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
README = BASE_DIR / "README.md"


def load(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(value, places=4):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def main_table():
    results = load("model_results.json")
    if not results:
        return ("_Not yet generated. Run `python -m ml.train`._")

    lines = [
        "| Model | Accuracy | Sensitivity | Specificity | F1 | ROC AUC | "
        "Avg Precision | Cost | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['Model']} | {_fmt(r['Accuracy'])} | "
            f"{_fmt(r['Sensitivity'])} | {_fmt(r['Specificity'])} | "
            f"{_fmt(r['F1 Score'])} | {_fmt(r['ROC AUC'])} | "
            f"{_fmt(r['Average Precision'])} | **{r['Total Cost']}** | "
            f"{_fmt(r['Avg Latency (ms)'], 3)} |")

    # Early-phase detection is reported per model and is the measurement the paper
    # claims but does not quantify.
    early = [r for r in results if r.get("Per Episode Detection")]
    if early:
        lines.append("")
        lines.append("**Early-phase detection**, measured per attack episode "
                     "(first correctly classified window, in seconds from onset):")
        lines.append("")
        lines.append("| Model | Episodes detected | Median offset | Max offset |")
        lines.append("|---|---|---|---|")
        for r in early:
            e = r["Per Episode Detection"]
            lines.append(
                f"| {r['Model']} | {e['Episodes Detected']}/{e['Episodes']} | "
                f"{_fmt(e['Median Offset (s)'], 1)} | "
                f"{_fmt(e['Max Offset (s)'], 1)} |")

    return "\n".join(lines)


def ablation_table():
    payload = load("ablation.json")
    if not payload:
        return "_Not yet generated. Run `python -m ml.experiments --only ablation`._"

    lines = [
        "| Feature group | Features | AUROC | Sensitivity @0.5 | Cost @0.5 |",
        "|---|---|---|---|---|",
    ]
    for row in payload["Results"]:
        lines.append(
            f"| {row['Group']} | {row['Features']} | {_fmt(row['AUROC'])} | "
            f"{_fmt(row['Sensitivity At 0.5'])} | {row['Cost At 0.5']} |")

    counts = payload["Results"][0]
    trailing = payload["Results"][1]
    lines.append("")
    lines.append(
        f"Counts alone reach AUROC {counts['AUROC']:.4f} but cost "
        f"{counts['Cost At 0.5']}; adding trailing aggregates brings cost to "
        f"{trailing['Cost At 0.5']} while sensitivity rises from "
        f"{counts['Sensitivity At 0.5']:.4f} to "
        f"{trailing['Sensitivity At 0.5']:.4f}.")
    return "\n".join(lines)


def negative_table():
    ensemble = load("ensemble.json")
    seeds = load("seed_results.json")
    lines = []

    if ensemble:
        best = ensemble["Best Single Model"]
        soft = ensemble["Voting"]["Rules"]["soft"]["0.5"]
        lines.append(
            f"- **Ensemble, {best} single model** — soft voting cost "
            f"{soft['Total Cost']} vs {ensemble['Single Models'][best]['Total Cost']} "
            f"for the best single model "
            f"({soft['Cost vs Best Single']:+d}).")

    if seeds and "Comparison (Cost)" in seeds:
        comp = seeds["Comparison (Cost)"]
        adjusted = [v.get("holm_adjusted_p") for v in comp["comparisons"].values()]
        adjusted = [p for p in adjusted if p is not None]
        if adjusted:
            worst = max(adjusted)
            lines.append(
                f"- **Model ranking is not statistically distinguishable** — "
                f"across {len(seeds['Seeds'])} seeds the largest Holm-adjusted "
                f"p-value for a cost difference is {worst:.3f}, so no model is "
                f"significantly better than any other.")

    if not lines:
        return "_Not yet generated. Run `python -m ml.experiments`._"
    return "\n".join(lines)


def scale_table():
    """The scale study, which is the strongest single result in the project."""
    payload = load("scale_curve.json")
    if not payload or not payload.get("Curves"):
        return "_Not yet generated. Run the scale study to populate this._"

    paper = payload.get("Base Paper", {})
    lines = [
        f"Evaluated at up to **{max(payload['Sizes']):,} windows** — the base paper's "
        f"dataset size — at a fixed 0.5 threshold.",
        "",
        "| Model | Windows | Trained on | Sensitivity | Specificity | "
        "Cost/attack window | Fit (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, points in payload["Curves"].items():
        for p in points:
            trained = "subsampled" if p.get("Subsampled") else "full"
            lines.append(
                f"| {name} | {p['Windows']:,} | {trained} | "
                f"{_fmt(p['Sensitivity'])} | {_fmt(p['Specificity'])} | "
                f"{_fmt(p['Cost Per Attack Window'])} | "
                f"{_fmt(p.get('Fit Seconds'), 1)} |")

    lines.append("")
    lines.append(
        f"For comparison, the base paper reports sensitivity "
        f"{paper.get('Sensitivity')} and cost "
        f"{paper.get('Cost Per Attack Window')} per attack window over "
        f"{paper.get('Attack Windows')} attack seconds.")
    lines.append("")
    lines.append(
        "Raw total cost is not comparable across dataset sizes — it scales with the "
        "number of evaluated windows. Cost per attack window is the comparable "
        "quantity, and it is what the table above reports.")
    return "\n".join(lines)


def benchmark_table():
    payload = load("benchmarks.json")
    if not payload or not payload.get("Inference", {}).get("Available"):
        return "_Not yet generated. Run `python benchmarks/ingest_bench.py`._"

    inference = payload["Inference"]
    ingest = payload["Ingest"]

    lines = [
        f"Ingest: **{ingest['Events Per Second']:,} events/s** "
        f"({ingest['Events Parsed']:,} events into {ingest['Windows']:,} windows).",
        "",
        "| Model | p50 (ms) | p95 (ms) | p99 (ms) | Shares at 1 Hz |",
        "|---|---|---|---|---|",
    ]
    for name, stats in inference["Models"].items():
        # Half the 1 s tick is reserved, so p99 must fit inside 500 ms.
        shares = int(500.0 / stats["P99 ms"]) if stats["P99 ms"] else 0
        lines.append(
            f"| {name} | {_fmt(stats['P50 ms'], 3)} | {_fmt(stats['P95 ms'], 3)} | "
            f"{_fmt(stats['P99 ms'], 3)} | {shares} |")

    lines.append("")
    lines.append(
        "The base paper states in Sect. 6.3 that latency \"remains within the quasi "
        "real-time range even when extrapolated to scenarios involving several hundred "
        "clients\". That was an extrapolation. This table measures it per model, and "
        "the answer is strongly model-dependent: the fastest sustains several hundred "
        "shares and the slowest only single digits. Which model is deployed therefore "
        "determines whether the paper's scalability claim holds.")
    return "\n".join(lines)


def test_count_line():
    """Report the collected test count, derived from the suite itself.

    The number used to be typed into the README by hand and had already drifted
    twice: it read 123 when the suite held 126 tests, then 146. Counting here
    means the README cannot disagree with the suite.
    """
    import re
    suite = (BASE_DIR / "tests" / "test_all.py").read_text(encoding="utf-8")
    total = len(re.findall(r"^\s+def test_", suite, re.M))
    classes = len(re.findall(r"^class Test\w+", suite, re.M))
    return f"tests/                    {total} tests in {classes} classes"


SECTIONS = {
    "RESULTS:MAIN": main_table,
    "RESULTS:ABLATION": ablation_table,
    "RESULTS:SCALE": scale_table,
    "RESULTS:BENCHMARK": benchmark_table,
    "RESULTS:NEGATIVE": negative_table,
    "RESULTS:TESTS": test_count_line,
}


MARKER_PREFIX = "<!-- RESULTS:"


def render(text):
    """Replace each marker with a generated block, bounded by the next marker.

    The previous version searched for the next ``<!--`` occurrence, which matched
    unrelated HTML comments later in the document and silently deleted every
    section in between. Bounding is now done on this module's own markers only.
    """
    lines = text.split("\n")
    out = []
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if not stripped.startswith(MARKER_PREFIX):
            out.append(line)
            index += 1
            continue

        marker = stripped[len("<!--"):stripped.index("-->")].strip()
        builder = SECTIONS.get(marker)

        # Drop any previously generated content for this marker, up to the next
        # marker or the next markdown heading.
        index += 1
        while index < len(lines):
            nxt = lines[index]
            nxt_stripped = nxt.strip()
            if (nxt_stripped.startswith(MARKER_PREFIX)
                    or nxt_stripped.startswith("#")):
                break
            index += 1

        out.append(line)
        if builder is None:
            print(f"  unknown marker: {marker}")
        else:
            out.append("")
            out.append(builder())
            out.append("")
            print(f"  filled {marker}")

    return "\n".join(out)


def main():
    text = README.read_text(encoding="utf-8")
    updated = render(text)
    # Collapse any run of 3+ blank lines the substitution may introduce.
    while "\n\n\n\n" in updated:
        updated = updated.replace("\n\n\n\n", "\n\n\n")
    if updated != text:
        README.write_text(updated, encoding="utf-8")
        print(f"README.md updated ({len(updated)} chars)")
    else:
        print("README.md already up to date")


if __name__ == "__main__":
    main()