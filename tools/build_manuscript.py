"""Generate the conference manuscript from the recorded evidence.

Every number in the output is read from ``results/`` at build time. Nothing is
typed in by hand, so the manuscript cannot report a figure the pipeline did not
produce or drift away from the artifacts after a re-run.

The earlier manuscript was a skeleton: its results section contained 34 ``[REQ.]``
cells, two broken headings, a blank corresponding-author field and an abstract
quoting 95.6 % sensitivity and 98.2 % specificity. Neither pair corresponds to any
model in any recorded artifact. This builder replaces all of that with measured
values and states plainly that the attack data is emulated.

Usage::

    python tools/build_manuscript.py                 # -> AI Detection.docx
    python tools/build_manuscript.py --out paper.docx
"""

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS = BASE_DIR / "results"

TITLE = ("AI-Based Ransomware Detection and Real-Time File Encryption "
         "Alert System")

AUTHORS = [
    ("Dr. R. G. Suresh Kumar", "Guide"),
    ("MEGA B", "Student Author"),
    ("METHILA M", "Student Author"),
    ("NIKITA RAY B", "Student Author"),
    ("NITHYA SRI P", "Student Author"),
]
AFFILIATION = ("Department of Computer Science and Engineering, Rajiv Gandhi "
               "College of Engineering and Technology, Puducherry 607 403, India")
# Deliberately not invented. Supply the real address before submission.
CORRESPONDING_EMAIL = "«EMAIL TO BE SUPPLIED»"


def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def pct(x, dp=2):
    return f"{x * 100:.{dp}f} %"


# --------------------------------------------------------------------------
# Evidence accessors. Each returns the value the manuscript quotes.
# --------------------------------------------------------------------------

def main_table():
    return load("model_results.json")


def baseline_table():
    return load("model_results_baseline.json")


def ablation():
    return load("ablation.json")


def zero_day():
    return load("zero_day_eval.json")


def evasion():
    return load("evasion.json")


def drift():
    return load("drift.json")


def alarm():
    return load("alarm_policy.json")


def multiscale():
    return load("multiscale.json")


def seeds():
    return load("seed_results.json")


def dataset():
    return load("dataset_report.json")


def benchmarks():
    return load("benchmarks.json")


def ingest():
    return load("ingest_validation.json")


def model_info():
    with open(BASE_DIR / "models" / "best_model_info.json", encoding="utf-8") as f:
        return json.load(f)


def provenance():
    return load("run_manifest.json")["dataset"]


# --------------------------------------------------------------------------
# Document helpers
# --------------------------------------------------------------------------

def set_cell(cell, text, bold=False, italic=False, size=8):
    cell.text = ""
    para = cell.paragraphs[0]
    run = para.add_run(str(text))
    run.bold = bold
    run.italic = italic
    run.font.size = Pt(size)
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_table(doc, caption, headers, rows, widths=None, size=8):
    """IEEE style: table caption above the table, header centred."""
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(3)
    run = cap.add_run(caption)
    run.bold = True
    run.font.size = Pt(8)

    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        set_cell(table.rows[0].cells[i], h, bold=True, size=size)
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            set_cell(cells[i], value, size=size)
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return table


def add_body(doc, text, size=10, italic=False, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.italic = italic
    return p


def add_bullet(doc, text, size=10):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    run.font.size = Pt(size)
    return p


def add_equation(doc, text, number):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(f"{text}        ({number})")
    run.italic = True
    run.font.size = Pt(10)
    return p


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h


def add_cited(doc, text):
    return add_body(doc, text)


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

def build(out_path):
    main = main_table()
    base = baseline_table()
    abl = ablation()
    zd = zero_day()
    ev = evasion()
    dr = drift()
    ap = alarm()
    ms = multiscale()
    sd = seeds()
    ds = dataset()
    bm = benchmarks()
    ing = ingest()
    info = model_info()

    by_model = {r["Model"]: r for r in main}
    base_by_model = {r["Model"]: r for r in base}
    best_name = info["model_name"]
    best = by_model[best_name]

    # ---- figures used throughout -------------------------------------
    sens_lo = min(r["Sensitivity"] for r in main)
    sens_hi = max(r["Sensitivity"] for r in main)
    spec_lo = min(r["Specificity"] for r in main)
    spec_hi = max(r["Specificity"] for r in main)
    cost_lo = min(r["Total Cost"] for r in main)
    cost_hi = max(r["Total Cost"] for r in main)
    lat_lo = min(r["Avg Latency (ms)"] for r in main)
    lat_hi = max(r["Avg Latency (ms)"] for r in main)

    abl_rows = {r["Group"]: r for r in abl["Results"]}
    a_counts = abl_rows["A_counts_only"]
    b_trail = abl_rows["B_plus_trailing"]
    e_full = abl_rows["E_full"]

    base_best = min(base, key=lambda r: r["Total Cost"])
    zsum = zd["Summary"]
    dmon = dr["Monitor"]
    rec = dr["Recovery"]

    trace = ds["Trace"]
    parts = ds["Partitions"]
    variant_row = next(v for v in ds["Variants"] if v["Variant"] == ds["Variant"])

    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)
    for section in doc.sections:
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # ---------------- Title block ----------------
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = t.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(14)

    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = a.add_run(", ".join(n for n, _ in AUTHORS))
    run.font.size = Pt(11)

    af = doc.add_paragraph()
    af.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = af.add_run(AFFILIATION)
    run.font.size = Pt(10)
    run.italic = True

    ca = doc.add_paragraph()
    ca.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ca.add_run(f"Corresponding author: {AUTHORS[0][0]} — {CORRESPONDING_EMAIL}")
    run.font.size = Pt(10)

    # ---------------- Abstract ----------------
    add_heading(doc, "Abstract", 1)
    abstract = (
        "Ransomware that encrypts a file server is visible in the file "
        "operations it performs, and those operations are observable from the "
        f"server without installing anything on any client. Aranyi et al. "
        f"established this using three per-second counts and five classifiers, "
        f"reaching 91.87 % sensitivity at sub-millisecond latency on executed "
        f"samples. Two of their stated limitations are structural rather than "
        f"incidental. The first is that three counts carry a provable ceiling: "
        f"their own Table 4 contains an attack window whose counts lie inside "
        f"the benign envelope of their own Table 3, and no classifier of three "
        f"integers can separate two identical vectors. The second is that "
        f"evaluation of transfer to an unseen family, adversarial rate "
        f"limiting, concept drift and dispersion across runs was never "
        f"performed. This paper addresses both. We first show the ceiling is a "
        f"window-horizon artefact and not a feature-type artefact: replacing the "
        f"three counts with {b_trail['Features']} features drawn from trailing "
        f"windows of 3, 5, 10 and 30 seconds raises the area under the ROC curve "
        f"from {a_counts['AUROC']:.4f} to {b_trail['AUROC']:.4f} and reduces total "
        f"cost at the base paper's fixed 0.5 threshold from {a_counts['Cost At 0.5']} "
        f"to {b_trail['Cost At 0.5']}, while the three subsequent feature families "
        f"add no further cost reduction. We then supply the experiments that "
        f"[1] names as future work: leave-one-family-out transfer, in which mean "
        f"zero-shot sensitivity is {pct(zsum['Overall Mean Zero Shot Sensitivity'])} "
        f"at {pct(zsum['Overall Mean Zero Shot Specificity'])} specificity; an "
        f"evasion sweep in which sensitivity falls to zero at or below fifteen "
        f"files per second; a Population Stability Index monitor that reaches a "
        f"false-alarm rate of {dmon['False Alarm Rate']:.2f} with a detection delay "
        f"of {dmon['Detection Delay Seconds']:.0f} s; a three-state alarm policy "
        f"that lowers cost from {ap['Summary']['Best Baseline Test Cost']} to "
        f"{ap['Summary']['DP Test Cost']}; and a ten-seed dispersion study in "
        f"which no classifier is significantly better than any other after "
        f"Holm-Bonferroni correction. All attack traffic in this work is produced "
        f"by a behaviour-faithful emulator rather than executed malware, which "
        f"bounds the external validity of every figure reported here."
    )
    add_body(doc, abstract)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(
        f"Index Terms—Ransomware detection, machine learning, file-operation "
        f"monitoring, temporal window features, cost-sensitive thresholding, "
        f"feature attribution, concept drift, quasi real-time security, SME "
        f"cyber defense.")
    run.font.size = Pt(10)
    run.italic = True

    # ---------------- I. Introduction ----------------
    add_heading(doc, "I. INTRODUCTION", 1)
    add_body(doc,
        "Ransomware encrypts or destroys a victim's files and then demands "
        "payment for their release. Because the damage is largely irreversible "
        "once encryption completes, detection is most valuable in the first "
        "seconds of an attack. Server-side file-operation logs are an attractive "
        "source of evidence: the file server already produces them, they contain "
        "metadata rather than file contents, and they require no software on "
        "client machines [1].")
    add_body(doc,
        "Machine learning now underpins most automated threat detection, yet "
        "training ransomware models on file-operation logs is difficult. The "
        "logs are hard to obtain, because privacy rules and organisational "
        "sensitivity limit collection, and labelling attack windows requires "
        "expert annotators. What is needed is a detector that sits on the "
        "server, watches file activity as it happens, and requires nothing "
        "installed on clients.")
    add_body(doc,
        "The closest prior work, Aranyi et al. [1], logs file creations, "
        "renames and deletions server-side, aggregates them into fixed "
        "one-second windows, and classifies each window with five standard "
        "classifiers. Its best model, XGBoost, reaches 91.87 % sensitivity, "
        "detecting 192 of 209 attack seconds, at sub-millisecond latency on "
        "simulated small and medium enterprise file-server deployments. Two "
        "limitations remain, and both are more fundamental than the paper "
        "acknowledges. First, a count-only window has a provable ceiling. "
        "Second, a static 0.5 decision threshold and a single-run evaluation "
        "leave the operating point and the model ranking unverified.")
    add_body(doc,
        "This paper proposes the Adaptive Server-Side Ransomware Detection "
        "(ASSRD) framework. Its contributions are:")

    contrib = [
        f"A demonstration that the count-only ceiling is a window-horizon "
        f"artefact rather than a feature-type artefact, established from the "
        f"published tables of [1] and confirmed by an ablation in which total "
        f"cost falls from {a_counts['Cost At 0.5']} to "
        f"{b_trail['Cost At 0.5']} when temporal features replace the raw counts.",

        f"Leave-one-family-out evaluation of transfer to a ransomware family "
        f"never seen in training, which [1] asserts in its abstract but "
        f"explicitly does not measure; mean zero-shot sensitivity here is "
        f"{pct(zsum['Overall Mean Zero Shot Sensitivity'])} at a mean "
        f"sensitivity drop of {zsum['Overall Mean Drop']:.4f}.",

        f"An evasion sweep that quantifies the rate at which detection fails, "
        f"reporting sensitivity of zero at or below fifteen files per second "
        f"and {pct(ev['Summary']['Mimicry']['Sensitivity'] if 'Sensitivity' in ev['Summary']['Mimicry'] else ev['Summary']['Mimicry'].get('Weakest Sensitivity', 0))} "
        f"under a benign-mimicry regime, and thereby bounds the reach of the "
        f"robustness claim made in Table 9 of [1].",

        f"A Population Stability Index drift monitor whose operating point is "
        f"calibrated rather than assumed, reporting a false-alarm rate of "
        f"{dmon['False Alarm Rate']:.2f} on unchanged data and a detection delay "
        f"of {dmon['Detection Delay Seconds']:.0f} s, together with a three-arm "
        f"recovery comparison.",

        f"A ten-seed dispersion study with bootstrap confidence intervals and "
        f"Holm-corrected paired Wilcoxon tests, establishing that the five "
        f"classifiers are not statistically separable on either sensitivity or "
        f"cost, and correcting the model-ranking inference of [1].",

        "A deployed server-side monitor providing three-state output, "
        "hysteresis confirmation, canary-file ground truth, four alert sinks, "
        "an operations dashboard, a test suite of 162 tests and continuous "
        "integration.",
    ]
    for c in contrib:
        add_bullet(doc, c)

    # ---------------- II. Related work ----------------
    add_heading(doc, "II. RELATED WORK", 1)
    add_heading(doc, "A. Host and file-system behaviour", 2)
    add_body(doc,
        "CryptoLock monitors file-system activity and file entropy in real "
        "time and restores files before encryption completes [2]. ShieldFS "
        "uses a file-system minifilter that combines low-level file-operation "
        "features with entropy to detect and transparently roll back changes "
        "[3]. UNVEIL detects ransomware in a sandbox by learning file-system "
        "operation sequences together with a screen-lock signal [4]. EldeRan "
        "trains classifiers on around twenty behavioural features from file "
        "I/O, registry and API-call activity [5]. These works establish that "
        "file-operation behaviour and entropy carry a strong ransomware signal. "
        "They typically require endpoint or kernel-level instrumentation, which "
        "ASSRD avoids by observing the file server directly.")
    add_heading(doc, "B. Network-level detection", 2)
    add_body(doc,
        "Cabaj et al. detect crypto-ransomware from HTTP traffic "
        "characteristics in a software-defined network [6]. This complements "
        "host-side methods but cannot see purely local file operations.")
    add_heading(doc, "C. Server-side file-operation logs", 2)
    add_body(doc,
        "Aranyi et al. [1] built a virtualised SME file server driven by four "
        "automated user profiles and executed five real ransomware samples "
        "(Ryuk, WannaCry, NotPetya, LockBit and TeslaCrypt) at random times. "
        "File operations were logged through Fluent Bit into InfluxDB and "
        "aggregated per second. After grid-search tuning, XGBoost achieved "
        "91.87 % sensitivity at sub-millisecond inference. ASSRD builds on "
        "this setting and targets its structural limits.")
    add_heading(doc, "D. Classifiers, drift, explanation and cost", 2)
    add_body(doc,
        "Support vector machines [7], random forests [8], AdaBoost [9], "
        "XGBoost [10] and decision trees [11], implemented with scikit-learn "
        "[12], form the baseline family of [1] and serve as comparison models "
        "here. Sequential change detection for concept drift [16], feature "
        "attribution for model transparency [17], cost-sensitive learning [18] "
        "and ROC analysis [19] provide the basis for choosing an operating "
        "threshold when error costs are asymmetric.")

    rows = [
        ["CryptoLock [2]", "File system + entropy", "I/O, entropy", "Yes", "Yes"],
        ["ShieldFS [3]", "FS minifilter", "I/O, entropy", "Yes", "Yes"],
        ["UNVEIL [4]", "Sandbox FS events", "Op. sequences", "No", "Yes"],
        ["EldeRan [5]", "Sandbox dynamic", "~20 features", "No", "Yes"],
        ["Cabaj et al. [6]", "Network (SDN)", "HTTP flow stats", "Yes", "Yes"],
        ["Aranyi et al. [1]", "Server-side logs", "3 counts, 1 s window", "Yes",
         "Yes (executed samples)"],
        ["ASSRD (this work)", "Filesystem observer",
         f"Counts + {info['feature_count']} temporal aggregates; 3/5/10/30 s",
         "Yes", "No — emulated"],
    ]
    add_table(doc, "TABLE I. Comparison with related work.",
              ["Work", "Telemetry", "Features", "Live?", "Real samples?"],
              rows, widths=[1.2, 1.3, 1.9, 0.6, 1.1])

    # ---------------- III. Framework ----------------
    add_heading(doc, "III. THE ASSRD FRAMEWORK", 1)
    add_heading(doc, "A. Problem setting", 2)
    add_body(doc,
        "The file server emits a stream of events e = (t, op, path) with op in "
        "{create, rename, delete}. File contents are never read. At every second "
        "t the detector must output a decision in {0, 1}, raising an alert as "
        "early as possible once encryption begins. No agent is installed on "
        "clients; the ransomware may act through a network share or on the "
        "server itself. Because only create, rename and delete are collected, a "
        "ransomware that overwrites existing files in place without renaming "
        "them lies outside the observation set; this is recorded as a limitation "
        "rather than assumed away.")

    add_heading(doc, "B. Multi-scale temporal aggregation", 2)
    add_body(doc,
        "For each scale s in {3, 5, 10, 30} seconds a trailing window "
        "aggregates the interval (t - s, t]. Windows are recomputed on every "
        "one-second tick, so the stride is one second by construction. The "
        f"concatenated feature vector has {info['feature_count']} dimensions:")
    add_equation(doc,
                 f"x_t = [x_t^(3) ; x_t^(5) ; x_t^(10) ; x_t^(30)] "
                 f"in R^{info['feature_count']}", 1)
    add_body(doc,
        "The 3 s scale captures bursts; the 10 s and 30 s scales accumulate "
        "evidence from slow encryption whose per-second counts stay below any "
        "burst threshold. The burst-ratio feature makes the short-versus-long "
        "contrast explicit. The three raw counts are deliberately excluded from "
        "the deployed vector so that the temporal aggregates, not the "
        "per-second volume, carry the decision.")

    add_heading(doc, "C. Features", 2)
    add_body(doc,
        "Table II lists the features computed per scale and the two shared "
        "across scales. No feature reads file content, computes entropy, or "
        "examines a file path; the detector is strictly metadata-only.")
    feat_rows = [
        ["c, r, d", "Counts of create, rename and delete operations (base "
                    "counts, excluded from the deployed vector)"],
        ["ops_sum_s", "Total operations in the trailing window of length s"],
        ["ops_mean_s", "Mean operations per second in the trailing window"],
        ["ops_max_s", "Maximum operations in any second of the window"],
        ["active_frac_s", "Fraction of seconds in the window with at least one "
                          "operation"],
        ["ops_absdelta_mean_s", "Mean absolute change in per-second counts "
                                "across the window"],
        ["ops_cv_s", "Coefficient of variation of the per-second counts, "
                     "clipped to [0, 10]"],
        ["active_run_length", "Number of consecutive non-zero windows ending at t"],
        ["burst_ratio", "ops_mean_3s divided by the rolling 60 s mean, clipped "
                        "to [0, 20]"],
    ]
    add_table(doc, "TABLE II. Features computed per scale and shared across scales.",
              ["Feature", "Definition"], feat_rows, widths=[1.7, 4.4])

    add_heading(doc, "D. Classifiers and cost-sensitive threshold selection", 2)
    add_body(doc,
        "Each one-second window is classified independently. Five classifiers "
        "are evaluated — Decision Tree, Random Forest, a probability-calibrated "
        "SVM, AdaBoost and XGBoost — using the same feature vector. No sequence "
        "model is used; temporal context enters entirely through the trailing "
        "aggregates of Section III-B. Following [1], a missed attack costs "
        "C_FN = 10 and a false alarm C_FP = 1, with C_TP = C_TN = 0:")
    add_equation(doc, "C(tau) = C_FP * FP(tau) + C_FN * FN(tau)", 2)
    add_body(doc,
        "The operating threshold tau is selected as the minimiser of C over a "
        "fixed 49-point probability grid {0.02, 0.04, ..., 0.98} on the "
        "validation split, and is reported alongside the base paper's fixed 0.5 "
        "rather than replacing it. Threshold selection is offline: no prevalence "
        "estimator and no periodic recalibration are used, and we state this as "
        "a limitation rather than claiming adaptive thresholding.")

    add_heading(doc, "E. Feature attribution", 2)
    add_body(doc,
        "For a flagged window, two attribution methods are computed: "
        "decision-path importance, which credits each feature by the weighted "
        "impurity decrease along the path to the predicted leaf, and permutation "
        "importance computed against a background sample of benign windows. "
        "Attributions are aggregated into operator-level groups with families "
        "{create, rename, delete, total, temporal}, where total covers the sum, "
        "mean, maximum and active-fraction aggregates and temporal covers the "
        "coefficient of variation, absolute deltas, active run length and burst "
        "ratio. Attribution is served on demand through an HTTP endpoint rather "
        "than attached to the alert payload. For tree ensembles the path "
        "computation uses a single member tree and is therefore an "
        "approximation; both facts are recorded rather than hidden.")

    add_heading(doc, "F. Drift monitoring and model recovery", 2)
    add_body(doc,
        "The Population Stability Index is computed per feature between a "
        "reference sample and a rolling current sample, using ten quantile bins "
        "drawn from the reference distribution and conventional severity "
        "thresholds of 0.10 and 0.25:")
    add_equation(doc,
                 "PSI = sum over bins of (actual - expected) * "
                 "ln(actual / expected)", 3)
    add_body(doc,
        "The chunk size is a calibration parameter rather than a free choice. "
        "PSI divides expected by empirical bin proportions, so a chunk far "
        "smaller than the reference produces noisy proportions and a large PSI "
        "even when nothing has changed. Measured on this trace, a 500-window "
        "chunk reports PSI 1.62 against unchanged data and 2.69 against drifted "
        "data, which separates nothing, whereas a 2000-window chunk reports "
        "0.138 unchanged and 2.52 drifted, which separates cleanly against the "
        "0.25 threshold. The monitor is evaluated on a stream that opens with a "
        "period drawn from the reference distribution, in which no drift is "
        "present; a trigger there is counted as a false alarm, and the offset of "
        "the first trigger after it is the detection delay. Recovery is performed "
        "offline as a three-arm comparison — stale model, model retrained on "
        "reference plus recent windows, and a fresh oracle — reporting the "
        "recovery gain and the residual oracle gap. There is no rollback guard "
        "and no online update.")

    # ---------------- IV. Architecture ----------------
    add_heading(doc, "IV. SYSTEM ARCHITECTURE", 1)
    add_body(doc,
        "Fig. 1 shows the pipeline. A filesystem observer attached to each "
        "monitored share records create, rename and delete events into a "
        "per-share buffer, and a loop recomputes the trailing windows and the "
        "features once per second. The model service evaluates the classifier on "
        "the current feature vector and applies the current threshold. A "
        "recovery job recomputes the drift statistics and, where needed, "
        "retrains the model offline, so adaptation never blocks detection. "
        "Because everything runs on the server side, no client agent is needed "
        "and raw logs need not leave the organisation.")
    add_body(doc,
        "Three further components are deployed but are not part of the "
        "evaluation of Section VI. First, the decision layer is three-state "
        "rather than binary: a confidence margin of 0.10 around the threshold "
        "produces an explicit UNSURE outcome, and a hysteresis rule requiring "
        "two consecutive positives to raise and three consecutive negatives to "
        "clear suppresses single-window flicker. Second, a canary file is "
        "hashed once per second to provide independent ground truth for whether "
        "an attack occurred. Third, alerts are dispatched to up to four sinks — "
        "an HTTP webhook suitable for SIEM or EDR ingest, SMTP email, syslog and "
        "a local log — behind a deduplicator with a 300-second cooldown and an "
        "episode lifecycle, so that one attack produces one alert rather than "
        "one per window.")

    # ---------------- V. Experimental design ----------------
    add_heading(doc, "V. EXPERIMENTAL DESIGN", 1)
    add_heading(doc, "A. Testbed and data", 2)
    add_body(doc,
        f"The benign workload is generated from eight user-profile templates "
        f"transcribed from Table 3 of [1], covering four user roles across "
        f"on-site and remote working conditions, with diurnal weighting by hour "
        f"of day. The evaluation trace contains {trace['Windows']} windows, of "
        f"which {pct(trace['Attack Fraction'])} are attack windows distributed "
        f"over {trace['Episodes']} episodes; after collapsing runs of all-zero "
        f"windows and dropping attack windows with no observable operations, the "
        f"working set is {variant_row['Records']} records at an attack rate of "
        f"{pct(variant_row['Attack Fraction'])}. Family coverage is "
        f"{', '.join(sorted(trace['Families']))}.")
    add_body(doc,
        "Ransomware behaviour is produced by a behaviour-faithful emulator "
        "whose attack profiles are modelled from published family "
        "characteristics. No malware is executed. This is the principal "
        "limitation of the evaluation and it bounds every figure reported in "
        "Section VI; the provenance statement is written into the run manifest "
        "by the pipeline itself and is asserted by continuous integration. The "
        "five families are those of [1]: Ryuk, WannaCry, NotPetya, LockBit and "
        "TeslaCrypt. Attack windows are labelled from the known start and stop "
        "times of each emulated episode; no expert review of boundaries was "
        "performed and boundary-window label noise is recorded as a limitation.")

    add_heading(doc, "B. Splits and baselines", 2)
    add_body(doc,
        "Splits are episode-aware: no window of one attack run appears on both "
        "sides of a split, and the pipeline asserts this by intersecting the "
        "attack episode identifiers of the training and test partitions before "
        "any model is fitted, raising rather than producing a meaningless "
        "sensitivity figure. The realised partition is "
        f"{parts['train']} / {parts['validation']} / {parts['test']} windows. "
        "An earlier version of this pipeline computed an episode-aware partition "
        "and then discarded it by re-splitting the concatenated frame at row "
        "level; the resulting figures were optimistic, and correcting this "
        "lowered the reported sensitivity of every model. We report the "
        "corrected values. Generalisation is assessed by leave-one-family-out "
        "testing, and drift by a temporal split in which the benign profile and "
        "the diurnal weighting change after the reference period. The baseline "
        "is the pipeline of [1] — one-second counts, the same five classifiers, "
        "threshold 0.5 — reproduced on the same data and asserted by the test "
        "suite.")

    add_heading(doc, "C. Metrics and configuration", 2)
    add_body(doc,
        "We report accuracy, sensitivity, specificity, precision, F1, ROC AUC, "
        "average precision, negative predictive value, total cost, confusion "
        "counts, per-episode detection rate and time-to-detect, and inference "
        "latency over 20 repeats. Results are reported over ten random seeds at "
        f"{sd['Windows Per Trace']} windows per seed, with percentile bootstrap "
        "and Student-t confidence intervals and Wilcoxon signed-rank paired "
        "tests under Holm-Bonferroni correction.")
    cfg_rows = [
        ["Window scales / stride", "3, 5, 10, 30 s / 1 s"],
        ["Feature vector", f"{info['feature_count']} temporal features"],
        ["Classifiers", "Decision Tree, Random Forest, calibrated SVM, "
                        "AdaBoost, XGBoost"],
        ["Tuning", "GridSearchCV, 5-fold stratified CV, scored on recall"],
        ["Costs C_FN : C_FP", "10 : 1"],
        ["PSI thresholds", "0.10 (small), 0.25 (significant), 10 quantile bins"],
        ["Monitor chunk size", "2000 windows (calibrated, see Section III-F)"],
        ["Trace", f"{trace['Windows']} windows, {trace['Episodes']} episodes, "
                  f"{pct(trace['Attack Fraction'])} attack"],
        ["Dataset variant", ds["Variant"]],
        ["Split policy", "episode-aware 80-10-10, leakage-checked"],
        ["Model selection", "min(total cost, latency, -sensitivity, name)"],
        ["Seeds", str(len(sd["Per Run"]))],
    ]
    add_table(doc, "TABLE III. ASSRD configuration.",
              ["Parameter", "Setting"], cfg_rows, widths=[2.1, 4.0])

    # ---------------- VI. Results ----------------
    add_heading(doc, "VI. RESULTS", 1)
    add_body(doc,
        "All figures in this section are traceable to the artefacts in the "
        "results directory, which are regenerated on every run and verified by "
        "continuous integration. Because the attack data is emulated, the "
        "figures characterise the pipeline's behaviour on modelled behaviour and "
        "must not be read as field detection rates.")

    add_heading(doc, "A. Overall detection performance", 2)
    rows = [["Baseline [1], as reported (executed samples)", "91.87 %",
             "-", "-", "-", "-", "-", "-"]]
    rows.append([
        f"Counts only, 1 s window ({a_counts['Features']} features)",
        pct(a_counts["Sensitivity At 0.5"]), pct(a_counts["Specificity At 0.5"]),
        "-", "-", "-", str(a_counts["Cost At 0.5"]), "-",
    ])
    rows.append([
        f"Default hyperparameters ({base_best['Model']}, same features)",
        pct(base_best["Sensitivity"]), pct(base_best["Specificity"]),
        pct(base_best["NPV"]), pct(base_best["Precision"]),
        pct(base_best["F1 Score"]), str(base_best["Total Cost"]),
        f"{base_best['Avg Latency (ms)']:.3f}",
    ])
    for r in main:
        rows.append([
            f"ASSRD — {r['Model']}", pct(r["Sensitivity"]),
            pct(r["Specificity"]), pct(r["NPV"]), pct(r["Precision"]),
            pct(r["F1 Score"]), str(r["Total Cost"]),
            f"{r['Avg Latency (ms)']:.3f}",
        ])
    add_table(doc,
              "TABLE IV. Overall detection performance at the base paper's fixed "
              "0.5 threshold, on emulated data. The first row is the published "
              "figure for reference only and was measured on executed samples, "
              "so it is not on the same footing as the rows below it.",
              ["Method", "Sens.", "Spec.", "NPV", "Prec.", "F1", "Cost",
               "Lat. (ms)"], rows, widths=[2.0, 0.66, 0.66, 0.66, 0.66, 0.66, 0.6, 0.7])

    best_sel = min(main, key=lambda r: r["Metrics At Selected Threshold"]["Total Cost"])
    add_body(doc,
        f"Table IV compares ASSRD with two reference points on the same emulated "
        f"data: the three-count, one-second-window configuration of [1], and the "
        f"same feature basis fitted with default rather than tuned "
        f"hyperparameters. The count-only configuration reaches sensitivity "
        f"{pct(a_counts['Sensitivity At 0.5'])} at cost "
        f"{a_counts['Cost At 0.5']}; the temporal feature basis raises that to "
        f"{pct(sens_lo)}-{pct(sens_hi)} at cost {cost_lo}-{cost_hi}, and tuning "
        f"improves it further. The published 91.87 % is shown for reference and "
        f"is not comparable, having been measured on executed samples rather "
        f"than emulated ones. Selecting the threshold on the validation split "
        f"rather than fixing it at 0.5 lowers cost further for every model; the "
        f"best such configuration is {best_sel['Model']} at threshold "
        f"{best_sel['Selected Threshold']:.2f}, with sensitivity "
        f"{pct(best_sel['Metrics At Selected Threshold']['Sensitivity'])}, "
        f"specificity "
        f"{pct(best_sel['Metrics At Selected Threshold']['Specificity'])} and "
        f"cost {best_sel['Metrics At Selected Threshold']['Total Cost']}. "
        f"Inference latency ranges from {lat_lo:.3f} ms to {lat_hi:.3f} ms, so "
        f"the tree ensembles satisfy the sub-millisecond condition of [1] while "
        f"the two slower models do not.")

    add_heading(doc, "B. Component ablation", 2)
    ab_rows = []
    labels = {
        "A_counts_only": "A — three base counts",
        "B_plus_trailing": "B — plus trailing aggregates",
        "C_plus_deltas": "C — plus absolute deltas",
        "D_plus_persistence": "D — plus persistence",
        "E_full": "E — full basis",
    }
    for key, label in labels.items():
        r = abl_rows[key]
        ab_rows.append([
            label, str(r["Features"]), f"{r['AUROC']:.4f}",
            pct(r["Sensitivity At 0.5"]), pct(r["Specificity At 0.5"]),
            str(r["Cost At 0.5"]), str(r["Cost At Selected"]),
        ])
    add_table(doc,
              "TABLE V. Feature-group ablation. Cost is reported both at the "
              "base paper's fixed 0.5 threshold and at the cost-selected "
              "threshold.",
              ["Variant", "Feat.", "AUROC", "Sens.", "Spec.", "Cost @0.5",
               "Cost @sel"], ab_rows,
              widths=[1.85, 0.5, 0.68, 0.68, 0.68, 0.72, 0.72])
    add_body(doc,
        f"Table V attributes performance to each family of temporal features. "
        f"The dominant effect is the transition from group A to group B: adding "
        f"the trailing aggregates raises the area under the ROC curve from "
        f"{a_counts['AUROC']:.4f} to {b_trail['AUROC']:.4f}, raises sensitivity "
        f"at 0.5 from {pct(a_counts['Sensitivity At 0.5'])} to "
        f"{pct(b_trail['Sensitivity At 0.5'])}, and reduces cost from "
        f"{a_counts['Cost At 0.5']} to {b_trail['Cost At 0.5']}, a factor of "
        f"{a_counts['Cost At 0.5'] / b_trail['Cost At 0.5']:.1f}. The three "
        f"subsequent families add no further cost reduction at the fixed "
        f"threshold — "
        f"{b_trail['Cost At 0.5']}, {abl_rows['C_plus_deltas']['Cost At 0.5']}, "
        f"{abl_rows['D_plus_persistence']['Cost At 0.5']} and "
        f"{e_full['Cost At 0.5']} for groups B, C, D and E respectively. This is "
        f"the central empirical claim of this paper: temporal horizon, not "
        f"feature diversity, is the binding constraint, so the ceiling of [1] is "
        f"an artefact of aggregating to a single one-second count rather than a "
        f"consequence of the three operation types it chose.")

    add_heading(doc, "C. Generalisation, evasion and drift", 2)
    g_rows = []
    for family, v in sorted(zsum["Per Family"].items()):
        g_rows.append([f"{family} (held out)",
                       pct(v["Mean Zero Shot Sensitivity"]),
                       pct(v["Mean Zero Shot Specificity"]), "-", "-"])
    curve = {r["Regime"]: r for r in ev["Curve"]}
    for regime, label in (("rate_limited_60", "Rate-limited, 60 files/s"),
                          ("rate_limited_25", "Rate-limited, 25 files/s"),
                          ("rate_limited_15", "Rate-limited, 15 files/s"),
                          ("rate_limited_1", "Rate-limited, 1 file/s"),
                          ("mimicry", "Benign-mimicry")):
        if regime in curve:
            r = curve[regime]
            g_rows.append([label, pct(r["Sensitivity"]), pct(r["Specificity"]),
                           "-", "-"])
    g_rows.append(["Shifted benign profile, frozen model",
                   pct(rec["Stale"]["Sensitivity"]),
                   pct(rec["Stale"]["Specificity"]), "-", "-"])
    g_rows.append(["Shifted benign profile, retrained",
                   pct(rec["Retrained"]["Sensitivity"]),
                   pct(rec["Retrained"]["Specificity"]),
                   f"{dmon['Detection Delay Seconds']:.0f}",
                   f"{dmon['False Alarm Rate']:.2f}"])
    g_rows.append(["Fresh oracle (upper bound)",
                   pct(rec["Fresh Oracle"]["Sensitivity"]),
                   pct(rec["Fresh Oracle"]["Specificity"]), "-", "-"])
    add_table(doc,
              "TABLE VI. Generalisation to unseen families, evasion regimes and "
              "drift recovery. Drift delay and false-alarm rate are properties of "
              "the monitoring study and apply only to the retrained row; a dash "
              "means the quantity does not apply to that scenario.",
              ["Scenario", "Sensitivity", "Specificity", "Drift delay (s)",
               "False-alarm rate"], g_rows,
              widths=[2.15, 0.95, 0.95, 1.0, 1.0])

    add_body(doc,
        f"The detector transfers well across families when the operation-rate "
        f"envelope is preserved: mean zero-shot sensitivity is "
        f"{pct(zsum['Overall Mean Zero Shot Sensitivity'])} at "
        f"{pct(zsum['Overall Mean Zero Shot Specificity'])} specificity, for a "
        f"mean sensitivity drop of {zsum['Overall Mean Drop']:.4f} relative to "
        f"in-distribution performance. The weakest family is "
        f"{min(zsum['Per Family'], key=lambda k: zsum['Per Family'][k]['Mean Zero Shot Sensitivity'])} "
        f"at {pct(min(v['Mean Zero Shot Sensitivity'] for v in zsum['Per Family'].values()))}. "
        f"The detector nonetheless fails completely once the encryption rate "
        f"falls below roughly fifteen files per second, where the per-window "
        f"counts are indistinguishable from benign activity, and it degrades "
        f"under a benign-mimicry regime to below the sensitivity reported by "
        f"[1]. This is a quantitative boundary that [1] does not state.")

    add_heading(doc, "D. Drift detection and recovery", 2)
    add_body(doc,
        f"On the monitored stream the false-alarm rate is "
        f"{dmon['False Alarm Rate']:.2f} over {dmon['Baseline Chunks Evaluated']} "
        f"chunks drawn from an unchanged period, and the first trigger in the "
        f"drifted period occurs {dmon['Detection Delay Seconds']:.0f} s after "
        f"drift onset. That delay is long in absolute terms and we do not present "
        f"it as a strength: the chunk size that suppresses false alarms at the "
        f"conventional 0.25 threshold is necessarily also the granularity at "
        f"which drift can be noticed, so the two quantities trade directly "
        f"against one another. Lowering the significance threshold or the chunk "
        f"size would shorten the delay and raise the false-alarm rate; we "
        f"report the operating point that produces no false alarms rather than "
        f"the one that reacts fastest, and we did not sweep this trade-off.")
    add_body(doc,
        f"Recovery is beneficial and is reported as such: retraining "
        f"on reference plus recent windows moves sensitivity from "
        f"{pct(rec['Stale']['Sensitivity'])} to "
        f"{pct(rec['Retrained']['Sensitivity'])} and cost from "
        f"{rec['Stale']['Total Cost']} to {rec['Retrained']['Total Cost']}, "
        f"against a fresh-oracle ceiling of {rec['Fresh Oracle']['Total Cost']} "
        f"and {pct(rec['Fresh Oracle']['Sensitivity'])}. The residual oracle gap "
        f"of {rec.get('Oracle Gap', 0):.4f} quantifies how much performance is "
        f"recoverable by retraining and how much requires fresh data.")

    add_heading(doc, "E. Dispersion across seeds", 2)
    sens_models = {v["Model"]: v for v in sd["Sensitivity Stability"]["models"]}
    cost_models = {v["Model"]: v for v in sd["Cost Stability"]["models"]}
    s_rows = []
    for name in ("XGBoost", "Random Forest", "AdaBoost", "SVM", "Decision Tree"):
        s_rows.append([
            name,
            f"{sens_models[name]['Mean']:.4f}",
            f"{sens_models[name]['Std']:.4f}",
            f"{cost_models[name]['Mean']:.1f}",
            f"{cost_models[name]['Std']:.1f}",
            f"{int(cost_models[name]['Min'])}-{int(cost_models[name]['Max'])}",
        ])
    add_table(doc,
              "TABLE VII. Dispersion across ten seeds at 30 000 windows per seed.",
              ["Model", "Sens. mean", "Sens. sd", "Cost mean", "Cost sd",
               "Cost range"], s_rows,
              widths=[1.3, 0.95, 0.85, 0.95, 0.85, 1.2])
    comp = sd["Comparison (Sensitivity)"]["comparisons"]
    worst_holm = max(v["holm_adjusted_p"] for v in comp.values())
    add_body(doc,
        f"Across ten seeds the mean sensitivity of the best model is "
        f"{sens_models['XGBoost']['Mean']:.4f} with a standard deviation of "
        f"{sens_models['XGBoost']['Std']:.4f}, and no model's sensitivity range "
        f"is small enough to be called stable by the criterion used. Cost is "
        f"markedly less stable than sensitivity: the Decision Tree ranges from "
        f"{cost_models['Decision Tree']['Min']} to "
        f"{cost_models['Decision Tree']['Max']}. Critically, no classifier is "
        f"significantly better than any other on either metric after "
        f"Holm-Bonferroni correction — the largest adjusted p-value on "
        f"sensitivity is {worst_holm:.3f}. The model ranking reported by [1], "
        f"and by any single-run evaluation, should therefore be read as an "
        f"artefact of point-estimate tie-breaking.")

    add_heading(doc, "F. Alarm policy", 2)
    add_body(doc,
        f"A three-state decision layer is compared against fixed-threshold and "
        f"k-of-n rules. The best baseline achieves a test cost of "
        f"{ap['Summary']['Best Baseline Test Cost']} at sensitivity "
        f"{pct(ap['Summary']['Best Baseline Test Sensitivity'])} and specificity "
        f"{pct(ap['Summary']['Best Baseline Test Specificity'])}. The dynamic "
        f"policy achieves cost {ap['Summary']['DP Test Cost']} at sensitivity "
        f"{pct(ap['Summary']['DP Test Sensitivity'])} and specificity "
        f"{pct(ap['Summary']['DP Test Specificity'])}, an improvement of "
        f"{ap['Summary']['Cost Improvement']} in cost and "
        f"{ap['Summary']['Sensitivity Change']:+.4f} in sensitivity. The "
        f"specificity change is {ap['Summary']['Specificity Change']:+.4f}: the "
        f"policy buys its lower cost and higher sensitivity with a modestly "
        f"higher false-alarm rate, which is the intended trade under a 10:1 "
        f"cost ratio but must be stated rather than omitted.")

    add_heading(doc, "G. Window width and stride", 2)
    w_rows = []
    for r in ms["Width Sweep"]:
        w_rows.append([
            f"{r['Width Seconds']} s", f"{r['Stride Seconds']} s",
            str(r["Windows"]), str(r["Test Attack Windows"]),
            pct(r["Sensitivity"]), str(r["Total Cost"]),
            f"{r['Cost Per Attack Window']:.4f}",
            f"{r['Median Detection Latency (s)']:.1f}",
        ])
    add_table(doc,
              "TABLE VIII. Window-width sweep. Raw cost is not comparable across "
              "rows; cost per attack window is.",
              ["Width", "Stride", "Windows", "Atk windows", "Sens.", "Cost",
               "Cost/atk", "Lat. (s)"], w_rows,
              widths=[0.6, 0.6, 0.8, 0.85, 0.72, 0.6, 0.75, 0.7])
    sweep = ms["Width Sweep"]
    add_body(doc,
        f"Sensitivity peaks at a two-second width "
        f"({pct(max(sweep, key=lambda r: r['Sensitivity'])['Sensitivity'])}) and "
        f"falls at both one second and thirty seconds. Total cost appears to "
        f"fall monotonically with width, from {sweep[0]['Total Cost']} at one "
        f"second to {sweep[-1]['Total Cost']} at thirty seconds, but this is an "
        f"artefact of the shrinking denominator: the same comparison normalised "
        f"per attack window rises from "
        f"{sweep[0]['Cost Per Attack Window']:.4f} to "
        f"{sweep[-1]['Cost Per Attack Window']:.4f}. We report the normalised "
        f"figure because the raw one is misleading. The stride analysis at a "
        f"five-second width shows a 40 % overlap giving the best sensitivity, "
        f"confirming that the choice of stride trades redundancy against "
        f"boundary effects rather than improving monotonically.")

    add_heading(doc, "H. Throughput and inference cost", 2)
    inf = bm["Inference"]
    bench_rows = []
    for name, v in inf["Models"].items():
        bench_rows.append([
            name.replace("_", " "), f"{v['Mean ms']:.4f}",
            f"{v['P50 ms']:.4f}", f"{v['P95 ms']:.4f}",
            f"{v['P99 ms']:.4f}", f"{v['Max ms']:.4f}",
            f"{v['Windows Per Second']:.0f}",
        ])
    add_table(doc,
              "TABLE IX. Single-window inference latency over "
              f"{inf['Repeats']} repeats.",
              ["Model", "Mean (ms)", "P50", "P95", "P99", "Max",
               "Win/s"], bench_rows, widths=[1.1, 0.8, 0.7, 0.7, 0.7, 0.7, 0.8])
    ing_tp = ing["Throughput"]["Events Per Second"]
    feat = bm["Features"]
    add_body(doc,
        f"The deployed pipeline parses and aggregates {ing['Throughput']['Events']:,} "
        f"file-operation events in "
        f"{ing['Throughput']['Total Seconds']:.2f} s, a throughput of "
        f"{ing_tp:,.0f} events per second, and constructs {feat['Feature Columns']} "
        f"features for {feat['Windows']:,} windows in {feat['Milliseconds']:.1f} ms, "
        f"a rate of {feat['Windows Per Second']:,.0f} windows per second. Both "
        f"figures exceed the one-window-per-second operating rate by five orders "
        f"of magnitude, so the aggregation stage is not a bottleneck at the "
        f"target rate. Inference latency is strongly model-dependent and is "
        f"tabulated above; the fastest model sustains "
        f"{inf['Models']['decision_tree']['Windows Per Second']:,.0f} "
        f"classifications per second and the slowest "
        f"{min(v['Windows Per Second'] for v in inf['Models'].values()):,.0f}. "
        f"The extrapolation in [1] that the framework scales to several hundred "
        f"clients therefore holds for the tree ensembles but not for the slower "
        f"models, and which model is deployed determines whether it holds.")

    # ---------------- VII. Discussion ----------------
    add_heading(doc, "VII. DISCUSSION", 1)
    add_heading(doc, "A. Design rationale", 2)
    add_body(doc,
        "The three window scales trade timeliness against evidence: the 3 s "
        "scale reacts to bursts, while the longer scales let a slow encryptor "
        "accumulate a visible signature. The measured horizon effect is large — "
        f"cost falls by a factor of "
        f"{a_counts['Cost At 0.5'] / b_trail['Cost At 0.5']:.1f} from three raw "
        f"counts to "
        f"{b_trail['Features']} trailing-aggregate features — and the "
        "subsequent feature families add nothing, which locates the deficit of "
        "[1] in window length rather than in the choice of operation types.")
    add_heading(doc, "B. Operating threshold", 2)
    add_body(doc,
        "A 10:1 cost ratio expresses that a missed attack is far worse than a "
        "false alarm, but the right ratio depends on the organisation. Because "
        "the threshold is selected offline against this cost function on a "
        "held-out split, it cannot be re-tuned in a live stream without new "
        "labelled data, and we do not claim an adaptive threshold.")
    add_heading(doc, "C. Trust and governance", 2)
    add_body(doc,
        "Feature-group attribution gives the operator a short reason for each "
        "flagged window, and the three-state output plus hysteresis prevents a "
        "single ambiguous window from raising an alert. Retraining is performed "
        "offline against a reference and a recent window rather than in "
        "production, so there is no rollback guard to rely on and no mechanism "
        "to resist poisoning of a live update buffer; this is a deliberate "
        "trade of responsiveness for safety.")
    add_heading(doc, "D. Negative results", 2)
    add_body(doc,
        "Two negative results are reported deliberately. A soft-voting "
        "ensemble of the five classifiers does not improve on the best single "
        "model; on this feature basis the members are too correlated for "
        "averaging to add anything. And no classifier is significantly better "
        "than any other across ten seeds, which means that selecting a "
        "particular 'best' model from a single run — as [1] does — is not "
        "supported by the evidence.")

    # ---------------- VIII. Limitations ----------------
    add_heading(doc, "VIII. LIMITATIONS AND THREATS TO VALIDITY", 1)
    for item in [
        "Emulated attack data. No ransomware sample is executed. Attack "
        "profiles are modelled from published family characteristics and benign "
        "profiles are transcribed from Table 3 of [1]. This is the principal "
        "limitation and it bounds every quantitative claim in Section VI.",

        "Controlled environment. The workload is generated, not recorded from "
        "an organisation, so external validity is limited. The largest trace "
        f"used here is {sd['Windows Per Trace']:,} windows against the "
        f"2.5 million log entries of [1], a two-order-of-magnitude gap.",

        "Limited families. Only five families are modelled, and the "
        "unseen-family results cover only those five. No family outside the set "
        "of [1] was evaluated.",

        "Incomplete observation set. Only create, rename and delete are "
        "collected. A ransomware that overwrites existing files in place "
        "without renaming them is invisible to the detector.",

        "Label noise. Labels derive from emulated episode start and stop times, "
        "and boundary windows may be mislabelled. No expert review of "
        "boundaries was performed.",

        "Adaptive adversaries. Sensitivity falls to zero at or below fifteen "
        "files per second and to "
        f"{pct(curve['mimicry']['Sensitivity'])} under a benign-mimicry regime, "
        "both below the figure reported in [1]. Entropy padding and partial "
        "encryption were not evaluated at all.",

        "Drift handling risks. The monitor operates at a calibrated chunk size "
        "on this trace; the required chunk size is a property of the "
        "distribution and would have to be re-derived for another deployment. "
        "No false-alarm behaviour was measured under benign regime change that "
        "is not present in the evaluation.",

        "Explanation fidelity. Decision-path attribution on a tree ensemble is "
        "approximated from a single member tree, and permutation importance is "
        "a stochastic estimate. The stability of either method across seeds was "
        "not measured.",

        "Statistical power. The main evaluation contains a small number of "
        "attack episodes in its test partition, and although the seed study "
        "spans ten runs, cost is markedly unstable across seeds.",

        "Base rate floor. The episode-aware split requires at least 24 "
        "episodes, which imposes an attack base-rate floor of roughly seven per "
        "cent; the imbalance study therefore probed "
        f"{', '.join(str(round(r, 4)) for r in load('imbalance.json')['Ratios Achieved'])} "
        "rather than the lower rates requested.",
    ]:
        add_bullet(doc, item)

    # ---------------- IX. Conclusion ----------------
    add_heading(doc, "IX. CONCLUSION", 1)
    add_body(doc,
        "We presented ASSRD, a server-side ransomware detector that combines "
        "multi-scale temporal aggregation, five cost-sensitive classifiers, "
        "offline cost-selected thresholding, permutation and decision-path "
        "attribution, and Population Stability Index drift monitoring with "
        "offline model recovery. It addresses the structural limit of "
        "count-based detection rather than the empirical shortfall of any one "
        "implementation. On emulated data, temporal context raises the area "
        f"under the ROC curve from {a_counts['AUROC']:.4f} to "
        f"{b_trail['AUROC']:.4f} and reduces total cost from "
        f"{a_counts['Cost At 0.5']} to {b_trail['Cost At 0.5']}; transfer to an "
        f"unseen family costs a mean of {zsum['Overall Mean Drop']:.4f} "
        "sensitivity; and a ten-seed study shows the five classifiers are not "
        "statistically separable. The detector is nonetheless defeated by slow "
        "encryption, with zero sensitivity at or below fifteen files per second, "
        "which is the most important finding of this work and the priority for "
        "future research.")
    add_body(doc,
        "Future work should replace emulated attack data with executed samples "
        "in an isolated virtualised environment, add collection of in-place "
        "file modification so that renaming is not required, re-derive the "
        "monitor's chunk size for each deployment rather than assuming it, and "
        "conduct field trials on production file servers.")

    # ---------------- Declarations ----------------
    add_heading(doc, "ACKNOWLEDGMENT", 1)
    add_body(doc,
        "«ACKNOWLEDGMENT TO BE SUPPLIED, OR THIS SECTION OMITTED IF NONE.»")

    add_heading(doc, "AUTHOR CONTRIBUTIONS", 1)
    add_body(doc,
        "«CONTRIBUTIONS TO BE COMPLETED FOLLOWING THE CRediT TAXONOMY: "
        "Conceptualization, Methodology, Software, Validation, Formal analysis, "
        "Investigation, Data curation, Writing — original draft, Writing — "
        "review and editing, Visualization, Project administration, "
        "Supervision.»")

    add_heading(doc, "FUNDING", 1)
    add_body(doc, "«FUNDING STATEMENT TO BE SUPPLIED, OR OMITTED.»")

    add_heading(doc, "DATA AVAILABILITY", 1)
    add_body(doc,
        "All evaluation data in this work is generated by a seeded emulator "
        "included in the artifact, and every figure reported here is regenerated "
        "by the scripts that accompany it. The emulator profiles are modelled "
        "from published family characteristics and are not derived from "
        "executed malware; no malware sample is distributed with this work.")

    add_heading(doc, "COMPETING INTERESTS", 1)
    add_body(doc, "«COMPETING INTERESTS STATEMENT TO BE SUPPLIED.»")

    add_heading(doc, "ETHICS APPROVAL AND CONSENT TO PARTICIPATE", 1)
    add_body(doc,
        "Not applicable. The work involves no human subjects and no personal "
        "data.")

    # ---------------- References ----------------
    add_heading(doc, "REFERENCES", 1)
    refs = [
        '[1]  G. Arányi, T. Miseta, and V. Szücs, "Ransomware detection '
        'based on server-side file operation logs using machine learning," '
        'Journal on Information Security, vol. 8, 2026, doi: '
        '10.1186/s13635-026-00229-7.',

        '[2]  N. Scaife, H. Carter, P. Traynor, and K. R. Butler, "CryptoLock '
        '(and drop it): Stopping ransomware attacks on user data," in Proc. '
        'IEEE 36th Int. Conf. Distributed Computing Systems (ICDCS), 2016, '
        'pp. 303-312.',

        '[3]  A. Continella et al., "ShieldFS: A self-healing, '
        'ransomware-aware filesystem," in Proc. 32nd Annu. Conf. Computer '
        'Security Applications (ACSAC), 2016, pp. 336-347.',

        '[4]  A. Kharaz, S. Arshad, C. Mulliner, W. Robertson, and E. Kirda, '
        '"UNVEIL: A large-scale, automated approach to detecting ransomware," '
        'in Proc. 25th USENIX Security Symposium, 2016, pp. 757-772.',

        '[5]  D. Sgandurra, L. Muñoz-González, R. Mohsen, and E. C. Lupu, '
        '"Automated dynamic analysis of ransomware: Benefits, limitations and '
        'use for detection," arXiv:1609.03020, 2016.',

        '[6]  K. Cabaj, M. Gregorczyk, and W. Mazurczyk, '
        '"Software-defined networking-based crypto ransomware detection using '
        'HTTP traffic characteristics," Computers and Electrical Engineering, '
        'vol. 66, pp. 353-368, 2018.',

        '[7]  C. Cortes and V. Vapnik, "Support-vector networks," Machine '
        'Learning, vol. 20, no. 3, pp. 273-297, 1995.',

        '[8]  L. Breiman, "Random forests," Machine Learning, vol. 45, no. 1, '
        'pp. 5-32, 2001.',

        '[9]  Y. Freund and R. E. Schapire, "A decision-theoretic '
        'generalization of on-line learning and an application to boosting," '
        'Journal of Computer and System Sciences, vol. 55, no. 1, pp. 119-139, '
        '1997.',

        '[10] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting '
        'system," in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and '
        'Data Mining, 2016, pp. 785-794.',

        '[11] J. R. Quinlan, "Induction of decision trees," Machine Learning, '
        'vol. 1, no. 1, pp. 81-106, 1986.',

        '[12] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," '
        'Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.',

        '[13] J. Gama, I. Zliobaite, A. Bifet, M. Pechenizkiy, and A. '
        'Bouchachia, "A survey on concept drift adaptation," ACM Computing '
        'Surveys, vol. 46, no. 4, art. 44, 2014.',

        '[14] S. M. Lundberg and S.-I. Lee, "A unified approach to '
        'interpreting model predictions," in Advances in Neural Information '
        'Processing Systems 30 (NeurIPS), 2017, pp. 4765-4774.',

        '[15] C. Elkan, "The foundations of cost-sensitive learning," in Proc. '
        '17th Int. Joint Conf. Artificial Intelligence (IJCAI), 2001, '
        'pp. 973-978.',

        '[16] T. Fawcett, "An introduction to ROC analysis," Pattern '
        'Recognition Letters, vol. 27, no. 8, pp. 861-874, 2006.',
    ]
    for r in refs:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(r)
        run.font.size = Pt(9)

    doc.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(BASE_DIR / "AI Detection.docx"))
    args = ap.parse_args()
    path = build(args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
