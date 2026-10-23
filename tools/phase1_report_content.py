"""Report content and assembly for the Phase 1 project report.

Every factual claim traces to a project file:
  * project phase1/Phase1_Abstract.docx  - abstract, proposed framework, figures
  * Base Paper.pdf                        - Aranyi, Miseta & Szücs (2026)
  * conference/literature_review_matrix.md - literature survey entries
  * conference/research_gap_and_contributions.md - gap and contribution scope
  * conference/project_inventory.md       - implementation inventory
  * conference/prototype_technical_audit.md - implemented vs planned status
  * data/model_results.json               - measured baseline results
  * prototype/ (ASSRD prototype)          - six-model rebaseline results
"""

from __future__ import annotations

from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from build_phase1_abstract import ABSTRACT_PARAGRAPHS, KEYWORDS
from build_phase1_report import (
    BODY_PT, DOCX_PATH, PROJECT_TITLE, bullet, caption, chapter, figure,
    new_document, para, placeholder, reference, section, subsection, table,
    _set_font, _spacing,
)

FIG = "project phase1/figures"

TITLE = PROJECT_TITLE


# ---------------------------------------------------------------------------
def title_page(doc):
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, after=18)
    _set_font(p.add_run(TITLE), 18.0, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, after=48)
    _set_font(p.add_run("PHASE I PROJECT REPORT"), 16.0, bold=True)

    for label in ("Student Name", "Roll Number / Registration Number",
                  "Supervisor", "Department", "Institution",
                  "Date of Submission"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _spacing(p, after=10)
        _set_font(p.add_run(f"{label}: "), BODY_PT, bold=True)
        _set_font(p.add_run("________________________________"), BODY_PT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, before=40)
    _set_font(p.add_run("The fields above are placeholders. They are not "
                        "available in the project materials and must be "
                        "completed before submission."), 10.0, italic=True)
    doc.add_page_break()


def abstract_page(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, after=12)
    _set_font(p.add_run("ABSTRACT"), 14.0, bold=True)

    for text in ABSTRACT_PARAGRAPHS:
        para(doc, text)

    para(doc, "")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Pt(36)
    _spacing(p)
    _set_font(p.add_run(KEYWORDS), BODY_PT, italic=True)
    # No page break here: chapter() already opens with one, and adding both
    # produces a blank page.


# ---------------------------------------------------------------------------
def chapter_1(doc):
    chapter(doc, 1, "INTRODUCTION")

    para(doc,
         "Ransomware has become one of the most costly forms of cyber-attack "
         "because it converts an availability failure into a permanent data "
         "loss. An endpoint agent can watch a workstation closely, but it "
         "requires host instrumentation on every machine, it consumes endpoint "
         "resources, and in many organisations it cannot be deployed at all "
         "because of administrative policy, third-party contractual limits or "
         "privacy regulation. The file server, however, is already generating "
         "the evidence that matters. Every file creation, rename and deletion "
         "passes through it and is already recorded in the server's own logs. "
         "Those records contain no user content, only metadata, so they can be "
         "analysed centrally without touching client machines at all. [1]")
    figure(doc, "fig1_1_lifecycle.png",
           "Figure 1.1: Ransomware lifecycle as observed in server-side file "
           "telemetry, and the early-detection window it defines")

    para(doc,
         "The central difficulty is that file-operation logs are extremely "
         "voluminous and individually unremarkable. A user who is actively "
         "working creates, renames and deletes files continuously, and the "
         "per-second counts produced by ordinary work overlap the counts "
         "produced by encryption. The signal that separates the two is not the "
         "size of any single operation but its persistence: an attack sustains "
         "elevated operation counts across consecutive windows, whereas a "
         "user's activity is a burst followed by quiet. Any detector that reads "
         "one window in isolation is therefore reading the wrong variable. [2]")
    figure(doc, "fig1_2_ops.png",
           "Figure 1.2: Overlap between benign and attack operation counts in a "
           "single one-second window")

    section(doc, "1.1", "Problem Statement and Motivation")
    para(doc,
         "The problem addressed by this project is stated precisely as "
         "follows: given only the stream of create, rename and delete events "
         "recorded by a file server, decide within seconds, for each "
         "one-second window, whether the window belongs to a ransomware "
         "encryption episode, and do so at an operating point that reflects the "
         "real asymmetry between the cost of a missed attack and the cost of a "
         "false alarm. The motivation for solving the three parts together "
         "rather than separately is that they interact. Feature design only "
         "matters if the decision rule can exploit persistence. A cost-aware "
         "threshold only matters if the underlying score actually ranks "
         "persistent activity above bursty activity. And an alerting path only "
         "has value if the score is stable enough that the alert rate is "
         "governable.")
    for item in [
        "Class imbalance: at realistic server event rates, attack windows are a "
        "small minority of all windows, so a detector that reports high "
        "accuracy may be raising no alarms at all. Accuracy alone is "
        "therefore not an acceptable measure.",
        "Rate-limited evasion: an attacker who limits encryption to a rate "
        "that stays inside the benign envelope defeats any count-only rule, "
        "and this project reports that limitation explicitly rather than "
        "assuming it away.",
        "Fixed operating points: a threshold fixed at 0.5 implicitly assumes "
        "that a false alarm and a missed attack cost the same, which is not "
        "true of ransomware.",
        "Unstable operating points: when the number of independent labelled "
        "episodes is small, the cost-minimising threshold itself moves, so the "
        "operating point has to be reported with its stability rather than as "
        "a single number.",
        "No ground truth on live servers: production file-server logs are "
        "unlabelled, so a model cannot be validated on the stream it will run "
        "on.",
    ]:
        bullet(doc, item)

    section(doc, "1.2", "Background of Server-Side Detection")
    para(doc,
         "The server-side approach was formalised by Aranyi, Miseta and Szücs "
         "(2026), who constructed a virtualised small and medium-sized "
         "enterprise environment, instrumented it with a nanosecond-scale "
         "timestamped logging mechanism built on Fluent Bit and InfluxDB, "
         "triggered attacks using samples attributed to the Ryuk, NotPetya, "
         "Lockbit, Teslacrypt and WannaCry families, aggregated file operations "
         "into one-second intervals, and trained five machine-learning models "
         "on the resulting windows. Their reported result is that all five "
         "models achieved reliable detection performance, with XGBoost "
         "performing best at a sensitivity of 91.87 per cent and prediction "
         "speeds below one millisecond. [1] That work is the direct precedent "
         "for this project and its pipeline is the baseline reproduced here.")
    figure(doc, "fig1_3_pipeline.png",
           "Figure 1.3: The server-side detection pipeline adopted as the "
           "starting point for this project")

    section(doc, "1.3", "Objectives of the Project")
    para(doc,
         "The objectives are stated so that each can be evidenced or explicitly "
         "withdrawn.")
    for i, obj in enumerate([
        "Reproduce the server-side count-based pipeline of Aranyi et al. (2026) "
        "end-to-end and transparently, so that its reported behaviour can be "
        "inspected rather than trusted.",
        "Replace the three per-window raw counts with multi-scale trailing "
        "temporal features over 3, 5, 10 and 30-second windows, and quantify by "
        "controlled ablation what the additional feature groups are worth.",
        "Replace the fixed 0.5 decision threshold with a cost-minimising "
        "threshold selected on validation data only, under an explicit "
        "asymmetry of C_FN = 10 against C_FP = 1.",
        "Compare six cost-sensitive classifiers, comprising the five "
        "paper-inspired models plus a standardised Logistic Regression linear "
        "baseline, under a single leakage-safe evaluation protocol.",
        "Quantify the stability of the selected operating point by resampling "
        "whole episodes rather than individual windows, and report that "
        "stability rather than a single threshold value.",
        "Attribute detections to operator-level feature groups so that an "
        "operator can understand why an alert was raised.",
        "Specify a concept-drift monitor based on the Population Stability "
        "Index together with an offline retraining and recovery procedure.",
        "Design the real-time alerting component and state precisely which of "
        "its parts are implemented and which remain proposed.",
    ], 1):
        para(doc, f"Objective {i}. {obj}", indent=True)

    section(doc, "1.4", "Scope of the Project")
    para(doc,
         "In scope: one-second windowing of server-side file-operation events; "
         "twenty-six temporal model inputs replacing the baseline's three raw "
         "counts, with a combined twenty-nine-input arm retained for ablation; "
         "six cost-sensitive classifiers; "
         "cost-based threshold selection; group-level attribution; an "
         "episode-level drift and threshold-stability protocol; an offline "
         "evaluation harness with fixed seeds and persisted artifacts; and the "
         "specification of a real-time alerting path.")
    para(doc,
         "Out of scope: endpoint or kernel-level instrumentation, which the "
         "project deliberately avoids; execution of real malware; evaluation on "
         "production or third-party server logs; multi-server or distributed "
         "deployment; and any claim of production-grade protection.")

    section(doc, "1.5", "Significance of the Project")
    para(doc,
         "The significance is threefold. First, it targets an organisational "
         "constraint that endpoint approaches cannot satisfy, because it needs "
         "no client software and touches no file content. Second, it treats "
         "persistence rather than volume as the discriminating property, which "
         "is a testable claim rather than an intuition. Third, it reports the "
         "conditions under which the method fails as prominently as the "
         "conditions under which it succeeds, which is what makes the result "
         "usable for a security decision.")

    section(doc, "1.6", "Organisation of the Report")
    para(doc,
         "Chapter 2 presents the literature survey. Chapter 3 studies the "
         "existing system, identifies its issues, states the research gap and "
         "describes the proposed system. Chapter 4 specifies the requirements, "
         "including dataset planning, feature extraction, algorithm selection "
         "and the evaluation metrics. Chapter 5 describes the implementation "
         "actually carried out and separates it from what remains proposed. "
         "Chapter 6 states the expected outcomes, the limitations and the work "
         "plan. Chapter 7 contains appendices and Chapter 8 lists the "
         "references.")


# ---------------------------------------------------------------------------
def chapter_2(doc):
    chapter(doc, 2, "LITERATURE SURVEY")
    para(doc,
         "The literature is grouped into three themes: host and file-system "
         "behavioural ransomware detection, which is the family this project "
         "belongs to; network and traffic-based detection, which observes a "
         "different telemetry layer; and the machine-learning foundations of "
         "the classifiers used. The base paper is treated separately because it "
         "is the direct methodological precedent rather than a comparison point.")
    para(doc,
         "The entries below are drawn from the project's literature review "
         "matrix. The matrix records that the project directory holds the base "
         "paper only, and that the remaining entries are general-background "
         "works included so that the project can be situated. They are marked "
         "as external and none of them reports this project's numbers.")

    section(doc, "2.1", "Literature Survey Table")
    table(doc,
          ["Ref", "Citation", "Problem addressed", "Method",
           "Strengths", "Limitations", "Relation to this project"],
          [
              ["A1", "Scaife, Carter, Traynor and Butler, CryptoLock, IEEE "
                     "ICDCS, 2016",
               "Stop ransomware at the file-system layer",
               "Real-time entropy and I/O monitoring with backup",
               "Early and practical; stops damage",
               "Needs a driver or agent; entropy signal noisy for some formats",
               "Same real-time telemetry concept; this project uses counts "
               "rather than content entropy"],
              ["A2", "Continella et al., ShieldFS, ACSAC, 2016",
               "Transparent detection with rollback",
               "File-system minifilter with process I/O and entropy features",
               "Strong accuracy and self-healing",
               "Kernel complexity; Windows-only",
               "This project performs statistical detection without rollback "
               "and is less invasive"],
              ["A3", "Kharaz et al., UNVEIL, USENIX Security, 2016",
               "Detect ransomware in a sandbox",
               "Dynamic analysis of file-operation sequences",
               "Works on unseen samples",
               "Sandbox-only, not live",
               "Closest to the operation-sequence idea but performed offline"],
              ["A4", "Sgandurra et al., Automated Dynamic Analysis of "
                     "Ransomware, 2016",
               "Automated dynamic feature extraction",
               "Twenty features from file I/O, registry and API calls",
               "Rich feature set",
               "Sandbox and offline",
               "This project uses three aggregate features, which is far "
               "lighter but less expressive"],
              ["B1", "Cabaj, Gregorczyk and Mazurczyk, SDN-based "
                     "crypto-ransomware detection, 2018",
               "Network-level crypto-ransomware detection",
               "SDN with HTTP flow statistics",
               "No endpoint agent required",
               "Only network-observable ransomware",
               "Different telemetry layer; this project uses server-side "
               "file-operation logs"],
              ["P", "Aranyi, Miseta and Szücs (2026), the base paper",
               "Ransomware detection from server-side file-operation logs",
               "Count features per window, five classifiers, cost matrix",
               "Server-side and low overhead",
               "Single window length, fixed threshold, one environment",
               "The direct precedent reproduced and extended by this project"],
          ],
      widths=[0.35, 1.05, 0.95, 1.05, 0.8, 0.9, 1.1],
      caption_text="Table 2.1: Literature survey grouped by theme")

    section(doc, "2.2", "Classifier Foundations")
    para(doc,
         "The classifiers used in this project are standard, and their "
         "foundational references are recorded so that the selection is "
         "traceable rather than arbitrary: support-vector networks by Cortes "
         "and Vapnik; random forests by Breiman; AdaBoost by Freund and "
         "Schapire; XGBoost by Chen and Guestrin; and induction of decision "
         "trees by Quinlan. Logistic Regression is included as a linear control "
         "and is treated in the same way as a standard baseline rather than as "
         "a contribution.")

    section(doc, "2.3", "Comparative Discussion")
    para(doc,
         "Two points of agreement and two of disagreement emerge from the "
         "survey. In agreement: all of the host and behavioural works confirm "
         "that file-operation behaviour carries a usable ransomware signal, "
         "which supports the premise of this project. In disagreement, the "
         "first concerns feature richness. The works A1 and A2 use content "
         "entropy together with per-process context, and A3 and A4 use dozens "
         "of features, whereas the count-based pipeline uses three. That is a "
         "deliberate simplification inherited from the base paper, and it "
         "directly increases exposure to low-and-slow evasion. In disagreement, "
         "the second and more important point concerns evaluation. The published "
         "works evaluate on real ransomware, whereas this project evaluates on "
         "a synthetic generator. No number reported here is comparable with "
         "the accuracies reported by A1 to A4, and none is presented as though "
         "it were.")
    para(doc,
         "One problem remains unresolved by the surveyed works and by this "
         "project: none of them demonstrates robustness to behavioural drift or "
         "to an adversarial attacker who stays below a fixed count threshold. "
         "That remains the central open problem and is addressed directly in "
         "the proposed methodology.")


# ---------------------------------------------------------------------------
def chapter_3(doc):
    chapter(doc, 3, "SYSTEM STUDY")

    section(doc, "3.1", "Existing System")
    para(doc,
         "The existing system for reference is the server-side count-based "
         "pipeline of Aranyi et al. (2026). A virtualised enterprise "
         "environment emits file operations; a logging layer built on Fluent Bit "
         "and InfluxDB timestamps them; the events are aggregated into "
         "one-second intervals; each interval is reduced to three counts, "
         "namely the number of file creations, renames and deletions; five "
         "machine-learning models are trained on those counts; and each new "
         "window is classified in near real time.")
    para(doc,
         "A working implementation of this baseline already exists inside the "
         "project. It captures operating-system file events through a "
         "filesystem observer, aggregates them in a one-second loop, scores "
         "each window with the selected classifier, stores every prediction in "
         "a SQLite database, and exposes the result through a REST interface "
         "and a browser dashboard. That implementation is genuine and is "
         "described further in Chapter 5.")

    section(doc, "3.2", "Issues Identified in the Existing System")
    for item in [
        "A single fixed one-second window. One window is too short to "
        "represent persistence and too long to remain cheap at very high event "
        "rates. The system therefore cannot distinguish a burst of work from a "
        "sustained attack on principle, not merely on average.",
        "Three raw counts per window. Counts discard ordering, regularity and "
        "duration, which are exactly the properties that distinguish sustained "
        "encryption from productive work.",
        "A static decision threshold of 0.5. This implicitly weights a missed "
        "attack and a false alarm equally, which contradicts the operational "
        "reality that a missed attack causes irreversible loss.",
        "Training in a single environment. A model tuned on one workload has "
        "no guarantee of transferring to a different server, a different shift "
        "pattern or an unseen ransomware family.",
        "Cost treated as a selection criterion only. Cost is applied after "
        "fitting as a scoring rule rather than during fitting, so the learner "
        "never sees the asymmetry.",
        "Robustness not evaluated. Neither behaviour drift nor evasion is "
        "measured, so the operating envelope of the detector is unknown.",
        "Explainability not provided. An operator receiving an alert cannot "
        "see which measurements produced it.",
    ]:
        bullet(doc, item)
    figure(doc, "fig3_1_existing.png",
           "Figure 3.1: Architecture of the conventional count-based detection "
           "pipeline")

    section(doc, "3.3", "Identified Research Gap")
    para(doc,
         "The gap is stated carefully, because the project inventory and the "
         "contribution assessment both warn against claiming novelty that the "
         "evidence does not support.")
    para(doc,
         "Server-side file-operation telemetry is a credible and inexpensive "
         "detection surface, supported by the host-behavioural literature and "
         "by the base paper. Independent, reproducible implementations of that "
         "approach, however, are scarce, and the published evaluations either "
         "require real malware or heavier instrumentation. A gap therefore "
         "exists for a reproducible engineering testbed with a transparent "
         "evaluation that implements the count-based pipeline end to end, "
         "exposes its cost-sensitive decision rule, and characterises "
         "explicitly where the approach fails.")
    para(doc,
         "This is an engineering and reproducibility gap. It is not a claim of "
         "a new detection algorithm. The contribution assessment is explicit "
         "that no new algorithm, feature or decision rule is introduced that is "
         "demonstrably beyond the base paper, and the project is framed "
         "accordingly.")

    section(doc, "3.4", "Proposed System")
    subsection(doc, "3.4.1", "System Overview")
    para(doc,
         "The proposed Adaptive Server-Side Ransomware Detection framework "
         "retains the server-side collection point and replaces everything "
         "downstream of it. It multiplies the temporal resolution of the "
         "feature extractor, replaces the fixed threshold with a cost-selected "
         "one, adds attribution and drift monitoring, and specifies a "
         "three-state alerting path with hysteresis and an offline recovery "
         "procedure.")
    figure(doc, "fig3_2_proposed.png",
           "Figure 3.2: Architecture of the proposed ASSRD framework")

    subsection(doc, "3.4.2", "Module Design")
    table(doc,
          ["Module", "Responsibility", "Status"],
          [
              ["Event capture", "Observe operating-system file events on the "
               "monitored share and timestamp them", "Implemented"],
              ["Window aggregation", "Partition the event stream into "
               "one-second windows and prune expired events", "Implemented"],
              ["Behavioural emulator", "Generate benign and ransomware-like "
               "traces with controllable families, intensity and rate",
               "Implemented in the prototype"],
              ["Feature extractor", "Three raw counts plus twenty-six trailing "
               "temporal features over 3, 5, 10 and 30-second windows",
               "Implemented in the prototype"],
              ["Episode-aware splitter", "Partition whole attack episodes "
               "across training, validation and test sets",
               "Implemented in the prototype"],
              ["Classifier bank", "Six cost-sensitive models fitted at fixed "
               "hyper-parameters with cost-weighted samples",
               "Implemented in the prototype"],
              ["Threshold selector", "Minimise validation cost under "
               "C_FN = 10 and C_FP = 1, breaking ties towards the higher "
               "threshold", "Implemented in the prototype"],
              ["Attribution", "Group-level decision-path and permutation "
               "importance", "Specified; offline only"],
              ["Drift monitor", "Population Stability Index over feature "
               "distributions with offline retraining",
               "Specified; offline only"],
              ["Three-state alerting", "Normal, unsure and ransomware states "
               "with hysteresis, then dispatch to alert sinks",
               "Specified; not integrated with the live path"],
          ],
          widths=[1.15, 3.0, 1.55],
          caption_text="Table 3.1: Modules of the proposed framework and their "
                       "implementation status")


# ---------------------------------------------------------------------------
def chapter_4(doc):
    chapter(doc, 4, "SYSTEM REQUIREMENTS")

    section(doc, "4.1", "Dataset Planning and Behavioural Data Generation")
    para(doc,
         "No public dataset exists that pairs a file server's complete "
         "create, rename and delete stream with ransomware labels, and "
         "obtaining one would require instrumenting a production server for "
         "long enough to observe a real attack. The project therefore plans "
         "its data deliberately rather than assuming a corpus.")
    para(doc,
         "Two sources are used. The first is the behavioural emulator built "
         "for this project, which generates a continuous one-second window "
         "series with labelled attack episodes. Benign profiles are modelled on "
         "the per-role create, rename and delete ranges of the base paper, "
         "split between working and non-working shifts, and are modulated by a "
         "diurnal factor so that activity is concentrated in office hours. "
         "Attack profiles are modelled from published characteristics of the "
         "five families and are generated as episodes of configurable length "
         "with a configurable share of rate-limited, low-and-slow episodes. The "
         "second source is the live path of the implemented baseline, which "
         "supplies genuine operating-system events for integration testing but "
         "not labelled evaluation data.")
    para(doc,
         "The emulator is a deliberate and declared limitation. It is not "
         "malware and it is not a sample; it is a generator whose parameters "
         "are visible and whose failure modes are therefore known. Every "
         "result in this report is bounded by that fact.")

    section(doc, "4.2", "Feature Extraction and the Temporal-Window Concept")
    para(doc,
         "Let the create, rename and delete counts in window i be c_i, r_i and "
         "d_i. The total operation count is T_i = c_i + r_i + d_i, the active "
         "indicator is A_i = 1 when T_i is positive and 0 otherwise, and the "
         "first difference is D_i = T_i minus T_(i-1), with D_0 defined as "
         "zero.")
    para(doc,
         "For each trailing width w in the set {3, 5, 10, 30} seconds, the "
         "trailing window includes the current window and the preceding w "
         "minus one windows, and six aggregates are computed over it: the sum "
         "of T, the mean of T, the maximum of T, the mean of A, the mean of "
         "the absolute value of D, and the coefficient of variation of T, that "
         "is its standard deviation divided by its mean, taken as zero when the "
         "mean is zero. That gives twenty-four features. Two windowless "
         "features complete the set: the length of the current run of "
         "consecutive active windows, and the ratio of the mean of T over the "
         "shortest trailing window to the mean of T over the preceding sixty "
         "windows, which contrasts a short-term rate against a longer baseline "
         "and is clipped to the interval from zero to twenty.")
    para(doc,
         "The deployed model therefore receives twenty-six inputs, the temporal "
         "features alone; the three raw counts are deliberately excluded so that "
         "the trailing aggregates, and not the per-second volume, carry the "
         "decision. A combined arm retaining all twenty-nine inputs is evaluated "
         "in the ablation so that the cost of discarding the raw counts is "
         "measured rather than assumed. Aggregates include the current window "
         "deliberately, "
         "because the window has elapsed before it is classified, and because "
         "the identical rolling operator is applied at training and at "
         "inference time, so nothing leaks across a split boundary.")
    figure(doc, "fig5_2_windows.png",
           "Figure 4.1: Overlapping trailing windows used for feature "
           "construction")
    figure(doc, "fig5_3_multiscale.png",
           "Figure 4.2: Window-width sweep showing sensitivity and normalised "
           "cost as the trailing width increases")
    table(doc,
          ["Feature family", "Count", "Windows", "What it captures"],
          [
              ["Raw counts", "3", "1 s",
               "Create, rename and delete counts of the current window"],
              ["Level aggregates", "12", "3, 5, 10, 30 s",
               "Sum, mean and maximum of the operation count"],
              ["Activity", "4", "3, 5, 10, 30 s",
               "Mean of the active indicator"],
              ["Volatility", "8", "3, 5, 10, 30 s",
               "Mean absolute first difference and coefficient of variation"],
              ["Windowless", "2", "n/a",
               "Run length of active windows and the burst ratio"],
              ["Total", "29", "n/a", "Model inputs"],
          ],
          widths=[1.3, 0.6, 1.15, 2.65],
          caption_text="Table 4.1: Feature families in the proposed extractor")

    section(doc, "4.3", "Machine-Learning Algorithm Selection")
    para(doc,
         "Six classifiers are evaluated. The first five are those of the base "
         "paper and constitute the paper-inspired baseline. The sixth, "
         "Logistic Regression, was added during implementation as a linear "
         "control and is reported separately from the baseline.")
    table(doc,
          ["Model", "Origin", "Scale sensitive", "Cost-weighted fitting",
           "Reason for inclusion"],
          [
              ["Decision Tree", "Base paper", "No", "Per-sample weights",
               "Transparent, cheap, interpretable baseline"],
              ["Random Forest", "Base paper", "No", "Per-sample weights",
               "Bagged trees; robust to variance"],
              ["Support Vector Machine (RBF, calibrated)", "Base paper", "Yes",
               "Per-sample weights",
               "Distance-based margin; calibrated for probabilities"],
              ["AdaBoost", "Base paper", "No", "Per-sample weights",
               "Boosted weak learners"],
              ["XGBoost", "Base paper", "No", "Per-sample weights",
               "Regularised gradient boosting; strongest in the base paper"],
              ["Logistic Regression", "Added in implementation", "Yes",
               "Per-sample weights",
               "Linear control: shows how much complexity actually earns"],
          ],
          widths=[1.5, 1.15, 0.85, 1.15, 1.5],
          caption_text="Table 4.2: The six evaluated classifiers")
    para(doc,
         "Logistic Regression is fitted on standardised inputs because its "
         "regularised objective is not invariant to rescaling, whereas the tree "
         "ensembles are and are therefore deliberately left unscaled. Its "
         "solver is fixed rather than searched, because the search grid must "
         "remain valid in every point; the grid crosses the inverse "
         "regularisation parameter over four values with the L1 and L2 "
         "penalties and with and without class balancing, giving sixteen "
         "evaluated configurations.")
    figure(doc, "fig5_4_models.png",
           "Figure 4.3: Detection metrics of the five paper-inspired models on "
           "the implemented baseline at a fixed 0.5 threshold")

    section(doc, "4.4", "Evaluation Metrics and Cost-Sensitive Evaluation")
    para(doc,
         "Because attack windows are a small minority of all windows, accuracy "
         "is reported but is never used on its own. The reported set is "
         "sensitivity, also known as recall or the true positive rate; "
         "specificity; precision; the F1 score; the area under the receiver "
         "operating characteristic curve; average precision; and the "
         "confusion matrix from which all of them derive.")
    para(doc,
         "The cost function is the operational quantity of interest. A false "
         "alarm is charged one unit and a missed attack is charged ten units, "
         "so the total cost of a window set is C_FP multiplied by the number "
         "of false positives plus C_FN multiplied by the number of false "
         "negatives. The asymmetry expresses the fact that a false alarm costs "
         "an operator's time, whereas a missed attack costs data. Cost enters "
         "twice and for two different reasons: as per-sample weights during "
         "fitting, so that the minimiser sees the asymmetry, and again as the "
         "criterion for choosing the operating threshold on validation data.")
    para(doc,
         "Precision is prevalence-bound and is interpreted accordingly. At a "
         "prevalence near five per cent, a detector with high sensitivity pays "
         "for it in false alarms, and a high precision figure reflects the "
         "class balance at least as much as the model.")
    figure(doc, "fig5_5_ablation.png",
           "Figure 4.4: Effect of replacing the raw counts with the "
           "multi-scale temporal feature groups")

    section(doc, "4.5", "Hardware Requirements")
    table(doc,
          ["Component", "Minimum", "Recommended"],
          [["Processor", "Dual core, 2.5 GHz", "Quad core, 3.0 GHz"],
           ["Memory", "8 GB", "16 GB"],
           ["Disk", "20 GB free", "50 GB SSD"],
           ["Network", "Not required for offline evaluation",
            "1 Gbps for live capture"]],
          widths=[1.4, 2.1, 2.1],
          caption_text="Table 4.3: Hardware requirements")

    section(doc, "4.6", "Software Requirements")
    table(doc,
          ["Component", "Version", "Role"],
          [["Python", "3.12", "Runtime"],
           ["NumPy", "2.2 or later", "Numerical computation"],
           ["pandas", "2.2 or later", "Tabular processing"],
           ["scikit-learn", "1.6 or later", "Classifiers, cross-validation, metrics"],
           ["XGBoost", "2.1 or later", "Gradient boosting"],
           ["scipy", "1.18 or later", "Statistical routines"],
           ["joblib", "1.3 or later", "Model persistence and parallelism"],
           ["watchdog", "6.0 or later", "Operating-system file events"],
           ["Flask", "3.0 or later", "REST interface and dashboard"]],
          widths=[1.3, 1.5, 2.8],
          caption_text="Table 4.4: Software requirements")
    para(doc,
         "One environment caveat is recorded because it was actually "
         "encountered. The installed combination of scikit-learn 1.6 and "
         "XGBoost 2.1.3 orders their class-inheritance hierarchies in a way "
         "that breaks the tag interface scikit-learn uses, and the project "
         "carries an explicit compatibility repair applied at import time "
         "rather than masking the fault with a version pin.")


# ---------------------------------------------------------------------------
def chapter_5(doc):
    chapter(doc, 5, "IMPLEMENTATION")
    para(doc,
         "This chapter separates what has been implemented and executed from "
         "what has only been specified. The distinction is maintained "
         "throughout and is repeated wherever it matters.")

    section(doc, "5.1", "What Is Implemented and What Is Proposed")
    table(doc,
          ["Component", "State", "Evidence"],
          [["Server-side event capture and one-second aggregation",
            "Implemented",
            "Filesystem observer with a threaded aggregation loop"],
           ["REST interface, prediction history and browser dashboard",
            "Implemented", "Flask application with a SQLite history table"],
           ["Behavioural emulator and labelled traces", "Implemented",
            "Prototype emulator with configurable families and rates"],
           ["Twenty-nine feature extractor", "Implemented",
            "Prototype feature module with the documented definitions"],
           ["Episode-aware train, validation and test splitting", "Implemented",
            "Prototype splitter with a composition manifest"],
           ["Six cost-sensitive classifiers and fixed-hyperparameter fitting",
            "Implemented", "Prototype model module and its test suite"],
           ["Cost-selected threshold on validation data only", "Implemented",
            "Prototype threshold selector and its isolation test"],
           ["Six-model rebaseline and comparison", "Implemented",
            "Re-baseline orchestrator with nine protected original artifacts"],
           ["Group-level attribution", "Specified",
            "Design only; not integrated into the live path"],
           ["Population Stability Index drift monitor", "Specified",
            "Design only; offline evaluation in the prototype"],
           ["Three-state alerting with hysteresis and sinks", "Specified",
            "Design only; not integrated with the live monitoring path"],
           ["Evaluation on real server or malware data", "Not performed",
            "No such data is available to the project"]],
          widths=[2.35, 1.15, 2.2],
          caption_text="Table 5.1: Implementation status by component")

    section(doc, "5.2", "Implemented Baseline Pipeline")
    para(doc,
         "The implemented baseline runs as follows. A filesystem observer "
         "records created, deleted and moved events on a monitored share and "
         "appends them to a timestamped buffer under a lock. A one-second loop "
         "prunes expired events, counts the creations, renames and deletions "
         "in the retained window, and passes that three-value vector to the "
         "selected classifier. The predicted label and its confidence are "
         "written to a SQLite history table, exposed through a REST interface "
         "and rendered on a dashboard. Activity simulators for both benign and "
         "ransomware-like behaviour exercise the live path for demonstration "
         "and integration testing.")

    section(doc, "5.3", "Prototype Evaluation Harness")
    para(doc,
         "The prototype evaluation harness exists so that the proposed method "
         "can be tested without touching production systems. It generates a "
         "continuous labelled trace, extracts the twenty-nine features on the "
         "whole trace in temporal order, and then splits it. The split is the "
         "part that most affects whether a result means anything, so it is "
         "handled explicitly rather than left to a library default.")
    figure(doc, "fig5_1_dfd.png",
           "Figure 5.1: Data-flow diagram of the prototype evaluation harness")
    para(doc,
         "Consecutive windows of one attack episode are strongly correlated, so "
         "a row-level split would place near-duplicates on both sides of the "
         "boundary and inflate every score. The splitter therefore assigns whole "
         "attack episodes to exactly one of training, validation and test, and "
         "groups benign windows into contiguous blocks so that adjacent benign "
         "windows are not scattered. It also declares a minimum number of "
         "stealth episodes for validation and test, and if those targets cannot "
         "be met it raises with a reported shortfall rather than quietly "
         "producing a split in which one stratum is empty.")
    para(doc,
         "Two further boundaries are enforced in code rather than in prose. "
         "Tuning sees the training split only, and the threshold is selected on "
         "validation data only, with the test split scored exactly once at the "
         "already-frozen threshold. The harness also computes a diagnostic test "
         "oracle, that is, the best threshold the test data would have "
         "supported, purely to expose the generalisation gap. That oracle is "
         "stored in a separate block, is marked as not achievable deployment "
         "performance, and is checked three ways so that it can never influence "
         "a model, a feature set, a hyper-parameter or a threshold.")

    section(doc, "5.4", "Six-Model Re-Baselining")
    para(doc,
         "Adding Logistic Regression created a provenance problem that had to "
         "be solved rather than ignored. The tuning stage records a winning "
         "configuration per model, and it had recorded five. For the new model "
         "there was no recorded configuration, so the loader would have "
         "substituted a built-in default while using properly tuned values for "
         "the other five. That is a silent mix of tuned and untuned models, and "
         "it is not acceptable in a comparison.")
    para(doc,
         "The resolution was to re-run the tuning stage with all six models and "
         "to pass the resulting six configurations explicitly into the "
         "subsequent stage, so that the parameter provenance is explicit in the "
         "artifacts. Nine original result files from the earlier five-model "
         "work were protected by recording their cryptographic digests in the "
         "orchestrator itself and re-verifying them on every command. They are "
         "byte-for-byte unchanged, which is what allows the earlier results to "
         "remain reproducible while the newer results sit beside them.")
    para(doc,
         "A comparison gate then confirmed that the five pre-existing models "
         "reproduced their earlier winning configurations exactly, so the "
         "re-baselining changed nothing except the addition of one model. The "
         "new models were refitted without a parameter search, using the "
         "recorded configurations, and a test asserts that the re-baselining "
         "path never constructs a search at all.")

    section(doc, "5.5", "Verification and Auditability")
    para(doc,
         "The prototype carries a test suite of 114 tests, of which one is "
         "skipped on this platform because creating symbolic links requires "
         "privileges that are unavailable. Tests assert structure and safety "
         "rather than detection quality: that the feature boundaries are "
         "correct, that the grids are valid, that the splitter shares no "
         "episode across partitions, that the threshold selector receives only "
         "validation labels, that the oracle is isolated, that undefined "
         "metrics are reported as absent rather than as zero, that repeated "
         "runs with a fixed seed are identical, and that the nine protected "
         "original artifacts remain unchanged.")
    para(doc,
         "Every run records its configuration, seeds, dataset provenance, "
         "parameter source and environment in its own result file, together "
         "with the digests of the artifacts it depends upon. Mutating commands "
         "take an exclusive lock located outside the directory they modify, and "
         "any destructive operation first captures a sealed forensic snapshot "
         "that is independently verified before anything is deleted.")


# ---------------------------------------------------------------------------
def chapter_6(doc):
    chapter(doc, 6, "EXPECTED OUTCOMES AND LIMITATIONS")

    section(doc, "6.1", "Expected Outcomes")
    for item in [
        "A reproducible implementation of the server-side count-based pipeline "
        "whose behaviour can be inspected rather than trusted.",
        "A quantified answer to what multi-scale temporal features are worth "
        "over three raw counts, obtained by controlled ablation rather than by "
        "assertion.",
        "An operating point chosen by explicit cost rather than by a fixed "
        "default, together with a report of how stable that operating point "
        "is.",
        "Six classifiers compared under one leakage-safe protocol, separating "
        "the paper-inspired five from the linear control added during "
        "implementation.",
        "An explicit, measured statement of the conditions under which the "
        "method fails.",
        "A specified alerting and drift-recovery design that can be implemented "
        "and reviewed against a written specification.",
    ]:
        bullet(doc, item)

    section(doc, "6.2", "Limitations")
    para(doc,
         "The limitations are stated first because they bound every number in "
         "this report.")
    for item in [
        "All attack traffic is generated by a behaviour-faithful emulator. No "
        "malware is executed anywhere in the project, and no figure in this "
        "report is a field detection rate.",
        "The detector is defeated by rate-limited encryption below "
        "approximately fifteen files per second, where per-window counts "
        "become indistinguishable from benign activity. This is the principal "
        "limitation and it is reported as such in the project's own evaluation.",
        "Precision is prevalence-bound at approximately five per cent attack "
        "prevalence; high sensitivity is paid for in false alarms.",
        "The absolute performance reported for the five paper-inspired models "
        "is close to perfect because the synthetic task is highly separable, "
        "which is a property of the generator and not a demonstration of "
        "detection capability.",
        "The baseline implementation binds its development server to all "
        "interfaces and, although a token has since been added to the "
        "state-changing endpoints, it is not a production deployment.",
        "Cost-derived thresholds were measured to be unstable across "
        "resampled episodes, which reflects the small number of independent "
        "labelled episodes rather than a defect in the selection rule.",
        "Logistic Regression coefficients are unstable in magnitude and sign "
        "across near-equivalent rescalings because the temporal features are "
        "collinear, so they are reported as diagnostic only and never as a "
        "causal ranking.",
        "University, supervisor and student details are not present in the "
        "project materials and are left as placeholders.",
    ]:
        bullet(doc, item)

    section(doc, "6.3", "Future Work")
    for item in [
        "Acquire real timestamped server-side file-operation logs, or a public "
        "equivalent, and repeat the evaluation, which is the single change that "
        "would convert the project from a reproduction into a contribution.",
        "Add at least one feature family that is not an aggregate count, for "
        "example a rename-to-extension ratio or a per-window path diversity, "
        "and ablate it in isolation.",
        "Integrate the multi-scale extractor and the cost-selected threshold "
        "into the live monitoring loop behind a feature flag, retaining the "
        "count-based path as a fallback.",
        "Implement the three-state alerting path with hysteresis and a "
        "de-duplicating dispatcher.",
        "Report confidence intervals across seeds and cost curves rather than "
        "single-run point estimates.",
        "Extend the drift monitor to separate benign drift from adversarial "
        "adaptation.",
    ]:
        bullet(doc, item)

    section(doc, "6.4", "Work Plan")
    para(doc,
         "The plan below orders the remaining work so that each phase produces "
         "an artefact that the next phase can be checked against.")
    table(doc,
          ["Phase", "Activity", "Output"],
          [["Phase 1", "Literature review, problem definition, research gap, "
            "requirements, prototype evaluation harness",
            "This report and the verified prototype"],
           ["Phase 2", "Explainability: group-level attribution offline",
            "Attribution artifacts and a written interpretation"],
           ["Phase 3", "Drift analysis: controlled benign and attack "
            "behaviour changes; threshold stability",
            "Drift and stability artifacts"],
           ["Phase 4", "Integration: multi-scale extractor and cost threshold "
            "behind a feature flag",
            "Integrated live path with regression evidence"],
           ["Phase 5", "Alerting: three states, hysteresis, dispatch sinks",
            "Alert path with a documented escalation policy"],
           ["Phase 6", "Evaluation on acquired real server-side logs",
            "Real-data evaluation, or a documented statement that it was not "
            "obtainable"]],
          widths=[0.75, 2.85, 2.1],
          caption_text="Table 6.1: Work plan for the remaining phases")


# ---------------------------------------------------------------------------
def chapter_7(doc):
    chapter(doc, 7, "APPENDIX")
    section(doc, "7.1", "Appendix A: Feature Dictionary")
    para(doc,
         "Each feature is listed with its definition. For a trailing width w, "
         "the trailing window contains the current window and the preceding w "
         "minus one windows, and the aggregate is taken over the operation "
         "count T unless stated otherwise.")
    table(doc,
          ["Feature", "Definition"],
          [["create_count, rename_count, delete_count",
            "Counts of the three operation types in the current window"],
           ["ops_sum_ws", "Sum of T over the trailing w window"],
           ["ops_mean_ws", "Mean of T over the trailing w window"],
           ["ops_max_ws", "Maximum of T over the trailing w window"],
           ["active_frac_ws", "Mean of the active indicator A over the window"],
           ["ops_absdelta_mean_ws",
            "Mean of the absolute first difference of T over the window"],
           ["ops_cv_ws",
            "Coefficient of variation of T over the window, zero when the mean "
            "is zero, clipped to ten"],
           ["active_run_length",
            "Length of the current run of consecutive active windows, zero at "
            "the first inactive window"],
           ["burst_ratio",
            "Mean of T over the shortest window divided by the mean of T over "
            "the preceding sixty windows, clipped to twenty"]],
          widths=[2.1, 3.6],
          caption_text="Table 7.1: Feature dictionary")

    section(doc, "7.2", "Appendix B: Cost Function and Support Flags")
    para(doc,
         "Total cost equals C_FP multiplied by the number of false positives "
         "plus C_FN multiplied by the number of false negatives, with C_FP = 1 "
         "and C_FN = 10, giving a ratio of ten to one. When several thresholds "
         "achieve the minimum validation cost, the largest is chosen, because "
         "among cost-equivalent operating points the higher threshold raises "
         "fewer alarms and is therefore the safer default for a detector "
         "expected to page a human.")
    para(doc,
         "Because evaluation windows are correlated within an attack episode, "
         "subgroup support is reported by episode count rather than by window "
         "count, using fixed bands: zero episodes is absent, one episode is "
         "insufficient, two to four episodes is thin, and five or more is "
         "adequate. No comparative claim is drawn from a cell marked absent or "
         "insufficient.")
    section(doc, "7.3", "Appendix C: Reproducibility Artifacts")
    para(doc,
         "The prototype persists its configuration, seeds, provenance and "
         "digests alongside every result file, and the re-baselining "
         "orchestrator additionally records a machine-readable manifest of "
         "which stage produced which artifact. Nine original result files from "
         "the five-model work are protected by digests compiled into the "
         "orchestrator, so their integrity can be confirmed without trusting "
         "any file that could itself have been altered.")


# ---------------------------------------------------------------------------
def chapter_8(doc):
    chapter(doc, 8, "REFERENCES")
    refs = [
        "Aranyi, G., Miseta, T. and Szücs, V. (2026) Ransomware detection based "
        "on server-side file operation logs using machine learning. Journal on "
        "Information Security, 2026:8. doi:10.1186/s13635-026-00229-7.",
        "The ASSRD prototype documentation set, project phase1 and prototype "
        "directories: behavioural emulator definitions, feature dictionary, "
        "six-model re-baselining manifests and recorded results.",
        "Project research documentation: literature review matrix, research gap "
        "and contributions assessment, project inventory and prototype "
        "technical audit, all held in the project repository.",
        "Scaife, N., Carter, H., Traynor, P. and Butler, K. (2016) CryptoLock "
        "(and drop it): stopping ransomware attacks on user data. 2016 IEEE "
        "36th International Symposium on Reliable Distributed Systems, "
        "pp. 303-312.",
        "Continella, A. et al. (2016) ShieldFS: a self-healing, "
        "ransomware-aware filesystem. Proceedings of the 32nd Annual "
        "Computer Security Applications Conference, pp. 336-347.",
        "Kharaz, A., Arshad, S., Mulliner, C., Robertson, W., Kirda, E. and "
        "Ross, R. (2016) UNVEIL: a large-scale, automated approach to "
        "detecting ransomware. 25th USENIX Security Symposium, pp. 757-772.",
        "Sgandurra, D., Munoz-Gonzalez, L., Mohsen, R. and Lupu, E. C. (2016) "
        "Automated dynamic analysis of ransomware. arXiv:1609.03020.",
        "Cabaj, K., Gregorczyk, M. and Mazurczyk, W. (2018) SDN-based "
        "crypto-ransomware detection using HTTP traffic characteristics. "
        "Computers and Electrical Engineering, 70, pp. 373-384.",
        "Cortes, C. and Vapnik, V. (1995) Support-vector networks. Machine "
        "Learning, 20(3), pp. 273-297.",
        "Breiman, L. (2001) Random forests. Machine Learning, 45(1), pp. 5-32.",
        "Freund, Y. and Schapire, R. E. (1997) A decision-theoretic "
        "generalization of on-line learning and an application to boosting. "
        "Journal of Computer and System Sciences, 55(1), pp. 119-139.",
        "Chen, T. and Guestrin, C. (2016) XGBoost: a scalable tree boosting "
        "system. Proceedings of the 22nd ACM SIGKDD International Conference on "
        "Knowledge Discovery and Data Mining, pp. 785-794.",
        "Quinlan, J. R. (1986) Induction of decision trees. Machine Learning, "
        "1(1), pp. 81-106.",
        "Pedregosa, F. et al. (2011) Scikit-learn: machine learning in Python. "
        "Journal of Machine Learning Research, 12, pp. 2825-2830.",
        "Harris, C. R. et al. (2020) Array programming with NumPy. Nature, 585, "
        "pp. 357-362.",
        "McKinney, W. (2010) Data structures for statistical computing in "
        "Python. Proceedings of the 9th Python in Science Conference, pp. 56-61.",
        "Virtanen, P. et al. (2020) SciPy 1.0: fundamental algorithms for "
        "scientific computing in Python. Nature Methods, 17, pp. 261-272.",
    ]
    for i, text in enumerate(refs, 1):
        reference(doc, i, text)


# ---------------------------------------------------------------------------
def build():
    doc = new_document()
    title_page(doc)
    abstract_page(doc)
    chapter_1(doc)
    chapter_2(doc)
    chapter_3(doc)
    chapter_4(doc)
    chapter_5(doc)
    chapter_6(doc)
    chapter_7(doc)
    chapter_8(doc)
    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX_PATH))
    print(f"written: {DOCX_PATH}")


if __name__ == "__main__":
    build()