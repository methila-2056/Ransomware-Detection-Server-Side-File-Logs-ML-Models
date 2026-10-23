"""Build the standalone Phase 1 abstract as a Word document.

The full report (``tools/phase1_report_content.py``) embeds the same abstract in
its ``abstract_page()``. This module writes that text on its own so the abstract
can be circulated without regenerating all eight chapters.

Formatting is inherited from ``build_phase1_report`` rather than redefined, so
the standalone document matches the report template exactly: A4, Times New Roman
12 pt, justified, 0.5" first-line indent, 1.5 spacing.

Every quantitative claim traces to a recorded artefact:
  * results/ablation.json       - 0.956 -> 0.999 AUC, 0.581 -> 0.963 sensitivity,
                                  2504 -> 226 cost (A_counts_only -> B_plus_trailing)
  * results/model_results.json  - XGBoost cost-selected cost 62 at sensitivity 0.9925
  * models/best_model_info.json - Logistic Regression deployed at threshold 0.46,
                                  26 features, lowest cost (200) at the fixed 0.5
  * results/zero_day_eval.json  - mean zero-shot sensitivity 0.8499 over five families
  * results/drift.json          - 0.9606 -> 0.9721 after offline retraining,
                                  0.0 false alarms, 6000 s detection delay
  * results/evasion.json        - sensitivity 0.0 at 15 files/second
  * results/model_results.json + conference/reference_verification_report.md
                                - Aranyi et al. (2026) 91.87% sensitivity, prior work

Usage:  python tools/build_phase1_abstract.py
"""

from __future__ import annotations

from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from build_phase1_report import (
    BODY_PT,
    OUT_DIR,
    PROJECT_TITLE,
    _set_font,
    _spacing,
    new_document,
    para,
)

DOCX_PATH = OUT_DIR / "Phase1_Abstract.docx"

TITLE = PROJECT_TITLE

ABSTRACT_PARAGRAPHS = [
    "Ransomware is a serious cybersecurity threat because it encrypts files "
    "and can cause significant data loss if not detected early. Endpoint-based "
    "detection requires client-side monitoring, which raises privacy concerns, "
    "so server-side file-operation logs offer an alternative by recording "
    "metadata such as file creation, renaming and deletion. Aranyi, Miseta and "
    "Szücs (2026) investigated this approach using fixed one-second windows and "
    "five machine-learning models — Decision Tree, Random Forest, Support "
    "Vector Machine (SVM), AdaBoost and XGBoost — reporting 91.87% sensitivity "
    "for XGBoost in a simulated file-server environment with sub-millisecond "
    "inference. However, three raw operation counts per window and a fixed "
    "decision threshold of 0.5 limit the detection of slow, low-and-slow "
    "encryption and of previously unseen ransomware families, and a single "
    "fixed threshold ignores the asymmetry between a missed attack and a false "
    "alarm.",

    "This project proposes the Adaptive Server-Side Ransomware Detection "
    "(ASSRD) framework to address these limitations. It replaces the three "
    "per-second counts with twenty-six trailing temporal features — six "
    "aggregate families (sum, mean, maximum, active fraction, mean absolute "
    "first difference and coefficient of variation) over 3-, 5-, 10- and "
    "30-second windows, together with an active-run-length and a "
    "short-term-to-baseline burst ratio — so that both burst and slow "
    "encryption are represented in the decision. Six cost-sensitive classifiers "
    "are evaluated: the five paper-inspired models named above, which "
    "constitute the core comparison, together with a standardised Logistic "
    "Regression added as a linear control, all tuned under stratified "
    "cross-validation scored on recall. Rather than fixing the operating "
    "threshold at 0.5, it is selected on the validation split by minimising a "
    "cost function in which a missed attack is charged ten times a false alarm "
    "(C_FN = 10, C_FP = 1), with the fixed-threshold result reported alongside "
    "it for comparison. Detections are attributed to operator-level feature "
    "groups using decision-path and permutation importance, and distribution "
    "drift is monitored with a Population Stability Index supporting an offline "
    "retraining and recovery procedure.",

    "In a controlled ablation, replacing the three per-second counts with "
    "multi-scale trailing aggregates raises the area under the receiver "
    "operating characteristic curve from 0.956 to 0.999, lifts sensitivity at "
    "the 0.5 threshold from 0.581 to 0.963, and reduces total cost from 2504 "
    "to 226; cost-selected thresholds lower the cost further, with XGBoost "
    "reaching a total cost of 62 at a sensitivity of 0.9925, while Logistic "
    "Regression is deployed at a selected threshold of 0.46 on the lowest "
    "cost at the fixed 0.5 threshold. Leave-one-family-out testing "
    "over five families gives a mean zero-shot sensitivity of 0.8499 on a "
    "wholly unseen family, and drift monitoring followed by offline retraining "
    "raises sensitivity from 0.9606 to 0.9721 while incurring no false drift "
    "alarms, although the drift signal itself lagged the induced change by "
    "approximately 6000 seconds. The detector is nonetheless defeated by "
    "rate-limited encryption below roughly fifteen files per second, where "
    "per-window counts become indistinguishable from benign activity; this is "
    "reported as the principal limitation, and because the evaluation uses "
    "behaviour emulated from published family characteristics rather than "
    "executed malware, performance in real-world environments remains to be "
    "established. Planned extensions, not yet implemented or evaluated in this "
    "study, are BiLSTM-based sequence modelling over the operation-count time "
    "series and a threshold-stability study across resampled splits.",
]

KEYWORDS = (
    "Keywords: Ransomware detection; Server-side file-operation logs; Machine "
    "learning; Temporal features; Multi-scale windowing; Cost-sensitive "
    "learning; Logistic Regression; Explainability; Concept drift."
)


# --------------------------------------------------------------------------
def title_block(doc):
    """Project title, mirroring the report's title page without the metadata."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, after=6)
    _set_font(p.add_run(TITLE), 14.0, bold=True)


def abstract_page(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, before=6, after=12)
    _set_font(p.add_run("ABSTRACT"), 14.0, bold=True)

    for text in ABSTRACT_PARAGRAPHS:
        para(doc, text)

    para(doc, "")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Pt(36)
    _spacing(p)
    _set_font(p.add_run(KEYWORDS), BODY_PT, italic=True)


def build():
    doc = new_document()
    title_block(doc)
    abstract_page(doc)
    DOCX_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX_PATH))
    print(f"written: {DOCX_PATH}")


if __name__ == "__main__":
    build()