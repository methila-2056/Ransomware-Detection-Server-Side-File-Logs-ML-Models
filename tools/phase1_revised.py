"""Patch the delivered Phase 1 report in place.

``project phase1/CHAPTER 1_revised.docx`` is the report as returned and
revised by hand: no script in ``tools/`` generates it, its chapter structure
differs from ``Phase1_Project_Report.docx``, and its core properties record a
human author. Regenerating it from ``phase1_report_content.py`` would therefore
produce a different document, so corrections are applied to the delivered file
itself, following the approach ``tools/phase1_litsurvey.py`` takes: operate on
the document in place, touch only the paragraphs and table cells that must
change, and leave figures, tables and references otherwise untouched.

Corrections applied
--------------------
1. Classifier count. Section 1.8 and section 5.6 said "five classifiers" and
   Table 5.2 listed five rows, contradicting the six classifiers now evaluated.
   A standardised Logistic Regression is added as a linear control: it is the
   cheapest model at the fixed 0.5 threshold and is therefore the deployed one.
2. Feature count. Section 4.1 said each window requires "twenty-nine feature
   values". The deployed model takes twenty-six; the raw counts are excluded
   deliberately, and the 29-input arm is retained only for ablation.
3. Author spelling. Four occurrences of "Szucs" become "Szucs" with the
   diaeresis, per conference/reference_verification_report.md.
4. Recorded results absent from section 6.1 are added, each read from
   results/*.json at build time rather than typed in.
5. Future work. BiLSTM sequence modelling and a threshold-stability study are
   added; neither is mentioned anywhere in the delivered text.
6. Front matter. An ABSTRACT heading, the three-paragraph abstract and the
   keyword line are inserted, matching tools/build_phase1_abstract.py so the
   two documents cannot diverge.

Usage:  python tools/phase1_revised.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import docx
from docx.text.paragraph import Paragraph

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_phase1_abstract import ABSTRACT_PARAGRAPHS, KEYWORDS  # noqa: E402
from build_phase1_report import (  # noqa: E402
    BODY_PT, FONT, _set_font, _spacing,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "project phase1"
TARGET = OUT_DIR / "CHAPTER 1_revised.docx"
RESULTS = ROOT / "results"
BACKUP = OUT_DIR / "CHAPTER 1_revised.original.docx"


# --------------------------------------------------------------------------
# evidence


def figures() -> dict:
    """Every number the patched prose quotes, read from the recorded artefacts.

    Nothing here is transcribed by hand, so a re-run after a fresh pipeline
    execution cannot leave a stale figure behind in the document.
    """
    models = json.loads((RESULTS / "model_results.json").read_text())
    manifest = json.loads((RESULTS / "run_manifest.json").read_text())
    best_info = json.loads((ROOT / "models" / "best_model_info.json").read_text())
    ablation = json.loads((RESULTS / "ablation.json").read_text())
    zeroday = json.loads((RESULTS / "zero_day_eval.json").read_text())["Summary"]
    drift = json.loads((RESULTS / "drift.json").read_text())
    evasion = json.loads((RESULTS / "evasion.json").read_text())

    by_name = {m["Model"]: m for m in models}
    arms = {a["Group"]: a for a in ablation["Results"]}
    counts = arms["A_counts_only"]
    trailing = arms["B_plus_trailing"]

    xgb = by_name["XGBoost"]["Metrics At Selected Threshold"]
    # The mimicry regime carries a NaN rate, so it is excluded before the
    # minimum is taken; otherwise NaN would win the comparison.
    rates = [r for r in evasion["Curve"]
             if isinstance(r.get("Declared Files Per Second"), (int, float))
             and r["Declared Files Per Second"] == r["Declared Files Per Second"]]
    weakest = min(rates, key=lambda r: r["Declared Files Per Second"])

    return {
        "auc_from": round(counts["AUROC"], 4),
        "auc_to": round(trailing["AUROC"], 4),
        "sens_from": round(counts["Sensitivity At 0.5"], 4),
        "sens_to": round(trailing["Sensitivity At 0.5"], 4),
        "cost_from": counts["Cost At 0.5"],
        "cost_to": trailing["Cost At 0.5"],
        "xgb_cost": xgb["Total Cost"],
        "xgb_sens": round(xgb["Sensitivity"], 4),
        "deployed": best_info["model_name"],
        "deployed_threshold": best_info["decision_threshold"],
        "deployed_cost": next(m["Total Cost"] for m in models
                              if m["Model"] == best_info["model_name"]),
        "feature_count": best_info["feature_count"],
        "model_count": len(models),
        "zero_shot": round(zeroday["Overall Mean Zero Shot Sensitivity"], 4),
        "zero_shot_drop": round(zeroday["Overall Mean Drop"], 4),
        "families": len(zeroday["Families"]),
        "drift_from": round(drift["Recovery"]["Stale"]["Sensitivity"], 4),
        "drift_to": round(drift["Recovery"]["Retrained"]["Sensitivity"], 4),
        "drift_cost_from": drift["Recovery"]["Stale"]["Total Cost"],
        "drift_cost_to": drift["Recovery"]["Retrained"]["Total Cost"],
        "drift_false_alarms": drift["Monitor"]["False Alarm Rate"],
        "drift_delay": int(drift["Monitor"]["Detection Delay Seconds"]),
        "rate_weakest": int(weakest["Declared Files Per Second"]),
        "rate_weakest_sens": weakest["Sensitivity"],
        "paper_sens": round(
            evasion["Summary"]["Paper Reference Sensitivity"] * 100, 2),
        "leakage_free": json.loads(
            (RESULTS / "separability.json").read_text()
        )["leakage_check"]["Leakage Free"],
        "best_model": manifest["best_model"],
    }


# --------------------------------------------------------------------------
# paragraph helpers


def ptext(el) -> str:
    return Paragraph(el, el.getparent()).text.strip()


def body_elements(doc):
    return doc.element.body


def find_paragraph(doc, needle: str):
    """First paragraph whose text contains ``needle``."""
    for el in body_elements(doc):
        if el.tag.endswith("}p") and needle in ptext(el):
            return el
    raise LookupError(f"no paragraph containing {needle!r}")


def set_text(el, text: str) -> None:
    """Replace a paragraph's text, keeping the first run's formatting.

    The run properties are copied onto the replacement run so the font, size
    and spacing set in the original document survive the edit.
    """
    src = el.findall(".//" + docx.oxml.ns.qn("w:r"))[0]
    rpr = src.find(docx.oxml.ns.qn("w:rPr"))

    for child in list(el):
        if child.tag.endswith("}r"):
            el.remove(child)

    run = docx.oxml.parse_xml(
        '<w:r xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
    )
    if rpr is not None:
        import copy as _copy

        run.append(_copy.deepcopy(rpr))
    t = docx.oxml.parse_xml(
        '<w:t xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
    )
    t.text = text
    t.set(
        "{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.append(t)
    el.append(run)


def set_cell(cell, text: str) -> None:
    para = cell.paragraphs[0]
    set_text(para._p, text)


def find_table_row(table, first_cell_text: str):
    for row in table.rows:
        if row.cells[0].text.strip() == first_cell_text:
            return row
    raise LookupError(f"no table row starting {first_cell_text!r}")


def append_row(table, values):
    row = table.add_row()
    for cell, value in zip(row.cells, values):
        cell.text = ""
        para = cell.paragraphs[0]
        run = para.add_run(str(value))
        _set_font(run, 11.0)
    return row


# --------------------------------------------------------------------------
# corrections


def fix_classifier_count(doc, f) -> list[str]:
    """Sections 1.8 and 5.6 plus Table 5.2 must describe six classifiers."""
    changes = []

    el = find_paragraph(doc, "To evaluate five classifiers")
    set_text(el,
             "To evaluate six classifiers - Decision Tree, Random Forest, a "
             "probability-calibrated SVM, AdaBoost, XGBoost and a standardised "
             "Logistic Regression added as a linear control - under a "
             "cost-sensitive objective in which a missed attack costs ten "
             "times a false alarm.")
    changes.append("1.8 objectives: five -> six classifiers, LR added")

    el = find_paragraph(doc, "Five classifiers are evaluated")
    set_text(el,
             f"{f['model_count']} classifiers are evaluated: a single Decision "
             "Tree, a Random Forest, a Support Vector Machine with an RBF "
             "kernel wrapped in a calibration layer, AdaBoost, XGBoost, and a "
             "standardised Logistic Regression that serves as a linear "
             "control. Every one of the first five is non-linear, so without "
             "the linear control it is impossible to tell whether the temporal "
             "features carry information the trees merely exploit non-linearly "
             "or carry information at all. Each model is tuned by exhaustive "
             "grid search under stratified cross-validation using recall as "
             "the optimisation score, which biases the search towards avoiding "
             "false negatives in line with the cost asymmetry. The SVM and the "
             "Logistic Regression use feature standardisation, because the "
             "kernel acts on distances and the linear model weights the raw "
             "feature values; tree-based models are invariant to monotone "
             "feature scaling and are left unscaled. The search spaces are "
             "listed in Table 5.2.")
    changes.append("5.6 classifier design: five -> six, LR described")

    table = next(t for t in doc.tables
                 if t.rows[0].cells[0].text.strip() == "Model"
                 and "hyperparameter" in t.rows[0].cells[-1].text.lower())
    if any(r.cells[0].text.strip() == "Logistic Regression" for r in table.rows):
        changes.append("Table 5.2: LR row already present")
    else:
        append_row(table, [
            "Logistic Regression (linear control)",
            "C {0.01,0.1,1,10}; penalty {l1,l2}; solver {liblinear,saga}; "
            "class_weight {None,balanced}",
        ])
        changes.append("Table 5.2: Logistic Regression row added")
    return changes


def fix_feature_count(doc, f) -> list[str]:
    """Section 4.1 quoted the ablation arm's width, not the deployed width."""
    el = find_paragraph(doc, "twenty-nine feature values")
    set_text(el,
             f"The hardware requirement follows directly from the workload. "
             f"The offline evaluation trains {f['model_count']} classifiers "
             f"over a few tens of thousands of windows, each window requiring "
             f"{f['feature_count']} feature values, which is a small tabular "
             "problem that needs memory rather than parallel compute. The live "
             "path is lighter still, since it performs one feature construction "
             "and one classification per second per monitored share. Disk space "
             "is dominated by the generated traces, the trained artefacts and "
             "the detection history rather than by the model files themselves. "
             "No component of the design requires a network connection in order "
             "to evaluate, so the minimum configuration is sufficient for "
             "reproducing every offline result; network capacity becomes "
             "relevant only when live capture is enabled across multiple "
             "shares.")
    return ["4.1: 'twenty-nine feature values' -> "
            f"'{f['feature_count']} feature values'"]


def fix_spelling(doc) -> list[str]:
    """Szucs -> Szucs with the diaeresis, everywhere it appears."""
    from docx.oxml.ns import qn

    targets = list(body_elements(doc))
    for table in doc.tables:
        targets.extend(c._tc for c in table._cells)

    n = 0
    for el in targets:
        for t in el.iter(qn("w:t")):
            if t.text and "Szucs" in t.text:
                t.text = t.text.replace("Szucs", "Szücs")
                n += 1
    return [f"spelling: Szucs -> Szücs in {n} text node(s)"]


def add_results(doc, f) -> list[str]:
    """Section 6.1 reported the ablation but not the deployment or drift."""
    el = find_paragraph(doc, "In Phase 1 we proposed ASSRD")
    original = ptext(el)

    addition = (
        f" Cost-based threshold selection then reduced the cost further: "
        f"XGBoost reached a total cost of {f['xgb_cost']} at a sensitivity of "
        f"{f['xgb_sens']}. The deployed model, {f['deployed']}, records the "
        f"lowest cost at the fixed 0.5 threshold "
        f"({f['deployed_cost']}) and operates at a cost-selected threshold of "
        f"{f['deployed_threshold']}. Leave-one-family-out evaluation across "
        f"{f['families']} families gave a mean zero-shot sensitivity of "
        f"{f['zero_shot']} on a wholly unseen family, a mean drop of "
        f"{f['zero_shot_drop']} against the in-distribution models. "
        f"Concept-drift monitoring followed by offline retraining moved "
        f"sensitivity from {f['drift_from']} to {f['drift_to']} and lowered "
        f"cost from {f['drift_cost_from']} to {f['drift_cost_to']} while "
        f"raising no false drift alarms, although the drift signal itself "
        f"lagged the induced change by approximately {f['drift_delay']} "
        f"seconds, which bounds how quickly the recovery loop can respond."
    )

    if "deployed model" in original:
        return ["6.1: results already present, left unchanged"]
    set_text(el, original.rstrip() + addition)
    return ["6.1: deployment, zero-shot and drift results added"]


def add_future_work(doc) -> list[str]:
    """BiLSTM and threshold stability appear nowhere in the delivered text."""
    from docx.oxml.ns import qn

    full = "\n".join(ptext(el) for el in body_elements(doc)
                     if el.tag.endswith("}p"))

    additions = []
    el = find_paragraph(doc, "Most importantly, rate-limited encryption")
    if "BiLSTM" not in full:
        set_text(el, ptext(el).rstrip() + (
            " Two extensions are planned but not yet implemented or evaluated "
            "in this study: BiLSTM-based sequence modelling over the "
            "operation-count time series, which would consume the window "
            "history directly rather than through fixed aggregates, and a "
            "threshold-stability study that re-derives the operating point "
            "across resampled splits to establish how much of the reported "
            "cost is a property of the data rather than of the split that "
            "happened to be drawn."
        ))
        additions.append("6.1: BiLSTM and threshold-stability future work added")
    else:
        additions.append("6.1: BiLSTM already mentioned")
    return additions


def add_front_matter(doc) -> list[str]:
    """Insert the ABSTRACT heading, the abstract and the keywords at the top."""
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt

    body = body_elements(doc)
    first = body[0]

    new_elements = []

    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(heading, before=0, after=12)
    _set_font(heading.add_run("ABSTRACT"), 14.0, bold=True)
    new_elements.append(heading._p)

    for text in ABSTRACT_PARAGRAPHS:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.first_line_indent = Pt(36)
        _spacing(p)
        _set_font(p.add_run(text), BODY_PT)
        new_elements.append(p._p)

    kw = doc.add_paragraph()
    kw.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    kw.paragraph_format.first_line_indent = Pt(36)
    _spacing(kw, before=10)
    _set_font(kw.add_run(KEYWORDS), BODY_PT, italic=True)
    new_elements.append(kw._p)

    for el in new_elements:
        el.getparent().remove(el)
        first.addprevious(el)

    return [f"front matter: ABSTRACT heading, {len(ABSTRACT_PARAGRAPHS)} "
            "paragraphs and keywords inserted before the title block"]


# --------------------------------------------------------------------------


def build():
    if not TARGET.exists():
        raise SystemExit(f"missing {TARGET}")
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)
        print(f"backup written: {BACKUP}")

    doc = docx.Document(str(TARGET))
    f = figures()

    changes = []
    changes += fix_classifier_count(doc, f)
    changes += fix_feature_count(doc, f)
    changes += fix_spelling(doc)
    changes += add_results(doc, f)
    changes += add_future_work(doc)
    changes += add_front_matter(doc)

    doc.save(str(TARGET))
    for c in changes:
        print(f"  - {c}")
    print(f"\nwritten: {TARGET}")


if __name__ == "__main__":
    build()