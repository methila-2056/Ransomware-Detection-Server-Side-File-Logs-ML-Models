"""Rewrite Chapter 2 (Literature Survey) of the Phase I report in the
LITERATURE SURVEY OVERVIEW format and export DOCX + PDF.

The script operates on a copy of the delivered report so that the front matter,
chapters, figures, tables and references are preserved byte for byte. Only the
body of Section 2 is replaced; Table 2.1 and Table 2.2 are carried across.

Usage:
    python tools/phase1_litsurvey.py
"""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import docx
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

import phase1_litsurvey_content as C

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "project phase1"
TARGET = OUT_DIR / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.docx"
PRISTINE = Path(os.environ.get("TEMP", ".")) / "phase1_report_pristine.docx"


def pristine_source() -> Path:
    if not PRISTINE.exists():
        shutil.copy2(TARGET, PRISTINE)
    return PRISTINE


def ptext(el):
    return Paragraph(el, el.getparent()).text.strip()


def find_paragraph(body, text):
    for i, el in enumerate(body):
        if el.tag != qn("w:p"):
            continue
        if ptext(el) == text:
            return i
    raise LookupError(text)


def find_table_after(body, caption):
    idx = find_paragraph(body, caption)
    for el in body[idx + 1:]:
        if el.tag == qn("w:tbl"):
            return el
    raise LookupError(caption)


def replace_text(template_p, text):
    new_p = copy.deepcopy(template_p)
    runs = new_p.findall(qn("w:r"))
    for extra in runs[1:]:
        new_p.remove(extra)
    first = new_p.find(qn("w:r"))
    for t in first.findall(qn("w:t")):
        first.remove(t)
    node = first.makeelement(qn("w:t"), {})
    node.text = text
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    first.append(node)
    return new_p


def set_cell(tc, text, width, bold=False, size=16, align="left"):
    tcw = tc.find(qn("w:tcPr")).find(qn("w:tcW"))
    tcw.set(qn("w:w"), str(width))
    para = tc.find(qn("w:p"))
    for extra in para.findall(qn("w:p"))[1:]:
        tc.remove(extra)
    ppr = para.find(qn("w:pPr"))
    if ppr is None:
        ppr = para.makeelement(qn("w:pPr"), {})
        para.insert(0, ppr)
    jc = ppr.find(qn("w:jc"))
    if jc is None:
        jc = ppr.makeelement(qn("w:jc"), {})
        ppr.append(jc)
    jc.set(qn("w:val"), align)
    runs = para.findall(qn("w:r"))
    for extra in runs[1:]:
        para.remove(extra)
    run = runs[0]
    for node in list(run):
        if node.tag in (qn("w:t"), qn("w:tab")):
            run.remove(node)
    rpr = run.find(qn("w:rPr"))
    if rpr is None:
        rpr = run.makeelement(qn("w:rPr"), {})
        run.insert(0, rpr)
    for tag, value in ((qn("w:b"), "1" if bold else "0"), (qn("w:sz"), str(size))):
        node = rpr.find(tag)
        if node is None:
            node = rpr.makeelement(tag, {})
            rpr.append(node)
        node.set(qn("w:val"), value)
    node = run.makeelement(qn("w:t"), {})
    node.text = text
    node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.append(node)


def build_survey_table(template_tbl, headers, rows, widths):
    tbl = copy.deepcopy(template_tbl)
    template_rows = copy.deepcopy(template_tbl.findall(qn("w:tr")))
    for tr in tbl.findall(qn("w:tr")):
        tbl.remove(tr)
    grid = tbl.find(qn("w:tblGrid"))
    for gc in list(grid):
        grid.remove(gc)
    for width in widths:
        grid.append(grid.makeelement(qn("w:gridCol"), {qn("w:w"): str(width)}))

    centred = {0, 3}

    def emit(values, template_tr, bold):
        tr = copy.deepcopy(template_tr)
        cells = tr.findall(qn("w:tc"))
        while len(cells) < len(values):
            tr.append(copy.deepcopy(cells[-1]))
            cells = tr.findall(qn("w:tc"))
        for index, (tc, value, width) in enumerate(zip(cells, values, widths)):
            align = "center" if bold or index in centred else "left"
            set_cell(tc, value, width, bold=bold, align=align)
        tbl.append(tr)

    emit(headers, template_rows[0], True)
    for tr in tbl.findall(qn("w:tr"))[:1]:
        trpr = tr.find(qn("w:trPr"))
        if trpr is None:
            trpr = tr.makeelement(qn("w:trPr"), {})
            tr.insert(0, trpr)
        trpr.append(trpr.makeelement(qn("w:tblHeader"), {}))
    for row in rows:
        emit(row, template_rows[1], False)
    return tbl


def main():
    shutil.copy2(pristine_source(), TARGET)

    doc = docx.Document(str(TARGET))
    body = doc.element.body

    start = find_paragraph(body, "2. LITERATURE SURVEY")
    end = find_paragraph(body, "3. SYSTEM STUDY")
    anchor = body[end]

    tbl_comparison = find_table_after(body, "Table 2.1: Comparison with related work")
    tbl_summary = find_table_after(body, "Table 2.2: Summary of the surveyed literature")
    cap_comparison = copy.deepcopy(
        [el for el in body[start:end] if el.tag == qn("w:p")
         and ptext(el).startswith("Table 2.1:")][0])
    cap_summary = copy.deepcopy(
        [el for el in body[start:end] if el.tag == qn("w:p")
         and ptext(el).startswith("Table 2.2:")][0])
    cap_summary = replace_text(cap_summary, "Table 2.3: Summary of the surveyed literature")

    heading_template = copy.deepcopy(body[start])
    old = list(body[start + 1:end])
    body_template = None
    for el in old:
        if el.tag == qn("w:p"):
            if ptext(el).startswith("Scaife et al. propose CryptoLock"):
                body_template = copy.deepcopy(el)
                break
    if body_template is None:
        raise LookupError("body paragraph template not found")

    for el in old:
        body.remove(el)

    blocks = [
        (C.OVERVIEW_HEADING, C.OVERVIEW),
        (C.PAPER1_HEADING, C.PAPER1),
        (C.PAPER2_HEADING, C.PAPER2),
        (C.PAPER3_HEADING, C.PAPER3),
        (C.PAPER4_HEADING, C.PAPER4),
        (C.PAPER5_HEADING, C.PAPER5),
        (C.PAPER6_HEADING, C.PAPER6),
        (C.COMPARATIVE_HEADING, C.COMPARATIVE_INTRO),
        (C.SURVEY_TABLE_HEADING, C.SURVEY_TABLE_INTRO),
        (C.RESEARCH_GAP_HEADING, C.RESEARCH_GAP),
        (C.IDEAS_HEADING, C.IDEAS),
        (C.PROBLEM_HEADING, C.PROBLEM),
    ]

    for heading, paragraphs in blocks:
        anchor.addprevious(replace_text(heading_template, heading))
        for para in paragraphs:
            anchor.addprevious(replace_text(body_template, para))
        if heading == C.COMPARATIVE_HEADING:
            anchor.addprevious(cap_comparison)
            anchor.addprevious(copy.deepcopy(tbl_comparison))
        if heading == C.SURVEY_TABLE_HEADING:
            cap = replace_text(cap_comparison, C.SURVEY_TABLE_CAPTION)
            anchor.addprevious(cap)
            anchor.addprevious(build_survey_table(
                tbl_comparison, C.SURVEY_TABLE_HEADERS, C.SURVEY_TABLE_ROWS,
                (520, 2100, 1250, 560, 1900, 3590)))
        if heading == C.IDEAS_HEADING:
            anchor.addprevious(cap_summary)
            anchor.addprevious(copy.deepcopy(tbl_summary))

    doc.save(str(TARGET))
    print("rewritten:", TARGET)


if __name__ == "__main__":
    main()