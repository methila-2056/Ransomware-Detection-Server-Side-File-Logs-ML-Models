"""Rewrite Chapter 8 (References) of the Phase I report in the author-year style
used by the LITERATURE SURVEY OVERVIEW format.

The existing numbered IEEE entries are removed and replaced with an alphabetised,
unnumbered author-year list that matches the in-text citations of the literature
survey. Every paragraph after the "8. REFERENCES" heading is discarded first, so
the step is idempotent.

Usage:
    python tools/phase1_references.py
"""

from __future__ import annotations

import copy
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

from phase1_references_content import REFERENCES

ROOT = Path(__file__).resolve().parents[1]
DOCX = (ROOT / "project phase1"
        / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.docx")
HEADING = "8. REFERENCES"
XML_SPACE = "{http://www.w3.org/XML/1998/namespace}space"


def ptext(el):
    return Paragraph(el, el.getparent()).text.strip()


def text_run(template, text, italic):
    run = copy.deepcopy(template)
    for node in list(run):
        if node.tag != qn("w:rPr"):
            run.remove(node)
    rpr = run.find(qn("w:rPr"))
    node = rpr.find(qn("w:i"))
    if node is None:
        node = rpr.makeelement(qn("w:i"), {})
        rpr.append(node)
    node.set(qn("w:val"), "1" if italic else "0")
    node = run.makeelement(qn("w:t"), {})
    node.text = text
    node.set(XML_SPACE, "preserve")
    run.append(node)
    return run


def reference_paragraph(template, entry):
    para = copy.deepcopy(template)
    for run in para.findall(qn("w:r")):
        para.remove(run)
    template_run = copy.deepcopy(template.findall(qn("w:r"))[0])
    for index, chunk in enumerate(entry.split("*")):
        if not chunk:
            continue
        para.append(text_run(template_run, chunk, index % 2 == 1))
    return para


def main():
    doc = Document(str(DOCX))
    body = doc.element.body
    children = list(body)

    start = next(i for i, el in enumerate(children)
                 if el.tag == qn("w:p") and ptext(el) == HEADING)
    template = copy.deepcopy(children[start + 1])

    for el in children[start + 1:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)

    anchor = next(el for el in body if el.tag == qn("w:sectPr"))
    for entry in REFERENCES:
        anchor.addprevious(reference_paragraph(template, entry))

    doc.save(str(DOCX))
    print(f"references rewritten: {len(REFERENCES)} entries")


if __name__ == "__main__":
    main()