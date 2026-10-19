"""Rebuild the Table of Contents, List of Tables and List of Figures page numbers
of the Phase I report from the rendered PDF.

The front matter of the report is static text rather than Word fields, so it is
refreshed explicitly whenever the body pagination changes.

Usage:
    python tools/phase1_frontmatter.py
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pymupdf
from docx import Document
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph

import phase1_litsurvey_content as C

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "project phase1"
DOCX = OUT_DIR / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.docx"
PDF = OUT_DIR / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.pdf"

SECTION2 = [
    C.OVERVIEW_HEADING,
    C.PAPER1_HEADING,
    C.PAPER2_HEADING,
    C.PAPER3_HEADING,
    C.PAPER4_HEADING,
    C.PAPER5_HEADING,
    C.PAPER6_HEADING,
    C.COMPARATIVE_HEADING,
    C.SURVEY_TABLE_HEADING,
    C.RESEARCH_GAP_HEADING,
    C.IDEAS_HEADING,
    C.PROBLEM_HEADING,
]


def ptext(el):
    return Paragraph(el, el.getparent()).text


def normalise(s):
    return re.sub(r"\s+", " ", s).strip().lower().replace("\u2013", "-").replace("\u2014", "-")


def page_index():
    doc = pymupdf.open(str(PDF))
    pages = [normalise(page.get_text()) for page in doc]
    body_start = None
    for i, text in enumerate(pages):
        if "1. introduction" in text and "1.1 overview of ransomware" in text:
            body_start = i
    if body_start is None:
        raise RuntimeError("could not locate body start")
    return pages, body_start


def locate(pages, body_start, label, is_caption):
    needle = normalise(label if is_caption else label)
    for candidate in (needle, *prefixes(needle)):
        for i in range(body_start, len(pages)):
            if candidate in pages[i]:
                return i + 1 - body_start
    return None


def prefixes(needle):
    """Shorter probes used when a long heading is split across text blocks."""
    words = needle.split()
    return [" ".join(words[:n]) for n in range(min(9, len(words) - 1), 2, -1)]


def set_paragraph_text(el, text):
    runs = el.findall(qn("w:r"))
    for extra in runs[1:]:
        el.remove(extra)
    first = runs[0]
    for t in first.findall(qn("w:t")):
        first.remove(t)
    for br in first.findall(qn("w:tab")):
        first.remove(br)
    parts = text.split("\t")
    first.append(first.makeelement(qn("w:t"), {}))
    first[-1].text = parts[0]
    first[-1].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for extra in parts[1:]:
        el.append(el.makeelement(qn("w:r"), {}))
        el[-1].append(el[-1].makeelement(qn("w:tab"), {}))
        el[-1].append(el[-1].makeelement(qn("w:t"), {}))
        el[-1][-1].text = extra
        el[-1][-1].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def main():
    pages, body_start = page_index()
    doc = Document(str(DOCX))
    body = doc.element.body
    children = list(body)

    def index_of(text):
        for i, el in enumerate(children):
            if el.tag == qn("w:p") and ptext(el).strip() == text:
                return i
        raise LookupError(text)

    lot = index_of("LIST OF TABLES")
    lof = index_of("LIST OF FIGURES")
    abbr = index_of("LIST OF ABBREVIATIONS")
    toc = index_of("TABLE OF CONTENTS")

    # ---- List of Tables and List of Figures ----
    for start, stop, prefix in ((lot + 1, lof, "Table"), (lof + 1, abbr, "Figure")):
        for el in children[start:stop]:
            if el.tag != qn("w:tbl"):
                continue
            rows = el.findall(qn("w:tr"))
            for tr in rows:
                cells = tr.findall(qn("w:tc"))
                if len(cells) != 3:
                    continue
                texts = ["".join(t.text or "" for t in tc.iter(qn("w:t"))).strip()
                         for tc in cells]
                if not re.fullmatch(r"\d+\.\d+", texts[0]):
                    continue
                if prefix == "Table" and texts[1].startswith("Summary of the surveyed"):
                    texts[0] = "2.3"
                    set_paragraph_text(cells[0].find(qn("w:p")), "2.3")
                page = locate(pages, body_start,
                              f"{prefix} {texts[0]}: {texts[1]}", True)
                if page is None:
                    print("  ! not found:", prefix, texts[0], texts[1])
                    continue
                for p in cells[2].findall(qn("w:p")):
                    set_paragraph_text(p, str(page))
            if prefix != "Table":
                continue
            numbers = ["".join(t.text or "" for t in tr.findall(qn("w:tc"))[0].iter(qn("w:t"))).strip()
                       for tr in rows[1:] if len(tr.findall(qn("w:tc"))) == 3]
            if "2.2" in numbers:
                continue
            anchor = next(tr for tr in rows
                          if "".join(t.text or "" for t in tr.findall(qn("w:tc"))[0].iter(qn("w:t"))).strip()
                          == "2.1")
            new_row = copy.deepcopy(anchor)
            cells = new_row.findall(qn("w:tc"))
            set_paragraph_text(cells[0].find(qn("w:p")), "2.2")
            set_paragraph_text(cells[1].find(qn("w:p")), "Literature survey table")
            page = locate(pages, body_start, "Table 2.2: Literature survey table", True)
            for p in cells[2].findall(qn("w:p")):
                set_paragraph_text(p, str(page if page else ""))
            anchor.addnext(new_row)

    # ---- Table of Contents ----
    entries = []
    for el in children[toc + 1:index_of("1. INTRODUCTION")]:
        if el.tag != qn("w:p"):
            continue
        text = ptext(el).strip()
        if not text or text.startswith("CHAPTER NO."):
            continue
        entries.append(el)

    labels = [re.sub(r"\t(?:[ivx]+|\d+)$", "", ptext(el).strip()).strip() for el in entries]
    roman_entries = {
        "ABSTRACT", "LIST OF TABLES", "LIST OF FIGURES", "LIST OF ABBREVIATIONS"
    }
    roman_pages = {label: ptext(el).strip() for label, el in zip(labels, entries)
                   if label in roman_entries}

    templates = {}
    for label, el in zip(labels, entries):
        if label in roman_entries:
            templates.setdefault("front", el)
        elif re.fullmatch(r"\d+\.\s.+", label):
            templates.setdefault("chapter", el)
        elif re.fullmatch(r"\d+\.\d+\s.+", label):
            templates.setdefault("sub", el)
    for key in ("front", "chapter", "sub"):
        if key not in templates:
            raise RuntimeError(f"missing TOC template: {key}")

    new_labels = []
    for label in labels:
        if label == "2. LITERATURE SURVEY":
            new_labels.append(label)
            new_labels.extend(SECTION2)
            continue
        if re.fullmatch(r"2\.\d+.*", label):
            continue
        new_labels.append(label)

    parent = entries[0].getparent()
    anchor = entries[-1].getnext()
    for el in entries:
        parent.remove(el)

    def style_for(label):
        if label in roman_entries:
            return "front"
        if re.fullmatch(r"\d+\.\s.+", label):
            return "chapter"
        return "sub"

    for label in new_labels:
        el = copy.deepcopy(templates[style_for(label)])
        if label in roman_entries:
            set_paragraph_text(el, roman_pages[label])
            anchor.addprevious(el)
            continue
        page = locate(pages, body_start, label, False)
        if page is None:
            print("  ! toc entry not found:", label)
            page = ""
        set_paragraph_text(el, f"{label}\t{page}")
        anchor.addprevious(el)

    doc.save(str(DOCX))
    print("front matter refreshed:", DOCX)


if __name__ == "__main__":
    main()