"""Build the Phase 1 project report for the ASSRD ransomware-detection project.

Formatting is reproduced from the reference document ``K_FULL_REPORT.pdf``
(59pp, A4). Measured characteristics of that document:

===========================  ===========================================
Page size                    A4, 595.3 x 841.9 pt (210 x 297 mm)
Margins                      left 90.0 pt (1.25"), right 65.2 pt (0.91"),
                             top 70.7 pt (0.98"), bottom 74.1 pt (1.03")
Body text                    Times New Roman 12 pt, justified,
                             first-line indent 36 pt (0.5"), ~1.5 spacing
Chapter label "CHAPTER N"    Times New Roman Bold 14 pt, centred
Chapter title "N. TITLE"     Times New Roman Bold 14 pt, left
Section "N.M TITLE"          Times New Roman Bold 12 pt, left
Sub-section "N.M.K TITLE"    Times New Roman Bold Italic 12 pt, left
Figure/Table caption         Times New Roman Italic 12 pt, centred
Table body                   Times New Roman 11 pt
Page number                  centred in the footer
Citation style               numeric square brackets, [n]
===========================  ===========================================

Content is this project's own: its abstract, literature review, research gap,
methodology and implementation. No content is taken from the reference
document, which concerns an unrelated financial-fraud system.

Usage:  python tools/build_phase1_report.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "project phase1"
FIG_DIR = OUT_DIR / "figures"
DOCX_PATH = OUT_DIR / "Phase1_Project_Report.docx"

PROJECT_TITLE = ("AI-Based Ransomware Detection and "
                 "Real-Time File Encryption Alert System")
FONT = "Times New Roman"
BODY_PT = 12.0
TABLE_PT = 11.0

# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------


def _set_font(run, size, bold=False, italic=False, color=None):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), FONT)


def _spacing(par, before=0, after=0, line=1.5):
    pf = par.paragraph_format
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line
    # Word multiplies the 12 pt font by 1.15 internally; match the reference.
    pf.line_spacing = line * 1.15


def _field(par, instr):
    run = par.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "begin")
    run._r.append(fld)
    run = par.add_run()
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    run._r.append(instr_el)
    run = par.add_run()
    fld = OxmlElement("w:fldChar")
    fld.set(qn("w:fldCharType"), "end")
    run._r.append(fld)


# --------------------------------------------------------------------------
# document set-up
# --------------------------------------------------------------------------


def new_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Pt(595.3)
    sec.page_height = Pt(841.9)
    sec.left_margin = Pt(90.0)
    sec.right_margin = Pt(65.2)
    sec.top_margin = Pt(70.7)
    sec.bottom_margin = Pt(74.1)
    sec.footer_distance = Pt(42)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(BODY_PT)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    normal.paragraph_format.line_spacing = 1.5 * 1.15

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _field(footer, " PAGE ")
    for run in footer.runs:
        _set_font(run, BODY_PT)
    return doc


# --------------------------------------------------------------------------
# content blocks
# --------------------------------------------------------------------------


def para(doc, text, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
         size=BODY_PT, italic=False, bold=False, after=0):
    p = doc.add_paragraph()
    p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Pt(36)
    _spacing(p, after=after)
    run = p.add_run(text)
    _set_font(run, size, bold=bold, italic=italic)
    return p


def bullet(doc, text, marker="•"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Pt(56)
    p.paragraph_format.first_line_indent = Pt(-18)
    _spacing(p)
    _set_font(p.add_run(marker + "\t"), BODY_PT)
    _set_font(p.add_run(text), BODY_PT)
    return p


def chapter(doc, number, title):
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, before=0, after=6)
    _set_font(p.add_run(f"CHAPTER {number}"), 14.0, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    _spacing(p, before=0, after=14)
    _set_font(p.add_run(f"{number}. {title}"), 14.0, bold=True)


def section(doc, number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = True
    _spacing(p, before=12, after=8)
    _set_font(p.add_run(f"{number} {title}"), BODY_PT, bold=True)


def subsection(doc, number, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.keep_with_next = True
    _spacing(p, before=10, after=6)
    _set_font(p.add_run(f"{number} {title}"), BODY_PT, bold=True, italic=True)


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, before=6, after=8)
    _set_font(p.add_run(text), BODY_PT, italic=True)
    return p


def figure(doc, filename, cap):
    path = FIG_DIR / filename
    if not path.exists():
        caption(doc, cap + "  [figure file not available]")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _spacing(p, before=8, after=2)
    p.add_run().add_picture(str(path), width=Inches(5.9))
    caption(doc, cap)


def _shade(cell, hexcolor):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hexcolor)
    tc_pr.append(shd)


def table(doc, headers, rows, widths=None, caption_text=None, size=TABLE_PT):
    if caption_text:
        caption(doc, caption_text)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _spacing(p, before=2, after=2, line=1.0)
        _set_font(p.add_run(str(head)), size, bold=True)
        _shade(cell, "DCE6F1")
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            _spacing(p, before=2, after=2, line=1.0)
            _set_font(p.add_run(str(value)), size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    return t


def reference(doc, number, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.left_indent = Pt(34)
    p.paragraph_format.first_line_indent = Pt(-34)
    _spacing(p, after=4)
    _set_font(p.add_run(f"[{number}]\t{text}"), BODY_PT)


def placeholder(doc, label):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.left_indent = Pt(36)
    _spacing(p, after=4)
    _set_font(p.add_run(f"[ {label} — TO BE COMPLETED BY THE STUDENT ]"),
              BODY_PT, bold=True, italic=True,
              color=RGBColor(0xB0, 0x00, 0x00))