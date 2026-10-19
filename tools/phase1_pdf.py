"""Export the Phase I report to PDF with Word and report the page count.

Usage:
    python tools/phase1_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import win32com.client as win32

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "project phase1"
DOCX = OUT_DIR / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.docx"
PDF = OUT_DIR / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.pdf"
WD_PAGES = 2


def main() -> int:
    word = win32.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(DOCX), ReadOnly=False)
        pages = doc.ComputeStatistics(WD_PAGES)
        doc.SaveAs(str(PDF), FileFormat=17)
        doc.Close(SaveChanges=0)
    finally:
        word.Quit()
    print(f"pages={pages}")
    return pages


if __name__ == "__main__":
    main()