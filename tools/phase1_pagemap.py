"""Report the printed page number of every heading in the Phase I PDF.

Usage:
    python tools/phase1_pagemap.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "project phase1" / "AI-Based_Ransomware_Detection_PhaseI_Report_v2.pdf"

PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)\.?\s+[A-Z][A-Za-z].{0,70}$|^[A-Z][A-Z \-]{6,}$"
)


def main():
    doc = pymupdf.open(str(PDF))
    print("pdf pages:", doc.page_count)
    for i, page in enumerate(doc):
        for line in page.get_text().splitlines():
            line = line.strip()
            if not line or PATTERN.match(line):
                print(f"{i + 1:>3} | {line[:90]}")


if __name__ == "__main__":
    main()