#!/usr/bin/env python3
"""Render original paper Figure 3 without changes. Requires pypdfium2."""

from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    document = pdfium.PdfDocument(str(ROOT / "assets/paper-method-source.pdf"))
    image = document[0].render(scale=3).to_pil()
    image.save(ROOT / "assets/figure3.png")
    print("Rendered original Figure 3 without changes")
