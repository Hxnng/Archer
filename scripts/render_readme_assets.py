#!/usr/bin/env python3
"""Generate transparent Archer branding and flat navigation badges."""

from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
THEMES = {
    "light": {"ink": "#17212d", "muted": "#596574", "orange": "#c85a2c"},
    "dark": {"ink": "#edf2f7", "muted": "#a5b2c2", "orange": "#ff9a6b"},
}


def text(x, y, value, size=16, fill="ink", weight=400, extra=""):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{{{fill}}}" font-weight="{weight}" {extra}>{escape(str(value))}</text>'


def mark(x=0, y=0, scale=1):
    return f'''<g transform="translate({x} {y}) scale({scale})" fill="none" stroke-linecap="round" stroke-linejoin="round">
      <path d="M31 20 Q-6 64 31 108" stroke="{{orange}}" stroke-width="9"/>
      <path d="M31 20 L77 64 L31 108" stroke="{{ink}}" stroke-width="5"/>
      <path d="M15 64 H105 M89 49 L105 64 L89 79" stroke="{{orange}}" stroke-width="7"/>
      <path d="M15 64 L5 54 M15 64 L5 74" stroke="{{orange}}" stroke-width="5"/>
    </g>'''


def svg(name, width, height, title, body, theme):
    body = body.format(**THEMES[theme])
    content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title">
<title id="title">{escape(title)}</title>
<g font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif">{body}</g></svg>\n'''
    (ASSETS / f"{name}-{theme}.svg").write_text(content)


def render(theme):
    body = mark(250, 8, 1.24)
    body += text(416, 128, "Archer", 96, weight=700, extra='letter-spacing="-5"')
    body += text(500, 189, "Efficient diffusion. Every token stays revisable.", 23, "muted", extra='text-anchor="middle"')
    svg("readme-hero", 1000, 220, "Archer — Efficient diffusion. Every token stays revisable.", body, theme)
    svg("logo", 128, 128, "Archer logo", mark(9, 0), theme)
    social = mark(68, 52, 1.25)
    social += text(225, 174, "Archer", 96, weight=700, extra='letter-spacing="-5"')
    social += text(74, 270, "Efficient diffusion.", 45, weight=650)
    social += text(74, 326, "Every token stays revisable.", 45, weight=650)
    social += text(76, 388, "Training-free KV caching for rollback-capable diffusion language models", 24, "muted")
    social += text(76, 490, "Up to 4.33× speedup", 43, "orange", 700)
    social += text(76, 535, "Across code generation and reasoning benchmarks", 24, "muted")
    social += text(76, 606, "github.com/Hxnng/Archer", 19, "muted")
    social += text(1204, 606, "Speedup vs. LLaDA-Confidence", 17, "muted", extra='text-anchor="end"')
    svg("social-preview", 1280, 640, "Archer project preview", social, theme)


def badges():
    directory = ASSETS / "badges"
    directory.mkdir(exist_ok=True)
    entries = [
        ("paper", "arXiv", "2608.08086", 46, 90, "#b31b1b"),
        ("results", "Results", "8 benchmarks", 58, 100, "#c85a2c"),
        ("quick-start", "Quick start", "Run", 78, 40, "#2563eb"),
        ("docs", "Docs", "Reproduce", 46, 86, "#237b69"),
    ]
    for name, label, value, left, right, color in entries:
        width = left + right
        title = escape(f"{label}: {value}")
        content = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="24" viewBox="0 0 {width} 24" role="img" aria-labelledby="title">
<title id="title">{title}</title>
<path fill="#111827" d="M0 0h{left}v24H0z"/>
<path fill="{color}" d="M{left} 0h{right}v24H{left}z"/>
<g fill="#fff" text-anchor="middle" font-family="Verdana,Arial,sans-serif" font-size="11.5">
<text x="{left / 2}" y="16.1">{escape(label)}</text>
<text x="{left + right / 2}" y="16.1">{escape(value)}</text>
</g></svg>\n'''
        (directory / f"{name}.svg").write_text(content)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    for theme in THEMES:
        render(theme)
    badges()
    print("Generated 6 transparent branding SVGs and 4 navigation badges")
