#!/usr/bin/env python3
"""Render one transparent eight-benchmark overview. Requires matplotlib."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "archer-readme-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("LLaDA-Confidence", "Saber", "Archer")
PALETTES = {
    "light": dict(ink="#17212d", muted="#596574", line="#dce2e9", orange="#c85a2c", blue="#6e8da6"),
    "dark": dict(ink="#edf2f7", muted="#a5b2c2", line="#303b49", orange="#ff9a6b", blue="#86abc9"),
}


def load_data():
    main = json.loads((ROOT / "results/table1_reference.json").read_text())
    methods = {row["method"]: row for row in main["rows"]}
    rows = []
    for key, name in [("mbpp", "MBPP"), ("livecodebench", "LiveCodeBench"), ("humaneval", "HumanEval")]:
        baseline = methods["LLaDA-Confidence"][key + "_time"]
        rows.append({"name": name, "domain": "Code generation", "methods": {
            method: {"quality": methods[method][key], "et": methods[method].get(key + "_et"),
                     "speedup": round(baseline / methods[method][key + "_time"], 2)}
            for method in METHODS}})
    noncode = json.loads((ROOT / "results/noncode_reference.json").read_text())
    for bench in noncode["benchmarks"]:
        rows.append({"name": bench["name"], "domain": bench["domain"], "methods": {
            method: {"quality": bench["methods"][method]["accuracy"], "et": None,
                     "speedup": bench["methods"][method]["speedup"]}
            for method in METHODS}})
    return rows


def render(theme, rows):
    p = PALETTES[theme]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(12, 8.8))
    fig.patch.set_alpha(0)
    fig.text(.025, .955, "Across code and reasoning", size=23, weight="bold", color=p["ink"])
    fig.text(.025, .913, "LLaDA-8B-Instruct · 8 benchmarks · reported results", size=12, color=p["muted"])
    fig.text(.975, .955, "Up to 4.33× faster", ha="right", size=17, weight="bold", color=p["orange"])
    fig.text(.025, .842, "Benchmark", size=12, weight="bold", color=p["ink"])
    fig.text(.38, .875, "Pass@1 / accuracy (%)", ha="center", size=12, weight="bold", color=p["ink"])
    columns = [("LLaDA-Confidence", .265, p["muted"]), ("Saber", .38, p["blue"]), ("Archer", .49, p["orange"])]
    for method, x, color in columns:
        fig.text(x, .842, method, ha="center", color=color, size=11, weight="bold")
    ax = fig.add_axes([.615, .17, .35, .62], facecolor="none")
    ax.set_title("Speedup vs. LLaDA-Confidence", loc="left", size=12, weight="bold", color=p["ink"], pad=29)
    ax.set_ylim(-.5, 7.5)
    ax.set_xlim(0, 5.2)
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4, 5], ["0", "1×", "2×", "3×", "4×", "5×"])
    ax.tick_params(axis="x", colors=p["muted"], labelsize=10, length=0, pad=8)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color(p["line"])
    ax.grid(axis="x", color=p["line"], linewidth=.65)
    ax.set_axisbelow(True)
    ax.axvline(1, color=p["muted"], linewidth=.9, linestyle=(0, (3, 3)), zorder=1)
    for index, bench in enumerate(rows):
        y = 7 - index
        fy = .17 + .62 * (y + .5) / 8
        fig.text(.025, fy + .007, bench["name"], va="center", color=p["ink"], size=13, weight="bold")
        fig.text(.025, fy - .019, bench["domain"], va="center", color=p["muted"], size=9)
        for method, x, color in columns:
            result = bench["methods"][method]
            fig.text(x, fy + .007, f"{result['quality']:.2f}", va="center", ha="center", color=color, size=14, weight="bold" if method == "Archer" else "normal")
            if result["et"] is not None:
                fig.text(x, fy - .019, f"{result['et']:.2f} ET", va="center", ha="center", color=color, size=9)
        for method, offset, color in [("Saber", .14, p["blue"]), ("Archer", -.14, p["orange"])]:
            value = bench["methods"][method]["speedup"]
            ax.barh(y + offset, value, height=.23, color=color, zorder=3)
            ax.text(value + .07, y + offset, f"{value:.2f}×", color=color, va="center", size=10, weight="bold" if method == "Archer" else "normal")
    fig.add_artist(plt.Line2D([.025, .975], [.81, .81], color=p["line"], linewidth=.8, transform=fig.transFigure))
    group_y = .17 + .62 * 5 / 8
    fig.add_artist(plt.Line2D([.025, .56], [group_y, group_y], color=p["line"], linewidth=.8, transform=fig.transFigure))
    ax.axhline(4.5, color=p["line"], linewidth=.8)
    fig.text(.025, .098, "Code: Pass@1; other tasks: accuracy. ET scores add tests to the same tasks and share their decoding time.", color=p["muted"], size=10)
    fig.text(.025, .067, "MMLU-Pro pools Engineering + Physics (2,268 questions). FinQA reports numerical-answer accuracy.", color=p["muted"], size=10)
    fig.text(.025, .036, "Speedup uses the same LLaDA-Confidence baseline throughout. Full comparisons and protocols: docs/RESULTS.md.", color=p["muted"], size=10)
    output = ROOT / "assets" / f"benchmark-overview-{theme}.svg"
    fig.savefig(output, transparent=True)
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines()) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    data = load_data()
    for theme in PALETTES:
        render(theme, data)
    print("Generated transparent light/dark eight-benchmark overview")
