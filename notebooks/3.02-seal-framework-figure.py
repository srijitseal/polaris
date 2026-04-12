#!/usr/bin/env python
"""Framework overview figure — evaluation workflow schematic.

Generates a conceptual figure showing the decision workflow:
Dataset → Characterization → Deployment scenario → Splitting strategy →
Failure-mode analyses → Identified failure modes.

Usage:
    pixi run -e cheminformatics python notebooks/3.02-seal-framework-figure.py

Outputs:
    data/processed/3.02-seal-framework-figure/framework_overview.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import typer
from loguru import logger

from polaris_generalization.config import PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()


def draw_box(ax, xy, width, height, text, facecolor="white", edgecolor="black",
             fontsize=11, fontweight="normal", text_color="black", linewidth=1.5,
             rounded=True, zorder=2):
    """Draw a rounded rectangle with centered text."""
    style = "round,pad=0.02" if rounded else "square,pad=0"
    box = mpatches.FancyBboxPatch(
        xy, width, height, boxstyle=style,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth, zorder=zorder
    )
    ax.add_patch(box)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color, zorder=zorder + 1,
            linespacing=1.4)


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "3.02-seal-framework-figure", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 20,
        "font.family": "sans-serif",
    })

    fig, ax = plt.subplots(figsize=(22, 16))
    ax.set_xlim(-1, 19)
    ax.set_ylim(-0.5, 15)
    ax.axis("off")

    # ── Row 1: Dataset ────────────────────────────────────────────────
    draw_box(ax, (5.5, 12.5), 7, 1.5,
             "ADMET Dataset\n(chemical series, temporal ordering, multi-endpoint)",
             facecolor="#E8EAF6", edgecolor="#3F51B5", fontsize=22, fontweight="bold")

    # ── Row 2: Dataset Characterization ───────────────────────────────
    draw_box(ax, (4.5, 10.6), 9, 1.5,
             "Dataset Characterization\nDistance distributions · Butina clustering · Endpoint coverage · Temporal structure",
             facecolor="#E3F2FD", edgecolor="#1976D2", fontsize=18, fontweight="bold")

    # ── Row 3: Deployment scenario label ──────────────────────────────
    draw_box(ax, (4.0, 9.3), 10, 0.9,
             "Match deployment scenario → splitting strategy",
             facecolor="#FFF8E1", edgecolor="#F9A825", fontsize=20, fontweight="bold")

    # ── Row 4: Three deployment scenarios ─────────────────────────────
    scenario_w = 5.0
    scenario_h = 2.0
    sc_y = 6.8

    draw_box(ax, (0.5, sc_y), scenario_w, scenario_h,
             "Hit Identification\nVirtual Screening\n\nCluster-based splitting\n(5 folds, structural separation)",
             facecolor="#E8F5E9", edgecolor="#388E3C", fontsize=16)

    draw_box(ax, (6.5, sc_y), scenario_w, scenario_h,
             "Prospective Deployment\nCampaign Evaluation\n\nTime-based splitting\n(4 folds, expanding window)",
             facecolor="#FFF3E0", edgecolor="#E65100", fontsize=16)

    draw_box(ax, (12.5, sc_y), scenario_w, scenario_h,
             "Lead Optimization\nValue Extrapolation\n\nTarget-value splitting\n(4 folds, expanding window)",
             facecolor="#FCE4EC", edgecolor="#C62828", fontsize=16)

    # ── Row 5: Complementary analyses label ───────────────────────────
    draw_box(ax, (4.0, 5.7), 10, 0.9,
             "Complementary failure-mode analyses",
             facecolor="#F3E5F5", edgecolor="#7B1FA2", fontsize=20, fontweight="bold")

    # ── Row 6: Four analysis types ────────────────────────────────────
    analysis_y = 3.7
    analysis_h = 1.6
    analysis_w = 4.0
    x_positions = [0.3, 4.7, 9.1, 13.5]

    analyses = [
        ("Performance over\ndistance curves\n(Fig 4)", "#E8F5E9", "#388E3C"),
        ("IID vs OOD\nseries comparison\n(Fig 5)", "#E3F2FD", "#1565C0"),
        ("Activity cliff\nevaluation\n(Fig S4)", "#FFF3E0", "#E65100"),
        ("Molecular variant\nconsistency\n(Fig S5–S8)", "#FCE4EC", "#C62828"),
    ]

    for i, (text, fc, ec) in enumerate(analyses):
        draw_box(ax, (x_positions[i], analysis_y), analysis_w, analysis_h,
                 text, facecolor=fc, edgecolor=ec, fontsize=16)

    # ── Row 7: Failure modes label ────────────────────────────────────
    draw_box(ax, (3.0, 2.5), 12, 0.9,
             "Failure modes identified",
             facecolor="#ECEFF1", edgecolor="#37474F", fontsize=20, fontweight="bold")

    # ── Row 8: Four failure mode boxes ────────────────────────────────
    fm_y = 0.5
    fm_h = 1.6
    fm_w = 4.0
    fm_x_positions = [0.0, 4.5, 9.0, 13.5]

    failure_modes = [
        ("Extrapolation\nfailure\n(cross-series, value range)", "#FFCDD2", "#B71C1C"),
        ("Interpolation\nfailure\n(activity cliffs)", "#FFE0B2", "#E65100"),
        ("Representation\nfailure\n(stereochemistry, resonance)", "#C8E6C9", "#1B5E20"),
        ("Evaluation\nfailure\n(scaffold ≈ random splits)", "#BBDEFB", "#0D47A1"),
    ]

    for i, (text, fc, ec) in enumerate(failure_modes):
        draw_box(ax, (fm_x_positions[i], fm_y), fm_w, fm_h,
                 text, facecolor=fc, edgecolor=ec, fontsize=16, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "framework_overview.png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    logger.info("Saved framework_overview.png")
    plt.close("all")


if __name__ == "__main__":
    app()
