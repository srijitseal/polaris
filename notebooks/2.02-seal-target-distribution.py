#!/usr/bin/env python
"""Target/endpoint distribution analysis.

Characterizes the 9 ADME endpoint distributions, coverage patterns, and
train/test comparisons. Informs which endpoints have sufficient data for
meaningful splitting strategies.

Usage:
    pixi run -e cheminformatics python notebooks/2.02-seal-target-distribution.py

Outputs:
    data/processed/2.02-seal-target-distribution/endpoint_distributions.png
    data/processed/2.02-seal-target-distribution/endpoint_counts.png
    data/processed/2.02-seal-target-distribution/coverage_heatmap.png
    data/processed/2.02-seal-target-distribution/endpoint_stats.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from scipy import stats

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()

ENDPOINTS = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]

# Short labels for plots
SHORT_LABELS = {
    "LogD": "LogD",
    "KSOL": "KSOL (uM)",
    "HLM CLint": "HLM CLint\n(mL/min/kg)",
    "MLM CLint": "MLM CLint\n(mL/min/kg)",
    "Caco-2 Permeability Papp A>B": "Caco-2 Papp\n(1e-6 cm/s)",
    "Caco-2 Permeability Efflux": "Caco-2 Efflux",
    "MPPB": "MPPB (% unbound)",
    "MBPB": "MBPB (% unbound)",
    "MGMB": "MGMB (% unbound)",
}

# Endpoints that are typically log-normal
LOG_ENDPOINTS = {"KSOL", "HLM CLint", "MLM CLint", "Caco-2 Permeability Papp A>B"}


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.02-seal-target-distribution", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    logger.info(f"Loaded {len(df)} molecules ({len(train)} train, {len(test)} test)")

    # ── 2. Per-endpoint distribution plots (3×3 grid) ─────────────────
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()

    for i, ep in enumerate(ENDPOINTS):
        ax = axes[i]
        train_vals = train[ep].dropna()
        test_vals = test[ep].dropna()

        use_log = ep in LOG_ENDPOINTS and (train_vals > 0).all() and (test_vals > 0).all()

        if use_log:
            train_plot = np.log10(train_vals)
            test_plot = np.log10(test_vals)
            xlabel = f"log10({SHORT_LABELS[ep]})"
        else:
            train_plot = train_vals
            test_plot = test_vals
            xlabel = SHORT_LABELS[ep]

        bins = np.linspace(
            min(train_plot.min(), test_plot.min()) if len(test_plot) > 0 else train_plot.min(),
            max(train_plot.max(), test_plot.max()) if len(test_plot) > 0 else train_plot.max(),
            41,
        )
        ax.hist(train_plot, bins=bins, density=True, alpha=0.6, color="steelblue",
                label=f"Train (n={len(train_vals)})", edgecolor="white")
        if len(test_plot) > 0:
            ax.hist(test_plot, bins=bins, density=True, alpha=0.6, color="coral",
                    label=f"Test (n={len(test_vals)})", edgecolor="white")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=8)

    fig.suptitle("Endpoint distributions: train vs test", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved endpoint_distributions.png")
    plt.close("all")

    # ── 3. Endpoint coverage bar chart ────────────────────────────────
    coverage_data = []
    for ep in ENDPOINTS:
        n_train = train[ep].notna().sum()
        n_test = test[ep].notna().sum()
        pct_train = 100 * n_train / len(train)
        pct_test = 100 * n_test / len(test)
        coverage_data.append({
            "endpoint": ep,
            "train_count": n_train,
            "test_count": n_test,
            "total_count": n_train + n_test,
            "train_pct": pct_train,
            "test_pct": pct_test,
            "total_pct": 100 * (n_train + n_test) / len(df),
        })
        logger.info(f"  {ep}: {n_train + n_test} total ({pct_train:.0f}% train, {pct_test:.0f}% test)")

    cov_df = pd.DataFrame(coverage_data)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(ENDPOINTS))
    width = 0.35
    ax.bar(x - width / 2, cov_df["train_count"], width, label="Train", color="steelblue", edgecolor="white")
    ax.bar(x + width / 2, cov_df["test_count"], width, label="Test", color="coral", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[ep].replace("\n", " ") for ep in ENDPOINTS], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of molecules with data")
    ax.set_title("Endpoint coverage: non-null counts per endpoint")
    ax.legend()

    # Annotate totals
    for i, row in cov_df.iterrows():
        ax.text(i, row["train_count"] + row["test_count"] + 50,
                f"{row['total_pct']:.0f}%", ha="center", fontsize=8, color="gray")

    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_counts.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved endpoint_counts.png")
    plt.close("all")

    # ── 4. Coverage heatmap (aggregated by n_endpoints) ───────────────
    # Fraction of molecules with each endpoint, grouped by n_endpoints
    groups = df.groupby("n_endpoints")
    heatmap_data = []
    for n_ep, group in groups:
        row = {"n_endpoints": int(n_ep), "count": len(group)}
        for ep in ENDPOINTS:
            row[ep] = group[ep].notna().mean()
        heatmap_data.append(row)
    heatmap_df = pd.DataFrame(heatmap_data).set_index("n_endpoints")
    counts = heatmap_df.pop("count")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
        xticklabels=[SHORT_LABELS[ep].replace("\n", " ") for ep in ENDPOINTS],
        vmin=0, vmax=1, linewidths=0.5,
    )
    # Add molecule counts as row labels
    ytick_labels = [f"{n} endpoints (n={int(counts.iloc[i])})" for i, n in enumerate(heatmap_df.index)]
    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=9)
    ax.set_title("Endpoint coverage by molecule completeness\n(fraction of molecules with each endpoint)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_heatmap.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved coverage_heatmap.png")
    plt.close("all")

    # ── 5. Summary statistics ─────────────────────────────────────────
    stats_rows = []
    for ep in ENDPOINTS:
        for split_name, split_df in [("train", train), ("test", test), ("all", df)]:
            vals = split_df[ep].dropna()
            if len(vals) == 0:
                continue
            stats_rows.append({
                "endpoint": ep,
                "split": split_name,
                "count": len(vals),
                "pct_coverage": 100 * len(vals) / len(split_df),
                "mean": vals.mean(),
                "std": vals.std(),
                "median": vals.median(),
                "q1": vals.quantile(0.25),
                "q3": vals.quantile(0.75),
                "min": vals.min(),
                "max": vals.max(),
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_dir / "endpoint_stats.csv", index=False)
    logger.info(f"Saved endpoint_stats.csv ({len(stats_df)} rows)")

    # ── 6. Train vs test comparison (KS tests) ───────────────────────
    logger.info("Train vs test distribution comparison (KS tests):")
    for ep in ENDPOINTS:
        train_vals = train[ep].dropna()
        test_vals = test[ep].dropna()
        if len(train_vals) > 0 and len(test_vals) > 0:
            ks_stat, ks_p = stats.ks_2samp(train_vals, test_vals)
            sig = "***" if ks_p < 0.001 else "**" if ks_p < 0.01 else "*" if ks_p < 0.05 else "ns"
            logger.info(f"  {ep}: KS={ks_stat:.3f}, p={ks_p:.2e} {sig}")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
