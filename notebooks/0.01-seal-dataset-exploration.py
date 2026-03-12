#!/usr/bin/env python
"""Initial exploration of the Expansion Tx ADMET dataset.

Load ML-ready CSVs (train + test), count molecules/endpoints, check missingness,
endpoint coverage per molecule, ordinal ordering analysis, and physicochemical
property characterization (RNA-binding vs protein modulator hypothesis).

Usage:
    pixi run -e cheminformatics python notebooks/0.01-seal-dataset-exploration.py

Outputs:
    data/processed/0.01-seal-dataset-exploration/endpoint_missingness.png
    data/processed/0.01-seal-dataset-exploration/molecule_coverage.png
    data/processed/0.01-seal-dataset-exploration/endpoint_distributions.png
    data/processed/0.01-seal-dataset-exploration/ordinal_ordering.png
    data/processed/0.01-seal-dataset-exploration/physicochemical_properties.png
    data/processed/0.01-seal-dataset-exploration/summary_statistics.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from polaris_generalization.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()

# 9 ML-ready endpoints (skip RLM CLint — only in raw)
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


def compute_physicochemical(smiles_series: pd.Series) -> pd.DataFrame:
    """Compute physicochemical properties from SMILES using RDKit."""
    records = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append({k: np.nan for k in ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]})
            continue
        records.append(
            {
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "TPSA": Descriptors.TPSA(mol),
                "RotBonds": Lipinski.NumRotatableBonds(mol),
                "AromaticRings": Descriptors.NumAromaticRings(mol),
                "FractionCSP3": Descriptors.FractionCSP3(mol),
            }
        )
    return pd.DataFrame(records)


@app.command()
def main(
    output_dir: Path = typer.Option(PROCESSED_DATA_DIR / "0.01-seal-dataset-exploration", help="Output directory"),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading datasets")
    train = pd.read_csv(RAW_DATA_DIR / "expansion_data_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "expansion_data_test.csv")

    for name, df in [("train", train), ("test", test)]:
        logger.info(f"{name}: {df.shape[0]} rows, {df.shape[1]} cols — {list(df.columns)}")

    # Combine ML-ready for analysis
    train["split"] = "train"
    test["split"] = "test"
    ml_ready = pd.concat([train, test], ignore_index=True)
    logger.info(f"Combined: {len(ml_ready)} molecules, {len(ENDPOINTS)} endpoints")

    # ── 2. Molecule counts and overlap ────────────────────────────────
    logger.info("Checking molecule overlap")
    train_mols = set(train["Molecule Name"])
    test_mols = set(test["Molecule Name"])
    overlap = train_mols & test_mols
    logger.info(f"Train: {len(train_mols)}, Test: {len(test_mols)}, Overlap: {len(overlap)}")

    # ── 3. Endpoint missingness ───────────────────────────────────────
    logger.info("Analyzing endpoint missingness")

    # Bar chart: non-null counts per endpoint
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts_ml = ml_ready[ENDPOINTS].notna().sum().sort_values(ascending=True)
    counts_ml.plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_xlabel("Non-null count")
    axes[0].set_title(f"Endpoint coverage (ML-ready, N={len(ml_ready)})")
    for i, v in enumerate(counts_ml):
        axes[0].text(v + 20, i, f"{v} ({v / len(ml_ready) * 100:.0f}%)", va="center", fontsize=9)

    # Per-molecule coverage histogram
    n_endpoints = ml_ready[ENDPOINTS].notna().sum(axis=1)
    axes[1].hist(n_endpoints, bins=range(0, len(ENDPOINTS) + 2), edgecolor="white", color="steelblue", align="left")
    axes[1].set_xlabel("Number of endpoints measured")
    axes[1].set_ylabel("Number of molecules")
    axes[1].set_title("Endpoints per molecule")
    axes[1].set_xticks(range(0, len(ENDPOINTS) + 1))

    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_missingness.png", dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved endpoint_missingness.png")

    # Separate molecule coverage plot
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(n_endpoints, bins=range(0, len(ENDPOINTS) + 2), edgecolor="white", color="steelblue", align="left")
    ax2.set_xlabel("Number of endpoints measured")
    ax2.set_ylabel("Number of molecules")
    ax2.set_title("Endpoints per molecule")
    ax2.set_xticks(range(0, len(ENDPOINTS) + 1))
    fig2.tight_layout()
    fig2.savefig(output_dir / "molecule_coverage.png", dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved molecule_coverage.png")
    plt.close("all")

    # ── 4. Endpoint distributions ─────────────────────────────────────
    logger.info("Plotting endpoint distributions")
    n_cols = 3
    n_rows = (len(ENDPOINTS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, ep in enumerate(ENDPOINTS):
        ax = axes[i]
        vals = ml_ready[ep].dropna()
        ax.hist(vals, bins=50, edgecolor="white", color="steelblue", alpha=0.8)
        ax.set_title(ep, fontsize=11)
        ax.set_ylabel("Count")
        # Annotate
        text = f"n={len(vals)}\nmed={vals.median():.2f}\nstd={vals.std():.2f}"
        ax.text(0.97, 0.97, text, transform=ax.transAxes, ha="right", va="top", fontsize=8, bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})

    # Hide unused axes
    for j in range(len(ENDPOINTS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Endpoint distributions (ML-ready, train+test)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved endpoint_distributions.png")
    plt.close("all")

    # ── 5. Ordinal ordering analysis ──────────────────────────────────
    logger.info("Analyzing ordinal ordering")
    ml_ready["mol_index"] = ml_ready["Molecule Name"].str.extract(r"E-(\d+)").astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution of indices
    axes[0].hist(ml_ready["mol_index"], bins=100, edgecolor="white", color="steelblue")
    axes[0].set_xlabel("Molecule index (from E-XXXXXXX)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of molecule indices")

    # Train vs test
    for split, color in [("train", "steelblue"), ("test", "coral")]:
        subset = ml_ready[ml_ready["split"] == split]
        axes[1].hist(subset["mol_index"], bins=80, alpha=0.6, label=split, color=color, edgecolor="white")
    axes[1].set_xlabel("Molecule index")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Train vs test ordinal ordering")
    axes[1].legend()

    train_median = ml_ready[ml_ready["split"] == "train"]["mol_index"].median()
    test_median = ml_ready[ml_ready["split"] == "test"]["mol_index"].median()
    logger.info(f"Median molecule index — train: {train_median:.0f}, test: {test_median:.0f}")

    fig.tight_layout()
    fig.savefig(output_dir / "ordinal_ordering.png", dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved ordinal_ordering.png")
    plt.close("all")

    # ── 6. Physicochemical properties ─────────────────────────────────
    logger.info("Computing physicochemical properties from SMILES")
    physchem = compute_physicochemical(ml_ready["SMILES"])
    props = list(physchem.columns)

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()
    for i, prop in enumerate(props):
        sns.violinplot(y=physchem[prop].dropna(), ax=axes[i], color="steelblue", inner="quartile")
        axes[i].set_title(prop)
        axes[i].set_ylabel("")

    # Add Lipinski reference lines where applicable
    lipinski_limits = {"MW": 500, "LogP": 5, "HBA": 10, "HBD": 5}
    for i, prop in enumerate(props):
        if prop in lipinski_limits:
            axes[i].axhline(lipinski_limits[prop], color="red", linestyle="--", alpha=0.7, label=f"Lipinski ({lipinski_limits[prop]})")
            axes[i].legend(fontsize=8)

    fig.suptitle("Physicochemical properties (RNA-targeting compounds)", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "physicochemical_properties.png", dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved physicochemical_properties.png")
    plt.close("all")

    # ── 7. Summary statistics ─────────────────────────────────────────
    logger.info("Generating summary statistics")
    stats = []
    for ep in ENDPOINTS:
        vals = ml_ready[ep].dropna()
        stats.append(
            {
                "endpoint": ep,
                "count": len(vals),
                "pct_missing": (1 - len(vals) / len(ml_ready)) * 100,
                "mean": vals.mean(),
                "median": vals.median(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    logger.info(f"Saved summary_statistics.csv")
    logger.info(f"\n{stats_df.to_string(index=False)}")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
