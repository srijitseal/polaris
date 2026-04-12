#!/usr/bin/env python
"""ECFP fingerprint and Tanimoto distance exploration.

Compute 2048-bit ECFP4 fingerprints, all-pairwise Tanimoto distances,
1-NN and 5-NN distance distributions. Characterizes chemical diversity
of the Expansion Tx dataset (cf. Cas's Biogen HLM case study).

Usage:
    pixi run -e cheminformatics python notebooks/0.02-seal-ecfp-distance-exploration.py

Outputs:
    data/processed/0.02-seal-ecfp-distance-exploration/nn_distance_distributions.png
    data/processed/0.02-seal-ecfp-distance-exploration/all_pairwise_distances.png
    data/processed/0.02-seal-ecfp-distance-exploration/train_vs_test_distances.png
    data/processed/0.02-seal-ecfp-distance-exploration/distance_stats.csv
    data/interim/tanimoto_distance_matrix.npz
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import squareform

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()


def compute_ecfp4(smiles_list: list[str], nbits: int = 2048, radius: int = 2) -> list:
    """Compute ECFP4 fingerprints from SMILES."""
    fps = []
    failed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            failed += 1
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True))
    if failed > 0:
        logger.warning(f"{failed} SMILES failed to parse")
    return fps


def pairwise_tanimoto_distance_matrix(fps: list) -> np.ndarray:
    """Compute full pairwise Tanimoto distance matrix. Returns condensed form."""
    n = len(fps)
    logger.info(f"Computing pairwise Tanimoto distances for {n} molecules ({n * (n - 1) // 2:,} pairs)")
    condensed = np.zeros(n * (n - 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Row {i}/{n}")
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        length = len(sims)
        condensed[idx : idx + length] = 1.0 - np.array(sims, dtype=np.float32)
        idx += length
    return condensed


def get_knn_distances(dist_matrix_square: np.ndarray, k: int) -> np.ndarray:
    """Get k-th nearest neighbor distance for each molecule."""
    n = dist_matrix_square.shape[0]
    knn = np.zeros(n)
    for i in range(n):
        row = dist_matrix_square[i]
        # Exclude self (distance 0)
        sorted_dists = np.sort(row[row > 0])
        knn[i] = sorted_dists[k - 1] if len(sorted_dists) >= k else np.nan
    return knn


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "0.02-seal-ecfp-distance-exploration", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data and compute fingerprints ─────────────────────────
    logger.info("Loading datasets")
    train = pd.read_csv(RAW_DATA_DIR / "expansion_data_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "expansion_data_test.csv")
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)
    logger.info(f"Combined: {len(df)} molecules")

    logger.info("Computing ECFP4 fingerprints (2048-bit, radius 2, chirality-aware)")
    fps = compute_ecfp4(df["SMILES"].tolist())

    # Filter out failed parses
    valid_mask = [fp is not None for fp in fps]
    if not all(valid_mask):
        n_invalid = sum(1 for v in valid_mask if not v)
        logger.warning(f"Dropping {n_invalid} molecules with invalid SMILES")
        df = df[valid_mask].reset_index(drop=True)
        fps = [fp for fp in fps if fp is not None]

    n = len(fps)
    logger.info(f"Valid molecules: {n}")

    # ── 2. Compute pairwise distance matrix ───────────────────────────
    condensed = pairwise_tanimoto_distance_matrix(fps)

    # Save for reuse
    np.savez_compressed(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", condensed=condensed, molecule_names=df["Molecule Name"].values)
    logger.info(f"Saved distance matrix to {INTERIM_DATA_DIR / 'tanimoto_distance_matrix.npz'}")

    # Convert to square form for NN queries
    dist_square = squareform(condensed)

    # ── 3. NN distance distributions ──────────────────────────────────
    logger.info("Computing 1-NN and 5-NN distances")
    nn1 = get_knn_distances(dist_square, k=1)
    nn5 = get_knn_distances(dist_square, k=5)

    # ── 4. All-pairwise distance histogram ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    # Subsample for plotting if needed (29M points is a lot for a histogram)
    if len(condensed) > 5_000_000:
        rng = np.random.default_rng(42)
        sample_idx = rng.choice(len(condensed), size=5_000_000, replace=False)
        plot_dists = condensed[sample_idx]
        ax.set_title(f"All-pairwise Tanimoto distances (5M subsample of {len(condensed):,})")
    else:
        plot_dists = condensed
        ax.set_title(f"All-pairwise Tanimoto distances (N={len(condensed):,})")
    ax.hist(plot_dists, bins=100, edgecolor="white", color="steelblue", alpha=0.8)
    ax.set_xlabel("Tanimoto distance")
    ax.set_ylabel("Count")
    text = f"mean={condensed.mean():.3f}\nmed={np.median(condensed):.3f}"
    ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=9, bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5})
    fig.tight_layout()
    fig.savefig(output_dir / "all_pairwise_distances.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved all_pairwise_distances.png")
    plt.close("all")

    # ── 5. Combined NN plot (the "hero" figure) ──────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 101)

    # All-pairwise (subsampled)
    ax.hist(plot_dists, bins=bins, density=True, alpha=0.3, color="gray", label="All pairwise")
    # 1-NN
    ax.hist(nn1, bins=bins, density=True, alpha=0.6, color="steelblue", label="1-NN", edgecolor="white")
    # 5-NN
    ax.hist(nn5, bins=bins, density=True, alpha=0.6, color="coral", label="5-NN", edgecolor="white")

    ax.set_xlabel("Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Distance distributions (N={n} molecules, ECFP4 2048-bit)")
    ax.legend()

    # Annotate medians
    for label, vals, color in [("1-NN", nn1, "steelblue"), ("5-NN", nn5, "coral")]:
        med = np.nanmedian(vals)
        ax.axvline(med, color=color, linestyle="--", alpha=0.7)
        ax.text(med + 0.01, ax.get_ylim()[1] * 0.9, f"{label} med={med:.3f}", color=color, fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "nn_distance_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved nn_distance_distributions.png")
    plt.close("all")

    # ── 6. Train vs test distance analysis ────────────────────────────
    logger.info("Computing train-to-test distances")
    train_idx = df.index[df["split"] == "train"].values
    test_idx = df.index[df["split"] == "test"].values

    # 1-NN from each test molecule to training set
    test_to_train_1nn = np.array([dist_square[ti, train_idx].min() for ti in test_idx])
    # 1-NN within training set
    train_within_1nn = np.array([np.sort(dist_square[ti, train_idx])[1] for ti in train_idx])  # [1] to skip self

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 51)
    weights_train = np.ones_like(train_within_1nn) / len(train_within_1nn)
    weights_test = np.ones_like(test_to_train_1nn) / len(test_to_train_1nn)
    ax.hist(train_within_1nn, bins=bins, weights=weights_train, alpha=0.6, color="steelblue", label=f"Within-train 1-NN (n={len(train_idx):,})", edgecolor="white", linewidth=0.5)
    ax.hist(test_to_train_1nn, bins=bins, weights=weights_test, alpha=0.6, color="coral", label=f"Test-to-train 1-NN (n={len(test_idx):,})", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Jaccard distance")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(0, 1)
    ax.legend(frameon=True, fontsize=9)

    for vals, color in [(train_within_1nn, "steelblue"), (test_to_train_1nn, "coral")]:
        med = np.median(vals)
        ax.axvline(med, color=color, linestyle="--", alpha=0.7, linewidth=1)

    fig.tight_layout()
    fig.savefig(output_dir / "train_vs_test_distances.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved train_vs_test_distances.png")
    plt.close("all")

    # ── 7. Summary statistics ─────────────────────────────────────────
    stats = []
    for label, vals in [("all_pairwise", condensed), ("1-NN", nn1), ("5-NN", nn5), ("test_to_train_1nn", test_to_train_1nn), ("train_within_1nn", train_within_1nn)]:
        v = vals[~np.isnan(vals)] if isinstance(vals, np.ndarray) else vals
        stats.append(
            {
                "metric": label,
                "count": len(v),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "q1": float(np.percentile(v, 25)),
                "q3": float(np.percentile(v, 75)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
        )
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "distance_stats.csv", index=False)
    logger.info(f"Saved distance_stats.csv")
    logger.info(f"\n{stats_df.to_string(index=False)}")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
