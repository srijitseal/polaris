#!/usr/bin/env python
"""Cross-dataset characterization: ExpansionRx vs Biogen ADME.

An isolated case study showing that the splitting protocol must follow dataset
characterization, not precede it. ExpansionRx is built from a few concentrated
Butina chemical series (which is what makes cluster- and IID-vs-OOD-series
splits constructible), whereas the public Biogen ADME dataset is structurally
diffuse: it has no large series, so a series-based split cannot be built and
even a random split already places test molecules far from training.

All chemistry uses the same representation as the rest of the paper: 2048-bit
ECFP4 (radius 2, chirality-aware), Tanimoto distance, Butina clustering at a
0.7 distance cutoff. ExpansionRx reuses the precomputed interim artifacts;
Biogen is characterized from scratch.

Data:
    data/external/biogen_adme_public_set_3521.csv
        Fang et al. 2023, J. Chem. Inf. Model. 63, 3263;
        github.com/molecularinformatics/Computational-ADME

Usage:
    pixi run -e cheminformatics python notebooks/4.01-seal-cross-dataset-characterization.py

Outputs:
    data/processed/4.01-seal-cross-dataset-characterization/cross_dataset_characterization.png
    data/processed/4.01-seal-cross-dataset-characterization/cross_dataset_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import squareform

from polaris_generalization.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()

BUTINA_CUTOFF = 0.7
RANDOM_SPLIT_SEEDS = (0, 1, 2, 3, 4)
TEST_FRAC = 0.2

DATASET_COLORS = {"ExpansionRx": "steelblue", "Biogen ADME": "darkorange"}


def compute_ecfp4(smiles_list: list[str], nbits: int = 2048, radius: int = 2) -> list:
    """ECFP4 fingerprints (chirality-aware), None for unparseable SMILES."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        fps.append(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True)
            if mol is not None else None
        )
    return fps


def pairwise_tanimoto_square(fps: list) -> np.ndarray:
    """Full pairwise Tanimoto distance matrix (square form)."""
    n = len(fps)
    logger.info(f"Computing {n * (n - 1) // 2:,} pairwise distances for {n} molecules")
    condensed = np.zeros(n * (n - 1) // 2, dtype=np.float32)
    idx = 0
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        condensed[idx:idx + len(sims)] = 1.0 - np.array(sims, dtype=np.float32)
        idx += len(sims)
    return squareform(condensed)


def butina_cluster_sizes(dist_square: np.ndarray, cutoff: float = BUTINA_CUTOFF) -> list[int]:
    """Butina cluster sizes, sorted largest first."""
    n = dist_square.shape[0]
    dists = [float(dist_square[i, j]) for i in range(1, n) for j in range(i)]
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    return sorted((len(c) for c in clusters), reverse=True)


def intra_set_nn1(dist_square: np.ndarray) -> np.ndarray:
    """Nearest-neighbour (1-NN) distance within the set for each molecule."""
    d = dist_square.copy()
    np.fill_diagonal(d, np.inf)
    return d.min(axis=1)


def random_split_median_nn(dist_square: np.ndarray) -> float:
    """Median test-to-train 1-NN distance under a random split, averaged over seeds."""
    n = dist_square.shape[0]
    n_test = int(n * TEST_FRAC)
    meds = []
    for seed in RANDOM_SPLIT_SEEDS:
        perm = np.random.default_rng(seed).permutation(n)
        test, train = perm[:n_test], perm[n_test:]
        meds.append(float(np.median(dist_square[np.ix_(test, train)].min(axis=1))))
    return float(np.mean(meds))


def characterize(name: str, dist_square: np.ndarray, cluster_sizes: list[int]) -> dict:
    """Compute the structural-diversity summary for one dataset."""
    n = dist_square.shape[0]
    total = sum(cluster_sizes)
    nn1 = intra_set_nn1(dist_square)
    cum_coverage = np.cumsum(cluster_sizes) / total * 100
    return {
        "name": name,
        "n_molecules": n,
        "n_clusters": len(cluster_sizes),
        "largest_cluster_pct": 100 * cluster_sizes[0] / total,
        "top10_cluster_pct": 100 * sum(cluster_sizes[:10]) / total,
        "singleton_pct": 100 * sum(s == 1 for s in cluster_sizes) / len(cluster_sizes),
        "median_intra_nn1": float(np.median(nn1)),
        "random_split_median_nn": random_split_median_nn(dist_square),
        "_nn1": nn1,
        "_cum_coverage": cum_coverage,
    }


def plot_characterization(results: list[dict], output_dir: Path, dpi: int) -> None:
    """Three-panel figure: cluster concentration, NN distance, random-split distance."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Cumulative cluster coverage vs cluster rank (log x).
    for r in results:
        ranks = np.arange(1, len(r["_cum_coverage"]) + 1)
        axes[0].plot(ranks, r["_cum_coverage"], color=DATASET_COLORS[r["name"]],
                     linewidth=2, label=r["name"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Butina cluster rank (largest first)")
    axes[0].set_ylabel("Cumulative % of molecules")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=True)
    axes[0].set_title("(a) Chemical-series concentration", fontsize=11)

    # (b) Intra-set nearest-neighbour distance (relative frequency).
    bins = np.linspace(0, 1, 41)
    for r in results:
        w = np.ones_like(r["_nn1"]) / len(r["_nn1"])
        axes[1].hist(r["_nn1"], bins=bins, weights=w, histtype="stepfilled", alpha=0.45,
                     color=DATASET_COLORS[r["name"]], edgecolor=DATASET_COLORS[r["name"]],
                     linewidth=1.2, label=f"{r['name']} (med={r['median_intra_nn1']:.2f})")
    axes[1].set_xlabel("Intra-set 1-NN Jaccard distance")
    axes[1].set_ylabel("Relative frequency")
    axes[1].set_xlim(0, 1)
    axes[1].legend(frameon=True)
    axes[1].set_title("(b) Structural scatter", fontsize=11)

    # (c) Random-split test-to-train distance (the add-on).
    names = [r["name"] for r in results]
    vals = [r["random_split_median_nn"] for r in results]
    axes[2].bar(np.arange(len(names)), vals,
                color=[DATASET_COLORS[n] for n in names], edgecolor="white")
    axes[2].set_xticks(np.arange(len(names)))
    axes[2].set_xticklabels(names)
    axes[2].set_ylabel("Median test-to-train 1-NN Jaccard distance")
    for i, v in enumerate(vals):
        axes[2].annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=9)
    axes[2].set_title("(c) Difficulty of a random split", fontsize=11)

    fig.tight_layout()
    fig.savefig(output_dir / "cross_dataset_characterization.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved cross_dataset_characterization.png")
    plt.close("all")


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "4.01-seal-cross-dataset-characterization", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    biogen_csv: Path = typer.Option(
        EXTERNAL_DATA_DIR / "biogen_adme_public_set_3521.csv", help="Biogen ADME CSV"
    ),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. ExpansionRx: reuse precomputed interim artifacts ───────────
    logger.info("Loading ExpansionRx precomputed distance matrix and Butina clusters")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    exp_dist = squareform(npz["condensed"])
    butina = pd.read_parquet(INTERIM_DATA_DIR / "butina_clusters.parquet")
    exp_sizes = sorted(butina.groupby("cluster_id").size().tolist(), reverse=True)

    # ── 2. Biogen ADME: characterize from scratch ─────────────────────
    logger.info(f"Loading Biogen ADME from {biogen_csv}")
    biogen = pd.read_csv(biogen_csv)
    fps = compute_ecfp4(biogen["SMILES"].tolist())
    valid = [fp is not None for fp in fps]
    n_invalid = valid.count(False)
    if n_invalid:
        logger.warning(f"Dropping {n_invalid} Biogen molecules with invalid SMILES")
    fps = [fp for fp, ok in zip(fps, valid) if ok]
    logger.info(f"Biogen valid molecules: {len(fps)}")
    bio_dist = pairwise_tanimoto_square(fps)
    bio_sizes = butina_cluster_sizes(bio_dist)

    # ── 3. Characterize both ──────────────────────────────────────────
    results = [
        characterize("ExpansionRx", exp_dist, exp_sizes),
        characterize("Biogen ADME", bio_dist, bio_sizes),
    ]

    plot_characterization(results, output_dir, dpi)

    summary = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    summary.to_csv(output_dir / "cross_dataset_summary.csv", index=False)
    logger.info("Saved cross_dataset_summary.csv")
    for r in results:
        logger.info(
            f"  {r['name']}: n={r['n_molecules']}, clusters={r['n_clusters']}, "
            f"largest={r['largest_cluster_pct']:.1f}%, top10={r['top10_cluster_pct']:.1f}%, "
            f"singletons={r['singleton_pct']:.1f}%, median_1NN={r['median_intra_nn1']:.3f}, "
            f"random-split_1NN={r['random_split_median_nn']:.3f}"
        )

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
