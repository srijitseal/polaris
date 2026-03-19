#!/usr/bin/env python
"""Chemical space characterization via Butina clustering.

Reuses precomputed ECFP4 Tanimoto distance matrix from 0.02 and canonical
dataset from 1.01. Performs Butina clustering (cutoff 0.7), visualizes
cluster size distribution and representative molecules from top clusters.

Usage:
    pixi run -e cheminformatics python notebooks/2.01-seal-chemical-space-analysis.py

Outputs:
    data/processed/2.01-seal-chemical-space-analysis/cluster_size_distribution.png
    data/processed/2.01-seal-chemical-space-analysis/top_clusters_grid.png
    data/processed/2.01-seal-chemical-space-analysis/cluster_stats.csv
    data/interim/butina_clusters.parquet
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.ML.Cluster import Butina
from scipy.spatial.distance import squareform

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()


def butina_cluster(dist_square: np.ndarray, cutoff: float = 0.7) -> list[tuple]:
    """Run Butina clustering from a square distance matrix.

    Butina.ClusterData expects a flat list in lower-triangle, row-major order:
    d(1,0), d(2,0), d(2,1), d(3,0), d(3,1), d(3,2), ...
    """
    n = dist_square.shape[0]
    # Build the flat distance list Butina expects
    dists = []
    for i in range(1, n):
        for j in range(i):
            dists.append(float(dist_square[i, j]))

    logger.info(f"Running Butina clustering: {n} molecules, cutoff={cutoff}")
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    logger.info(f"Found {len(clusters)} clusters")
    return clusters


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.01-seal-chemical-space-analysis", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    cutoff: float = typer.Option(0.7, help="Butina distance cutoff"),
    n_top_clusters: int = typer.Option(5, help="Number of top clusters to visualize"),
    mols_per_cluster: int = typer.Option(6, help="Molecules to show per cluster"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    logger.info(f"Loaded {len(df)} molecules")

    logger.info("Loading precomputed distance matrix")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    condensed = npz["condensed"]
    molecule_names = npz["molecule_names"]
    logger.info(f"Distance matrix: {len(molecule_names)} molecules, {len(condensed):,} pairs")

    # Convert to square form
    dist_square = squareform(condensed)

    # Verify alignment
    assert len(molecule_names) == len(df), (
        f"Distance matrix ({len(molecule_names)}) != dataset ({len(df)})"
    )

    # ── 2. Butina clustering ──────────────────────────────────────────
    clusters = butina_cluster(dist_square, cutoff=cutoff)

    # Assign cluster IDs sorted by size (largest first)
    clusters = sorted(clusters, key=len, reverse=True)
    cluster_ids = np.full(len(df), -1, dtype=int)
    cluster_sizes_map = {}
    for cid, members in enumerate(clusters):
        for idx in members:
            cluster_ids[idx] = cid
        cluster_sizes_map[cid] = len(members)

    df["cluster_id"] = cluster_ids
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes_map)

    # ── 3. Cluster statistics ─────────────────────────────────────────
    sizes = [len(c) for c in clusters]
    n_singletons = sum(1 for s in sizes if s == 1)
    logger.info(f"Total clusters: {len(clusters)}")
    logger.info(f"Singletons: {n_singletons} ({100 * n_singletons / len(clusters):.1f}%)")
    logger.info(f"Top 5 cluster sizes: {sizes[:5]}")
    logger.info(f"Median cluster size: {np.median(sizes):.0f}")

    # ── 4. Cluster size distribution ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of top 20 clusters
    top_n = min(20, len(sizes))
    axes[0].bar(range(top_n), sizes[:top_n], color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Cluster rank")
    axes[0].set_ylabel("Cluster size")
    axes[0].set_title(f"Top {top_n} clusters (of {len(clusters)} total)")

    # Right: histogram of all cluster sizes (log scale)
    axes[1].hist(sizes, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Cluster size")
    axes[1].set_ylabel("Count")
    axes[1].set_yscale("log")
    axes[1].set_title(f"Cluster size distribution\n{n_singletons} singletons, cutoff={cutoff}")

    fig.tight_layout()
    fig.savefig(output_dir / "cluster_size_distribution.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved cluster_size_distribution.png")
    plt.close("all")

    # ── 5. Visualize top clusters ─────────────────────────────────────
    rng = np.random.default_rng(42)
    smiles_list = df["SMILES"].values
    names_list = df["Molecule Name"].values

    grid_images = []
    for rank in range(min(n_top_clusters, len(clusters))):
        members = list(clusters[rank])
        n_show = min(mols_per_cluster, len(members))
        if len(members) > n_show:
            sample_idx = rng.choice(members, size=n_show, replace=False)
        else:
            sample_idx = members[:n_show]

        mols = []
        legends = []
        for idx in sample_idx:
            mol = Chem.MolFromSmiles(smiles_list[idx])
            if mol is not None:
                mols.append(mol)
                legends.append(f"{names_list[idx]}\nCluster {rank} (n={len(members)})")

        if mols:
            img = Draw.MolsToGridImage(
                mols, molsPerRow=3, subImgSize=(300, 300), legends=legends
            )
            img.save(output_dir / f"cluster_{rank}_examples.png")
            grid_images.append((rank, len(members), img))

    logger.info(f"Saved {len(grid_images)} individual cluster grids")

    # Combined figure
    if grid_images:
        n_grids = len(grid_images)
        fig, axes = plt.subplots(n_grids, 1, figsize=(10, 4 * n_grids))
        if n_grids == 1:
            axes = [axes]
        for ax, (rank, size, img) in zip(axes, grid_images):
            ax.imshow(img)
            ax.set_title(f"Cluster {rank} — {size} molecules", fontsize=12)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_dir / "top_clusters_grid.png", dpi=dpi, bbox_inches="tight")
        logger.info("Saved top_clusters_grid.png")
        plt.close("all")

    # ── 6. Save cluster stats CSV ─────────────────────────────────────
    stats_rows = []
    for cid, members in enumerate(clusters):
        example_names = [str(names_list[i]) for i in members[:3]]
        stats_rows.append({
            "cluster_id": cid,
            "size": len(members),
            "example_molecules": "; ".join(example_names),
        })
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_dir / "cluster_stats.csv", index=False)
    logger.info(f"Saved cluster_stats.csv ({len(stats_df)} clusters)")

    # ── 7. Save cluster assignments for reuse ─────────────────────────
    clusters_parquet = INTERIM_DATA_DIR / "butina_clusters.parquet"
    df[["Molecule Name", "cluster_id", "cluster_size"]].to_parquet(clusters_parquet, index=False)
    logger.info(f"Saved {clusters_parquet}")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
