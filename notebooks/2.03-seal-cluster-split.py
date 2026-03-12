#!/usr/bin/env python
"""Cluster-based cross-validation splitting via EKM + Mini Batch K-Means.

For each ADME endpoint, projects molecules (with non-null values) into
Euclidean space via Empirical Kernel Map (KernelPCA on Tanimoto similarity),
then clusters with Mini Batch K-Means into 5 folds. Repeated 5 times with
different random seeds for 5x5 CV.

Usage:
    pixi run -e cheminformatics python notebooks/2.03-seal-cluster-split.py

Outputs:
    data/processed/2.03-seal-cluster-split/fold_sizes_grid.png
    data/processed/2.03-seal-cluster-split/fold_distance_distributions.png
    data/processed/2.03-seal-cluster-split/fold_size_variation.png
    data/processed/2.03-seal-cluster-split/fold_stats.csv
    data/interim/cluster_cv_folds.parquet
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.spatial.distance import squareform
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import KernelPCA

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


def make_endpoint_folds(
    dist_square: np.ndarray,
    mask: np.ndarray,
    n_folds: int = 5,
    n_components: int = 50,
    n_clusters: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Assign fold IDs via EKM + MiniBatchKMeans for molecules indicated by mask.

    Uses more clusters than folds (default 20), then greedily assigns clusters
    to folds to balance sizes. This produces much more balanced folds than
    using k=n_folds directly.

    Args:
        dist_square: Full square distance matrix (N x N).
        mask: Boolean array of length N indicating which molecules to include.
        n_folds: Number of folds.
        n_components: KernelPCA components.
        n_clusters: Number of k-means clusters (should be > n_folds).
        seed: Random state for reproducibility.

    Returns:
        Array of fold IDs (length = mask.sum()), values 0 to n_folds-1.
    """
    idx = np.where(mask)[0]
    n = len(idx)
    n_comp = min(n_components, n - 1)
    n_clust = min(n_clusters, n - 1)

    # Sub-distance matrix → similarity
    D_sub = dist_square[np.ix_(idx, idx)]
    K_sub = 1.0 - D_sub

    # EKM: KernelPCA on precomputed kernel
    kpca = KernelPCA(n_components=n_comp, kernel="precomputed", random_state=seed)
    X_proj = kpca.fit_transform(K_sub)

    # Mini Batch K-Means into n_clusters groups
    kmeans = MiniBatchKMeans(n_clusters=n_clust, random_state=seed, batch_size=min(1024, n), n_init=3)
    cluster_ids = kmeans.fit_predict(X_proj)

    # Greedy assignment: sort clusters by size (largest first), assign each to
    # the fold with the fewest molecules so far
    cluster_sizes = [(c, int((cluster_ids == c).sum())) for c in range(n_clust)]
    cluster_sizes.sort(key=lambda x: -x[1])

    fold_totals = np.zeros(n_folds, dtype=int)
    cluster_to_fold = {}
    for c, size in cluster_sizes:
        fold = int(np.argmin(fold_totals))
        cluster_to_fold[c] = fold
        fold_totals[fold] += size

    folds = np.array([cluster_to_fold[c] for c in cluster_ids])
    return folds


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.03-seal-cluster-split", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    n_folds: int = typer.Option(5, help="Number of CV folds"),
    n_repeats: int = typer.Option(5, help="Number of CV repeats"),
    n_components: int = typer.Option(50, help="KernelPCA components"),
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
    dist_square = squareform(condensed)
    logger.info(f"Distance matrix: {dist_square.shape[0]} x {dist_square.shape[1]}")

    # ── 2. Per-endpoint fold assignments (5x5 CV) ────────────────────
    all_fold_records = []
    fold_stats_rows = []

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        n_mol = mask.sum()
        logger.info(f"Endpoint {ep}: {n_mol} molecules")

        for repeat in range(n_repeats):
            seed = repeat
            folds = make_endpoint_folds(dist_square, mask, n_folds=n_folds, n_components=n_components, seed=seed)

            mol_names = df.loc[mask, "Molecule Name"].values
            for name, fold_id in zip(mol_names, folds):
                all_fold_records.append({
                    "Molecule Name": name,
                    "endpoint": ep,
                    "repeat": repeat,
                    "fold": int(fold_id),
                })

            # Stats
            for fold_id in range(n_folds):
                fold_size = int((folds == fold_id).sum())
                fold_stats_rows.append({
                    "endpoint": ep,
                    "repeat": repeat,
                    "fold": fold_id,
                    "size": fold_size,
                    "pct": 100 * fold_size / n_mol,
                })

    folds_df = pd.DataFrame(all_fold_records)
    stats_df = pd.DataFrame(fold_stats_rows)

    logger.info(f"Total fold assignments: {len(folds_df)}")

    # ── 3. Fold size validation (3x3 grid, repeat 0) ─────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for i, ep in enumerate(ENDPOINTS):
        ax = axes[i]
        ep_stats = stats_df[(stats_df["endpoint"] == ep) & (stats_df["repeat"] == 0)]
        ax.bar(ep_stats["fold"], ep_stats["size"], color="steelblue", edgecolor="white")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Size")
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_folds))

        sizes = ep_stats["size"].values
        ratio = sizes.max() / sizes.min() if sizes.min() > 0 else float("inf")
        ax.text(0.95, 0.95, f"max/min={ratio:.2f}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")

    fig.suptitle("Fold sizes per endpoint (repeat 0)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_sizes_grid.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_sizes_grid.png")
    plt.close("all")

    # ── 4. Per-fold distance analysis (LogD, repeat 0) ────────────────
    rep_ep = "LogD"
    mask = df[rep_ep].notna().values
    idx = np.where(mask)[0]
    ep_folds = folds_df[(folds_df["endpoint"] == rep_ep) & (folds_df["repeat"] == 0)]
    fold_ids = ep_folds["fold"].values
    D_sub = dist_square[np.ix_(idx, idx)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n_folds))

    for fold_id in range(n_folds):
        test_mask = fold_ids == fold_id
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        # 1-NN from test to train
        test_to_train = D_sub[np.ix_(test_idx, train_idx)]
        nn1 = test_to_train.min(axis=1)

        ax.hist(nn1, bins=50, density=True, alpha=0.5, color=colors[fold_id],
                label=f"Fold {fold_id} (n={len(test_idx)}, med={np.median(nn1):.3f})",
                edgecolor="white")

    ax.set_xlabel("Test-to-train 1-NN Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Test-to-train 1-NN distances per fold ({rep_ep}, repeat 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_distance_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_distance_distributions.png")
    plt.close("all")

    # ── 5. Fold size variation across repeats ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: LogD fold sizes across repeats
    ep_stats = stats_df[stats_df["endpoint"] == rep_ep]
    pivot = ep_stats.pivot(index="repeat", columns="fold", values="size")
    pivot.plot(kind="bar", ax=axes[0], colormap="Set2", edgecolor="white")
    axes[0].set_xlabel("Repeat")
    axes[0].set_ylabel("Fold size")
    axes[0].set_title(f"Fold sizes across repeats ({rep_ep})")
    axes[0].legend(title="Fold", fontsize=8)
    axes[0].tick_params(axis="x", rotation=0)

    # Right: max/min ratio per endpoint across repeats
    ratio_data = []
    for ep in ENDPOINTS:
        for repeat in range(n_repeats):
            ep_r = stats_df[(stats_df["endpoint"] == ep) & (stats_df["repeat"] == repeat)]
            sizes = ep_r["size"].values
            ratio = sizes.max() / sizes.min() if sizes.min() > 0 else float("inf")
            ratio_data.append({"endpoint": ep, "repeat": repeat, "max_min_ratio": ratio})
    ratio_df = pd.DataFrame(ratio_data)
    mean_ratios = ratio_df.groupby("endpoint")["max_min_ratio"].mean()
    mean_ratios = mean_ratios.reindex(ENDPOINTS)
    axes[1].bar(range(len(ENDPOINTS)), mean_ratios.values, color="steelblue", edgecolor="white")
    axes[1].set_xticks(range(len(ENDPOINTS)))
    axes[1].set_xticklabels([ep[:12] for ep in ENDPOINTS], rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean max/min fold size ratio")
    axes[1].set_title("Fold balance across endpoints (mean over repeats)")
    axes[1].axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="ratio=2")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "fold_size_variation.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_size_variation.png")
    plt.close("all")

    # ── 6. Save fold stats CSV ────────────────────────────────────────
    stats_df.to_csv(output_dir / "fold_stats.csv", index=False)
    logger.info(f"Saved fold_stats.csv ({len(stats_df)} rows)")

    # ── 7. Save fold assignments ──────────────────────────────────────
    folds_path = INTERIM_DATA_DIR / "cluster_cv_folds.parquet"
    folds_df.to_parquet(folds_path, index=False)
    logger.info(f"Saved {folds_path} ({len(folds_df)} rows)")

    # Summary
    for ep in ENDPOINTS:
        ep_r0 = stats_df[(stats_df["endpoint"] == ep) & (stats_df["repeat"] == 0)]
        sizes = ep_r0["size"].values
        ratio = sizes.max() / sizes.min() if sizes.min() > 0 else float("inf")
        logger.info(f"  {ep}: folds={list(sizes)}, ratio={ratio:.2f}")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
