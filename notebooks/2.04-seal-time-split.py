#!/usr/bin/env python
"""Time-based cross-validation splitting using ordinal molecule indices.

Uses the ordinal molecule naming (E-XXXXXXX) as a temporal proxy to create
expanding-window CV folds. Earlier molecules form training data, later
molecules form test sets. Compares distance characteristics to cluster-based
splits from 2.03.

Usage:
    pixi run -e cheminformatics python notebooks/2.04-seal-time-split.py

Outputs:
    data/processed/2.04-seal-time-split/fold_sizes_grid.png
    data/processed/2.04-seal-time-split/fold_temporal_ranges.png
    data/processed/2.04-seal-time-split/fold_distance_distributions.png
    data/processed/2.04-seal-time-split/time_vs_cluster_distances.png
    data/processed/2.04-seal-time-split/fold_stats.csv
    data/interim/time_cv_folds.parquet
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from scipy.spatial.distance import squareform

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


def make_time_folds(
    mol_indices: np.ndarray,
    n_windows: int = 4,
) -> np.ndarray:
    """Assign expanding-window CV fold IDs based on ordinal molecule indices.

    Fold k: train = first (k+1)*chunk molecules, test = next chunk.
    Where chunk = n // (n_windows + 1).

    Args:
        mol_indices: Ordinal molecule indices (used for sorting).
        n_windows: Number of test windows (folds).

    Returns:
        Array of fold IDs (length = len(mol_indices)).
        -1 for molecules in the first chunk (train-only in fold 0).
        0..n_windows-1 for test window assignments.
    """
    n = len(mol_indices)
    sort_order = np.argsort(mol_indices)
    chunk = n // (n_windows + 1)

    # Each molecule gets the fold where it serves as test
    # First chunk is always train; subsequent chunks are test for folds 0..n_windows-1
    fold_ids = np.full(n, -1, dtype=int)
    for k in range(n_windows):
        start = (k + 1) * chunk
        end = (k + 2) * chunk if k < n_windows - 1 else n
        fold_ids[sort_order[start:end]] = k

    return fold_ids


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.04-seal-time-split", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    n_windows: int = typer.Option(4, help="Number of expanding window folds"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    logger.info(f"Loaded {len(df)} molecules")

    logger.info("Loading precomputed distance matrix")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    dist_square = squareform(npz["condensed"])

    logger.info("Loading cluster CV folds for comparison")
    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")

    # ── 2. Per-endpoint time-split fold assignment ────────────────────
    all_fold_records = []
    fold_stats_rows = []

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        n_mol = len(ep_df)
        logger.info(f"Endpoint {ep}: {n_mol} molecules")

        mol_indices = ep_df["mol_index"].values
        fold_ids = make_time_folds(mol_indices, n_windows)

        for name, fold_id in zip(ep_df["Molecule Name"].values, fold_ids):
            all_fold_records.append({
                "Molecule Name": name,
                "endpoint": ep,
                "repeat": 0,
                "fold": int(fold_id),
            })

        # Stats per fold
        for fold_id in range(-1, n_windows):
            fold_mask = fold_ids == fold_id
            fold_size = int(fold_mask.sum())
            fold_mol_idx = mol_indices[fold_mask]
            label = f"train-only" if fold_id == -1 else f"fold {fold_id}"
            fold_stats_rows.append({
                "endpoint": ep,
                "fold": fold_id,
                "size": fold_size,
                "pct": 100 * fold_size / n_mol,
                "mol_index_min": int(fold_mol_idx.min()) if fold_size > 0 else 0,
                "mol_index_max": int(fold_mol_idx.max()) if fold_size > 0 else 0,
            })
            if fold_id >= 0:
                logger.info(f"  {label}: n={fold_size}, idx=[{fold_mol_idx.min()}-{fold_mol_idx.max()}]")

    folds_df = pd.DataFrame(all_fold_records)
    stats_df = pd.DataFrame(fold_stats_rows)
    logger.info(f"Total fold assignments: {len(folds_df)}")

    # ── 3. Fold size validation (3×3 grid) ────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for i, ep in enumerate(ENDPOINTS):
        ax = axes[i]
        ep_stats = stats_df[(stats_df["endpoint"] == ep) & (stats_df["fold"] >= 0)]
        ax.bar(ep_stats["fold"], ep_stats["size"], color="coral", edgecolor="white")
        ax.set_xlabel("Fold (test window)")
        ax.set_ylabel("Test size")
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_windows))

    fig.suptitle("Time-split fold sizes per endpoint (expanding window)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_sizes_grid.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_sizes_grid.png")
    plt.close("all")

    # ── 4. Temporal ranges per fold ───────────────────────────────────
    rep_ep = "LogD"
    ep_stats = stats_df[stats_df["endpoint"] == rep_ep]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["gray"] + list(plt.cm.Set2(np.linspace(0, 1, n_windows)))
    for _, row in ep_stats.iterrows():
        fold = row["fold"]
        label = "train-only" if fold == -1 else f"Fold {fold} test"
        y = fold + 1  # shift so train-only is at 0
        ax.barh(y, row["mol_index_max"] - row["mol_index_min"],
                left=row["mol_index_min"], height=0.6,
                color=colors[fold + 1], edgecolor="white", label=label)
        ax.text(row["mol_index_max"] + 200, y, f"n={row['size']}", va="center", fontsize=9)

    ax.set_yticks(range(n_windows + 1))
    ax.set_yticklabels(["Train-only"] + [f"Fold {k}" for k in range(n_windows)])
    ax.set_xlabel("Molecule index (ordinal)")
    ax.set_title(f"Temporal ranges per fold ({rep_ep}, expanding window)")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_temporal_ranges.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_temporal_ranges.png")
    plt.close("all")

    # ── 5. Per-fold distance analysis (LogD) ──────────────────────────
    mask = df[rep_ep].notna().values
    ep_df = df[mask].copy()
    idx_in_full = np.where(mask)[0]
    D_sub = dist_square[np.ix_(idx_in_full, idx_in_full)]
    mol_indices = ep_df["mol_index"].values
    fold_ids = make_time_folds(mol_indices, n_windows)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, n_windows))
    medians = []

    for fold_id in range(n_windows):
        test_mask = fold_ids == fold_id
        # Train = everything before this test window (folds -1 through fold_id-1)
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= (fold_ids == k)

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        test_to_train = D_sub[np.ix_(test_idx, train_idx)]
        nn1 = test_to_train.min(axis=1)
        med = np.median(nn1)
        medians.append(med)

        ax.hist(nn1, bins=50, density=True, alpha=0.5, color=colors[fold_id],
                label=f"Fold {fold_id} (n={len(test_idx)}, med={med:.3f})",
                edgecolor="white")

    ax.set_xlabel("Test-to-train 1-NN Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Time-split: test-to-train 1-NN per fold ({rep_ep})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_distance_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_distance_distributions.png")
    logger.info(f"  Median 1-NN per fold: {[f'{m:.3f}' for m in medians]}")
    plt.close("all")

    # ── 6. Comparison: time-split vs cluster-split ────────────────────
    # Get cluster-split fold 0 distances for LogD
    cluster_ep = cluster_folds[
        (cluster_folds["endpoint"] == rep_ep) & (cluster_folds["repeat"] == 0)
    ]
    # Map cluster fold IDs back to position indices
    ep_names = ep_df["Molecule Name"].values
    cluster_fold_map = dict(zip(cluster_ep["Molecule Name"], cluster_ep["fold"]))
    cluster_fold_ids = np.array([cluster_fold_map.get(n, -1) for n in ep_names])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: time-split median distances per fold
    ax = axes[0]
    ax.bar(range(n_windows), medians, color="coral", edgecolor="white")
    ax.set_xlabel("Fold (temporal order)")
    ax.set_ylabel("Median test-to-train 1-NN distance")
    ax.set_title("Time-split: increasing distance over time")
    ax.set_xticks(range(n_windows))

    # Right: overlay time-split vs cluster-split 1-NN distributions
    ax = axes[1]

    # Time-split: pool all folds
    time_nn1_all = []
    for fold_id in range(n_windows):
        test_mask = fold_ids == fold_id
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= (fold_ids == k)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        nn1 = D_sub[np.ix_(test_idx, train_idx)].min(axis=1)
        time_nn1_all.extend(nn1)

    # Cluster-split: pool all folds
    cluster_nn1_all = []
    n_cluster_folds = cluster_fold_ids.max() + 1
    for fold_id in range(n_cluster_folds):
        test_mask = cluster_fold_ids == fold_id
        train_mask = ~test_mask & (cluster_fold_ids >= 0)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            nn1 = D_sub[np.ix_(test_idx, train_idx)].min(axis=1)
            cluster_nn1_all.extend(nn1)

    bins = np.linspace(0, 1, 51)
    ax.hist(time_nn1_all, bins=bins, density=True, alpha=0.6, color="coral",
            label=f"Time-split (med={np.median(time_nn1_all):.3f})", edgecolor="white")
    ax.hist(cluster_nn1_all, bins=bins, density=True, alpha=0.6, color="steelblue",
            label=f"Cluster-split (med={np.median(cluster_nn1_all):.3f})", edgecolor="white")
    ax.set_xlabel("Test-to-train 1-NN Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Time-split vs cluster-split ({rep_ep})")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "time_vs_cluster_distances.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved time_vs_cluster_distances.png")
    logger.info(f"  Time-split pooled median: {np.median(time_nn1_all):.3f}")
    logger.info(f"  Cluster-split pooled median: {np.median(cluster_nn1_all):.3f}")
    plt.close("all")

    # ── 7. Save fold stats and assignments ────────────────────────────
    stats_df.to_csv(output_dir / "fold_stats.csv", index=False)
    logger.info(f"Saved fold_stats.csv ({len(stats_df)} rows)")

    folds_path = INTERIM_DATA_DIR / "time_cv_folds.parquet"
    folds_df.to_parquet(folds_path, index=False)
    logger.info(f"Saved {folds_path} ({len(folds_df)} rows)")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
