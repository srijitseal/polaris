#!/usr/bin/env python
"""Target-distribution-based cross-validation splitting.

Sorts molecules by endpoint value and creates expanding-window CV folds
so the test set contains target values progressively beyond the training
range. Tests whether models can extrapolate in target space (lead
optimization scenario).

Usage:
    pixi run -e cheminformatics python notebooks/2.05-seal-target-split.py

Outputs:
    data/processed/2.05-seal-target-split/fold_sizes_grid.png
    data/processed/2.05-seal-target-split/fold_target_ranges.png
    data/processed/2.05-seal-target-split/fold_target_distributions.png
    data/processed/2.05-seal-target-split/fold_distance_distributions.png
    data/processed/2.05-seal-target-split/fold_stats.csv
    data/interim/target_cv_folds.parquet
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


def make_value_folds(values: np.ndarray, n_windows: int = 4) -> np.ndarray:
    """Assign expanding-window CV fold IDs based on sorted endpoint values.

    Identical to time-split logic but sorted by value instead of time.
    Fold k: train = all molecules with lower values, test = next chunk.

    Returns:
        Array of fold IDs. -1 = train-only (lowest chunk), 0..n_windows-1 = test windows.
    """
    n = len(values)
    sort_order = np.argsort(values)
    chunk = n // (n_windows + 1)

    fold_ids = np.full(n, -1, dtype=int)
    for k in range(n_windows):
        start = (k + 1) * chunk
        end = (k + 2) * chunk if k < n_windows - 1 else n
        fold_ids[sort_order[start:end]] = k

    return fold_ids


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.05-seal-target-split", help="Output directory"
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

    # ── 2. Per-endpoint target-split fold assignment ──────────────────
    all_fold_records = []
    fold_stats_rows = []

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        n_mol = len(ep_df)
        logger.info(f"Endpoint {ep}: {n_mol} molecules")

        values = ep_df[ep].values
        fold_ids = make_value_folds(values, n_windows)

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
            fold_vals = values[fold_mask]
            label = "train-only" if fold_id == -1 else f"fold {fold_id}"
            fold_stats_rows.append({
                "endpoint": ep,
                "fold": fold_id,
                "size": fold_size,
                "pct": 100 * fold_size / n_mol,
                "value_min": float(fold_vals.min()) if fold_size > 0 else 0,
                "value_max": float(fold_vals.max()) if fold_size > 0 else 0,
                "value_median": float(np.median(fold_vals)) if fold_size > 0 else 0,
            })
            if fold_id >= 0:
                logger.info(f"  {label}: n={fold_size}, values=[{fold_vals.min():.2f}-{fold_vals.max():.2f}]")

    folds_df = pd.DataFrame(all_fold_records)
    stats_df = pd.DataFrame(fold_stats_rows)
    logger.info(f"Total fold assignments: {len(folds_df)}")

    # ── 3. Fold size validation (3×3 grid) ────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    axes = axes.ravel()

    for i, ep in enumerate(ENDPOINTS):
        ax = axes[i]
        ep_stats = stats_df[(stats_df["endpoint"] == ep) & (stats_df["fold"] >= 0)]
        ax.bar(ep_stats["fold"], ep_stats["size"], color="forestgreen", edgecolor="white")
        ax.set_xlabel("Fold (value window)")
        ax.set_ylabel("Test size")
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.set_xticks(range(n_windows))

    fig.suptitle("Target-split fold sizes per endpoint (expanding window)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_sizes_grid.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_sizes_grid.png")
    plt.close("all")

    # ── 4. Target value ranges per fold (representative endpoint) ─────
    rep_ep = "LogD"
    ep_stats_rep = stats_df[stats_df["endpoint"] == rep_ep]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["gray"] + list(plt.cm.Greens(np.linspace(0.3, 0.9, n_windows)))
    for _, row in ep_stats_rep.iterrows():
        fold = row["fold"]
        label = "train-only" if fold == -1 else f"Fold {fold} test"
        y = fold + 1
        ax.barh(y, row["value_max"] - row["value_min"],
                left=row["value_min"], height=0.6,
                color=colors[fold + 1], edgecolor="white", label=label)
        ax.text(row["value_max"] + 0.1, y,
                f"n={row['size']}, med={row['value_median']:.2f}", va="center", fontsize=9)

    ax.set_yticks(range(n_windows + 1))
    ax.set_yticklabels(["Train-only"] + [f"Fold {k}" for k in range(n_windows)])
    ax.set_xlabel(f"{rep_ep} value")
    ax.set_title(f"Target value ranges per fold ({rep_ep}, expanding window)")
    fig.tight_layout()
    fig.savefig(output_dir / "fold_target_ranges.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_target_ranges.png")
    plt.close("all")

    # ── 5. Train vs test target distributions per fold ────────────────
    mask = df[rep_ep].notna().values
    ep_df = df[mask].copy()
    values = ep_df[rep_ep].values
    fold_ids = make_value_folds(values, n_windows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for fold_id in range(n_windows):
        ax = axes[fold_id]
        test_mask = fold_ids == fold_id
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= (fold_ids == k)

        train_vals = values[train_mask]
        test_vals = values[test_mask]

        bins = np.linspace(values.min(), values.max(), 41)
        ax.hist(train_vals, bins=bins, density=True, alpha=0.6, color="steelblue",
                label=f"Train (n={len(train_vals)})", edgecolor="white")
        ax.hist(test_vals, bins=bins, density=True, alpha=0.6, color="forestgreen",
                label=f"Test (n={len(test_vals)})", edgecolor="white")
        ax.set_xlabel(rep_ep)
        ax.set_ylabel("Density")
        ax.set_title(f"Fold {fold_id}")
        ax.legend(fontsize=8)

    fig.suptitle(f"Train vs test {rep_ep} distributions per fold", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_target_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_target_distributions.png")
    plt.close("all")

    # ── 6. Per-fold distance analysis ─────────────────────────────────
    idx_in_full = np.where(mask)[0]
    D_sub = dist_square[np.ix_(idx_in_full, idx_in_full)]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_plot = plt.cm.Greens(np.linspace(0.3, 0.9, n_windows))

    for fold_id in range(n_windows):
        test_mask = fold_ids == fold_id
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= (fold_ids == k)

        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]

        test_to_train = D_sub[np.ix_(test_idx, train_idx)]
        nn1 = test_to_train.min(axis=1)
        med = np.median(nn1)

        ax.hist(nn1, bins=50, density=True, alpha=0.5, color=colors_plot[fold_id],
                label=f"Fold {fold_id} (n={len(test_idx)}, med={med:.3f})",
                edgecolor="white")
        logger.info(f"  Fold {fold_id}: median 1-NN = {med:.3f}")

    ax.set_xlabel("Test-to-train 1-NN Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title(f"Target-split: test-to-train 1-NN per fold ({rep_ep})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fold_distance_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_distance_distributions.png")
    plt.close("all")

    # ── 7. Save fold stats and assignments ────────────────────────────
    stats_df.to_csv(output_dir / "fold_stats.csv", index=False)
    logger.info(f"Saved fold_stats.csv ({len(stats_df)} rows)")

    folds_path = INTERIM_DATA_DIR / "target_cv_folds.parquet"
    folds_df.to_parquet(folds_path, index=False)
    logger.info(f"Saved {folds_path} ({len(folds_df)} rows)")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
