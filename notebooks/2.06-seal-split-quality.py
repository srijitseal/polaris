#!/usr/bin/env python
"""Split quality diagnostics across all three splitting strategies.

Compares cluster-split, time-split, and target-split on LogD using four
checks: fold sizes, target distributions, test-to-train distances, and
structural overlap. Includes UMAP visualization.

Usage:
    pixi run -e cheminformatics python notebooks/2.06-seal-split-quality.py

Outputs:
    data/processed/2.06-seal-split-quality/fold_sizes_comparison.png
    data/processed/2.06-seal-split-quality/target_distributions_by_strategy.png
    data/processed/2.06-seal-split-quality/distance_distributions_by_strategy.png
    data/processed/2.06-seal-split-quality/structural_overlap.png
    data/processed/2.06-seal-split-quality/umap_by_strategy.png
    data/processed/2.06-seal-split-quality/split_quality_summary.csv
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import umap
from loguru import logger
from scipy import stats
from scipy.spatial.distance import squareform

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()

ENDPOINT = "LogD"


def get_train_test_masks(fold_ids: np.ndarray, fold_id: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Get train and test boolean masks for a given fold."""
    test_mask = fold_ids == fold_id
    if strategy == "cluster":
        train_mask = (fold_ids != fold_id) & (fold_ids >= 0)
    else:
        # Expanding window: train = all folds before test
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= (fold_ids == k)
    return train_mask, test_mask


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.06-seal-split-quality", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    overlap_threshold: float = typer.Option(0.1, help="1-NN distance threshold for structural overlap"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading data")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    dist_square = squareform(npz["condensed"])

    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    time_folds = pd.read_parquet(INTERIM_DATA_DIR / "time_cv_folds.parquet")
    target_folds = pd.read_parquet(INTERIM_DATA_DIR / "target_cv_folds.parquet")

    # Filter to LogD
    mask = df[ENDPOINT].notna().values
    ep_df = df[mask].copy()
    ep_names = ep_df["Molecule Name"].values
    ep_values = ep_df[ENDPOINT].values
    idx_in_full = np.where(mask)[0]
    D_sub = dist_square[np.ix_(idx_in_full, idx_in_full)]
    n_mol = len(ep_df)
    logger.info(f"{ENDPOINT}: {n_mol} molecules")

    # Build fold ID arrays per strategy
    strategies = {}
    for name, folds_df, repeat_filter in [
        ("cluster", cluster_folds, True),
        ("time", time_folds, True),
        ("target", target_folds, True),
    ]:
        sub = folds_df[folds_df["endpoint"] == ENDPOINT]
        if repeat_filter and "repeat" in sub.columns:
            sub = sub[sub["repeat"] == 0]
        fold_map = dict(zip(sub["Molecule Name"], sub["fold"]))
        fold_ids = np.array([fold_map.get(n, -99) for n in ep_names])
        n_folds = max(f for f in fold_ids if f >= 0) + 1
        strategies[name] = {"fold_ids": fold_ids, "n_folds": n_folds}
        logger.info(f"  {name}: {n_folds} folds, assigned={np.sum(fold_ids >= -1)}")

    # ── 2. Check 1: Fold sizes ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x_offset = 0
    colors = {"cluster": "steelblue", "time": "coral", "target": "forestgreen"}
    tick_positions = []
    tick_labels = []

    for strat_name, strat in strategies.items():
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]
        for fold_id in range(n_folds):
            _, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)
            size = test_mask.sum()
            ax.bar(x_offset, size, color=colors[strat_name], edgecolor="white", width=0.8)
            tick_positions.append(x_offset)
            tick_labels.append(f"{strat_name[0].upper()}{fold_id}")
            x_offset += 1
        x_offset += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
    ax.set_ylabel("Test fold size")
    ax.set_title(f"Fold sizes across strategies ({ENDPOINT})")
    ax.axhline(y=n_mol * 0.1, color="red", linestyle="--", alpha=0.3, label="10% threshold")
    ax.axhline(y=n_mol * 0.4, color="red", linestyle="--", alpha=0.3, label="40% threshold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fold_sizes_comparison.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_sizes_comparison.png")
    plt.close("all")

    # ── 3. Check 2: Target distribution per fold ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (strat_name, strat) in zip(axes, strategies.items()):
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]

        box_data = []
        box_labels = []
        for fold_id in range(n_folds):
            train_mask, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)
            box_data.append(ep_values[test_mask])
            box_labels.append(f"F{fold_id} test")

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[strat_name])
            patch.set_alpha(0.6)
        ax.set_ylabel(ENDPOINT)
        ax.set_title(f"{strat_name.capitalize()}-split")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle(f"Target distributions per fold ({ENDPOINT})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "target_distributions_by_strategy.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved target_distributions_by_strategy.png")
    plt.close("all")

    # ── 4. Check 3: Test-to-train 1-NN distances ─────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    summary_rows = []

    for ax, (strat_name, strat) in zip(axes, strategies.items()):
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]
        fold_colors = plt.cm.Set2(np.linspace(0, 1, n_folds))

        for fold_id in range(n_folds):
            train_mask, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            if len(test_idx) == 0 or len(train_idx) == 0:
                continue

            nn1 = D_sub[np.ix_(test_idx, train_idx)].min(axis=1)
            med = np.median(nn1)
            overlap_frac = np.mean(nn1 < overlap_threshold)

            # KS test on target values
            ks_stat, ks_p = stats.ks_2samp(ep_values[train_mask], ep_values[test_mask])

            summary_rows.append({
                "strategy": strat_name,
                "fold": fold_id,
                "test_size": int(test_mask.sum()),
                "train_size": int(train_mask.sum()),
                "median_1nn": float(med),
                "overlap_frac": float(overlap_frac),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            })

            ax.hist(nn1, bins=50, density=True, alpha=0.5, color=fold_colors[fold_id],
                    label=f"F{fold_id} (med={med:.3f})", edgecolor="white")

        ax.set_xlabel("Test-to-train 1-NN distance")
        ax.set_ylabel("Density")
        ax.set_title(f"{strat_name.capitalize()}-split")
        ax.legend(fontsize=7)

    fig.suptitle(f"Test-to-train 1-NN distances ({ENDPOINT})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_distributions_by_strategy.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved distance_distributions_by_strategy.png")
    plt.close("all")

    # ── 5. Check 4: Structural overlap ────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x_offset = 0
    tick_positions = []
    tick_labels_overlap = []

    for strat_name in strategies:
        strat_rows = summary_df[summary_df["strategy"] == strat_name]
        for _, row in strat_rows.iterrows():
            ax.bar(x_offset, row["overlap_frac"] * 100, color=colors[strat_name], edgecolor="white", width=0.8)
            tick_positions.append(x_offset)
            tick_labels_overlap.append(f"{strat_name[0].upper()}{int(row['fold'])}")
            x_offset += 1
        x_offset += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_overlap, fontsize=8, rotation=45)
    ax.set_ylabel(f"% test molecules with 1-NN < {overlap_threshold}")
    ax.set_title(f"Structural overlap: near-duplicates in train ({ENDPOINT})")

    legend_elements = [Patch(facecolor=c, label=n) for n, c in colors.items()]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(output_dir / "structural_overlap.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved structural_overlap.png")
    plt.close("all")

    # Log summary
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['strategy']} fold {row['fold']}: "
            f"test={row['test_size']}, med_1nn={row['median_1nn']:.3f}, "
            f"overlap={row['overlap_frac']:.1%}, KS={row['ks_statistic']:.3f}"
        )

    # ── 6. UMAP embedding ────────────────────────────────────────────
    logger.info("Computing UMAP embedding")
    reducer = umap.UMAP(metric="precomputed", n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(D_sub)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (strat_name, strat) in zip(axes, strategies.items()):
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]
        fold_colors_map = plt.cm.Set2(np.linspace(0, 1, max(n_folds, 5)))

        # Plot train-only as gray first
        train_only = fold_ids == -1
        if train_only.any():
            ax.scatter(embedding[train_only, 0], embedding[train_only, 1],
                       c="lightgray", s=3, alpha=0.3, label="train-only", rasterized=True)

        for fold_id in range(n_folds):
            fold_mask = fold_ids == fold_id
            if fold_mask.any():
                ax.scatter(embedding[fold_mask, 0], embedding[fold_mask, 1],
                           c=[fold_colors_map[fold_id]], s=5, alpha=0.5,
                           label=f"Fold {fold_id}", rasterized=True)

        ax.set_title(f"{strat_name.capitalize()}-split")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=7, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"UMAP of fold assignments ({ENDPOINT})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "umap_by_strategy.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved umap_by_strategy.png")
    plt.close("all")

    # ── 7. Save summary ──────────────────────────────────────────────
    summary_df.to_csv(output_dir / "split_quality_summary.csv", index=False)
    logger.info(f"Saved split_quality_summary.csv ({len(summary_df)} rows)")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
