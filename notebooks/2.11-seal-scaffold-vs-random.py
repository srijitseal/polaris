#!/usr/bin/env python
"""Scaffold vs random split comparison (paper Fig S1, ref: Greg Landrum's blog).

Demonstrates that naive Bemis-Murcko scaffold splits produce similar performance
to random splits — they don't create meaningful distribution shift. Compares both
to cluster-based splitting (from 2.03) which produces genuine structural separation.

Usage:
    pixi run -e cheminformatics python notebooks/2.11-seal-scaffold-vs-random.py

Outputs:
    data/processed/2.11-seal-scaffold-vs-random/summary_metrics.csv
    data/processed/2.11-seal-scaffold-vs-random/aggregated_metrics.csv
    data/processed/2.11-seal-scaffold-vs-random/distance_stats.csv
    data/processed/2.11-seal-scaffold-vs-random/ks_distance_tests.csv
    data/processed/2.11-seal-scaffold-vs-random/scaffold_group_stats.csv
    data/processed/2.11-seal-scaffold-vs-random/scaffold_group_sizes.png
    data/processed/2.11-seal-scaffold-vs-random/metric_comparison.png
    data/processed/2.11-seal-scaffold-vs-random/distance_distributions.png
    data/processed/2.11-seal-scaffold-vs-random/strategy_summary.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from dimorphite_dl import protonate_smiles as dimorphite_protonate
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, ks_2samp, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.tuning import tune_xgboost
from polaris_generalization.visualization import DEFAULT_DPI, set_style

RDLogger.DisableLog("rdApp.*")

app = typer.Typer()

ENDPOINTS = [
    "LogD", "KSOL", "HLM CLint", "MLM CLint",
    "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux",
    "MPPB", "MBPB", "MGMB",
]

LOG_TRANSFORM_ENDPOINTS = [ep for ep in ENDPOINTS if ep.lower() != "logd"]

ENDPOINT_PH = {
    "LogD": 7.4,
    "KSOL": 7.4,
    "HLM CLint": 7.4,
    "MLM CLint": 7.4,
    "Caco-2 Permeability Papp A>B": 6.5,
    "Caco-2 Permeability Efflux": 6.5,
    "MPPB": 7.4,
    "MBPB": 7.4,
    "MGMB": 7.4,
}

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)

STRATEGY_COLORS = {"scaffold": "coral", "random": "forestgreen", "cluster": "steelblue"}
STRATEGY_ORDER = ["scaffold", "random", "cluster"]


def clip_and_log_transform(x: np.ndarray) -> np.ndarray:
    """Log-transform matching competition evaluation: log10(clip(x, 1e-10) + 1)."""
    return np.log10(np.clip(x, 1e-10, None) + 1)


def protonate_at_ph(smiles_list: list[str], ph: float) -> list[str]:
    """Protonate SMILES at given pH using dimorphite_dl."""
    protonated = []
    for smi in smiles_list:
        try:
            result = dimorphite_protonate(smi, ph_min=ph - 0.5, ph_max=ph + 0.5, max_variants=1)
            protonated.append(result[0] if result else smi)
        except Exception:
            protonated.append(smi)
    return protonated


def compute_ecfp4(smiles_list: list[str], nbits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute ECFP4 fingerprints and return as dense numpy array."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(nbits, dtype=np.uint8))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True)
            fps.append(np.array(fp, dtype=np.uint8))
    return np.vstack(fps)


def compute_rdkit_descriptors(smiles_list: list[str]) -> np.ndarray:
    """Compute full RDKit 2D descriptor suite (~200 descriptors)."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(DESCRIPTOR_NAMES))
        else:
            rows.append(list(DESC_CALC.CalcDescriptors(mol)))
    arr = np.array(rows, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def get_scaffold(smi: str) -> str:
    """Get generic Murcko scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return smi


def make_scaffold_folds(smiles: list[str], n_folds: int = 5) -> np.ndarray:
    """Assign molecules to folds by naive scaffold grouping.

    Groups molecules by generic Murcko scaffold, sorts scaffolds by frequency
    (largest first), then greedily assigns each scaffold group to the fold
    with fewest molecules so far.

    Returns:
        Array of fold assignments (0..n_folds-1), one per molecule.
    """
    # Compute scaffolds
    scaffolds = [get_scaffold(s) for s in smiles]

    # Group molecule indices by scaffold
    scaffold_groups: dict[str, list[int]] = {}
    for i, scaf in enumerate(scaffolds):
        scaffold_groups.setdefault(scaf, []).append(i)

    # Sort scaffolds by group size (largest first)
    sorted_scaffolds = sorted(scaffold_groups.keys(), key=lambda s: len(scaffold_groups[s]), reverse=True)

    # Greedy assignment: assign each scaffold to the fold with fewest molecules
    fold_sizes = [0] * n_folds
    fold_ids = np.full(len(smiles), -1, dtype=int)

    for scaf in sorted_scaffolds:
        indices = scaffold_groups[scaf]
        # Pick fold with smallest current size
        target_fold = int(np.argmin(fold_sizes))
        for idx in indices:
            fold_ids[idx] = target_fold
        fold_sizes[target_fold] += len(indices)

    return fold_ids


def make_random_folds(n_molecules: int, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Assign molecules to n_folds randomly."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, size=n_molecules)


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.11-seal-scaffold-vs-random", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
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
    dist_mol_names = npz["molecule_names"]
    name_to_dist_idx = {str(n): i for i, n in enumerate(dist_mol_names)}

    logger.info("Loading cluster CV folds")
    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    cluster_folds = cluster_folds[cluster_folds["repeat"] == 0]

    # ── 2. Train and evaluate per strategy per endpoint ───────────────
    metric_rows = []
    distance_rows = []  # per-molecule test-to-train 1-NN distances

    for ep in ENDPOINTS:
        ph = ENDPOINT_PH[ep]
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        n_mol = len(ep_df)

        if n_mol < 50:
            logger.warning(f"Skipping {ep}: only {n_mol} molecules")
            continue

        names = ep_df["Molecule Name"].values
        smiles = ep_df["SMILES"].tolist()

        # Sub-distance-matrix for this endpoint
        idx_in_full = np.array([name_to_dist_idx[n] for n in names])
        D_sub = dist_square[np.ix_(idx_in_full, idx_in_full)]

        # Activity values
        raw_values = ep_df[ep].values
        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_all = clip_and_log_transform(raw_values)
        else:
            y_all = raw_values

        # Compute features once per endpoint
        prot_smiles = protonate_at_ph(smiles, ph)
        ecfp = compute_ecfp4(prot_smiles)
        desc = compute_rdkit_descriptors(prot_smiles)
        variance = desc.var(axis=0)
        desc = desc[:, variance > 0]
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc)
        X_all = np.hstack([ecfp, desc_scaled])

        # Generate fold assignments for each strategy
        fold_assignments = {}

        # Scaffold split
        fold_assignments["scaffold"] = make_scaffold_folds(smiles, n_folds=5)

        # Random split
        fold_assignments["random"] = make_random_folds(n_mol, n_folds=5, seed=42)

        # Cluster split (from precomputed)
        ep_cluster = cluster_folds[cluster_folds["endpoint"] == ep]
        cluster_map = dict(zip(ep_cluster["Molecule Name"], ep_cluster["fold"]))
        fold_assignments["cluster"] = np.array([cluster_map.get(n, -1) for n in names])

        logger.info(f"  {ep} ({n_mol} molecules)")

        for strategy in STRATEGY_ORDER:
            fold_ids = fold_assignments[strategy]
            unique_folds = sorted(set(fold_ids[fold_ids >= 0]))

            for fold_id in unique_folds:
                test_mask = fold_ids == fold_id
                train_mask = (fold_ids >= 0) & ~test_mask

                X_tr = X_all[train_mask]
                y_tr = y_all[train_mask]
                X_te = X_all[test_mask]
                y_te = y_all[test_mask]

                if len(y_te) < 10 or len(y_tr) < 10:
                    continue

                # Train and predict
                cache_dir = INTERIM_DATA_DIR / "optuna_cache"
                model, _, _ = tune_xgboost(X_tr, y_tr, cache_dir=cache_dir, cache_key=f"{ep}_{strategy}_fold{fold_id}")
                y_pred = model.predict(X_te)

                # Competition metrics
                mae = mean_absolute_error(y_te, y_pred)
                r2 = r2_score(y_te, y_pred)
                sp_r, _ = spearmanr(y_te, y_pred)
                kt, _ = kendalltau(y_te, y_pred)
                baseline_mad = np.mean(np.abs(y_te - np.mean(y_te)))
                rae = mae / baseline_mad if baseline_mad > 0 else np.nan

                metric_rows.append({
                    "endpoint": ep,
                    "strategy": strategy,
                    "fold": fold_id,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "mae": mae,
                    "r2": r2,
                    "spearman_r": sp_r,
                    "kendall_tau": kt,
                    "rae": rae,
                })

                # Test-to-train 1-NN distances
                test_idx = np.where(test_mask)[0]
                train_idx = np.where(train_mask)[0]
                nn1_dists = D_sub[np.ix_(test_idx, train_idx)].min(axis=1)

                for d in nn1_dists:
                    distance_rows.append({
                        "endpoint": ep,
                        "strategy": strategy,
                        "fold": fold_id,
                        "nn1_distance": d,
                    })

            logger.info(
                f"    {strategy}: {len(unique_folds)} folds, "
                f"sizes={[int((fold_ids == f).sum()) for f in unique_folds]}"
            )

    # ── 3. Save per-fold metrics ──────────────────────────────────────
    summary_df = pd.DataFrame(metric_rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    logger.info(f"Saved summary_metrics.csv ({len(summary_df)} rows)")

    # ── 4. Aggregate metrics (mean ± std per endpoint per strategy) ───
    agg_rows = []
    for (ep, strategy), grp in summary_df.groupby(["endpoint", "strategy"]):
        row = {"endpoint": ep, "strategy": strategy, "n_folds": len(grp)}
        for col in ["mae", "r2", "spearman_r", "kendall_tau", "rae"]:
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(output_dir / "aggregated_metrics.csv", index=False)
    logger.info(f"Saved aggregated_metrics.csv ({len(agg_df)} rows)")

    # Log MA-RAE per strategy
    for strategy in STRATEGY_ORDER:
        strat_sub = summary_df[summary_df["strategy"] == strategy]
        ma_rae = strat_sub.groupby("fold")["rae"].mean().mean()
        logger.info(f"MA-RAE ({strategy}): {ma_rae:.3f}")

    # ── 5. Distance statistics ────────────────────────────────────────
    dist_df = pd.DataFrame(distance_rows)
    dist_stats_rows = []
    for strategy in STRATEGY_ORDER:
        s_dists = dist_df[dist_df["strategy"] == strategy]["nn1_distance"].values
        dist_stats_rows.append({
            "strategy": strategy,
            "mean": np.mean(s_dists),
            "median": np.median(s_dists),
            "std": np.std(s_dists),
            "q25": np.percentile(s_dists, 25),
            "q75": np.percentile(s_dists, 75),
            "q90": np.percentile(s_dists, 90),
            "n": len(s_dists),
        })

    dist_stats_df = pd.DataFrame(dist_stats_rows)
    dist_stats_df.to_csv(output_dir / "distance_stats.csv", index=False)
    logger.info("Saved distance_stats.csv")
    for _, row in dist_stats_df.iterrows():
        logger.info(
            f"  {row['strategy']}: median 1-NN={row['median']:.3f}, "
            f"mean={row['mean']:.3f}, q75={row['q75']:.3f}"
        )

    # ── 6. Figure A: Per-metric comparison (grouped bar charts) ───────
    active_endpoints = sorted(agg_df["endpoint"].unique())
    n_ep = len(active_endpoints)
    x = np.arange(n_ep)
    width = 0.25

    metric_plots = [
        ("mae", "MAE"),
        ("r2", "R²"),
        ("spearman_r", "Spearman ρ"),
        ("kendall_tau", "Kendall τ"),
        ("rae", "RAE"),
    ]

    for col, ylabel in metric_plots:
        fig, ax = plt.subplots(figsize=(14, 6))

        for i, strategy in enumerate(STRATEGY_ORDER):
            means = []
            stds = []
            for ep in active_endpoints:
                row = agg_df[(agg_df["endpoint"] == ep) & (agg_df["strategy"] == strategy)]
                if row.empty:
                    means.append(np.nan)
                    stds.append(0)
                else:
                    means.append(row[f"{col}_mean"].values[0])
                    stds.append(row[f"{col}_std"].values[0])

            ax.bar(x + i * width, means, width, yerr=stds, label=strategy,
                   color=STRATEGY_COLORS[strategy], edgecolor="white", capsize=3, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by endpoint — scaffold vs random vs cluster split")
        ax.legend(fontsize=8)
        if col == "rae":
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
        if col == "r2":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"metric_{col}.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved metric_{col}.png")
        plt.close("all")

    # Combined metric comparison figure (main Fig S1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for ax_idx, (col, ylabel) in enumerate(metric_plots):
        ax = axes[ax_idx]
        for i, strategy in enumerate(STRATEGY_ORDER):
            means = []
            stds = []
            for ep in active_endpoints:
                row = agg_df[(agg_df["endpoint"] == ep) & (agg_df["strategy"] == strategy)]
                if row.empty:
                    means.append(np.nan)
                    stds.append(0)
                else:
                    means.append(row[f"{col}_mean"].values[0])
                    stds.append(row[f"{col}_std"].values[0])

            ax.bar(x + i * width, means, width, yerr=stds, label=strategy,
                   color=STRATEGY_COLORS[strategy], edgecolor="white", capsize=2, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        if col == "rae":
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
        if col == "r2":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    axes[5].set_visible(False)

    fig.suptitle("Scaffold vs random vs cluster split — performance comparison", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "metric_comparison.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved metric_comparison.png")
    plt.close("all")

    # ── 7. Figure B: Distance distributions ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in STRATEGY_ORDER:
        s_dists = dist_df[dist_df["strategy"] == strategy]["nn1_distance"].values
        ax.hist(s_dists, bins=80, density=True, alpha=0.5, color=STRATEGY_COLORS[strategy],
                label=f"{strategy} (med={np.median(s_dists):.3f})", edgecolor="white")

    ax.set_xlabel("Test-to-train 1-NN Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title("Test-to-train structural distance distributions by split strategy")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "distance_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved distance_distributions.png")
    plt.close("all")

    # ── 8. Figure C: Strategy summary (MA-RAE bar chart) ──────────────
    fig, ax = plt.subplots(figsize=(6, 4))

    ma_raes = []
    for strategy in STRATEGY_ORDER:
        strat_sub = summary_df[summary_df["strategy"] == strategy]
        # MA-RAE: mean RAE across endpoints, then mean across folds
        ma_rae = strat_sub.groupby("fold")["rae"].mean().mean()
        ma_raes.append(ma_rae)

    bars = ax.bar(STRATEGY_ORDER, ma_raes,
                  color=[STRATEGY_COLORS[s] for s in STRATEGY_ORDER],
                  edgecolor="white", alpha=0.8)

    for bar, val in zip(bars, ma_raes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("MA-RAE")
    ax.set_title("Mean-across-endpoints RAE by split strategy")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_summary.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved strategy_summary.png")
    plt.close("all")

    # ── 9. KS tests on distance distributions ──────────────────────
    logger.info("Computing KS tests on 1-NN distance distributions")

    comparisons = [
        ("scaffold", "random"),
        ("cluster", "random"),
        ("cluster", "scaffold"),
    ]

    ks_rows = []
    for s1, s2 in comparisons:
        d1 = dist_df[dist_df["strategy"] == s1]["nn1_distance"].values
        d2 = dist_df[dist_df["strategy"] == s2]["nn1_distance"].values
        stat, pval = ks_2samp(d1, d2)
        ks_rows.append({
            "comparison": f"{s1} vs {s2}",
            "strategy_1": s1,
            "strategy_2": s2,
            "n_1": len(d1),
            "n_2": len(d2),
            "ks_statistic": stat,
            "p_value": pval,
            "median_1": np.median(d1),
            "median_2": np.median(d2),
            "mean_1": np.mean(d1),
            "mean_2": np.mean(d2),
        })
        logger.info(
            f"  {s1} vs {s2}: D = {stat:.4f}, p = {pval:.2e} "
            f"(n1={len(d1)}, n2={len(d2)})"
        )

    ks_df = pd.DataFrame(ks_rows)
    ks_df.to_csv(output_dir / "ks_distance_tests.csv", index=False)
    logger.info("Saved ks_distance_tests.csv")

    # ── 10. Scaffold group size statistics ─────────────────────────
    logger.info("Computing scaffold group size statistics")

    all_smiles = df["SMILES"].tolist()
    all_scaffolds = [get_scaffold(s) for s in all_smiles]

    scaffold_groups_all: dict[str, list[int]] = {}
    for i, scaf in enumerate(all_scaffolds):
        scaffold_groups_all.setdefault(scaf, []).append(i)

    group_sizes = np.array([len(v) for v in scaffold_groups_all.values()])
    n_unique = len(scaffold_groups_all)
    n_singletons = int(np.sum(group_sizes == 1))

    stats = {
        "n_molecules": len(all_smiles),
        "n_unique_scaffolds": n_unique,
        "n_singletons": n_singletons,
        "pct_singletons": 100 * n_singletons / n_unique,
        "median_group_size": float(np.median(group_sizes)),
        "mean_group_size": float(np.mean(group_sizes)),
        "max_group_size": int(np.max(group_sizes)),
    }
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "scaffold_group_stats.csv", index=False)
    logger.info(
        f"  {n_unique} unique scaffolds, {n_singletons} singletons "
        f"({stats['pct_singletons']:.1f}%), median size {stats['median_group_size']}"
    )

    # Histogram of scaffold group sizes
    fig, ax = plt.subplots(figsize=(8, 4))
    max_display = 20
    bins = np.arange(1, max_display + 2) - 0.5
    clipped = np.clip(group_sizes, 1, max_display)
    ax.hist(clipped, bins=bins, color="coral", edgecolor="white", alpha=0.8)
    tick_labels = [str(i) for i in range(1, max_display)] + [f"{max_display}+"]
    ax.set_xticks(range(1, max_display + 1))
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Scaffold group size")
    ax.set_ylabel("Number of scaffolds")
    ax.set_title(
        f"Murcko scaffold group size distribution "
        f"({n_singletons}/{n_unique} = {stats['pct_singletons']:.0f}% singletons)"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "scaffold_group_sizes.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved scaffold_group_sizes.png")
    plt.close("all")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
