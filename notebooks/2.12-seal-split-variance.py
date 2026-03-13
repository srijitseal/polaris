#!/usr/bin/env python
"""Split variance study (paper Fig S2, ref: Pat Walters' blog).

Demonstrates that single-split evaluation is misleading by running multiple
repeats of random (20 seeds) and cluster-based (5 repeats) CV, showing the
variance in performance metrics. Motivates need for repeated CV with confidence
intervals rather than single point estimates.

Usage:
    pixi run -e cheminformatics python notebooks/2.12-seal-split-variance.py

Outputs:
    data/processed/2.12-seal-split-variance/summary_metrics.csv
    data/processed/2.12-seal-split-variance/repeat_aggregates.csv
    data/processed/2.12-seal-split-variance/variance_summary.csv
    data/processed/2.12-seal-split-variance/rae_distributions.png
    data/processed/2.12-seal-split-variance/r2_distributions.png
    data/processed/2.12-seal-split-variance/ma_rae_distribution.png
    data/processed/2.12-seal-split-variance/variance_heatmap.png
    data/processed/2.12-seal-split-variance/single_split_danger.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from dimorphite_dl import protonate_smiles as dimorphite_protonate
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
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

STRATEGY_COLORS = {"random": "forestgreen", "cluster": "steelblue"}
STRATEGY_ORDER = ["random", "cluster"]


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
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
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


def make_random_folds(n_molecules: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """Assign molecules to n_folds randomly."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_folds, size=n_molecules)


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.12-seal-split-variance", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    n_random_repeats: int = typer.Option(20, help="Number of random split repeats"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    logger.info(f"Loaded {len(df)} molecules")

    logger.info("Loading cluster CV folds (all 5 repeats)")
    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    n_cluster_repeats = cluster_folds["repeat"].nunique()
    logger.info(f"Cluster folds: {len(cluster_folds)} rows, {n_cluster_repeats} repeats")

    # ── 2. Train and evaluate per strategy per repeat per endpoint ────
    metric_rows = []

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

        # Activity values
        raw_values = ep_df[ep].values
        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_all = clip_and_log_transform(raw_values)
        else:
            y_all = raw_values

        # Compute features once per endpoint (shared across all repeats)
        prot_smiles = protonate_at_ph(smiles, ph)
        ecfp = compute_ecfp4(prot_smiles)
        desc = compute_rdkit_descriptors(prot_smiles)
        variance = desc.var(axis=0)
        desc = desc[:, variance > 0]
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc)
        X_all = np.hstack([ecfp, desc_scaled])

        logger.info(f"  {ep} ({n_mol} molecules, {X_all.shape[1]} features)")

        # ── Random repeats ────────────────────────────────────────────
        for repeat in range(n_random_repeats):
            fold_ids = make_random_folds(n_mol, n_folds=5, seed=repeat)
            unique_folds = sorted(set(fold_ids))

            for fold_id in unique_folds:
                test_mask = fold_ids == fold_id
                train_mask = ~test_mask

                X_tr, y_tr = X_all[train_mask], y_all[train_mask]
                X_te, y_te = X_all[test_mask], y_all[test_mask]

                if len(y_te) < 10 or len(y_tr) < 10:
                    continue

                model = XGBRegressor(random_state=42, verbosity=0)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                mae = mean_absolute_error(y_te, y_pred)
                r2 = r2_score(y_te, y_pred)
                sp_r, _ = spearmanr(y_te, y_pred)
                kt, _ = kendalltau(y_te, y_pred)
                baseline_mad = np.mean(np.abs(y_te - np.mean(y_te)))
                rae = mae / baseline_mad if baseline_mad > 0 else np.nan

                metric_rows.append({
                    "endpoint": ep,
                    "strategy": "random",
                    "repeat": repeat,
                    "fold": fold_id,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "mae": mae,
                    "r2": r2,
                    "spearman_r": sp_r,
                    "kendall_tau": kt,
                    "rae": rae,
                })

        logger.info(f"    random: {n_random_repeats} repeats done")

        # ── Cluster repeats ───────────────────────────────────────────
        ep_cluster = cluster_folds[cluster_folds["endpoint"] == ep]

        for repeat in range(n_cluster_repeats):
            rep_folds = ep_cluster[ep_cluster["repeat"] == repeat]
            fold_map = dict(zip(rep_folds["Molecule Name"], rep_folds["fold"]))
            fold_ids = np.array([fold_map.get(n, -1) for n in names])
            unique_folds = sorted(set(fold_ids[fold_ids >= 0]))

            for fold_id in unique_folds:
                test_mask = fold_ids == fold_id
                train_mask = (fold_ids >= 0) & ~test_mask

                X_tr, y_tr = X_all[train_mask], y_all[train_mask]
                X_te, y_te = X_all[test_mask], y_all[test_mask]

                if len(y_te) < 10 or len(y_tr) < 10:
                    continue

                model = XGBRegressor(random_state=42, verbosity=0)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)

                mae = mean_absolute_error(y_te, y_pred)
                r2 = r2_score(y_te, y_pred)
                sp_r, _ = spearmanr(y_te, y_pred)
                kt, _ = kendalltau(y_te, y_pred)
                baseline_mad = np.mean(np.abs(y_te - np.mean(y_te)))
                rae = mae / baseline_mad if baseline_mad > 0 else np.nan

                metric_rows.append({
                    "endpoint": ep,
                    "strategy": "cluster",
                    "repeat": repeat,
                    "fold": fold_id,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    "mae": mae,
                    "r2": r2,
                    "spearman_r": sp_r,
                    "kendall_tau": kt,
                    "rae": rae,
                })

        logger.info(f"    cluster: {n_cluster_repeats} repeats done")

    # ── 3. Save per-fold metrics ──────────────────────────────────────
    summary_df = pd.DataFrame(metric_rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    logger.info(f"Saved summary_metrics.csv ({len(summary_df)} rows)")

    # ── 4. Per-repeat aggregates (mean across folds) ──────────────────
    repeat_agg = (
        summary_df.groupby(["endpoint", "strategy", "repeat"])
        .agg(
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
            spearman_r_mean=("spearman_r", "mean"),
            kendall_tau_mean=("kendall_tau", "mean"),
            rae_mean=("rae", "mean"),
        )
        .reset_index()
    )
    repeat_agg.to_csv(output_dir / "repeat_aggregates.csv", index=False)
    logger.info(f"Saved repeat_aggregates.csv ({len(repeat_agg)} rows)")

    # ── 5. Variance summary per endpoint per strategy ─────────────────
    var_rows = []
    for (ep, strategy), grp in repeat_agg.groupby(["endpoint", "strategy"]):
        row = {"endpoint": ep, "strategy": strategy, "n_repeats": len(grp)}
        for col in ["mae_mean", "r2_mean", "spearman_r_mean", "kendall_tau_mean", "rae_mean"]:
            vals = grp[col].values
            metric_name = col.replace("_mean", "")
            row[f"{metric_name}_mean"] = np.mean(vals)
            row[f"{metric_name}_std"] = np.std(vals)
            row[f"{metric_name}_min"] = np.min(vals)
            row[f"{metric_name}_max"] = np.max(vals)
            row[f"{metric_name}_range"] = np.max(vals) - np.min(vals)
            row[f"{metric_name}_cv"] = np.std(vals) / abs(np.mean(vals)) if np.mean(vals) != 0 else np.nan
            # 95% CI (using t-distribution approximation for small n)
            row[f"{metric_name}_ci95"] = 1.96 * np.std(vals) / np.sqrt(len(vals))
        var_rows.append(row)

    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(output_dir / "variance_summary.csv", index=False)
    logger.info("Saved variance_summary.csv")

    for _, row in var_df.iterrows():
        logger.info(
            f"  {row['endpoint']} ({row['strategy']}, n={row['n_repeats']}): "
            f"RAE={row['rae_mean']:.3f}±{row['rae_std']:.3f} "
            f"[{row['rae_min']:.3f}–{row['rae_max']:.3f}], "
            f"R²={row['r2_mean']:.3f}±{row['r2_std']:.3f}"
        )

    # ── 6. Figure A: Per-endpoint RAE distributions ───────────────────
    active_endpoints = sorted(repeat_agg["endpoint"].unique())
    n_ep = len(active_endpoints)
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)

    for metric_col, metric_label, fig_name in [
        ("rae_mean", "RAE", "rae_distributions"),
        ("r2_mean", "R²", "r2_distributions"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes).ravel()

        for ax_idx, ep in enumerate(active_endpoints):
            ax = axes[ax_idx]
            ep_data = repeat_agg[repeat_agg["endpoint"] == ep]

            for i, strategy in enumerate(STRATEGY_ORDER):
                strat_data = ep_data[ep_data["strategy"] == strategy][metric_col].values
                if len(strat_data) == 0:
                    continue
                bp = ax.boxplot(
                    strat_data,
                    positions=[i],
                    widths=0.6,
                    patch_artist=True,
                    showfliers=True,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(STRATEGY_COLORS[strategy])
                    patch.set_alpha(0.7)

                # Overlay individual points
                jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(strat_data))
                ax.scatter(
                    np.full(len(strat_data), i) + jitter,
                    strat_data,
                    color=STRATEGY_COLORS[strategy],
                    s=20, alpha=0.6, zorder=3, edgecolors="white", linewidths=0.5,
                )

                rng = strat_data.max() - strat_data.min()
                ax.text(
                    i, strat_data.max() + rng * 0.15,
                    f"range={rng:.3f}",
                    ha="center", fontsize=7, color=STRATEGY_COLORS[strategy],
                )

            ax.set_xticks(range(len(STRATEGY_ORDER)))
            ax.set_xticklabels(STRATEGY_ORDER, fontsize=8)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.set_title(ep, fontsize=10, fontweight="bold")
            if metric_col == "rae_mean":
                ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
            if metric_col == "r2_mean":
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        for i in range(n_ep, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"Per-repeat {metric_label} distributions — random ({n_random_repeats} repeats) "
            f"vs cluster ({n_cluster_repeats} repeats)",
            fontsize=14, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{fig_name}.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {fig_name}.png")
        plt.close("all")

    # ── 7. Figure B: MA-RAE distribution across repeats ───────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    ma_rae_data = {}
    for strategy in STRATEGY_ORDER:
        strat_agg = repeat_agg[repeat_agg["strategy"] == strategy]
        # MA-RAE per repeat: mean RAE across endpoints for each repeat
        ma_raes = strat_agg.groupby("repeat")["rae_mean"].mean().values
        ma_rae_data[strategy] = ma_raes

    positions = []
    for i, strategy in enumerate(STRATEGY_ORDER):
        vals = ma_rae_data[strategy]
        pos = i
        positions.append(pos)

        bp = ax.boxplot(
            vals, positions=[pos], widths=0.5, patch_artist=True, showfliers=True,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(STRATEGY_COLORS[strategy])
            patch.set_alpha(0.7)

        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter, vals,
            color=STRATEGY_COLORS[strategy], s=30, alpha=0.7,
            zorder=3, edgecolors="white", linewidths=0.5,
        )

        ax.text(
            pos, vals.max() + 0.02,
            f"{vals.mean():.3f}±{vals.std():.3f}\nrange={vals.max() - vals.min():.3f}",
            ha="center", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(STRATEGY_ORDER, fontsize=10)
    ax.set_ylabel("MA-RAE", fontsize=11)
    ax.set_title("MA-RAE distribution across repeated splits", fontsize=13)
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "ma_rae_distribution.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved ma_rae_distribution.png")
    plt.close("all")

    # ── 8. Figure C: Variance summary heatmap ─────────────────────────
    metric_cols = ["mae", "r2", "spearman_r", "rae"]
    metric_labels = ["MAE", "R²", "Spearman ρ", "RAE"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, strategy in enumerate(STRATEGY_ORDER):
        ax = axes[ax_idx]
        strat_var = var_df[var_df["strategy"] == strategy].set_index("endpoint")

        heatmap_data = pd.DataFrame(index=active_endpoints, columns=metric_labels, dtype=float)
        for col, label in zip(metric_cols, metric_labels):
            for ep in active_endpoints:
                if ep in strat_var.index:
                    heatmap_data.loc[ep, label] = strat_var.loc[ep, f"{col}_cv"]

        heatmap_data = heatmap_data.astype(float)
        sns.heatmap(
            heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Coefficient of variation"},
        )
        ax.set_title(f"{strategy} — CV across repeats", fontsize=11)
        ax.set_ylabel("")

    fig.suptitle("Metric variance (coefficient of variation) across repeated splits", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_heatmap.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved variance_heatmap.png")
    plt.close("all")

    # ── 9. Figure D: Single-split danger illustration ─────────────────
    # Find endpoint with highest RAE range for cluster strategy
    cluster_var = var_df[var_df["strategy"] == "cluster"]
    if not cluster_var.empty:
        worst_ep = cluster_var.loc[cluster_var["rae_range"].idxmax(), "endpoint"]
    else:
        worst_ep = active_endpoints[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, strategy in enumerate(STRATEGY_ORDER):
        ax = axes[ax_idx]
        ep_data = repeat_agg[
            (repeat_agg["endpoint"] == worst_ep) & (repeat_agg["strategy"] == strategy)
        ].sort_values("repeat")

        vals = ep_data["rae_mean"].values
        repeats = ep_data["repeat"].values

        ax.bar(
            range(len(vals)), vals,
            color=STRATEGY_COLORS[strategy], edgecolor="white", alpha=0.8,
        )

        # Highlight best and worst
        best_idx = np.argmin(vals)
        worst_idx = np.argmax(vals)
        ax.bar(best_idx, vals[best_idx], color="green", edgecolor="white", alpha=0.9)
        ax.bar(worst_idx, vals[worst_idx], color="red", edgecolor="white", alpha=0.9)

        ax.axhline(y=np.mean(vals), color="black", linestyle="--", alpha=0.5, label=f"mean={np.mean(vals):.3f}")
        ax.set_xlabel("Repeat")
        ax.set_ylabel("RAE")
        ax.set_title(f"{strategy} — {worst_ep}")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([str(r) for r in repeats], fontsize=8)
        ax.legend(fontsize=8)

        ax.text(
            best_idx, vals[best_idx] - 0.02, "best", ha="center", va="top",
            fontsize=8, fontweight="bold", color="white",
        )
        ax.text(
            worst_idx, vals[worst_idx] + 0.01, "worst", ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="red",
        )

    fig.suptitle(
        f"Cherry-picking danger: RAE per repeat for {worst_ep}",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "single_split_danger.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved single_split_danger.png")
    plt.close("all")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
