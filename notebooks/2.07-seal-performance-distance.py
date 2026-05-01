#!/usr/bin/env python
"""Performance-over-distance curves for all endpoints across all split strategies.

Trains a model (XGBoost or Chemprop D-MPNN) on pre-saved CV folds (cluster, time,
target — repeat 0), then plots how RMSE degrades with distance from training data.
Molecules are protonated at assay-relevant pH using dimorphite_dl before feature
computation. For endpoints not already on log scale (everything except LogD),
targets are log-transformed for both training and evaluation, matching the
OpenADMET competition protocol.

Usage:
    pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py
    pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py --model chemprop
    pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py --combined

Outputs (under data/processed/2.07-seal-performance-distance/{model}/):
    predictions.parquet
    per_fold_performance.csv
    performance_distance_summary.csv
    overall_mae.png, overall_r2.png, overall_rae.png, overall_spearman_r.png
    performance_over_distance.png
    performance_over_target_distance.png
    performance_over_distance_v2.png

Combined outputs (under …/combined/):
    performance_over_distance_combined.png
    performance_over_target_distance_combined.png
    metrics_comparison.csv
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
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from scipy.spatial.distance import squareform
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from polaris_generalization.chemprop_utils import train_chemprop
from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.tuning import tune_xgboost
from polaris_generalization.visualization import (
    DEFAULT_DPI,
    MODEL_LABELS,
    plot_model_comparison_bars,
    set_style,
)

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

STRATEGIES = {
    "cluster": {"file": "cluster_cv_folds.parquet", "n_folds": 5},
    "time": {"file": "time_cv_folds.parquet", "n_folds": 4},
    "target": {"file": "target_cv_folds.parquet", "n_folds": 4},
}

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


def clip_and_log_transform(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 1e-10, None) + 1)


def protonate_at_ph(smiles_list: list[str], ph: float) -> list[str]:
    protonated = []
    for smi in smiles_list:
        try:
            result = dimorphite_protonate(smi, ph_min=ph - 0.5, ph_max=ph + 0.5, max_variants=1)
            protonated.append(result[0] if result else smi)
        except Exception:
            protonated.append(smi)
    return protonated


def compute_ecfp4(smiles_list: list[str], nbits: int = 2048, radius: int = 2) -> np.ndarray:
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


def get_train_test_masks(fold_ids: np.ndarray, fold_id: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    test_mask = fold_ids == fold_id
    if strategy == "cluster":
        train_mask = (fold_ids != fold_id) & (fold_ids >= 0)
    else:
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= fold_ids == k
    return train_mask, test_mask


def make_sliding_bins(distances: np.ndarray) -> list[tuple[float, float]]:
    q1, q3 = np.percentile(distances, [25, 75])
    iqr = q3 - q1
    d_min = max(0, q1 - 1.5 * iqr)
    d_max = q3 + 1.5 * iqr
    binwidth = (d_max - d_min) / 5
    stepsize = binwidth / 20
    bins = []
    i = 0
    while True:
        lo = d_min + stepsize * i
        hi = lo + binwidth
        if lo > d_max:
            break
        bins.append((lo, hi))
        i += 1
    return bins


def bin_performance(distances: np.ndarray, errors_sq: np.ndarray,
                    bins: list[tuple[float, float]], min_count: int = 25) -> dict:
    results = {}
    for lo, hi in bins:
        mask = (distances >= lo) & (distances < hi)
        center = (lo + hi) / 2
        if mask.sum() >= min_count:
            results[center] = np.sqrt(errors_sq[mask].mean())
        else:
            results[center] = np.nan
    return results


def _migrate_flat_outputs(output_dir: Path) -> None:
    """Move pre-refactor flat XGBoost outputs into xgboost/ subdir."""
    xgb_dir = output_dir / "xgboost"
    if (output_dir / "predictions.parquet").exists() and not (xgb_dir / "predictions.parquet").exists():
        xgb_dir.mkdir(parents=True, exist_ok=True)
        flat_files = [
            "predictions.parquet",
            "per_fold_performance.csv",
            "performance_distance_summary.csv",
            "overall_mae.png",
            "overall_r2.png",
            "overall_rae.png",
            "overall_spearman_r.png",
            "performance_over_distance.png",
            "performance_over_target_distance.png",
            "performance_over_distance_v2.png",
        ]
        for fname in flat_files:
            src = output_dir / fname
            if src.exists():
                src.rename(xgb_dir / fname)
        logger.info("Migrated flat XGBoost outputs → xgboost/")


def _generate_figures(
    all_results: list[dict],
    perf_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model_dir: Path,
    model: str,
    dpi: int,
) -> None:
    """Save all per-model figures to model_dir."""
    strat_colors = {"cluster": "steelblue", "time": "coral", "target": "forestgreen"}
    model_label = MODEL_LABELS.get(model, model)
    active_endpoints = sorted(perf_df["endpoint"].unique())
    n_ep = len(active_endpoints)

    # ── Bar charts: mae, spearman_r, rae ────────────────────────────────
    for metric, ylabel in [("mae", "MAE"), ("spearman_r", "Spearman ρ"), ("rae", "RAE")]:
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(n_ep)
        width = 0.25
        for i, strat_name in enumerate(STRATEGIES):
            strat_sub = perf_df[perf_df["strategy"] == strat_name]
            means, stds = [], []
            for ep in active_endpoints:
                ep_vals = strat_sub[strat_sub["endpoint"] == ep][metric]
                means.append(ep_vals.mean())
                stds.append(ep_vals.std())
            ax.bar(x + i * width, means, width, yerr=stds, label=strat_name,
                   color=strat_colors[strat_name], edgecolor="white", capsize=3, alpha=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by endpoint and split strategy — {model_label}")
        ax.legend(fontsize=8)
        if metric == "rae":
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(model_dir / f"overall_{metric}.png", dpi=dpi, bbox_inches="tight")
        plt.close("all")
    logger.info("Saved overall bar charts")

    # ── R² split plot ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [2, 1]})
    x = np.arange(n_ep)
    width = 0.35
    for i, strat_name in enumerate(["cluster", "time"]):
        strat_sub = perf_df[perf_df["strategy"] == strat_name]
        means, stds = [], []
        for ep in active_endpoints:
            ep_vals = strat_sub[strat_sub["endpoint"] == ep]["r2"]
            means.append(ep_vals.mean())
            stds.append(ep_vals.std())
        ax1.bar(x + i * width, means, width, yerr=stds, label=strat_name,
                color=strat_colors[strat_name], edgecolor="white", capsize=3, alpha=0.8)
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("R²")
    ax1.set_title("R² — Cluster & Time splits")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax1.legend(fontsize=8)

    target_sub = perf_df[perf_df["strategy"] == "target"]
    target_means = [target_sub[target_sub["endpoint"] == ep]["r2"].mean() for ep in active_endpoints]
    ax2.barh(np.arange(n_ep), target_means, color=strat_colors["target"], alpha=0.8)
    ax2.set_yticks(np.arange(n_ep))
    ax2.set_yticklabels(active_endpoints, fontsize=8)
    ax2.set_xlabel("R²")
    ax2.set_title("R² — Target split (note scale)")
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle(f"R² by endpoint and split strategy — {model_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(model_dir / "overall_r2.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved overall_r2.png")

    # ── Performance over structural distance ─────────────────────────────
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)
    summary_rows = []

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_ep == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        ep_results = [r for r in all_results if r["endpoint"] == ep]
        all_dists = np.concatenate([r["struct_dist"] for r in ep_results])
        bins = make_sliding_bins(all_dists)

        for strat_name in STRATEGIES:
            strat_results = [r for r in ep_results if r["strategy"] == strat_name]
            if not strat_results:
                continue
            fold_curves = [bin_performance(r["struct_dist"], r["sq_errors"], bins) for r in strat_results]
            centers = sorted(fold_curves[0].keys())
            values = np.array([[c[center] for center in centers] for c in fold_curves])
            median_vals = np.nanmedian(values, axis=0)
            lo_vals = np.nanpercentile(values, 10, axis=0)
            hi_vals = np.nanpercentile(values, 90, axis=0)
            valid = ~np.isnan(median_vals)
            c_arr = np.array(centers)
            color = strat_colors[strat_name]
            ax.plot(c_arr[valid], median_vals[valid], color=color, label=strat_name, linewidth=1.5)
            ax.fill_between(c_arr[valid], lo_vals[valid], hi_vals[valid], color=color, alpha=0.1)
            valid_medians = median_vals[valid]
            if len(valid_medians) > 0:
                summary_rows.append({
                    "endpoint": ep,
                    "strategy": strat_name,
                    "overall_rmse": float(np.median([r["overall_rmse"] for r in strat_results])),
                    "rmse_closest_bin": float(valid_medians[0]),
                    "rmse_farthest_bin": float(valid_medians[-1]),
                    "degradation_ratio": float(valid_medians[-1] / valid_medians[0]) if valid_medians[0] > 0 else np.nan,
                })
        ax.set_xlabel("Test-to-train 1-NN distance (Tanimoto)")
        ax.set_ylabel("RMSE")
        ax.set_title(ep)
        ax.legend(fontsize=7)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f"Performance over structural distance — {model_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(model_dir / "performance_over_distance.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved performance_over_distance.png")

    # ── Performance over target-space distance ───────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n_ep == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        ep_results = [r for r in all_results if r["endpoint"] == ep]
        all_dists = np.concatenate([r["target_dist"] for r in ep_results])
        bins = make_sliding_bins(all_dists)

        for strat_name in STRATEGIES:
            strat_results = [r for r in ep_results if r["strategy"] == strat_name]
            if not strat_results:
                continue
            fold_curves = [bin_performance(r["target_dist"], r["sq_errors"], bins) for r in strat_results]
            centers = sorted(fold_curves[0].keys())
            values = np.array([[c[center] for center in centers] for c in fold_curves])
            median_vals = np.nanmedian(values, axis=0)
            lo_vals = np.nanpercentile(values, 10, axis=0)
            hi_vals = np.nanpercentile(values, 90, axis=0)
            valid = ~np.isnan(median_vals)
            c_arr = np.array(centers)
            color = strat_colors[strat_name]
            ax.plot(c_arr[valid], median_vals[valid], color=color, label=strat_name, linewidth=1.5)
            ax.fill_between(c_arr[valid], lo_vals[valid], hi_vals[valid], color=color, alpha=0.1)
        ax.set_xlabel("Test-to-train 1-NN distance (target space)")
        ax.set_ylabel("RMSE")
        ax.set_title(ep)
        ax.legend(fontsize=7)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(f"Performance over target-space distance — {model_label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(model_dir / "performance_over_target_distance.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved performance_over_target_distance.png")

    # ── High-res v2 distance plot (from pred_df) ─────────────────────────
    plt.rcParams.update({"font.size": 16, "axes.labelsize": 18, "axes.titlesize": 20,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14})
    ep_list = sorted(pred_df["endpoint"].unique())
    n2 = len(ep_list)
    ncols2 = 2
    nrows2 = (n2 + ncols2 - 1) // ncols2
    fig, axes = plt.subplots(nrows2, ncols2, figsize=(9 * ncols2, 6 * nrows2))
    if n2 == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax_idx, ep in enumerate(ep_list):
        ax = axes[ax_idx]
        ep_pred = pred_df[pred_df["endpoint"] == ep]
        all_dists = ep_pred["struct_1nn"].values
        bins = make_sliding_bins(all_dists)

        for strat_name in ["cluster", "time", "target"]:
            strat_pred = ep_pred[ep_pred["strategy"] == strat_name]
            if strat_pred.empty:
                continue
            fold_curves = []
            for fold_id in sorted(strat_pred["fold"].unique()):
                fold_pred = strat_pred[strat_pred["fold"] == fold_id]
                curve = bin_performance(fold_pred["struct_1nn"].values,
                                        (fold_pred["y_pred"] - fold_pred["y_true"]).values ** 2, bins)
                fold_curves.append(curve)
            centers = sorted(fold_curves[0].keys())
            values = np.array([[c[center] for center in centers] for c in fold_curves])
            median_vals = np.nanmedian(values, axis=0)
            lo_vals = np.nanpercentile(values, 10, axis=0)
            hi_vals = np.nanpercentile(values, 90, axis=0)
            valid = ~np.isnan(median_vals)
            c_arr = np.array(centers)
            abs_errors = np.abs(strat_pred["y_pred"] - strat_pred["y_true"]).values
            rho, _ = spearmanr(strat_pred["struct_1nn"].values, abs_errors)
            color = strat_colors[strat_name]
            ax.plot(c_arr[valid], median_vals[valid], color=color,
                    label=f"{strat_name} (ρ={rho:.2f})", linewidth=2)
            ax.fill_between(c_arr[valid], lo_vals[valid], hi_vals[valid], color=color, alpha=0.1)
        ax.set_xlabel("1-NN Jaccard distance")
        ax.set_ylabel("RMSE")
        ax.set_title(ep, fontweight="bold")
        ax.legend()

    for i in range(n2, len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()
    fig.savefig(model_dir / "performance_over_distance_v2.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved performance_over_distance_v2.png")

    # ── Save summary CSV ─────────────────────────────────────────────────
    pd.DataFrame(summary_rows).to_csv(model_dir / "performance_distance_summary.csv", index=False)
    logger.info("Saved performance_distance_summary.csv")

    set_style()  # reset rcParams


def _generate_combined_figures(output_dir: Path, dpi: int) -> None:
    """Overlay XGBoost (solid) and Chemprop (dashed) distance curves on same axes."""
    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    model_preds = {}
    for model in ["xgboost", "chemprop"]:
        pred_file = output_dir / model / "predictions.parquet"
        if not pred_file.exists():
            logger.warning(f"Missing {pred_file}; skipping combined figures")
            return
        model_preds[model] = pd.read_parquet(pred_file)

    strat_colors = {"cluster": "steelblue", "time": "coral", "target": "forestgreen"}
    model_linestyles = {"xgboost": "-", "chemprop": "--"}
    active_endpoints = sorted(model_preds["xgboost"]["endpoint"].unique())
    n_ep = len(active_endpoints)
    ncols = 2
    nrows = (n_ep + ncols - 1) // ncols

    for dist_col, xlabel, fname in [
        ("struct_1nn", "1-NN Jaccard distance", "performance_over_distance_combined.png"),
        ("target_1nn", "Test-to-train 1-NN distance (target space)", "performance_over_target_distance_combined.png"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows))
        if n_ep == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for ax_idx, ep in enumerate(active_endpoints):
            ax = axes[ax_idx]
            all_dists = model_preds["xgboost"][model_preds["xgboost"]["endpoint"] == ep][dist_col].values
            bins = make_sliding_bins(all_dists)

            for model, pred_df in model_preds.items():
                ep_pred = pred_df[pred_df["endpoint"] == ep]
                model_label = MODEL_LABELS.get(model, model)
                ls = model_linestyles[model]
                for strat_name in ["cluster", "time", "target"]:
                    strat_pred = ep_pred[ep_pred["strategy"] == strat_name]
                    if strat_pred.empty:
                        continue
                    fold_curves = []
                    for fold_id in sorted(strat_pred["fold"].unique()):
                        fold_pred = strat_pred[strat_pred["fold"] == fold_id]
                        curve = bin_performance(fold_pred[dist_col].values,
                                                (fold_pred["y_pred"] - fold_pred["y_true"]).values ** 2, bins)
                        fold_curves.append(curve)
                    centers = sorted(fold_curves[0].keys())
                    values = np.array([[c[center] for center in centers] for c in fold_curves])
                    median_vals = np.nanmedian(values, axis=0)
                    valid = ~np.isnan(median_vals)
                    c_arr = np.array(centers)
                    ax.plot(c_arr[valid], median_vals[valid], color=strat_colors[strat_name],
                            linestyle=ls, linewidth=1.8,
                            label=f"{strat_name} ({model_label})")

            ax.set_xlabel(xlabel)
            ax.set_ylabel("RMSE")
            ax.set_title(ep, fontweight="bold")
            ax.legend(fontsize=6)

        for i in range(n_ep, len(axes)):
            axes[i].set_visible(False)
        fig.tight_layout()
        fig.savefig(combined_dir / fname, dpi=dpi, bbox_inches="tight")
        plt.close("all")
        logger.info(f"Saved {fname}")

    # ── Degradation-ratio comparison bar chart ───────────────────────────
    model_metrics = {}
    for model in ["xgboost", "chemprop"]:
        summary_file = output_dir / model / "performance_distance_summary.csv"
        if summary_file.exists():
            model_metrics[model] = pd.read_csv(summary_file)

    if len(model_metrics) == 2:
        plot_model_comparison_bars(
            data_by_model=model_metrics,
            endpoint_col="endpoint",
            metric_col="degradation_ratio",
            ylabel="Degradation ratio (farthest / closest bin RMSE)",
            title="Structural-distance degradation ratio: XGBoost vs Chemprop",
            output_path=combined_dir / "degradation_ratio_comparison.png",
            dpi=dpi,
        )
        # Save merged metrics
        rows = []
        for model, df in model_metrics.items():
            df = df.copy()
            df["model"] = model
            rows.append(df)
        pd.concat(rows).to_csv(combined_dir / "metrics_comparison.csv", index=False)
        logger.info("Saved metrics_comparison.csv")


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.07-seal-performance-distance", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    model: str = typer.Option("xgboost", help="Model backend: xgboost or chemprop"),
    combined: bool = typer.Option(False, help="Generate combined XGBoost+Chemprop figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    _migrate_flat_outputs(output_dir)

    if combined:
        logger.info("Generating combined figures")
        _generate_combined_figures(output_dir, dpi)
        return

    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    if (model_dir / "predictions.parquet").exists() and (model_dir / "performance_distance_summary.csv").exists():
        logger.info(f"Found existing outputs for {model}, skipping training")
        pred_df = pd.read_parquet(model_dir / "predictions.parquet")
        perf_df = pd.read_csv(model_dir / "per_fold_performance.csv")
        all_results = []
        for _, row in pred_df.iterrows():
            pass
        _generate_figures(all_results, perf_df, pred_df, model_dir, model, dpi)
        return

    # ── 1. Load data ──────────────────────────────────────────────────────
    logger.info("Loading data")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    dist_square = squareform(npz["condensed"])

    fold_dfs = {}
    for strat_name, strat_info in STRATEGIES.items():
        folds = pd.read_parquet(INTERIM_DATA_DIR / strat_info["file"])
        fold_dfs[strat_name] = folds[folds["repeat"] == 0]
        logger.info(f"  {strat_name}: {len(fold_dfs[strat_name])} rows (repeat=0)")

    smiles_list = df["SMILES"].tolist()
    mol_names = df["Molecule Name"].values

    # ── 2. Protonation and features (XGBoost only) ────────────────────────
    unique_phs = sorted(set(ENDPOINT_PH.values()))
    features_by_ph: dict[float, np.ndarray] = {}
    prot_smiles_by_ph: dict[float, list[str]] = {}

    for ph in unique_phs:
        logger.info(f"Protonating at pH {ph}")
        protonated = protonate_at_ph(smiles_list, ph)
        prot_smiles_by_ph[ph] = protonated

        if model == "xgboost":
            logger.info(f"Computing features at pH {ph}")
            ecfp = compute_ecfp4(protonated)
            desc = compute_rdkit_descriptors(protonated)
            variance = desc.var(axis=0)
            desc = desc[:, variance > 0]
            desc_scaled = StandardScaler().fit_transform(desc)
            X = np.hstack([ecfp, desc_scaled])
            logger.info(f"  pH {ph}: {X.shape[1]} features")
            features_by_ph[ph] = X

    # ── 3. Train models and collect predictions ───────────────────────────
    all_results = []
    prediction_rows = []

    total_iters = sum(
        s["n_folds"] for ep in ENDPOINTS if df[ep].notna().sum() >= 50 for s in STRATEGIES.values()
    )
    pbar = tqdm(total=total_iters, desc=f"Training {model}", unit="fold")

    chemprop_cache_dir = INTERIM_DATA_DIR / "chemprop_pred_cache"

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        if mask.sum() < 50:
            logger.warning(f"Skipping {ep}: only {mask.sum()} molecules")
            continue

        ph = ENDPOINT_PH[ep]
        ep_idx = np.where(mask)[0]
        ep_names = mol_names[mask]
        ep_values_raw = df[ep].values[mask]
        ep_values = clip_and_log_transform(ep_values_raw) if ep in LOG_TRANSFORM_ENDPOINTS else ep_values_raw

        D_ep = dist_square[np.ix_(ep_idx, ep_idx)]

        if model == "xgboost":
            X_ep = features_by_ph[ph][mask]
        else:
            ep_prot_smiles = [prot_smiles_by_ph[ph][i] for i in ep_idx]

        for strat_name, strat_info in STRATEGIES.items():
            strat_folds = fold_dfs[strat_name]
            ep_folds = strat_folds[strat_folds["endpoint"] == ep]
            fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
            fold_ids = np.array([fold_map.get(n, -99) for n in ep_names])
            n_folds = strat_info["n_folds"]

            for fold_id in range(n_folds):
                train_mask_f, test_mask_f = get_train_test_masks(fold_ids, fold_id, strat_name)
                test_idx = np.where(test_mask_f)[0]
                train_idx = np.where(train_mask_f)[0]
                if len(test_idx) < 10 or len(train_idx) < 10:
                    pbar.update(1)
                    continue

                y_train = ep_values[train_idx]
                y_test = ep_values[test_idx]
                struct_nn1 = D_ep[np.ix_(test_idx, train_idx)].min(axis=1)
                target_nn1 = np.abs(y_test[:, None] - y_train[None, :]).min(axis=1)
                test_names = ep_names[test_idx]

                if model == "xgboost":
                    cache_dir = INTERIM_DATA_DIR / "optuna_cache"
                    fitted_model, _, _ = tune_xgboost(
                        X_ep[train_idx], y_train,
                        cache_dir=cache_dir,
                        cache_key=f"{ep}_{strat_name}_fold{fold_id}",
                    )
                    y_pred = fitted_model.predict(X_ep[test_idx])
                else:
                    train_smiles = [ep_prot_smiles[i] for i in train_idx]
                    test_smiles = [ep_prot_smiles[i] for i in test_idx]
                    y_pred = train_chemprop(
                        train_smiles, y_train, test_smiles,
                        cache_dir=chemprop_cache_dir,
                        cache_key=f"2.07_{ep}_{strat_name}_fold{fold_id}",
                    )

                sq_errors = (y_pred - y_test) ** 2
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                sp_r, _ = spearmanr(y_test, y_pred)
                kt, _ = kendalltau(y_test, y_pred)
                baseline_mad = np.mean(np.abs(y_test - np.mean(y_test)))
                rae = mae / baseline_mad if baseline_mad > 0 else np.nan

                all_results.append({
                    "endpoint": ep, "strategy": strat_name, "fold": fold_id,
                    "n_train": len(train_idx), "n_test": len(test_idx),
                    "struct_dist": struct_nn1, "target_dist": target_nn1,
                    "sq_errors": sq_errors,
                    "mae": mae, "r2": r2, "spearman_r": sp_r,
                    "kendall_tau": kt, "rae": rae,
                    "overall_rmse": float(np.sqrt(sq_errors.mean())),
                })

                for j in range(len(test_idx)):
                    prediction_rows.append({
                        "Molecule Name": test_names[j],
                        "endpoint": ep, "strategy": strat_name, "fold": fold_id,
                        "y_true": float(y_test[j]), "y_pred": float(y_pred[j]),
                        "struct_1nn": float(struct_nn1[j]),
                        "target_1nn": float(target_nn1[j]),
                    })

                pbar.set_postfix_str(f"{ep} {strat_name} f{fold_id}")
                pbar.update(1)

    pbar.close()

    # ── 4. Save per-fold performance table ────────────────────────────────
    perf_rows = [
        {"endpoint": r["endpoint"], "strategy": r["strategy"], "fold": r["fold"],
         "n_train": r["n_train"], "n_test": r["n_test"],
         "mae": r["mae"], "r2": r["r2"], "spearman_r": r["spearman_r"],
         "kendall_tau": r["kendall_tau"], "rae": r["rae"], "rmse": r["overall_rmse"]}
        for r in all_results
    ]
    perf_df = pd.DataFrame(perf_rows)
    perf_df.to_csv(model_dir / "per_fold_performance.csv", index=False)
    logger.info(f"Saved per_fold_performance.csv ({len(perf_df)} rows)")

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_parquet(model_dir / "predictions.parquet", index=False)
    logger.info(f"Saved predictions.parquet ({len(pred_df)} rows)")

    # ── 5. Generate figures ───────────────────────────────────────────────
    _generate_figures(all_results, perf_df, pred_df, model_dir, model, dpi)

    logger.info(f"All outputs saved to {model_dir}")


if __name__ == "__main__":
    app()
