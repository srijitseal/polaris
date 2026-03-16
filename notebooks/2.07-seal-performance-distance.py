#!/usr/bin/env python
"""Performance-over-distance curves for all endpoints across all split strategies.

Trains XGBoost on ECFP4 + full RDKit 2D descriptors (~200) using pre-saved CV
folds (cluster, time, target — repeat 0), then plots how RMSE degrades with
distance from training data. Molecules are protonated at assay-relevant pH
using dimorphite_dl before feature computation. For endpoints not already on
log scale (everything except LogD), targets are log-transformed for both
training and evaluation, matching the OpenADMET competition protocol.

Usage:
    pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py

Outputs:
    data/processed/2.07-seal-performance-distance/overall_r2.png
    data/processed/2.07-seal-performance-distance/overall_rmse.png
    data/processed/2.07-seal-performance-distance/per_fold_performance.csv
    data/processed/2.07-seal-performance-distance/performance_over_distance.png
    data/processed/2.07-seal-performance-distance/performance_over_target_distance.png
    data/processed/2.07-seal-performance-distance/performance_distance_summary.csv
    data/processed/2.07-seal-performance-distance/predictions.parquet
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

STRATEGIES = {
    "cluster": {"file": "cluster_cv_folds.parquet", "n_folds": 5},
    "time": {"file": "time_cv_folds.parquet", "n_folds": 4},
    "target": {"file": "target_cv_folds.parquet", "n_folds": 4},
}

# Full RDKit 2D descriptor suite
DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


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


def get_train_test_masks(fold_ids: np.ndarray, fold_id: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Get train and test boolean masks for a given fold."""
    test_mask = fold_ids == fold_id
    if strategy == "cluster":
        train_mask = (fold_ids != fold_id) & (fold_ids >= 0)
    else:
        # Expanding window: train = all folds before test
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= fold_ids == k
    return train_mask, test_mask


def make_sliding_bins(distances: np.ndarray) -> list[tuple[float, float]]:
    """Create sliding window bins from pooled distances per outline protocol."""
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
    """Compute RMSE per bin. Returns dict of bin_center -> rmse (or NaN)."""
    results = {}
    for lo, hi in bins:
        mask = (distances >= lo) & (distances < hi)
        center = (lo + hi) / 2
        if mask.sum() >= min_count:
            results[center] = np.sqrt(errors_sq[mask].mean())
        else:
            results[center] = np.nan
    return results


def make_xgb_model() -> XGBRegressor:
    """Create XGBoost model with fixed general-purpose config."""
    return XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.4,
        min_child_weight=5,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.5,
        tree_method="hist",
        random_state=42,
        verbosity=0,
    )


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.07-seal-performance-distance", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading data")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    dist_square = squareform(npz["condensed"])

    # Load all three fold files
    fold_dfs = {}
    for strat_name, strat_info in STRATEGIES.items():
        folds = pd.read_parquet(INTERIM_DATA_DIR / strat_info["file"])
        fold_dfs[strat_name] = folds[folds["repeat"] == 0]
        logger.info(f"  {strat_name}: {len(fold_dfs[strat_name])} rows (repeat=0)")

    # ── 2. Protonate and compute features per pH ─────────────────────
    smiles_list = df["SMILES"].tolist()
    mol_names = df["Molecule Name"].values

    unique_phs = sorted(set(ENDPOINT_PH.values()))
    features_by_ph: dict[float, np.ndarray] = {}

    for ph in unique_phs:
        logger.info(f"Protonating at pH {ph}")
        protonated = protonate_at_ph(smiles_list, ph)

        logger.info(f"Computing features at pH {ph}")
        ecfp = compute_ecfp4(protonated)
        desc = compute_rdkit_descriptors(protonated)

        # Remove zero-variance descriptors
        variance = desc.var(axis=0)
        nonzero_var = variance > 0
        desc = desc[:, nonzero_var]

        # Scale descriptors
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc)

        X = np.hstack([ecfp, desc_scaled])
        logger.info(f"  pH {ph}: {X.shape[1]} features (2048 ECFP + {desc.shape[1]} RDKit 2D)")
        features_by_ph[ph] = X

    # ── 3. Train models and collect predictions ───────────────────────
    all_results = []
    prediction_rows = []

    # Count total iterations
    total_iters = 0
    for ep in ENDPOINTS:
        if df[ep].notna().sum() < 50:
            continue
        total_iters += sum(s["n_folds"] for s in STRATEGIES.values())

    pbar = tqdm(total=total_iters, desc="Training models", unit="fold")

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        if mask.sum() < 50:
            logger.warning(f"Skipping {ep}: only {mask.sum()} molecules")
            continue

        ph = ENDPOINT_PH[ep]
        ep_idx = np.where(mask)[0]
        ep_names = mol_names[mask]
        ep_values_raw = df[ep].values[mask]

        # Log-transform targets (except LogD)
        if ep in LOG_TRANSFORM_ENDPOINTS:
            ep_values = clip_and_log_transform(ep_values_raw)
        else:
            ep_values = ep_values_raw

        X_ep = features_by_ph[ph][mask]
        D_ep = dist_square[np.ix_(ep_idx, ep_idx)]

        for strat_name, strat_info in STRATEGIES.items():
            strat_folds = fold_dfs[strat_name]
            ep_folds = strat_folds[strat_folds["endpoint"] == ep]
            fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
            fold_ids = np.array([fold_map.get(n, -99) for n in ep_names])
            n_folds = strat_info["n_folds"]

            for fold_id in range(n_folds):
                train_mask, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)

                test_idx = np.where(test_mask)[0]
                train_idx = np.where(train_mask)[0]
                if len(test_idx) < 10 or len(train_idx) < 10:
                    pbar.update(1)
                    continue

                X_train, X_test = X_ep[train_idx], X_ep[test_idx]
                y_train, y_test = ep_values[train_idx], ep_values[test_idx]

                # Structural 1-NN distances
                struct_nn1 = D_ep[np.ix_(test_idx, train_idx)].min(axis=1)

                # Target-space 1-NN distances
                target_nn1 = np.abs(y_test[:, None] - y_train[None, :]).min(axis=1)

                test_names = ep_names[test_idx]

                model = make_xgb_model()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                sq_errors = (y_pred - y_test) ** 2

                # Competition metrics
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                sp_r, _ = spearmanr(y_test, y_pred)
                kt, _ = kendalltau(y_test, y_pred)
                baseline_mad = np.mean(np.abs(y_test - np.mean(y_test)))
                rae = mae / baseline_mad if baseline_mad > 0 else np.nan

                all_results.append({
                    "endpoint": ep,
                    "strategy": strat_name,
                    "fold": fold_id,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "struct_dist": struct_nn1,
                    "target_dist": target_nn1,
                    "sq_errors": sq_errors,
                    "mae": mae,
                    "r2": r2,
                    "spearman_r": sp_r,
                    "kendall_tau": kt,
                    "rae": rae,
                    "overall_rmse": np.sqrt(sq_errors.mean()),
                })

                # Store per-molecule predictions
                for j in range(len(test_idx)):
                    prediction_rows.append({
                        "Molecule Name": test_names[j],
                        "endpoint": ep,
                        "strategy": strat_name,
                        "fold": fold_id,
                        "y_true": y_test[j],
                        "y_pred": y_pred[j],
                        "struct_1nn": struct_nn1[j],
                        "target_1nn": target_nn1[j],
                    })

                pbar.set_postfix_str(f"{ep} {strat_name} f{fold_id} ({len(train_idx)}:{len(test_idx)})")
                pbar.update(1)

    pbar.close()

    # ── 4. Plot: Overall performance summary ────────────────────────
    logger.info("Plotting overall performance summary")
    perf_rows = []
    for r in all_results:
        perf_rows.append({
            "endpoint": r["endpoint"],
            "strategy": r["strategy"],
            "fold": r["fold"],
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "mae": r["mae"],
            "r2": r["r2"],
            "spearman_r": r["spearman_r"],
            "kendall_tau": r["kendall_tau"],
            "rae": r["rae"],
            "rmse": r["overall_rmse"],
        })
    perf_df = pd.DataFrame(perf_rows)

    active_endpoints = sorted(perf_df["endpoint"].unique())
    n_ep = len(active_endpoints)

    strat_colors = {"cluster": "steelblue", "time": "coral", "target": "forestgreen"}

    for metric, ylabel in [
        ("mae", "MAE"),
        ("spearman_r", "Spearman ρ"),
        ("rae", "RAE"),
    ]:
        fig, ax = plt.subplots(figsize=(14, 6))

        x = np.arange(n_ep)
        width = 0.25
        for i, strat_name in enumerate(STRATEGIES):
            strat_sub = perf_df[perf_df["strategy"] == strat_name]
            means = []
            stds = []
            for ep in active_endpoints:
                ep_vals = strat_sub[strat_sub["endpoint"] == ep][metric]
                means.append(ep_vals.mean())
                stds.append(ep_vals.std())
            ax.bar(x + i * width, means, width, yerr=stds, label=strat_name,
                   color=strat_colors[strat_name], edgecolor="white", capsize=3, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by endpoint and split strategy — XGBoost")
        ax.legend(fontsize=8)
        if metric == "rae":
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"overall_{metric}.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved overall_{metric}.png")
        plt.close("all")

    # R² plot: separate cluster+time (interpretable) from target (catastrophic)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={"width_ratios": [2, 1]})

    x = np.arange(n_ep)
    width = 0.35
    for i, strat_name in enumerate(["cluster", "time"]):
        strat_sub = perf_df[perf_df["strategy"] == strat_name]
        means = []
        stds = []
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

    # Target-split R² (log-scale of |R²| since values are deeply negative)
    target_sub = perf_df[perf_df["strategy"] == "target"]
    target_means = []
    for ep in active_endpoints:
        ep_vals = target_sub[target_sub["endpoint"] == ep]["r2"]
        target_means.append(ep_vals.mean())
    ax2.barh(np.arange(n_ep), target_means, color=strat_colors["target"], alpha=0.8)
    ax2.set_yticks(np.arange(n_ep))
    ax2.set_yticklabels(active_endpoints, fontsize=8)
    ax2.set_xlabel("R²")
    ax2.set_title("R² — Target split (note scale)")
    ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("R² by endpoint and split strategy — XGBoost", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_r2.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved overall_r2.png")
    plt.close("all")

    # Log MA-RAE per strategy
    for strat_name in STRATEGIES:
        strat_sub = perf_df[perf_df["strategy"] == strat_name]
        ma_rae = strat_sub.groupby("fold")["rae"].mean().mean()
        logger.info(f"MA-RAE ({strat_name}): {ma_rae:.3f}")

    # Save per-fold performance table
    perf_df.to_csv(output_dir / "per_fold_performance.csv", index=False)
    logger.info(f"Saved per_fold_performance.csv ({len(perf_df)} rows)")

    # ── 5. Plot: Performance over structural distance ─────────────────
    logger.info("Plotting performance over structural distance")
    active_endpoints = sorted(set(r["endpoint"] for r in all_results))
    n_ep = len(active_endpoints)
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

        # Pool all distances for shared binning
        all_dists = np.concatenate([r["struct_dist"] for r in ep_results])
        bins = make_sliding_bins(all_dists)

        for strat_name in STRATEGIES:
            strat_results = [r for r in ep_results if r["strategy"] == strat_name]
            if not strat_results:
                continue

            # Per-fold curves
            fold_curves = []
            for r in strat_results:
                curve = bin_performance(r["struct_dist"], r["sq_errors"], bins)
                fold_curves.append(curve)

            centers = sorted(fold_curves[0].keys())
            values = np.array([[c[center] for center in centers] for c in fold_curves])

            median_vals = np.nanmedian(values, axis=0)
            lo_vals = np.nanpercentile(values, 10, axis=0)
            hi_vals = np.nanpercentile(values, 90, axis=0)

            valid = ~np.isnan(median_vals)
            c_arr = np.array(centers)

            color = strat_colors[strat_name]
            ax.plot(c_arr[valid], median_vals[valid], color=color,
                    label=strat_name, linewidth=1.5)
            ax.fill_between(c_arr[valid], lo_vals[valid], hi_vals[valid],
                            color=color, alpha=0.1)

            # Summary stats
            overall = np.median([r["overall_rmse"] for r in strat_results])
            valid_medians = median_vals[valid]
            if len(valid_medians) > 0:
                summary_rows.append({
                    "endpoint": ep,
                    "strategy": strat_name,
                    "overall_rmse": float(overall),
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

    fig.suptitle("Performance over structural distance — XGBoost", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "performance_over_distance.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved performance_over_distance.png")
    plt.close("all")

    # ── 6. Plot: Performance over target-space distance ───────────────
    logger.info("Plotting performance over target-space distance")
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

            fold_curves = []
            for r in strat_results:
                curve = bin_performance(r["target_dist"], r["sq_errors"], bins)
                fold_curves.append(curve)

            centers = sorted(fold_curves[0].keys())
            values = np.array([[c[center] for center in centers] for c in fold_curves])

            median_vals = np.nanmedian(values, axis=0)
            lo_vals = np.nanpercentile(values, 10, axis=0)
            hi_vals = np.nanpercentile(values, 90, axis=0)

            valid = ~np.isnan(median_vals)
            c_arr = np.array(centers)

            color = strat_colors[strat_name]
            ax.plot(c_arr[valid], median_vals[valid], color=color,
                    label=strat_name, linewidth=1.5)
            ax.fill_between(c_arr[valid], lo_vals[valid], hi_vals[valid],
                            color=color, alpha=0.1)

        ax.set_xlabel("Test-to-train 1-NN distance (target space)")
        ax.set_ylabel("RMSE")
        ax.set_title(ep)
        ax.legend(fontsize=7)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Performance over target-space distance — XGBoost", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "performance_over_target_distance.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved performance_over_target_distance.png")
    plt.close("all")

    # ── 7. Save summary ──────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "performance_distance_summary.csv", index=False)
    logger.info(f"Saved performance_distance_summary.csv ({len(summary_df)} rows)")

    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['endpoint']} {row['strategy']}: "
            f"overall={row['overall_rmse']:.3f}, "
            f"closest={row['rmse_closest_bin']:.3f}, "
            f"farthest={row['rmse_farthest_bin']:.3f}, "
            f"degradation={row['degradation_ratio']:.2f}x"
        )

    # Save predictions
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_parquet(output_dir / "predictions.parquet", index=False)
    logger.info(f"Saved predictions.parquet ({len(pred_df)} rows)")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
