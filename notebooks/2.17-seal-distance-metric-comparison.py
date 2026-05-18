#!/usr/bin/env python
"""Compare performance-over-distance curves using Tanimoto vs RDKit 2D descriptor distance.

Uses existing predictions from 2.07 and recomputes the degradation curves with
Euclidean distance in scaled RDKit 2D descriptor space as the x-axis, alongside
the original Tanimoto (Jaccard) distance curves.

Methodology matches 2.07: bins defined from ALL strategies pooled, per-fold curves
computed, then MEDIAN across folds taken.

Usage:
    pixi run -e cheminformatics python notebooks/2.17-seal-distance-metric-comparison.py

Outputs (under data/processed/2.17-seal-distance-metric-comparison/):
    distance_metric_comparison.png
    degradation_comparison.csv
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
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

RDLogger.DisableLog("rdApp.*")

app = typer.Typer()

ENDPOINTS = [
    "LogD", "KSOL", "HLM CLint", "MLM CLint",
    "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux",
    "MPPB", "MBPB", "MGMB",
]

ENDPOINT_PH = {
    "LogD": 7.4, "KSOL": 7.4, "HLM CLint": 7.4, "MLM CLint": 7.4,
    "Caco-2 Permeability Papp A>B": 6.5, "Caco-2 Permeability Efflux": 6.5,
    "MPPB": 7.4, "MBPB": 7.4, "MGMB": 7.4,
}

STRATEGIES = {
    "cluster": {"file": "cluster_cv_folds.parquet", "n_folds": 5},
    "time": {"file": "time_cv_folds.parquet", "n_folds": 4},
    "target": {"file": "target_cv_folds.parquet", "n_folds": 4},
}

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


def protonate_at_ph(smiles_list: list[str], ph: float) -> list[str]:
    protonated = []
    for smi in smiles_list:
        try:
            result = dimorphite_protonate(smi, ph_min=ph - 0.5, ph_max=ph + 0.5, max_variants=1)
            protonated.append(result[0] if result else smi)
        except Exception:
            protonated.append(smi)
    return protonated


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


def bin_performance(
    distances: np.ndarray, errors_sq: np.ndarray, bins: list[tuple[float, float]], min_count: int = 25
) -> dict[float, float]:
    result = {}
    for lo, hi in bins:
        mask = (distances >= lo) & (distances < hi)
        center = (lo + hi) / 2
        if mask.sum() >= min_count:
            result[center] = float(np.sqrt(errors_sq[mask].mean()))
        else:
            result[center] = np.nan
    return result


@app.command()
def main(
    dpi: int = typer.Option(DEFAULT_DPI, help="Figure DPI"),
):
    output_dir = PROCESSED_DATA_DIR / "2.17-seal-distance-metric-comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    smiles_list = df["SMILES"].tolist()
    mol_names = df["Molecule Name"].values

    # Compute RDKit 2D descriptor features per pH
    logger.info("Computing RDKit 2D descriptors")
    desc_by_ph: dict[float, np.ndarray] = {}
    for ph in sorted(set(ENDPOINT_PH.values())):
        logger.info(f"  Protonating at pH {ph}")
        protonated = protonate_at_ph(smiles_list, ph)
        desc = compute_rdkit_descriptors(protonated)
        variance = desc.var(axis=0)
        desc = desc[:, variance > 0]
        desc_by_ph[ph] = desc
        logger.info(f"  pH {ph}: {desc.shape[1]} descriptors after variance filter")

    # Load fold assignments
    fold_dfs = {}
    for strat_name, strat_info in STRATEGIES.items():
        folds = pd.read_parquet(INTERIM_DATA_DIR / strat_info["file"])
        fold_dfs[strat_name] = folds[folds["repeat"] == 0]

    # Load existing XGBoost predictions (from 2.07)
    pred_file = PROCESSED_DATA_DIR / "2.07-seal-performance-distance" / "xgboost" / "predictions.parquet"
    if not pred_file.exists():
        logger.error(f"Run 2.07 first: {pred_file} not found")
        raise typer.Exit(1)
    pred_df = pd.read_parquet(pred_file)
    pred_df["sq_error"] = (pred_df["y_pred"] - pred_df["y_true"]) ** 2

    # For each endpoint/strategy/fold, compute descriptor-space 1-NN distances
    logger.info("Computing descriptor-space 1-NN distances per fold")
    desc_1nn_map = {}  # (mol_name, endpoint, strategy, fold) -> desc_1nn_dist

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        if mask.sum() < 50:
            continue
        ph = ENDPOINT_PH[ep]
        ep_idx = np.where(mask)[0]
        ep_names = mol_names[mask]
        desc_ep = desc_by_ph[ph][mask]

        for strat_name, strat_info in STRATEGIES.items():
            strat_folds = fold_dfs[strat_name]
            ep_folds = strat_folds[strat_folds["endpoint"] == ep]
            fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
            fold_ids = np.array([fold_map.get(n, -99) for n in ep_names])
            n_folds = strat_info["n_folds"]

            for fold_id in range(n_folds):
                if strat_name == "cluster":
                    test_mask = fold_ids == fold_id
                    train_mask = (fold_ids != fold_id) & (fold_ids >= 0)
                else:
                    test_mask = fold_ids == fold_id
                    train_mask = np.zeros(len(fold_ids), dtype=bool)
                    for k in range(-1, fold_id):
                        train_mask |= fold_ids == k

                test_idx = np.where(test_mask)[0]
                train_idx = np.where(train_mask)[0]
                if len(test_idx) < 10 or len(train_idx) < 10:
                    continue

                scaler = StandardScaler()
                desc_train = scaler.fit_transform(desc_ep[train_idx])
                desc_test = scaler.transform(desc_ep[test_idx])

                dists = cdist(desc_test, desc_train, metric="euclidean")
                desc_nn1 = dists.min(axis=1)

                for j, tidx in enumerate(test_idx):
                    key = (ep_names[tidx], ep, strat_name, fold_id)
                    desc_1nn_map[key] = desc_nn1[j]

    # Merge descriptor distances into predictions
    logger.info("Merging descriptor distances with predictions")
    pred_df["desc_1nn"] = pred_df.apply(
        lambda row: desc_1nn_map.get(
            (row["Molecule Name"], row["endpoint"], row["strategy"], row["fold"]), np.nan
        ), axis=1
    )
    n_matched = pred_df["desc_1nn"].notna().sum()
    logger.info(f"  Matched {n_matched}/{len(pred_df)} predictions with descriptor distances")

    # ── Per-fold median methodology (matching 2.07) ─────────────────────────
    # For each endpoint: pool distances from ALL strategies to define bins,
    # then compute per-fold curves per strategy, then take median across folds.
    set_style()
    ep_list = sorted(pred_df["endpoint"].unique())
    n_ep = len(ep_list)
    ncols = 3
    nrows = (n_ep + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten()

    summary_rows = []

    for ax_idx, ep in enumerate(ep_list):
        ax = axes[ax_idx]
        ep_pred = pred_df[pred_df["endpoint"] == ep]
        if ep_pred.empty:
            continue

        # ─── Tanimoto curves (per-fold median, bins from all strategies) ───
        # Pool Tanimoto distances from ALL strategies for this endpoint to define bins
        all_tani_dists = ep_pred["struct_1nn"].values
        tani_bins = make_sliding_bins(all_tani_dists)

        # Compute per-fold curves for cluster strategy, take median
        cluster_pred = ep_pred[ep_pred["strategy"] == "cluster"]
        cluster_folds = sorted(cluster_pred["fold"].unique())
        tani_fold_curves = []
        for fold_id in cluster_folds:
            fold_data = cluster_pred[cluster_pred["fold"] == fold_id]
            curve = bin_performance(fold_data["struct_1nn"].values, fold_data["sq_error"].values, tani_bins)
            tani_fold_curves.append(curve)

        if tani_fold_curves:
            centers = sorted(tani_fold_curves[0].keys())
            values = np.array([[c[center] for center in centers] for c in tani_fold_curves])
            tani_median = np.nanmedian(values, axis=0)
            valid = ~np.isnan(tani_median)
            c_arr = np.array(centers)

            ax.plot(c_arr[valid], tani_median[valid], color="steelblue", linewidth=2, label="Tanimoto (Jaccard)")
            ax.fill_between(
                c_arr[valid],
                np.nanpercentile(values, 10, axis=0)[valid],
                np.nanpercentile(values, 90, axis=0)[valid],
                color="steelblue", alpha=0.1,
            )

            valid_medians = tani_median[valid]
            if len(valid_medians) >= 2 and valid_medians[0] > 0:
                tani_ratio = float(valid_medians[-1] / valid_medians[0])
            else:
                tani_ratio = np.nan

            # Spearman on cluster data
            cluster_abs_err = np.abs(cluster_pred["y_pred"] - cluster_pred["y_true"]).values
            tani_rho, _ = spearmanr(cluster_pred["struct_1nn"].values, cluster_abs_err)
        else:
            tani_ratio = np.nan
            tani_rho = np.nan

        # ─── Descriptor-space curves (per-fold median, bins from all strategies) ───
        ep_pred_desc = ep_pred[ep_pred["desc_1nn"].notna()]
        all_desc_dists = ep_pred_desc["desc_1nn"].values

        if len(all_desc_dists) > 100:
            desc_bins = make_sliding_bins(all_desc_dists)

            cluster_desc = ep_pred_desc[ep_pred_desc["strategy"] == "cluster"]
            cluster_desc_folds = sorted(cluster_desc["fold"].unique())
            desc_fold_curves = []
            for fold_id in cluster_desc_folds:
                fold_data = cluster_desc[cluster_desc["fold"] == fold_id]
                curve = bin_performance(fold_data["desc_1nn"].values, fold_data["sq_error"].values, desc_bins)
                desc_fold_curves.append(curve)

            if desc_fold_curves:
                centers_d = sorted(desc_fold_curves[0].keys())
                values_d = np.array([[c[center] for center in centers_d] for c in desc_fold_curves])
                desc_median = np.nanmedian(values_d, axis=0)
                valid_d = ~np.isnan(desc_median)
                c_arr_d = np.array(centers_d)

                ax.plot(c_arr_d[valid_d], desc_median[valid_d], color="darkorange", linewidth=2, label="RDKit 2D (Euclidean)")
                ax.fill_between(
                    c_arr_d[valid_d],
                    np.nanpercentile(values_d, 10, axis=0)[valid_d],
                    np.nanpercentile(values_d, 90, axis=0)[valid_d],
                    color="darkorange", alpha=0.1,
                )

                valid_desc_medians = desc_median[valid_d]
                if len(valid_desc_medians) >= 2 and valid_desc_medians[0] > 0:
                    desc_ratio = float(valid_desc_medians[-1] / valid_desc_medians[0])
                else:
                    desc_ratio = np.nan

                cluster_desc_abs_err = np.abs(cluster_desc["y_pred"] - cluster_desc["y_true"]).values
                desc_rho, _ = spearmanr(cluster_desc["desc_1nn"].values, cluster_desc_abs_err)
            else:
                desc_ratio = np.nan
                desc_rho = np.nan
        else:
            desc_ratio = np.nan
            desc_rho = np.nan

        ax.set_xlabel("1-NN distance to training set")
        ax.set_ylabel("RMSE")
        ax.set_title(ep, fontweight="bold")
        ax.legend(fontsize=9)

        summary_rows.append({
            "endpoint": ep,
            "tanimoto_degradation_ratio": round(tani_ratio, 2) if not np.isnan(tani_ratio) else np.nan,
            "descriptor_degradation_ratio": round(desc_ratio, 2) if not np.isnan(desc_ratio) else np.nan,
            "tanimoto_spearman": round(tani_rho, 3) if not np.isnan(tani_rho) else np.nan,
            "descriptor_spearman": round(desc_rho, 3) if not np.isnan(desc_rho) else np.nan,
        })

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)
    fig.suptitle(
        "Performance degradation: Tanimoto vs RDKit 2D descriptor distance\n(XGBoost, cluster split, per-fold median)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "distance_metric_comparison.png", dpi=dpi, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved {output_dir / 'distance_metric_comparison.png'}")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "degradation_comparison.csv", index=False)
    logger.info(f"Saved degradation_comparison.csv")
    print("\nDegradation ratio comparison (farthest/closest bin RMSE, per-fold median):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    app()
