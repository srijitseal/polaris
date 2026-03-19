#!/usr/bin/env python
"""IID vs OOD on chemical series — the "hero" example (paper Fig 7).

Takes two large Butina clusters (chemical series), time-splits the largest
into train + IID validation, uses the smaller cluster as OOD test set, trains
default XGBoost models, and compares squared error intra-series (IID) vs
inter-series (OOD). Demonstrates that chemical series + time-split uniquely
enable this analysis.

Usage:
    pixi run -e cheminformatics python notebooks/2.09-seal-iid-vs-ood-series.py

Outputs:
    data/processed/2.09-seal-iid-vs-ood-series/iid_vs_ood_errors.csv
    data/processed/2.09-seal-iid-vs-ood-series/summary_metrics.csv
    data/processed/2.09-seal-iid-vs-ood-series/split_summary.csv
    data/processed/2.09-seal-iid-vs-ood-series/squared_error_distributions.png
    data/processed/2.09-seal-iid-vs-ood-series/distance_characterization.png
    data/processed/2.09-seal-iid-vs-ood-series/median_se_by_endpoint.png
    data/processed/2.09-seal-iid-vs-ood-series/mae_by_endpoint.png
    data/processed/2.09-seal-iid-vs-ood-series/r2_by_endpoint.png
    data/processed/2.09-seal-iid-vs-ood-series/spearman_by_endpoint.png
    data/processed/2.09-seal-iid-vs-ood-series/kendall_by_endpoint.png
    data/processed/2.09-seal-iid-vs-ood-series/rae_by_endpoint.png
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


def clip_and_log_transform(x: np.ndarray) -> np.ndarray:
    """Log-transform matching competition evaluation: log10(clip(x, 1e-10) + 1)."""
    return np.log10(np.clip(x, 1e-10, None) + 1)


def protonate_at_ph(smiles_list: list[str], ph: float) -> list[str]:
    """Protonate SMILES at given pH using dimorphite_dl. Returns most probable form."""
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


def compute_features(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECFP4 + full RDKit 2D descriptors."""
    ecfp = compute_ecfp4(smiles_list)
    desc = compute_rdkit_descriptors(smiles_list)
    return ecfp, desc


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.09-seal-iid-vs-ood-series", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    train_frac: float = typer.Option(0.8, help="Fraction of largest cluster for training (rest = IID val)"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    logger.info(f"Loaded {len(df)} molecules")

    logger.info("Loading Butina clusters")
    clusters_df = pd.read_parquet(INTERIM_DATA_DIR / "butina_clusters.parquet")
    df = df.merge(clusters_df[["Molecule Name", "cluster_id"]], on="Molecule Name", how="left")

    logger.info("Loading precomputed distance matrix")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    dist_square = squareform(npz["condensed"])
    dist_mol_names = npz["molecule_names"]

    # Build name-to-index map for the distance matrix
    name_to_dist_idx = {str(n): i for i, n in enumerate(dist_mol_names)}

    # ── 2. Select two largest clusters ────────────────────────────────
    cluster_sizes = df.groupby("cluster_id").size().sort_values(ascending=False)
    largest_cid = cluster_sizes.index[0]
    second_cid = cluster_sizes.index[1]
    logger.info(
        f"Largest cluster: {largest_cid} (n={cluster_sizes.iloc[0]}), "
        f"second: {second_cid} (n={cluster_sizes.iloc[1]})"
    )

    series_a = df[df["cluster_id"] == largest_cid].copy()
    series_b = df[df["cluster_id"] == second_cid].copy()

    # ── 3. Time-split cluster A into train + IID validation ───────────
    series_a = series_a.sort_values("mol_index")
    n_train = int(len(series_a) * train_frac)
    train_df = series_a.iloc[:n_train].copy()
    iid_df = series_a.iloc[n_train:].copy()
    ood_df = series_b.copy()

    train_df["set"] = "train"
    iid_df["set"] = "iid"
    ood_df["set"] = "ood"

    logger.info(f"Train: {len(train_df)}, IID val: {len(iid_df)}, OOD test: {len(ood_df)}")

    # ── 4. Distance characterization ──────────────────────────────────
    train_dist_idx = np.array([name_to_dist_idx[n] for n in train_df["Molecule Name"]])
    iid_dist_idx = np.array([name_to_dist_idx[n] for n in iid_df["Molecule Name"]])
    ood_dist_idx = np.array([name_to_dist_idx[n] for n in ood_df["Molecule Name"]])

    iid_to_train = dist_square[np.ix_(iid_dist_idx, train_dist_idx)]
    ood_to_train = dist_square[np.ix_(ood_dist_idx, train_dist_idx)]

    iid_nn1 = iid_to_train.min(axis=1)
    ood_nn1 = ood_to_train.min(axis=1)

    logger.info(f"IID 1-NN median: {np.median(iid_nn1):.3f}, OOD 1-NN median: {np.median(ood_nn1):.3f}")

    # Save split summary
    split_summary = pd.DataFrame([
        {"set": "train", "n": len(train_df), "cluster_id": int(largest_cid)},
        {"set": "iid", "n": len(iid_df), "cluster_id": int(largest_cid),
         "nn1_median": np.median(iid_nn1), "nn1_mean": np.mean(iid_nn1)},
        {"set": "ood", "n": len(ood_df), "cluster_id": int(second_cid),
         "nn1_median": np.median(ood_nn1), "nn1_mean": np.mean(ood_nn1)},
    ])
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    logger.info("Saved split_summary.csv")

    # ── 5. Train XGBoost per endpoint and collect errors ──────────────
    error_rows = []
    metric_rows = []

    # Precompute features per pH (protonation differs by pH)
    unique_phs = sorted(set(ENDPOINT_PH.values()))
    features_cache: dict[float, dict[str, tuple]] = {}

    for ph in unique_phs:
        logger.info(f"Protonating and featurizing at pH {ph}")
        train_prot = protonate_at_ph(train_df["SMILES"].tolist(), ph)
        iid_prot = protonate_at_ph(iid_df["SMILES"].tolist(), ph)
        ood_prot = protonate_at_ph(ood_df["SMILES"].tolist(), ph)

        ecfp_tr, desc_tr = compute_features(train_prot)
        ecfp_iid, desc_iid = compute_features(iid_prot)
        ecfp_ood, desc_ood = compute_features(ood_prot)

        # Remove zero-variance descriptors (fit on train)
        variance = desc_tr.var(axis=0)
        nonzero_var = variance > 0
        desc_tr = desc_tr[:, nonzero_var]
        desc_iid = desc_iid[:, nonzero_var]
        desc_ood = desc_ood[:, nonzero_var]

        # Scale descriptors
        scaler = StandardScaler()
        desc_tr_scaled = scaler.fit_transform(desc_tr)
        desc_iid_scaled = scaler.transform(desc_iid)
        desc_ood_scaled = scaler.transform(desc_ood)

        features_cache[ph] = {
            "train": (ecfp_tr, desc_tr_scaled),
            "iid": (ecfp_iid, desc_iid_scaled),
            "ood": (ecfp_ood, desc_ood_scaled),
        }
        logger.info(f"  pH {ph}: {ecfp_tr.shape[1] + desc_tr.shape[1]} features")

    for ep in ENDPOINTS:
        ph = ENDPOINT_PH[ep]
        ecfp_tr, desc_tr = features_cache[ph]["train"]
        ecfp_iid, desc_iid = features_cache[ph]["iid"]
        ecfp_ood, desc_ood = features_cache[ph]["ood"]

        # Masks for non-null endpoint values
        train_mask = train_df[ep].notna().values
        iid_mask = iid_df[ep].notna().values
        ood_mask = ood_df[ep].notna().values

        n_tr = train_mask.sum()
        n_iid = iid_mask.sum()
        n_ood = ood_mask.sum()

        if n_tr < 20 or n_iid < 10 or n_ood < 10:
            logger.warning(f"Skipping {ep}: insufficient data (train={n_tr}, iid={n_iid}, ood={n_ood})")
            continue

        X_tr = np.hstack([ecfp_tr[train_mask], desc_tr[train_mask]])
        X_iid = np.hstack([ecfp_iid[iid_mask], desc_iid[iid_mask]])
        X_ood = np.hstack([ecfp_ood[ood_mask], desc_ood[ood_mask]])

        y_tr_raw = train_df[ep].values[train_mask]
        y_iid_raw = iid_df[ep].values[iid_mask]
        y_ood_raw = ood_df[ep].values[ood_mask]

        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_tr = clip_and_log_transform(y_tr_raw)
            y_iid = clip_and_log_transform(y_iid_raw)
            y_ood = clip_and_log_transform(y_ood_raw)
        else:
            y_tr = y_tr_raw
            y_iid = y_iid_raw
            y_ood = y_ood_raw

        logger.info(f"  {ep}: train={n_tr}, iid={n_iid}, ood={n_ood}")

        model = XGBRegressor(random_state=42, verbosity=0)
        model.fit(X_tr, y_tr)

        y_pred_iid = model.predict(X_iid)
        y_pred_ood = model.predict(X_ood)

        se_iid = (y_iid - y_pred_iid) ** 2
        se_ood = (y_ood - y_pred_ood) ** 2

        iid_names = iid_df["Molecule Name"].values[iid_mask]
        ood_names = ood_df["Molecule Name"].values[ood_mask]

        for name, se in zip(iid_names, se_iid):
            error_rows.append({"Molecule Name": name, "endpoint": ep, "set": "iid", "squared_error": se})
        for name, se in zip(ood_names, se_ood):
            error_rows.append({"Molecule Name": name, "endpoint": ep, "set": "ood", "squared_error": se})

        for label, y_true, y_pred, se in [
            ("iid", y_iid, y_pred_iid, se_iid),
            ("ood", y_ood, y_pred_ood, se_ood),
        ]:
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            sp_r, _ = spearmanr(y_true, y_pred)
            kt, _ = kendalltau(y_true, y_pred)
            baseline_mad = np.mean(np.abs(y_true - np.mean(y_true)))
            rae = mae / baseline_mad if baseline_mad > 0 else np.nan

            logger.info(
                f"    {label}: MAE={mae:.3f}, R2={r2:.3f}, "
                f"Spearman={sp_r:.3f}, Kendall={kt:.3f}, RAE={rae:.3f}"
            )

            metric_rows.append({
                "endpoint": ep,
                "set": label,
                "n": len(y_true),
                "mae": mae,
                "r2": r2,
                "spearman_r": sp_r,
                "kendall_tau": kt,
                "rae": rae,
                "baseline_mad": baseline_mad,
                "median_se": np.median(se),
                "mean_se": np.mean(se),
            })

    errors_df = pd.DataFrame(error_rows)
    metrics_df = pd.DataFrame(metric_rows)

    errors_df.to_csv(output_dir / "iid_vs_ood_errors.csv", index=False)
    metrics_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    logger.info("Saved iid_vs_ood_errors.csv and summary_metrics.csv")

    # ── 6. Figure A: Squared error distributions (3×3 grid) ───────────
    active_endpoints = sorted(metrics_df["endpoint"].unique())
    n_ep = len(active_endpoints)
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        iid_se = errors_df[(errors_df["endpoint"] == ep) & (errors_df["set"] == "iid")]["squared_error"]
        ood_se = errors_df[(errors_df["endpoint"] == ep) & (errors_df["set"] == "ood")]["squared_error"]

        # Use log-scale bins for squared errors
        all_se = np.concatenate([iid_se.values, ood_se.values])
        lo = max(all_se.min(), 1e-8)
        hi = all_se.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)

        ax.hist(iid_se, bins=bins, density=True, alpha=0.6, color="steelblue",
                label=f"IID (med={np.median(iid_se):.3f})", edgecolor="white")
        ax.hist(ood_se, bins=bins, density=True, alpha=0.6, color="coral",
                label=f"OOD (med={np.median(ood_se):.3f})", edgecolor="white")

        ax.set_xscale("log")
        ax.set_xlabel("Squared error")
        ax.set_ylabel("Density")
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("IID vs OOD squared error distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "squared_error_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved squared_error_distributions.png")
    plt.close("all")

    # ── 7. Figure B: Distance characterization ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)
    ax.hist(iid_nn1, bins=bins, density=True, alpha=0.6, color="steelblue",
            label=f"IID (med={np.median(iid_nn1):.3f})", edgecolor="white")
    ax.hist(ood_nn1, bins=bins, density=True, alpha=0.6, color="coral",
            label=f"OOD (med={np.median(ood_nn1):.3f})", edgecolor="white")
    ax.set_xlabel("Test-to-train 1-NN Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title("Structural distance to training set: IID vs OOD")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "distance_characterization.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved distance_characterization.png")
    plt.close("all")

    # ── 8. Figure C: Per-metric IID vs OOD bar charts ───────────────
    x = np.arange(len(active_endpoints))
    width = 0.35

    metric_plots = [
        ("median_se", "Median squared error", "median_se"),
        ("mae", "MAE", "mae"),
        ("r2", "R²", "r2"),
        ("spearman_r", "Spearman ρ", "spearman"),
        ("kendall_tau", "Kendall τ", "kendall"),
        ("rae", "RAE", "rae"),
    ]

    for col, ylabel, fname in metric_plots:
        fig, ax = plt.subplots(figsize=(10, 5))

        iid_vals = []
        ood_vals = []
        for ep in active_endpoints:
            iid_row = metrics_df[(metrics_df["endpoint"] == ep) & (metrics_df["set"] == "iid")]
            ood_row = metrics_df[(metrics_df["endpoint"] == ep) & (metrics_df["set"] == "ood")]
            iid_vals.append(iid_row[col].values[0])
            ood_vals.append(ood_row[col].values[0])

        ax.bar(x - width / 2, iid_vals, width, label="IID",
               color="steelblue", edgecolor="white", alpha=0.8)
        ax.bar(x + width / 2, ood_vals, width, label="OOD",
               color="coral", edgecolor="white", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"IID vs OOD {ylabel} by endpoint")
        ax.legend()
        if col in ("r2",):
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"{fname}_by_endpoint.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {fname}_by_endpoint.png")
        plt.close("all")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
