#!/usr/bin/env python
"""Baseline XGBoost performance on the original train/test split.

Trains XGBoost on ECFP4 + full RDKit 2D descriptors (~200) using the
competition train/test split (5,326 / 2,282). Molecules are protonated at the
relevant assay pH using dimorphite_dl before feature computation:
  - pH 7.4: LogD, KSOL, HLM/MLM CLint, MPPB, MBPB, MGMB
  - pH 6.5: Caco-2 Papp A>B, Caco-2 Efflux (apical compartment)

Uses HalvingRandomSearchCV to tune hyperparameters per endpoint via 3-fold CV.
For endpoints not already on log scale (everything except LogD), both training
targets and evaluation are log-transformed via log10(clip(x, 1e-10) + 1),
matching the OpenADMET competition evaluation protocol.

Usage:
    pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py

Outputs:
    data/processed/2.08-seal-baseline-performance/overall_metrics.csv
    data/processed/2.08-seal-baseline-performance/predictions.parquet
    data/processed/2.08-seal-baseline-performance/best_params.csv
    data/processed/2.08-seal-baseline-performance/r2_by_endpoint.png
    data/processed/2.08-seal-baseline-performance/mae_by_endpoint.png
    data/processed/2.08-seal-baseline-performance/spearman_by_endpoint.png
    data/processed/2.08-seal-baseline-performance/scatter_predictions.png
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
from scipy.stats import kendalltau, randint, spearmanr, uniform
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import HalvingRandomSearchCV
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

# Assay pH per endpoint
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

# Full RDKit 2D descriptor suite
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


def compute_features(smiles_list: list[str]) -> np.ndarray:
    """Compute ECFP4 + full RDKit 2D descriptors, scaled and concatenated."""
    ecfp = compute_ecfp4(smiles_list)
    desc = compute_rdkit_descriptors(smiles_list)
    return ecfp, desc


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.08-seal-baseline-performance", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading data")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

    train_smiles = train_df["SMILES"].tolist()
    test_smiles = test_df["SMILES"].tolist()
    train_names = train_df["Molecule Name"].values
    test_names = test_df["Molecule Name"].values

    # ── 2. Protonate at each unique pH and compute features ──────────
    unique_phs = sorted(set(ENDPOINT_PH.values()))
    features_by_ph: dict[float, dict] = {}

    for ph in unique_phs:
        logger.info(f"Protonating at pH {ph}")
        train_prot = protonate_at_ph(train_smiles, ph)
        test_prot = protonate_at_ph(test_smiles, ph)

        logger.info(f"Computing features at pH {ph}")
        ecfp_tr, desc_tr = compute_features(train_prot)
        ecfp_te, desc_te = compute_features(test_prot)

        # Remove zero-variance descriptors (fit on train)
        variance = desc_tr.var(axis=0)
        nonzero_var = variance > 0
        desc_tr = desc_tr[:, nonzero_var]
        desc_te = desc_te[:, nonzero_var]

        # Scale descriptors
        scaler = StandardScaler()
        desc_tr_scaled = scaler.fit_transform(desc_tr)
        desc_te_scaled = scaler.transform(desc_te)

        X_tr = np.hstack([ecfp_tr, desc_tr_scaled])
        X_te = np.hstack([ecfp_te, desc_te_scaled])
        logger.info(f"  pH {ph}: {X_tr.shape[1]} features (2048 ECFP + {desc_tr.shape[1]} RDKit 2D)")

        features_by_ph[ph] = {"X_train": X_tr, "X_test": X_te}

    # ── 3. Hyperparameter search space ───────────────────────────────
    param_distributions = {
        "n_estimators": randint(100, 1000),
        "max_depth": randint(3, 12),
        "learning_rate": uniform(0.01, 0.29),  # 0.01 to 0.3
        "subsample": uniform(0.5, 0.5),  # 0.5 to 1.0
        "colsample_bytree": uniform(0.3, 0.7),  # 0.3 to 1.0
        "min_child_weight": randint(1, 10),
        "gamma": uniform(0, 5),
        "reg_alpha": uniform(0, 1),
        "reg_lambda": uniform(0.5, 2.5),
    }

    # ── 4. Train and evaluate per endpoint ────────────────────────────
    metric_rows = []
    prediction_rows = []
    best_params_rows = []

    for ep in ENDPOINTS:
        train_mask = train_df[ep].notna().values
        test_mask = test_df[ep].notna().values

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_test < 10:
            logger.warning(f"Skipping {ep}: only {n_test} test molecules")
            continue

        ph = ENDPOINT_PH[ep]
        X_tr = features_by_ph[ph]["X_train"][train_mask]
        X_te = features_by_ph[ph]["X_test"][test_mask]
        y_tr_raw = train_df[ep].values[train_mask]
        y_te_raw = test_df[ep].values[test_mask]
        te_names = test_names[test_mask]

        # Log-transform targets for training and evaluation (except LogD)
        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_tr = clip_and_log_transform(y_tr_raw)
            y_te = clip_and_log_transform(y_te_raw)
        else:
            y_tr = y_tr_raw
            y_te = y_te_raw

        # Baseline RAE denominator (competition formula: mean(|true - mean(true)|))
        baseline_mad = np.mean(np.abs(y_te - np.mean(y_te)))

        logger.info(f"  {ep} ({n_train} train, {n_test} test, pH={ph}) — tuning XGBoost")
        search = HalvingRandomSearchCV(
            XGBRegressor(
                tree_method="hist",
                random_state=42,
                verbosity=0,
            ),
            param_distributions,
            n_candidates=50,
            factor=3,
            cv=3,
            scoring="neg_mean_absolute_error",
            random_state=42,
            verbose=0,
        )
        search.fit(X_tr, y_tr)
        best = search.best_params_
        logger.info(f"    Best params: {best} (CV MAE={-search.best_score_:.3f})")

        best_params_rows.append({"endpoint": ep, "ph": ph, **best})

        model = search.best_estimator_
        y_pred = model.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        sp_r, _ = spearmanr(y_te, y_pred)
        kt, _ = kendalltau(y_te, y_pred)
        rae = mae / baseline_mad if baseline_mad > 0 else np.nan

        metric_rows.append({
            "endpoint": ep,
            "ph": ph,
            "n_train": n_train,
            "n_test": n_test,
            "mae": mae,
            "r2": r2,
            "spearman_r": sp_r,
            "kendall_tau": kt,
            "rae": rae,
            "baseline_mad": baseline_mad,
            "log_transformed": ep in LOG_TRANSFORM_ENDPOINTS,
        })

        logger.info(
            f"    MAE={mae:.3f}, R2={r2:.3f}, "
            f"Spearman={sp_r:.3f}, Kendall={kt:.3f}, RAE={rae:.3f}"
        )

        for j in range(len(y_te)):
            prediction_rows.append({
                "Molecule Name": te_names[j],
                "endpoint": ep,
                "y_true": y_te[j],
                "y_pred": y_pred[j],
            })

    metrics_df = pd.DataFrame(metric_rows)
    ma_rae = metrics_df["rae"].mean()
    logger.info(f"MA-RAE: {ma_rae:.3f}")

    # ── 5. Save data ─────────────────────────────────────────────────
    metrics_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    logger.info(f"Saved overall_metrics.csv ({len(metrics_df)} rows)")

    params_df = pd.DataFrame(best_params_rows)
    params_df.to_csv(output_dir / "best_params.csv", index=False)
    logger.info("Saved best_params.csv")

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_parquet(output_dir / "predictions.parquet", index=False)
    logger.info(f"Saved predictions.parquet ({len(pred_df)} rows)")

    # ── 6. Plots ─────────────────────────────────────────────────────
    active_endpoints = sorted(metrics_df["endpoint"].unique())
    x = np.arange(len(active_endpoints))
    width = 0.6

    for metric, ylabel, fname in [
        ("r2", "R²", "r2_by_endpoint"),
        ("mae", "MAE (log-scale)", "mae_by_endpoint"),
        ("spearman_r", "Spearman ρ", "spearman_by_endpoint"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [metrics_df[metrics_df["endpoint"] == ep][metric].values[0] for ep in active_endpoints]
        ax.bar(x, vals, width, color="steelblue", edgecolor="white", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by endpoint — XGBoost baseline (original split)")
        if metric == "r2":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / f"{fname}.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {fname}.png")
        plt.close("all")

    # Scatter: y_true vs y_pred (3x3 grid)
    n_ep = len(active_endpoints)
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = axes.flatten()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        ep_pred = pred_df[pred_df["endpoint"] == ep]
        ax.scatter(ep_pred["y_true"], ep_pred["y_pred"], s=5, alpha=0.3,
                   color="steelblue", rasterized=True)

        lo = min(ep_pred["y_true"].min(), ep_pred["y_pred"].min())
        hi = max(ep_pred["y_true"].max(), ep_pred["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

        row = metrics_df[metrics_df["endpoint"] == ep]
        r2_val = row["r2"].values[0]
        ax.text(0.05, 0.95, f"R²={r2_val:.3f}",
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                color="steelblue")

        suffix = " (log)" if ep in LOG_TRANSFORM_ENDPOINTS else ""
        ax.set_xlabel(f"True{suffix}")
        ax.set_ylabel(f"Predicted{suffix}")
        ax.set_title(ep)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Predictions vs true values — XGBoost baseline (original split)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_predictions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved scatter_predictions.png")
    plt.close("all")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
