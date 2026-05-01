#!/usr/bin/env python
"""Baseline model performance on the original train/test split.

Trains XGBoost or Chemprop D-MPNN on the competition split (5,326 / 2,282).
Molecules are protonated at the relevant assay pH using dimorphite_dl:
  - pH 7.4: LogD, KSOL, HLM/MLM CLint, MPPB, MBPB, MGMB
  - pH 6.5: Caco-2 Papp A>B, Caco-2 Efflux (apical compartment)

XGBoost uses ECFP4 + full RDKit 2D descriptors (~200), Optuna TPE-tuned per endpoint.
Chemprop D-MPNN uses default hyperparameters (hidden_dim=300, depth=3, 50 epochs).

For endpoints not on log scale (everything except LogD), targets are log-transformed
via log10(clip(x, 1e-10) + 1), matching the OpenADMET competition protocol.

Usage:
    pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py
    pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py --model chemprop
    pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py --combined

Outputs (per model):
    data/processed/2.08-seal-baseline-performance/{model}/overall_metrics.csv
    data/processed/2.08-seal-baseline-performance/{model}/predictions.parquet
    data/processed/2.08-seal-baseline-performance/{model}/best_params.csv  (xgboost only)
    data/processed/2.08-seal-baseline-performance/{model}/r2_by_endpoint.png
    data/processed/2.08-seal-baseline-performance/{model}/mae_by_endpoint.png
    data/processed/2.08-seal-baseline-performance/{model}/spearman_by_endpoint.png
    data/processed/2.08-seal-baseline-performance/{model}/scatter_predictions.png

Combined outputs:
    data/processed/2.08-seal-baseline-performance/combined/metrics_comparison.csv
    data/processed/2.08-seal-baseline-performance/combined/{metric}_comparison.png
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
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from polaris_generalization.chemprop_utils import train_chemprop
from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.tuning import tune_xgboost
from polaris_generalization.visualization import (
    DEFAULT_DPI,
    MODEL_COLORS,
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

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


# ── Utilities ─────────────────────────────────────────────────────────────────

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
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_features(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return compute_ecfp4(smiles_list), compute_rdkit_descriptors(smiles_list)


# ── Migration ─────────────────────────────────────────────────────────────────

def _migrate_flat_outputs(output_dir: Path) -> None:
    """Move pre-model-flag XGBoost outputs into xgboost/ subdirectory."""
    flat_files = [
        "overall_metrics.csv", "predictions.parquet", "best_params.csv",
        "r2_by_endpoint.png", "mae_by_endpoint.png", "spearman_by_endpoint.png",
        "scatter_predictions.png",
    ]
    xgboost_dir = output_dir / "xgboost"
    migrated = []
    for fname in flat_files:
        src = output_dir / fname
        dst = xgboost_dir / fname
        if src.exists() and not dst.exists():
            xgboost_dir.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            migrated.append(fname)
    if migrated:
        logger.info(f"Migrated {len(migrated)} existing files → xgboost/")


# ── Figure generation ─────────────────────────────────────────────────────────

def _generate_figures(
    pred_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model_dir: Path,
    model: str,
    dpi: int,
) -> None:
    """Generate all per-model figures from prediction and metrics dataframes."""
    active_endpoints = sorted(metrics_df["endpoint"].unique())
    x = np.arange(len(active_endpoints))
    width = 0.6
    model_label = MODEL_LABELS.get(model, model)
    model_color = MODEL_COLORS.get(model, "steelblue")

    ci_map = {
        "r2": ("r2_ci_lo", "r2_ci_hi"),
        "mae": ("mae_ci_lo", "mae_ci_hi"),
        "spearman_r": ("spearman_ci_lo", "spearman_ci_hi"),
    }

    for metric, ylabel, fname in [
        ("r2", "R²", "r2_by_endpoint"),
        ("mae", "MAE (log-scale)", "mae_by_endpoint"),
        ("spearman_r", "Spearman ρ", "spearman_by_endpoint"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [metrics_df.loc[metrics_df["endpoint"] == ep, metric].values[0] for ep in active_endpoints]

        lo_col, hi_col = ci_map[metric]
        ci_lo = [metrics_df.loc[metrics_df["endpoint"] == ep, lo_col].values[0] for ep in active_endpoints]
        ci_hi = [metrics_df.loc[metrics_df["endpoint"] == ep, hi_col].values[0] for ep in active_endpoints]
        yerr = np.array([[v - lo, hi - v] for v, lo, hi in zip(vals, ci_lo, ci_hi)]).T

        ax.bar(x, vals, width, color=model_color, edgecolor="white", alpha=0.8,
               yerr=yerr, capsize=4, error_kw={"linewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} by endpoint — {model_label} (original split)")
        if metric == "r2":
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        fig.tight_layout()
        fig.savefig(model_dir / f"{fname}.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {fname}.png")
        plt.close("all")

    # Scatter: y_true vs y_pred
    n_ep = len(active_endpoints)
    ncols = min(n_ep, 3)
    nrows = (n_ep + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        ep_pred = pred_df[pred_df["endpoint"] == ep]
        ax.scatter(ep_pred["y_true"], ep_pred["y_pred"], s=5, alpha=0.3,
                   color=model_color, rasterized=True)

        lo = min(ep_pred["y_true"].min(), ep_pred["y_pred"].min())
        hi = max(ep_pred["y_true"].max(), ep_pred["y_pred"].max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

        r2_val = metrics_df.loc[metrics_df["endpoint"] == ep, "r2"].values[0]
        ax.text(0.05, 0.95, f"R²={r2_val:.3f}", transform=ax.transAxes,
                fontsize=8, verticalalignment="top", color=model_color)

        suffix = " (log)" if ep in LOG_TRANSFORM_ENDPOINTS else ""
        ax.set_xlabel(f"True{suffix}")
        ax.set_ylabel(f"Predicted{suffix}")
        ax.set_title(ep)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Predictions vs true — {model_label} (original split)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(model_dir / "scatter_predictions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved scatter_predictions.png")
    plt.close("all")


# ── Combined figures ──────────────────────────────────────────────────────────

def _generate_combined_figures(output_dir: Path, dpi: int) -> None:
    """Load both models' metrics and produce side-by-side comparison figures."""
    xgb_path = output_dir / "xgboost" / "overall_metrics.csv"
    chemprop_path = output_dir / "chemprop" / "overall_metrics.csv"

    missing = [m for m, p in [("xgboost", xgb_path), ("chemprop", chemprop_path)] if not p.exists()]
    if missing:
        logger.error(f"Missing results for: {missing}. Run those models first.")
        return

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    xgb = pd.read_csv(xgb_path)
    chemprop = pd.read_csv(chemprop_path)

    # Comparison CSV
    merge_cols = ["endpoint", "mae", "r2", "spearman_r", "rae", "kendall_tau"]
    comparison_df = (
        xgb[merge_cols].rename(columns={c: f"{c}_xgboost" for c in merge_cols if c != "endpoint"})
        .merge(
            chemprop[merge_cols].rename(columns={c: f"{c}_chemprop" for c in merge_cols if c != "endpoint"}),
            on="endpoint",
        )
    )
    comparison_df.to_csv(combined_dir / "metrics_comparison.csv", index=False)
    logger.info("Saved metrics_comparison.csv")

    # Grouped bar charts
    data_by_model = {"xgboost": xgb, "chemprop": chemprop}
    for metric, ylabel, fname in [
        ("r2", "R²", "r2_comparison"),
        ("mae", "MAE", "mae_comparison"),
        ("spearman_r", "Spearman ρ", "spearman_comparison"),
        ("rae", "RAE", "rae_comparison"),
    ]:
        plot_model_comparison_bars(
            data_by_model, "endpoint", metric, ylabel,
            f"{ylabel} — XGBoost vs Chemprop (original split)",
            combined_dir / f"{fname}.png", dpi=dpi,
        )
        logger.info(f"Saved {fname}.png")

    logger.info(f"Combined figures saved to {combined_dir}")


# ── Main ─────────────────────────────────────────────────────────────────────

@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.08-seal-baseline-performance", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    model: str = typer.Option("xgboost", help="Model architecture: xgboost or chemprop"),
    combined: bool = typer.Option(False, help="Generate combined XGBoost+Chemprop comparison figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    if combined:
        _generate_combined_figures(output_dir, dpi)
        return

    if model not in ("xgboost", "chemprop"):
        logger.error(f"Unknown model: {model}. Choose xgboost or chemprop.")
        raise typer.Exit(1)

    _migrate_flat_outputs(output_dir)

    model_dir = output_dir / model
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip training if predictions already exist ────────────────────
    pred_path = model_dir / "predictions.parquet"
    metrics_path = model_dir / "overall_metrics.csv"
    if pred_path.exists() and metrics_path.exists():
        logger.info(f"Existing {model} predictions found — regenerating figures only")
        pred_df = pd.read_parquet(pred_path)
        metrics_df = pd.read_csv(metrics_path)
        _generate_figures(pred_df, metrics_df, model_dir, model, dpi)
        return

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

    # ── 2. Protonate at each unique pH (both models use pH-appropriate SMILES) ──
    unique_phs = sorted(set(ENDPOINT_PH.values()))
    prot_by_ph: dict[float, dict] = {}
    for ph in unique_phs:
        logger.info(f"Protonating at pH {ph}")
        prot_by_ph[ph] = {
            "train": protonate_at_ph(train_smiles, ph),
            "test": protonate_at_ph(test_smiles, ph),
        }

    # ── 3. XGBoost: compute ECFP4 + RDKit 2D features ────────────────
    features_by_ph: dict[float, dict] = {}
    if model == "xgboost":
        for ph in unique_phs:
            logger.info(f"Computing XGBoost features at pH {ph}")
            ecfp_tr, desc_tr = compute_features(prot_by_ph[ph]["train"])
            ecfp_te, desc_te = compute_features(prot_by_ph[ph]["test"])

            nonzero_var = desc_tr.var(axis=0) > 0
            desc_tr = desc_tr[:, nonzero_var]
            desc_te = desc_te[:, nonzero_var]

            scaler = StandardScaler()
            desc_tr_s = scaler.fit_transform(desc_tr)
            desc_te_s = scaler.transform(desc_te)

            X_tr = np.hstack([ecfp_tr, desc_tr_s])
            X_te = np.hstack([ecfp_te, desc_te_s])
            logger.info(f"  pH {ph}: {X_tr.shape[1]} features (2048 ECFP + {desc_tr.shape[1]} RDKit 2D)")
            features_by_ph[ph] = {"X_train": X_tr, "X_test": X_te}

    # ── 4. Train and evaluate per endpoint ───────────────────────────
    optuna_cache = INTERIM_DATA_DIR / "optuna_cache"
    chemprop_cache = INTERIM_DATA_DIR / "chemprop_pred_cache"

    metric_rows = []
    prediction_rows = []
    best_params_rows = []

    for ep in ENDPOINTS:
        train_mask = train_df[ep].notna().values
        test_mask = test_df[ep].notna().values
        n_train, n_test = train_mask.sum(), test_mask.sum()

        if n_test < 10:
            logger.warning(f"Skipping {ep}: only {n_test} test molecules")
            continue

        ph = ENDPOINT_PH[ep]
        y_tr_raw = train_df[ep].values[train_mask]
        y_te_raw = test_df[ep].values[test_mask]
        te_names = test_names[test_mask]

        y_tr = clip_and_log_transform(y_tr_raw) if ep in LOG_TRANSFORM_ENDPOINTS else y_tr_raw
        y_te = clip_and_log_transform(y_te_raw) if ep in LOG_TRANSFORM_ENDPOINTS else y_te_raw

        baseline_mad = np.mean(np.abs(y_te - np.mean(y_te)))

        if model == "xgboost":
            X_tr = features_by_ph[ph]["X_train"][train_mask]
            X_te = features_by_ph[ph]["X_test"][test_mask]
            logger.info(f"  {ep} ({n_train} train, {n_test} test) — tuning XGBoost")
            model_obj, best, _ = tune_xgboost(X_tr, y_tr, cache_dir=optuna_cache, cache_key=f"{ep}_baseline")
            best_params_rows.append({"endpoint": ep, "ph": ph, **best})
            y_pred = model_obj.predict(X_te)
        else:
            train_prot = [s for s, m in zip(prot_by_ph[ph]["train"], train_mask) if m]
            test_prot = [s for s, m in zip(prot_by_ph[ph]["test"], test_mask) if m]
            logger.info(f"  {ep} ({n_train} train, {n_test} test) — training Chemprop D-MPNN")
            y_pred = train_chemprop(
                train_prot, y_tr, test_prot,
                cache_dir=chemprop_cache,
                cache_key=f"2.08_{ep}_baseline",
                checkpoint_dir=model_dir / "models",
            )

        mae = mean_absolute_error(y_te, y_pred)
        r2 = r2_score(y_te, y_pred)
        sp_r, _ = spearmanr(y_te, y_pred)
        kt, _ = kendalltau(y_te, y_pred)
        rae = mae / baseline_mad if baseline_mad > 0 else np.nan

        logger.info(f"    MAE={mae:.3f}, R²={r2:.3f}, Spearman={sp_r:.3f}, RAE={rae:.3f}")

        metric_rows.append({
            "endpoint": ep, "ph": ph, "n_train": n_train, "n_test": n_test,
            "mae": mae, "r2": r2, "spearman_r": sp_r, "kendall_tau": kt,
            "rae": rae, "baseline_mad": baseline_mad,
            "log_transformed": ep in LOG_TRANSFORM_ENDPOINTS,
        })

        for j in range(len(y_te)):
            prediction_rows.append({
                "Molecule Name": te_names[j],
                "endpoint": ep,
                "y_true": float(y_te[j]),
                "y_pred": float(y_pred[j]),
            })

    metrics_df = pd.DataFrame(metric_rows)
    logger.info(f"MA-RAE: {metrics_df['rae'].mean():.3f}")

    # ── 5. Bootstrap 95% CIs ─────────────────────────────────────────
    logger.info("Computing bootstrap 95% CIs (1000 resamples)")
    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_rows = []

    for ep in metrics_df["endpoint"]:
        ep_preds = [r for r in prediction_rows if r["endpoint"] == ep]
        y_true = np.array([r["y_true"] for r in ep_preds])
        y_pred_arr = np.array([r["y_pred"] for r in ep_preds])
        baseline_mad = metrics_df.loc[metrics_df["endpoint"] == ep, "baseline_mad"].values[0]

        boot_mae, boot_r2, boot_sp, boot_rae = [], [], [], []
        for _ in range(n_boot):
            idx = rng.integers(0, len(y_true), size=len(y_true))
            yt, yp = y_true[idx], y_pred_arr[idx]
            b_mae = mean_absolute_error(yt, yp)
            boot_mae.append(b_mae)
            boot_r2.append(r2_score(yt, yp))
            boot_sp.append(spearmanr(yt, yp).statistic)
            boot_rae.append(b_mae / baseline_mad if baseline_mad > 0 else np.nan)

        boot_rows.append({
            "endpoint": ep,
            "mae_ci_lo": np.percentile(boot_mae, 2.5), "mae_ci_hi": np.percentile(boot_mae, 97.5),
            "r2_ci_lo": np.percentile(boot_r2, 2.5), "r2_ci_hi": np.percentile(boot_r2, 97.5),
            "spearman_ci_lo": np.percentile(boot_sp, 2.5), "spearman_ci_hi": np.percentile(boot_sp, 97.5),
            "rae_ci_lo": np.percentile(boot_rae, 2.5), "rae_ci_hi": np.percentile(boot_rae, 97.5),
        })
        logger.info(
            f"  {ep}: R²=[{boot_rows[-1]['r2_ci_lo']:.3f}, {boot_rows[-1]['r2_ci_hi']:.3f}]  "
            f"RAE=[{boot_rows[-1]['rae_ci_lo']:.3f}, {boot_rows[-1]['rae_ci_hi']:.3f}]"
        )

    metrics_df = metrics_df.merge(pd.DataFrame(boot_rows), on="endpoint")

    # ── 6. Save ───────────────────────────────────────────────────────
    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_parquet(pred_path, index=False)
    logger.info(f"Saved predictions.parquet ({len(pred_df)} rows)")

    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved overall_metrics.csv ({len(metrics_df)} rows)")

    if model == "xgboost" and best_params_rows:
        pd.DataFrame(best_params_rows).to_csv(model_dir / "best_params.csv", index=False)
        logger.info("Saved best_params.csv")

    # ── 7. Figures ────────────────────────────────────────────────────
    _generate_figures(pred_df, metrics_df, model_dir, model, dpi)
    logger.info(f"All outputs saved to {model_dir}")


if __name__ == "__main__":
    app()
