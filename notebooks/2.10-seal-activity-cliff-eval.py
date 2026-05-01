#!/usr/bin/env python
"""Activity cliff evaluation (paper Fig S3, ref: MoleculeACE).

Identifies activity cliffs — pairs of structurally similar molecules with large
activity differences — and evaluates XGBoost or Chemprop D-MPNN performance on
cliff vs non-cliff molecules. Uses cluster-split CV (repeat 0, 5 folds) to train
and evaluate. Demonstrates that smoothly interpolating models fail on activity cliffs.

Usage:
    pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py
    pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py --model chemprop
    pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py --combined

Model-agnostic outputs:
    data/processed/2.10-seal-activity-cliff-eval/cliff_stats.csv
    data/processed/2.10-seal-activity-cliff-eval/cliff_molecules.csv
    data/processed/2.10-seal-activity-cliff-eval/cliff_characterization.png

Outputs (per model):
    data/processed/2.10-seal-activity-cliff-eval/{model}/cliff_vs_noncliff_errors.csv
    data/processed/2.10-seal-activity-cliff-eval/{model}/summary_metrics.csv
    data/processed/2.10-seal-activity-cliff-eval/{model}/squared_error_distributions.png
    data/processed/2.10-seal-activity-cliff-eval/{model}/median_se_by_endpoint.png
    data/processed/2.10-seal-activity-cliff-eval/{model}/mae_by_endpoint.png
    data/processed/2.10-seal-activity-cliff-eval/{model}/r2_by_endpoint.png
    data/processed/2.10-seal-activity-cliff-eval/{model}/spearman_by_endpoint.png
    data/processed/2.10-seal-activity-cliff-eval/{model}/kendall_by_endpoint.png
    data/processed/2.10-seal-activity-cliff-eval/{model}/rae_by_endpoint.png
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

from polaris_generalization.chemprop_utils import train_chemprop
from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.tuning import tune_xgboost
from polaris_generalization.visualization import (
    DEFAULT_DPI,
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


def compute_features(smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Compute ECFP4 + full RDKit 2D descriptors."""
    ecfp = compute_ecfp4(smiles_list)
    desc = compute_rdkit_descriptors(smiles_list)
    return ecfp, desc


def identify_cliffs(
    dist_sub: np.ndarray,
    values: np.ndarray,
    sim_threshold: float = 0.85,
    min_diff: float = 2.0,
) -> tuple[set[int], list[tuple[int, int, float, float]], float]:
    """Identify activity cliff molecules.

    Args:
        dist_sub: Square distance sub-matrix for molecules with this endpoint.
        values: Activity values (log-transformed if applicable).
        sim_threshold: Minimum Tanimoto similarity for a pair to be considered
            structurally similar (distance < 1 - sim_threshold).
        min_diff: Absolute activity difference threshold to define a cliff.
            For log-transformed endpoints this is in log10(x+1) space;
            for LogD this is in raw units.

    Returns:
        cliff_indices: Set of local indices that participate in at least one cliff.
        cliff_pairs: List of (i, j, similarity, activity_diff) tuples.
        diff_threshold: The activity difference threshold used (= min_diff).
    """
    dist_cutoff = 1.0 - sim_threshold
    n = len(values)

    # Find all similar pairs and their activity differences
    similar_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist_sub[i, j] < dist_cutoff:
                diff = abs(values[i] - values[j])
                sim = 1.0 - dist_sub[i, j]
                similar_pairs.append((i, j, sim, diff))

    if not similar_pairs:
        return set(), [], 0.0

    diff_threshold = min_diff

    cliff_pairs = [(i, j, sim, d) for i, j, sim, d in similar_pairs if d >= diff_threshold]
    cliff_indices = set()
    for i, j, _, _ in cliff_pairs:
        cliff_indices.add(i)
        cliff_indices.add(j)

    return cliff_indices, cliff_pairs, diff_threshold


def _migrate_flat_outputs(output_dir: Path) -> None:
    """Move pre-model-flag XGBoost outputs into xgboost/ subdirectory."""
    flat_files = [
        "cliff_vs_noncliff_errors.csv", "summary_metrics.csv",
        "squared_error_distributions.png",
        "median_se_by_endpoint.png", "mae_by_endpoint.png", "r2_by_endpoint.png",
        "spearman_by_endpoint.png", "kendall_by_endpoint.png", "rae_by_endpoint.png",
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


def _generate_figures(errors_df: pd.DataFrame, metrics_df: pd.DataFrame,
                      model_dir: Path, dpi: int) -> None:
    """Generate all per-model figures."""
    active_endpoints = sorted(metrics_df["endpoint"].unique())
    n_ep = len(active_endpoints)

    # Figure A: Squared error distributions (3×3 grid)
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        cliff_se = errors_df[(errors_df["endpoint"] == ep) & (errors_df["is_cliff"])]["squared_error"]
        noncliff_se = errors_df[(errors_df["endpoint"] == ep) & (~errors_df["is_cliff"])]["squared_error"]

        if cliff_se.empty or noncliff_se.empty:
            ax.set_visible(False)
            continue

        all_se = np.concatenate([cliff_se.values, noncliff_se.values])
        lo = max(all_se.min(), 1e-8)
        hi = all_se.max()
        bins = np.logspace(np.log10(lo), np.log10(hi), 40)

        ax.hist(noncliff_se, bins=bins, density=True, alpha=0.6, color="steelblue",
                label=f"Non-cliff (med={np.median(noncliff_se):.3f})", edgecolor="white")
        ax.hist(cliff_se, bins=bins, density=True, alpha=0.6, color="coral",
                label=f"Cliff (med={np.median(cliff_se):.3f})", edgecolor="white")

        ax.set_xscale("log")
        ax.set_xlabel("Squared error")
        ax.set_ylabel("Density")
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Cliff vs non-cliff squared error distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(model_dir / "squared_error_distributions.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved squared_error_distributions.png")
    plt.close("all")

    # Figure B: Per-metric bar charts
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

        cliff_vals = []
        noncliff_vals = []
        for ep in active_endpoints:
            cliff_row = metrics_df[(metrics_df["endpoint"] == ep) & (metrics_df["set"] == "cliff")]
            noncliff_row = metrics_df[(metrics_df["endpoint"] == ep) & (metrics_df["set"] == "non-cliff")]
            cliff_vals.append(cliff_row[col].values[0] if not cliff_row.empty else np.nan)
            noncliff_vals.append(noncliff_row[col].values[0] if not noncliff_row.empty else np.nan)

        ax.bar(x - width / 2, noncliff_vals, width, label="Non-cliff",
               color="steelblue", edgecolor="white", alpha=0.8)
        ax.bar(x + width / 2, cliff_vals, width, label="Cliff",
               color="coral", edgecolor="white", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(active_endpoints, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Cliff vs non-cliff {ylabel} by endpoint")
        ax.legend()
        if col in ("r2",):
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(model_dir / f"{fname}_by_endpoint.png", dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved {fname}_by_endpoint.png")
        plt.close("all")


def _generate_combined_figures(output_dir: Path, dpi: int) -> None:
    """Load both models' metrics and produce side-by-side comparison figures."""
    xgb_path = output_dir / "xgboost" / "summary_metrics.csv"
    chemprop_path = output_dir / "chemprop" / "summary_metrics.csv"

    missing = [m for m, p in [("xgboost", xgb_path), ("chemprop", chemprop_path)] if not p.exists()]
    if missing:
        logger.error(f"Missing results for: {missing}. Run those models first.")
        return

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    xgb = pd.read_csv(xgb_path)
    chemprop = pd.read_csv(chemprop_path)

    for set_label in ("cliff", "non-cliff"):
        xgb_set = xgb[xgb["set"] == set_label].copy()
        chemprop_set = chemprop[chemprop["set"] == set_label].copy()
        data_by_model = {"xgboost": xgb_set, "chemprop": chemprop_set}

        fname_label = set_label.replace("-", "")
        for metric, ylabel, fname in [
            ("mae", "MAE", "mae"),
            ("r2", "R²", "r2"),
            ("spearman_r", "Spearman ρ", "spearman"),
            ("rae", "RAE", "rae"),
        ]:
            plot_model_comparison_bars(
                data_by_model, "endpoint", metric, ylabel,
                f"{ylabel} — XGBoost vs Chemprop ({set_label})",
                combined_dir / f"{fname}_{fname_label}_comparison.png", dpi=dpi,
            )
            logger.info(f"Saved {fname}_{fname_label}_comparison.png")

    logger.info(f"Combined figures saved to {combined_dir}")


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.10-seal-activity-cliff-eval", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    sim_threshold: float = typer.Option(0.85, help="Tanimoto similarity threshold for similar pairs"),
    min_diff: float = typer.Option(1.0, help="Absolute activity difference threshold for cliff definition"),
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
    # Use repeat 0 only
    cluster_folds = cluster_folds[cluster_folds["repeat"] == 0]

    # ── 2. Identify activity cliffs per endpoint ──────────────────────
    cliff_stats_rows = []
    cliff_mol_rows = []
    # Map: endpoint -> set of cliff molecule names
    cliff_names_by_ep: dict[str, set[str]] = {}

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        n_mol = len(ep_df)
        names = ep_df["Molecule Name"].values

        # Sub-distance-matrix
        idx_in_full = np.array([name_to_dist_idx[n] for n in names])
        D_sub = dist_square[np.ix_(idx_in_full, idx_in_full)]

        # Activity values (log-transformed for non-LogD)
        raw_values = ep_df[ep].values
        if ep in LOG_TRANSFORM_ENDPOINTS:
            values = clip_and_log_transform(raw_values)
        else:
            values = raw_values

        cliff_indices, cliff_pairs, diff_threshold = identify_cliffs(
            D_sub, values, sim_threshold, min_diff
        )

        cliff_name_set = {names[i] for i in cliff_indices}
        cliff_names_by_ep[ep] = cliff_name_set

        n_cliff = len(cliff_indices)
        n_pairs = len(cliff_pairs)
        logger.info(
            f"  {ep}: {n_pairs} cliff pairs, {n_cliff}/{n_mol} cliff molecules "
            f"({100 * n_cliff / n_mol:.1f}%), diff_threshold={diff_threshold:.3f}"
        )

        cliff_stats_rows.append({
            "endpoint": ep,
            "n_molecules": n_mol,
            "n_similar_pairs": len([1 for i in range(len(values)) for j in range(i+1, len(values))
                                    if D_sub[i, j] < (1.0 - sim_threshold)]),
            "n_cliff_pairs": n_pairs,
            "n_cliff_molecules": n_cliff,
            "pct_cliff": 100 * n_cliff / n_mol,
            "diff_threshold": diff_threshold,
            "sim_threshold": sim_threshold,
        })

        # Per-molecule cliff info
        cliff_pair_count = {}
        for i, j, sim, d in cliff_pairs:
            cliff_pair_count[names[i]] = cliff_pair_count.get(names[i], 0) + 1
            cliff_pair_count[names[j]] = cliff_pair_count.get(names[j], 0) + 1

        for name, count in cliff_pair_count.items():
            cliff_mol_rows.append({
                "Molecule Name": name,
                "endpoint": ep,
                "n_cliff_pairs": count,
            })

    cliff_stats_df = pd.DataFrame(cliff_stats_rows)
    cliff_stats_df.to_csv(output_dir / "cliff_stats.csv", index=False)
    logger.info("Saved cliff_stats.csv")

    cliff_mol_df = pd.DataFrame(cliff_mol_rows)
    cliff_mol_df.to_csv(output_dir / "cliff_molecules.csv", index=False)
    logger.info(f"Saved cliff_molecules.csv ({len(cliff_mol_df)} rows)")

    # Cliff characterization figure (model-agnostic)
    ep_order = cliff_stats_df.sort_values("pct_cliff", ascending=False)["endpoint"]
    x = np.arange(len(ep_order))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    pcts = [cliff_stats_df[cliff_stats_df["endpoint"] == ep]["pct_cliff"].values[0] for ep in ep_order]
    ax.bar(x, pcts, color="coral", edgecolor="white", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ep_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% cliff molecules")
    ax.set_title(f"Activity cliff prevalence (sim>{sim_threshold})")

    ax = axes[1]
    ax.bar(x, [cliff_stats_df[cliff_stats_df["endpoint"] == ep]["n_cliff_pairs"].values[0] for ep in ep_order],
           color="coral", edgecolor="white", alpha=0.8, label="Cliff pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(ep_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of cliff pairs")
    ax.set_title("Cliff pair counts per endpoint")

    fig.tight_layout()
    fig.savefig(output_dir / "cliff_characterization.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved cliff_characterization.png")
    plt.close("all")

    # Skip training if outputs already exist
    errors_path = model_dir / "cliff_vs_noncliff_errors.csv"
    metrics_path = model_dir / "summary_metrics.csv"
    if errors_path.exists() and metrics_path.exists():
        logger.info(f"Existing {model} outputs found — regenerating figures only")
        errors_df = pd.read_csv(errors_path)
        metrics_df = pd.read_csv(metrics_path)
        _generate_figures(errors_df, metrics_df, model_dir, dpi)
        logger.info("Figures regenerated (no models retrained)")
        return

    # ── 3. Train model with cluster-split CV and evaluate ─────────────
    error_rows = []
    metric_rows = []

    optuna_cache = INTERIM_DATA_DIR / "optuna_cache"
    chemprop_cache = INTERIM_DATA_DIR / "chemprop_pred_cache"

    for ep in ENDPOINTS:
        ph = ENDPOINT_PH[ep]
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        names = ep_df["Molecule Name"].values
        cliff_name_set = cliff_names_by_ep.get(ep, set())

        # Get fold assignments for this endpoint
        ep_folds = cluster_folds[cluster_folds["endpoint"] == ep]
        fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
        fold_ids = np.array([fold_map.get(n, -1) for n in names])
        unique_folds = sorted(set(fold_ids[fold_ids >= 0]))

        if len(unique_folds) < 2:
            logger.warning(f"Skipping {ep}: not enough folds")
            continue

        # Activity values
        raw_values = ep_df[ep].values
        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_all = clip_and_log_transform(raw_values)
        else:
            y_all = raw_values

        # Compute features once for this endpoint (XGBoost only)
        if model == "xgboost":
            smiles = ep_df["SMILES"].tolist()
            prot_smiles = protonate_at_ph(smiles, ph)
            ecfp, desc = compute_features(prot_smiles)

            # Remove zero-variance descriptors
            variance = desc.var(axis=0)
            nonzero_var = variance > 0
            desc = desc[:, nonzero_var]

            # Scale descriptors
            scaler = StandardScaler()
            desc_scaled = scaler.fit_transform(desc)
            X_all = np.hstack([ecfp, desc_scaled])
        else:
            smiles = ep_df["SMILES"].tolist()
            all_prot = protonate_at_ph(smiles, ph)

        logger.info(f"  {ep}: {len(unique_folds)} folds, {len(cliff_name_set)} cliff molecules")

        for fold_id in unique_folds:
            test_mask = fold_ids == fold_id
            train_mask = (fold_ids >= 0) & ~test_mask

            y_tr = y_all[train_mask]
            y_te = y_all[test_mask]
            te_names = names[test_mask]

            if len(y_te) < 10:
                continue

            if model == "xgboost":
                X_tr = X_all[train_mask]
                X_te = X_all[test_mask]
                model_obj, _, _ = tune_xgboost(X_tr, y_tr, cache_dir=optuna_cache,
                                               cache_key=f"{ep}_cluster_fold{fold_id}")
                y_pred = model_obj.predict(X_te)
            else:
                train_prot = [all_prot[i] for i in np.where(train_mask)[0]]
                test_prot = [all_prot[i] for i in np.where(test_mask)[0]]
                y_pred = train_chemprop(
                    train_prot, y_tr, test_prot,
                    cache_dir=chemprop_cache,
                    cache_key=f"2.10_{ep}_cluster_fold{fold_id}",
                    checkpoint_dir=model_dir / "models",
                )

            se = (y_te - y_pred) ** 2
            is_cliff = np.array([n in cliff_name_set for n in te_names])

            for i in range(len(y_te)):
                error_rows.append({
                    "Molecule Name": te_names[i],
                    "endpoint": ep,
                    "fold": fold_id,
                    "is_cliff": bool(is_cliff[i]),
                    "y_true": y_te[i],
                    "y_pred": y_pred[i],
                    "squared_error": se[i],
                })

    errors_df = pd.DataFrame(error_rows)
    errors_df.to_csv(errors_path, index=False)
    logger.info(f"Saved cliff_vs_noncliff_errors.csv ({len(errors_df)} rows)")

    # ── 4. Aggregate metrics: cliff vs non-cliff ──────────────────────
    for ep in ENDPOINTS:
        ep_errors = errors_df[errors_df["endpoint"] == ep]
        if ep_errors.empty:
            continue

        for label, subset in [("cliff", True), ("non-cliff", False)]:
            sub = ep_errors[ep_errors["is_cliff"] == subset]
            if len(sub) < 5:
                logger.warning(f"  {ep}/{label}: only {len(sub)} molecules, skipping metrics")
                continue

            y_true = sub["y_true"].values
            y_pred = sub["y_pred"].values
            se = sub["squared_error"].values

            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            sp_r, _ = spearmanr(y_true, y_pred)
            kt, _ = kendalltau(y_true, y_pred)
            baseline_mad = np.mean(np.abs(y_true - np.mean(y_true)))
            rae = mae / baseline_mad if baseline_mad > 0 else np.nan

            logger.info(
                f"    {ep}/{label} (n={len(sub)}): MAE={mae:.3f}, R2={r2:.3f}, "
                f"Spearman={sp_r:.3f}, Kendall={kt:.3f}, RAE={rae:.3f}"
            )

            metric_rows.append({
                "endpoint": ep,
                "set": label,
                "n": len(sub),
                "mae": mae,
                "r2": r2,
                "spearman_r": sp_r,
                "kendall_tau": kt,
                "rae": rae,
                "baseline_mad": baseline_mad,
                "median_se": np.median(se),
                "mean_se": np.mean(se),
            })

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved summary_metrics.csv")

    # ── 5. Figures ────────────────────────────────────────────────────
    _generate_figures(errors_df, metrics_df, model_dir, dpi)
    logger.info(f"All outputs saved to {model_dir}")


if __name__ == "__main__":
    app()
