#!/usr/bin/env python
"""Molecular variant consistency analysis (paper Fig S4, ref: Srijit's LinkedIn).

Identifies groups of structurally related molecules (stereoisomers, scaffold
decorations) and tests whether model predictions are consistent within groups.
If ECFP fingerprints change significantly for minor structural changes, models
produce inconsistent predictions — exposing fingerprint artifacts rather than
learned chemistry.

Supports XGBoost (ECFP4 + RDKit 2D descriptors, Optuna-tuned) and Chemprop
D-MPNN. Use --combined to generate side-by-side comparison figures once both
models have been run.

Usage:
    pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py
    pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py --model chemprop
    pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py --combined

Model-agnostic outputs (saved directly to output_dir):
    data/processed/2.13-seal-molecular-variants/variant_groups.csv
    data/processed/2.13-seal-molecular-variants/fingerprint_similarity.csv
    data/processed/2.13-seal-molecular-variants/stereoisomer_activity_diffs.csv

Per-model outputs (saved to output_dir/{model}/):
    data/processed/2.13-seal-molecular-variants/{model}/consistency_metrics.csv
    data/processed/2.13-seal-molecular-variants/{model}/consistency_summary.csv
    data/processed/2.13-seal-molecular-variants/{model}/fingerprint_distances.png
    data/processed/2.13-seal-molecular-variants/{model}/prediction_consistency.png
    data/processed/2.13-seal-molecular-variants/{model}/consistency_heatmap.png
    data/processed/2.13-seal-molecular-variants/{model}/spread_scatter.png

Combined outputs:
    data/processed/2.13-seal-molecular-variants/combined/consistency_comparison.csv
    data/processed/2.13-seal-molecular-variants/combined/pred_cv_comparison.png
    data/processed/2.13-seal-molecular-variants/combined/consistency_ratio_comparison.png
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from dimorphite_dl import protonate_smiles as dimorphite_protonate
from loguru import logger
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
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

VARIANT_COLORS = {
    "stereoisomer": "coral",
    "scaffold_decoration": "steelblue",
    "random": "gray",
}
VARIANT_ORDER = ["stereoisomer", "scaffold_decoration", "random"]


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


def get_achiral_smiles(smi: str) -> str | None:
    """Strip stereochemistry and return achiral canonical SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def get_scaffold(smi: str) -> str | None:
    """Get generic Murcko scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return None


def compute_tanimoto_distances_within_group(
    smiles_list: list[str], nbits: int = 2048, radius: int = 2
) -> list[float]:
    """Compute all pairwise Tanimoto distances within a group of molecules."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    for mol in mols:
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True))

    distances = []
    for i, j in combinations(range(len(fps)), 2):
        if fps[i] is not None and fps[j] is not None:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1.0 - sim)
    return distances


# ── Migration ─────────────────────────────────────────────────────────────────

def _migrate_flat_outputs(output_dir: Path) -> None:
    """Move pre-model-flag XGBoost outputs into xgboost/ subdirectory."""
    flat_files = [
        "consistency_metrics.csv", "consistency_summary.csv",
        "fingerprint_distances.png", "prediction_consistency.png",
        "consistency_heatmap.png", "spread_scatter.png",
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


# ── Per-model figures ─────────────────────────────────────────────────────────

def _generate_figures(
    consistency_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    fp_sim_df: pd.DataFrame,
    model_dir: Path,
    dpi: int,
) -> None:
    """Generate per-model figures."""

    # Figure A: Fingerprint distance distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax_idx, vtype in enumerate(VARIANT_ORDER):
        ax = axes[ax_idx]
        vtype_dists = fp_sim_df[fp_sim_df["variant_type"] == vtype]["tanimoto_distance"].values

        if len(vtype_dists) == 0:
            ax.set_visible(False)
            continue

        ax.hist(
            vtype_dists, bins=50, density=True,
            color=VARIANT_COLORS[vtype], edgecolor="white", alpha=0.8,
        )
        ax.axvline(
            np.median(vtype_dists), color="black", linestyle="--", alpha=0.7,
            label=f"median={np.median(vtype_dists):.3f}",
        )
        ax.set_xlabel("Tanimoto distance")
        if ax_idx == 0:
            ax.set_ylabel("Density")
        ax.set_title(vtype.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Intra-group ECFP4 Tanimoto distances by variant type",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(model_dir / "fingerprint_distances.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fingerprint_distances.png")
    plt.close("all")

    # Figure B: Prediction consistency boxplots
    active_endpoints = sorted(consistency_df["endpoint"].unique())
    n_ep = len(active_endpoints)
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        ep_data = consistency_df[consistency_df["endpoint"] == ep]

        for i, vtype in enumerate(VARIANT_ORDER):
            vtype_data = ep_data[ep_data["variant_type"] == vtype]["pred_cv"].dropna().values
            if len(vtype_data) == 0:
                continue
            bp = ax.boxplot(
                vtype_data, positions=[i], widths=0.6,
                patch_artist=True, showfliers=False,
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(VARIANT_COLORS[vtype])
                patch.set_alpha(0.7)

        ax.set_xticks(range(len(VARIANT_ORDER)))
        ax.set_xticklabels(
            [v.replace("_", "\n") for v in VARIANT_ORDER], fontsize=7,
        )
        ax.set_ylabel("Prediction CV", fontsize=9)
        ax.set_title(ep, fontsize=10, fontweight="bold")

    for i in range(n_ep, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(
        "Within-group prediction CV by variant type and endpoint",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(model_dir / "prediction_consistency.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved prediction_consistency.png")
    plt.close("all")

    # Figure C: Consistency summary heatmap
    fig, ax = plt.subplots(figsize=(10, 7))

    heatmap_data = summary_df.pivot(
        index="endpoint", columns="variant_type", values="pred_cv_mean",
    )
    col_order = [c for c in VARIANT_ORDER if c in heatmap_data.columns]
    heatmap_data = heatmap_data[col_order]

    annot_data = summary_df.pivot(
        index="endpoint", columns="variant_type", values="n_groups",
    )
    annot_data = annot_data[col_order]
    annot_arr = np.empty(heatmap_data.shape, dtype=object)
    for i, ep in enumerate(heatmap_data.index):
        for j, col in enumerate(col_order):
            val = heatmap_data.loc[ep, col]
            n = annot_data.loc[ep, col]
            if pd.notna(val) and pd.notna(n):
                annot_arr[i, j] = f"{val:.3f}\n(n={int(n)})"
            else:
                annot_arr[i, j] = ""

    sns.heatmap(
        heatmap_data.astype(float), annot=annot_arr, fmt="",
        cmap="YlOrRd", ax=ax,
        cbar_kws={"label": "Mean prediction CV"},
    )
    ax.set_title("Prediction consistency: mean CV within variant groups", fontsize=12)
    ax.set_xlabel("Variant type")
    ax.set_ylabel("")

    fig.tight_layout()
    fig.savefig(model_dir / "consistency_heatmap.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved consistency_heatmap.png")
    plt.close("all")

    # Figure D: Predicted spread vs true spread scatter
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax_idx, vtype in enumerate(VARIANT_ORDER):
        ax = axes[ax_idx]
        vtype_data = consistency_df[consistency_df["variant_type"] == vtype]

        if len(vtype_data) == 0:
            ax.set_visible(False)
            continue

        ax.scatter(
            vtype_data["true_range"], vtype_data["pred_range"],
            color=VARIANT_COLORS[vtype], alpha=0.3, s=15, edgecolors="none",
        )

        max_val = max(
            vtype_data["true_range"].max(),
            vtype_data["pred_range"].max(),
        )
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="y = x")

        ax.set_xlabel("True activity range")
        ax.set_ylabel("Predicted range")
        ax.set_title(vtype.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Within-group predicted range vs true activity range",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(model_dir / "spread_scatter.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved spread_scatter.png")
    plt.close("all")


# ── Combined figures ──────────────────────────────────────────────────────────

def _generate_combined_figures(output_dir: Path, dpi: int) -> None:
    """Load both models' consistency metrics and produce comparison figures."""
    xgb_path = output_dir / "xgboost" / "consistency_summary.csv"
    chemprop_path = output_dir / "chemprop" / "consistency_summary.csv"

    missing = [m for m, p in [("xgboost", xgb_path), ("chemprop", chemprop_path)] if not p.exists()]
    if missing:
        logger.error(f"Missing results for: {missing}. Run those models first.")
        return

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    xgb = pd.read_csv(xgb_path)
    chemprop = pd.read_csv(chemprop_path)

    # Merge on (variant_type, endpoint) and save comparison CSV
    merge_cols = ["variant_type", "endpoint", "pred_cv_mean", "pred_cv_median",
                  "pred_std_mean", "consistency_ratio_mean", "consistency_ratio_median"]
    comparison_df = (
        xgb[merge_cols].rename(columns={c: f"{c}_xgboost" for c in merge_cols
                                        if c not in ("variant_type", "endpoint")})
        .merge(
            chemprop[merge_cols].rename(columns={c: f"{c}_chemprop" for c in merge_cols
                                                 if c not in ("variant_type", "endpoint")}),
            on=["variant_type", "endpoint"],
        )
    )
    comparison_df.to_csv(combined_dir / "consistency_comparison.csv", index=False)
    logger.info("Saved consistency_comparison.csv")

    for vtype in VARIANT_ORDER:
        xgb_vt = xgb[xgb["variant_type"] == vtype].copy()
        chemprop_vt = chemprop[chemprop["variant_type"] == vtype].copy()
        if xgb_vt.empty or chemprop_vt.empty:
            continue
        vt_data = {"xgboost": xgb_vt, "chemprop": chemprop_vt}
        vtype_label = vtype.replace("_", " ").title()

        for metric, ylabel, fname in [
            ("pred_cv_mean", "Mean prediction CV", f"pred_cv_{vtype}"),
            ("consistency_ratio_mean", "Mean consistency ratio", f"consistency_ratio_{vtype}"),
        ]:
            plot_model_comparison_bars(
                vt_data, "endpoint", metric, ylabel,
                f"{ylabel} — XGBoost vs Chemprop ({vtype_label})",
                combined_dir / f"{fname}_comparison.png", dpi=dpi,
            )
            logger.info(f"Saved {fname}_comparison.png")

    logger.info(f"Combined figures saved to {combined_dir}")


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.13-seal-molecular-variants", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    max_scaffold_group_size: int = typer.Option(20, help="Max scaffold group size for analysis"),
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

    logger.info("Loading cluster CV folds (repeat 0)")
    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    cluster_folds = cluster_folds[cluster_folds["repeat"] == 0]

    # ── 2. Identify variant groups ────────────────────────────────────
    smiles_all = df["SMILES"].tolist()
    names_all = df["Molecule Name"].values

    # Stereoisomers: group by achiral SMILES
    logger.info("Identifying stereoisomer groups")
    achiral = [get_achiral_smiles(s) for s in smiles_all]
    df["achiral_smiles"] = achiral

    stereo_groups: dict[str, list[int]] = {}
    for i, key in enumerate(achiral):
        if key is not None:
            stereo_groups.setdefault(key, []).append(i)
    # Keep only groups with ≥2 members
    stereo_groups = {k: v for k, v in stereo_groups.items() if len(v) >= 2}
    n_stereo_mols = sum(len(v) for v in stereo_groups.values())
    logger.info(
        f"Stereoisomer groups: {len(stereo_groups)} groups, "
        f"{n_stereo_mols} molecules ({100 * n_stereo_mols / len(df):.1f}%)"
    )

    # Scaffold decorations: group by Murcko scaffold
    logger.info("Identifying scaffold decoration groups")
    scaffolds = [get_scaffold(s) for s in smiles_all]
    df["scaffold_smiles"] = scaffolds

    scaffold_groups: dict[str, list[int]] = {}
    for i, key in enumerate(scaffolds):
        if key is not None:
            scaffold_groups.setdefault(key, []).append(i)
    # Keep groups with ≥2 members and ≤ max_scaffold_group_size
    scaffold_groups = {
        k: v for k, v in scaffold_groups.items()
        if 2 <= len(v) <= max_scaffold_group_size
    }
    n_scaffold_mols = sum(len(v) for v in scaffold_groups.values())
    logger.info(
        f"Scaffold decoration groups: {len(scaffold_groups)} groups, "
        f"{n_scaffold_mols} molecules ({100 * n_scaffold_mols / len(df):.1f}%)"
    )

    # Save variant group membership (model-agnostic)
    group_rows = []
    for group_id, (key, indices) in enumerate(stereo_groups.items()):
        for idx in indices:
            group_rows.append({
                "group_id": f"stereo_{group_id}",
                "variant_type": "stereoisomer",
                "group_key": key,
                "group_size": len(indices),
                "molecule_name": names_all[idx],
                "smiles": smiles_all[idx],
            })
    for group_id, (key, indices) in enumerate(scaffold_groups.items()):
        for idx in indices:
            group_rows.append({
                "group_id": f"scaffold_{group_id}",
                "variant_type": "scaffold_decoration",
                "group_key": key,
                "group_size": len(indices),
                "molecule_name": names_all[idx],
                "smiles": smiles_all[idx],
            })

    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(output_dir / "variant_groups.csv", index=False)
    logger.info(f"Saved variant_groups.csv ({len(group_df)} rows)")

    # Log group size distributions
    for vtype, groups in [("stereoisomer", stereo_groups), ("scaffold_decoration", scaffold_groups)]:
        sizes = [len(v) for v in groups.values()]
        logger.info(
            f"  {vtype}: sizes min={min(sizes)}, max={max(sizes)}, "
            f"median={np.median(sizes):.0f}, mean={np.mean(sizes):.1f}"
        )

    # ── 2b. Stereoisomer activity differences (assay noise baseline) ──
    logger.info("Computing within-stereoisomer-group activity differences")
    stereo_activity_rows = []
    for ep in ENDPOINTS:
        ep_diffs = []
        groups_with_data = 0
        for _key, indices in stereo_groups.items():
            vals = df.iloc[indices][ep].dropna().values
            if len(vals) >= 2:
                groups_with_data += 1
                for i in range(len(vals)):
                    for j in range(i + 1, len(vals)):
                        ep_diffs.append(abs(vals[i] - vals[j]))
        if ep_diffs:
            arr = np.array(ep_diffs)
            stereo_activity_rows.append({
                "endpoint": ep,
                "n_groups": groups_with_data,
                "n_pairs": len(arr),
                "mean_abs_diff": arr.mean(),
                "median_abs_diff": np.median(arr),
                "max_abs_diff": arr.max(),
                "pct_gt_0.1": 100 * (arr > 0.1).mean(),
                "pct_gt_0.5": 100 * (arr > 0.5).mean(),
                "pct_gt_1.0": 100 * (arr > 1.0).mean(),
            })
            logger.info(
                f"  {ep}: {groups_with_data} groups, {len(arr)} pairs, "
                f"median |Δ|={np.median(arr):.4f}, mean |Δ|={arr.mean():.4f}"
            )
    stereo_activity_df = pd.DataFrame(stereo_activity_rows)
    stereo_activity_df.to_csv(output_dir / "stereoisomer_activity_diffs.csv", index=False)
    logger.info("Saved stereoisomer_activity_diffs.csv")

    # ── 3. Fingerprint similarity within groups (model-agnostic) ──────
    # ECFP4 Tanimoto distances are used for variant group identification.
    # This is independent of model choice.
    logger.info("Computing intra-group Tanimoto distances")
    fp_sim_rows = []

    all_variant_groups = {
        "stereoisomer": stereo_groups,
        "scaffold_decoration": scaffold_groups,
    }

    for vtype, groups in all_variant_groups.items():
        all_distances = []
        for key, indices in groups.items():
            group_smiles = [smiles_all[i] for i in indices]
            dists = compute_tanimoto_distances_within_group(group_smiles)
            all_distances.extend(dists)
            for d in dists:
                fp_sim_rows.append({
                    "variant_type": vtype,
                    "group_key": key,
                    "tanimoto_distance": d,
                })

        if all_distances:
            logger.info(
                f"  {vtype}: {len(all_distances)} pairs, "
                f"mean dist={np.mean(all_distances):.3f}, "
                f"median={np.median(all_distances):.3f}"
            )

    # Random baseline: sample pairs of same sizes as stereoisomer groups
    rng = np.random.default_rng(42)
    n_random_pairs = sum(
        len(list(combinations(range(len(v)), 2))) for v in stereo_groups.values()
    )
    # Cap at 10000 for performance
    n_random_pairs = min(n_random_pairs, 10000)
    random_indices = rng.integers(0, len(smiles_all), size=(n_random_pairs, 2))
    for i, j in random_indices:
        if i == j:
            continue
        dists = compute_tanimoto_distances_within_group([smiles_all[i], smiles_all[j]])
        if dists:
            fp_sim_rows.append({
                "variant_type": "random",
                "group_key": "random",
                "tanimoto_distance": dists[0],
            })

    fp_sim_df = pd.DataFrame(fp_sim_rows)
    fp_sim_df.to_csv(output_dir / "fingerprint_similarity.csv", index=False)
    logger.info(f"Saved fingerprint_similarity.csv ({len(fp_sim_df)} rows)")

    random_dists = fp_sim_df[fp_sim_df["variant_type"] == "random"]["tanimoto_distance"].values
    logger.info(
        f"  random: {len(random_dists)} pairs, "
        f"mean dist={np.mean(random_dists):.3f}, "
        f"median={np.median(random_dists):.3f}"
    )

    # ── Skip training if model outputs already exist ──────────────────
    if (model_dir / "consistency_metrics.csv").exists():
        logger.info(f"Existing {model} outputs found — regenerating figures only")
        consistency_df = pd.read_csv(model_dir / "consistency_metrics.csv")
        summary_df = pd.read_csv(model_dir / "consistency_summary.csv")
        _generate_figures(consistency_df, summary_df, fp_sim_df, model_dir, dpi)
        logger.info(f"All outputs saved to {model_dir}")
        return

    # ── 4. Train and collect out-of-fold predictions ──────────────────
    logger.info(f"Training {model} models and collecting out-of-fold predictions")
    # Store predictions: molecule_name -> {endpoint -> predicted_value}
    oof_predictions: dict[str, dict[str, float]] = {}

    optuna_cache = INTERIM_DATA_DIR / "optuna_cache"
    chemprop_cache = INTERIM_DATA_DIR / "chemprop_pred_cache"

    # Protonated SMILES per molecule index, keyed by pH (always computed for Chemprop;
    # for XGBoost this is also the basis for feature computation below).
    prot_smiles_by_ph: dict[float, list[str]] = {}
    unique_phs = sorted(set(ENDPOINT_PH.values()))
    for ph in unique_phs:
        logger.info(f"Protonating all molecules at pH {ph}")
        prot_smiles_by_ph[ph] = protonate_at_ph(smiles_all, ph)

    for ep in ENDPOINTS:
        ph = ENDPOINT_PH[ep]
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        n_mol = len(ep_df)

        if n_mol < 50:
            logger.warning(f"Skipping {ep}: only {n_mol} molecules")
            continue

        ep_names = ep_df["Molecule Name"].values
        raw_values = ep_df[ep].values
        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_all = clip_and_log_transform(raw_values)
        else:
            y_all = raw_values

        # ECFP4 + RDKit features: XGBoost training only
        if model == "xgboost":
            prot_smiles_ep = [prot_smiles_by_ph[ph][i] for i in np.where(mask)[0]]
            ecfp = compute_ecfp4(prot_smiles_ep)
            desc = compute_rdkit_descriptors(prot_smiles_ep)
            variance = desc.var(axis=0)
            desc = desc[:, variance > 0]
            scaler = StandardScaler()
            desc_scaled = scaler.fit_transform(desc)
            X_all = np.hstack([ecfp, desc_scaled])

        # Get cluster folds for this endpoint
        ep_folds = cluster_folds[cluster_folds["endpoint"] == ep]
        fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
        fold_ids = np.array([fold_map.get(n, -1) for n in ep_names])
        unique_folds = sorted(set(fold_ids[fold_ids >= 0]))

        logger.info(f"  {ep}: {n_mol} molecules, {len(unique_folds)} folds")

        for fold_id in unique_folds:
            test_mask = fold_ids == fold_id
            train_mask = (fold_ids >= 0) & ~test_mask

            y_tr = y_all[train_mask]
            y_te = y_all[test_mask]

            if len(y_te) < 10 or len(y_tr) < 10:
                continue

            if model == "xgboost":
                X_tr = X_all[train_mask]
                X_te = X_all[test_mask]
                model_obj, _, _ = tune_xgboost(
                    X_tr, y_tr, cache_dir=optuna_cache,
                    cache_key=f"{ep}_cluster_fold{fold_id}",
                )
                y_pred = model_obj.predict(X_te)
            else:
                # Chemprop: use protonated SMILES per pH, indexed back to ep subset
                ep_indices = np.where(mask)[0]
                train_prot_ep = [prot_smiles_by_ph[ph][ep_indices[i]] for i in np.where(train_mask)[0]]
                test_prot_ep = [prot_smiles_by_ph[ph][ep_indices[i]] for i in np.where(test_mask)[0]]
                y_pred = train_chemprop(
                    train_prot_ep, y_tr, test_prot_ep,
                    cache_dir=chemprop_cache,
                    cache_key=f"2.13_{ep}_cluster_fold{fold_id}",
                )

            # Store out-of-fold predictions
            test_names = ep_names[test_mask]
            for name, pred, true in zip(test_names, y_pred, y_te):
                if name not in oof_predictions:
                    oof_predictions[name] = {}
                oof_predictions[name][ep] = {"pred": float(pred), "true": float(true)}

    logger.info(f"Collected predictions for {len(oof_predictions)} molecules")

    # ── 5. Compute within-group consistency ───────────────────────────
    logger.info("Computing within-group prediction consistency")
    consistency_rows = []

    for vtype, groups in all_variant_groups.items():
        for group_key, indices in groups.items():
            group_names = [names_all[i] for i in indices]

            for ep in ENDPOINTS:
                # Collect predictions and true values for group members
                preds = []
                trues = []
                for name in group_names:
                    if name in oof_predictions and ep in oof_predictions[name]:
                        preds.append(oof_predictions[name][ep]["pred"])
                        trues.append(oof_predictions[name][ep]["true"])

                if len(preds) < 2:
                    continue

                preds_arr = np.array(preds)
                trues_arr = np.array(trues)

                pred_std = np.std(preds_arr)
                pred_range = np.ptp(preds_arr)
                pred_mean = np.mean(preds_arr)
                pred_cv = pred_std / abs(pred_mean) if abs(pred_mean) > 1e-10 else np.nan

                true_std = np.std(trues_arr)
                true_range = np.ptp(trues_arr)

                consistency_ratio = pred_std / true_std if true_std > 1e-10 else np.nan

                consistency_rows.append({
                    "variant_type": vtype,
                    "group_key": group_key,
                    "endpoint": ep,
                    "n_members": len(preds),
                    "pred_mean": pred_mean,
                    "pred_std": pred_std,
                    "pred_range": pred_range,
                    "pred_cv": pred_cv,
                    "true_mean": np.mean(trues_arr),
                    "true_std": true_std,
                    "true_range": true_range,
                    "consistency_ratio": consistency_ratio,
                })

    # Random baseline: create random groups matching stereoisomer group sizes
    stereo_sizes = [len(v) for v in stereo_groups.values()]
    rng = np.random.default_rng(42)
    all_names_list = list(names_all)
    for size in stereo_sizes:
        random_names = rng.choice(all_names_list, size=size, replace=False)
        for ep in ENDPOINTS:
            preds = []
            trues = []
            for name in random_names:
                if name in oof_predictions and ep in oof_predictions[name]:
                    preds.append(oof_predictions[name][ep]["pred"])
                    trues.append(oof_predictions[name][ep]["true"])

            if len(preds) < 2:
                continue

            preds_arr = np.array(preds)
            trues_arr = np.array(trues)

            pred_std = np.std(preds_arr)
            pred_range = np.ptp(preds_arr)
            pred_mean = np.mean(preds_arr)
            pred_cv = pred_std / abs(pred_mean) if abs(pred_mean) > 1e-10 else np.nan

            true_std = np.std(trues_arr)
            true_range = np.ptp(trues_arr)

            consistency_ratio = pred_std / true_std if true_std > 1e-10 else np.nan

            consistency_rows.append({
                "variant_type": "random",
                "group_key": "random",
                "endpoint": ep,
                "n_members": len(preds),
                "pred_mean": pred_mean,
                "pred_std": pred_std,
                "pred_range": pred_range,
                "pred_cv": pred_cv,
                "true_mean": np.mean(trues_arr),
                "true_std": true_std,
                "true_range": true_range,
                "consistency_ratio": consistency_ratio,
            })

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(model_dir / "consistency_metrics.csv", index=False)
    logger.info(f"Saved consistency_metrics.csv ({len(consistency_df)} rows)")

    # ── 6. Aggregate consistency summary ──────────────────────────────
    summary_rows = []
    for (vtype, ep), grp in consistency_df.groupby(["variant_type", "endpoint"]):
        summary_rows.append({
            "variant_type": vtype,
            "endpoint": ep,
            "n_groups": len(grp),
            "pred_cv_mean": grp["pred_cv"].mean(),
            "pred_cv_median": grp["pred_cv"].median(),
            "pred_std_mean": grp["pred_std"].mean(),
            "pred_range_mean": grp["pred_range"].mean(),
            "true_std_mean": grp["true_std"].mean(),
            "true_range_mean": grp["true_range"].mean(),
            "consistency_ratio_mean": grp["consistency_ratio"].mean(),
            "consistency_ratio_median": grp["consistency_ratio"].median(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(model_dir / "consistency_summary.csv", index=False)
    logger.info("Saved consistency_summary.csv")

    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['variant_type']:>20s} | {row['endpoint']:>30s} | "
            f"n_groups={row['n_groups']:4d} | "
            f"pred_cv={row['pred_cv_mean']:.3f} | "
            f"consistency_ratio={row['consistency_ratio_mean']:.3f}"
        )

    # ── 7. Generate figures ───────────────────────────────────────────
    _generate_figures(consistency_df, summary_df, fp_sim_df, model_dir, dpi)

    logger.info(f"All outputs saved to {model_dir}")


if __name__ == "__main__":
    app()
