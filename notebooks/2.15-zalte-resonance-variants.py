#!/usr/bin/env python
"""Molecular variant consistency analysis: Resonance Structures.

Analyzes prediction sensitivity to resonance form representation by loading
pre-generated resonance forms from 2.14, training models (XGBoost or Chemprop),
and comparing predictions across canonical and resonance-form SMILES.

ECFP4 fingerprint instability analysis (Tanimoto distances between resonance
forms) is model-agnostic and always computed. Feature computation for XGBoost
training (ECFP4 + RDKit 2D) is XGBoost-only.

Supports XGBoost (ECFP4 + RDKit 2D descriptors, Optuna-tuned) and Chemprop
D-MPNN. Use --combined to generate side-by-side comparison figures.

Usage:
    pixi run -e cheminformatics python notebooks/2.15-zalte-resonance-variants.py
    pixi run -e cheminformatics python notebooks/2.15-zalte-resonance-variants.py --model chemprop
    pixi run -e cheminformatics python notebooks/2.15-zalte-resonance-variants.py --combined

Model-agnostic outputs (saved to output_dir):
    data/processed/2.15-zalte-resonance-variants/fingerprint_similarity.csv
    data/processed/2.15-zalte-resonance-variants/fingerprint_distances.png

Per-model outputs (saved to output_dir/{model}/):
    data/processed/2.15-zalte-resonance-variants/{model}/consistency_metrics.csv
    data/processed/2.15-zalte-resonance-variants/{model}/consistency_summary.csv
    data/processed/2.15-zalte-resonance-variants/{model}/resonance_sensitivity_panel.png

Combined outputs:
    data/processed/2.15-zalte-resonance-variants/combined/consistency_comparison.csv
    data/processed/2.15-zalte-resonance-variants/combined/pct_swing_comparison.png
"""

import json
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
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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

ENDPOINT_LABELS = {
    "Caco-2 Permeability Papp A>B": "Caco-2 Papp A>B",
    "Caco-2 Permeability Efflux":   "Caco-2 Efflux",
}

LOG_TRANSFORM_ENDPOINTS = [ep for ep in ENDPOINTS if ep.lower() != "logd"]

ENDPOINT_PH = {ep: 7.4 for ep in ENDPOINTS}
ENDPOINT_PH["Caco-2 Permeability Papp A>B"] = 6.5
ENDPOINT_PH["Caco-2 Permeability Efflux"] = 6.5

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)


# -----------------------------
# Utilities
# -----------------------------
def canonicalize_smiles(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def clip_and_log_transform(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(x, 1e-10, None) + 1)


def protonate_at_ph(smiles_list: list[str], ph: float) -> list[str]:
    out = []
    for smi in smiles_list:
        try:
            res = dimorphite_protonate(smi, ph_min=ph - 0.5, ph_max=ph + 0.5, max_variants=1)
            out.append(res[0] if res else smi)
        except Exception:
            out.append(smi)
    return out


def compute_bit_fp(smiles_list, nbits=2048, radius=2):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True)
            fps.append(np.array(fp, dtype=np.uint8))
        else:
            fps.append(np.zeros(nbits, dtype=np.uint8))
    return np.vstack(fps)


def compute_rdkit_descriptors(smiles_list):
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            rows.append(list(DESC_CALC.CalcDescriptors(mol)))
        else:
            rows.append([0.0] * len(DESCRIPTOR_NAMES))
    return np.array(rows, dtype=np.float64)


def compute_tanimoto_distances_within_group(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useChirality=True) if m else None for m in mols]

    dists = []
    for i, j in combinations(range(len(fps)), 2):
        if fps[i] and fps[j]:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dists.append(1 - sim)
    return dists


# ── Migration ─────────────────────────────────────────────────────────────────

def _migrate_flat_outputs(output_dir: Path) -> None:
    """Move pre-model-flag XGBoost outputs into xgboost/ subdirectory."""
    flat_files = [
        "consistency_metrics.csv", "consistency_summary.csv",
        "resonance_sensitivity_panel.png",
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


# ── Per-model figure ──────────────────────────────────────────────────────────

def _generate_figures(
    consistency_df: pd.DataFrame,
    ep_metrics: dict,
    model_dir: Path,
    dpi: int,
) -> None:
    """Generate the RIGR-aligned three-panel resonance sensitivity figure."""
    import matplotlib.gridspec as gridspec

    summary = pd.DataFrame(list(ep_metrics.values())).sort_values("pct_swing", ascending=False)
    ep_order = summary["endpoint"].tolist()

    COLOR_A       = "#2F7DBF"
    COLOR_IMPROV  = "#2D7D3A"
    COLOR_WORSEN  = "#C0392B"
    COLOR_BOX     = "#E07B54"
    COLOR_STRIP   = "#7B2D1E"
    BAR_EDGE      = "black"
    BAR_LW        = 0.6

    disp = [ENDPOINT_LABELS.get(ep, ep) for ep in ep_order]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.28, wspace=0.18)

    y_pos = np.arange(len(ep_order))

    # Panel A: Coverage (top-left)
    ax_a = fig.add_subplot(gs[0, 0])
    coverages = [ep_metrics[ep]["coverage_pct"] for ep in ep_order]
    ax_a.barh(y_pos, coverages, color=COLOR_A, height=0.6,
              edgecolor=BAR_EDGE, linewidth=BAR_LW)
    for i, (ep, cov) in enumerate(zip(ep_order, coverages)):
        n_m = ep_metrics[ep]["n_multi"]
        n_t = ep_metrics[ep]["n_total"]
        ax_a.text(cov + 0.8, i, f"{cov:.1f}%  ({n_m}/{n_t})", va="center", fontsize=10,
                  color="dimgray", clip_on=False)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(disp, fontsize=10)
    ax_a.set_xlabel("Molecules with multiple resonance forms (%)", fontsize=10)
    ax_a.set_xlim(0, max(coverages) * 1.35)
    ax_a.set_title("A", fontsize=16, fontweight="bold", loc="left")
    ax_a.invert_yaxis()
    ax_a.grid(False)
    for spine in ["top", "right"]:
        ax_a.spines[spine].set_visible(False)

    # Panel B: Diverging improvement/worsening (top-right)
    ax_b = fig.add_subplot(gs[0, 1])
    improves_neg = [-ep_metrics[ep]["improve_pct"] for ep in ep_order]
    worsens      = [ ep_metrics[ep]["worsen_pct"]  for ep in ep_order]

    ax_b.barh(y_pos, improves_neg, color=COLOR_IMPROV, height=0.6,
              edgecolor=BAR_EDGE, linewidth=BAR_LW, label="Best case")
    ax_b.barh(y_pos, worsens,      color=COLOR_WORSEN, height=0.6,
              edgecolor=BAR_EDGE, linewidth=BAR_LW, label="Worst case")

    max_improve = max(abs(v) for v in improves_neg)
    max_worsen  = max(worsens)
    ax_b.set_xlim(-max_improve * 1.9, max_worsen * 1.35)

    for i, (imp, wor) in enumerate(zip(improves_neg, worsens)):
        ax_b.text(imp - 0.2, i, f"{abs(imp):.1f}%", va="center", ha="right",
                  fontsize=10, color=COLOR_IMPROV, clip_on=False)
        ax_b.text(wor + 0.2, i, f"{wor:.1f}%", va="center", ha="left",
                  fontsize=10, color=COLOR_WORSEN, clip_on=False)

    ax_b.axvline(0, color="black", linewidth=1.0)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([""] * len(ep_order))
    ax_b.set_xlabel("Overall RMSE change (% of baseline)", fontsize=10)
    ax_b.set_title("B", fontsize=16, fontweight="bold", loc="left")
    ax_b.legend(loc="lower right", fontsize=8, framealpha=0.95, edgecolor="lightgray")
    ax_b.invert_yaxis()
    ax_b.grid(False)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)

    # Panel C: Per-molecule severity (bottom, full width)
    ax_c = fig.add_subplot(gs[1, :])

    plot_df = consistency_df.copy()
    plot_df["endpoint_label"] = plot_df["endpoint"].map(lambda e: ENDPOINT_LABELS.get(e, e))
    disp_order = [ENDPOINT_LABELS.get(ep, ep) for ep in ep_order]

    sns.boxplot(
        data=plot_df,
        x="endpoint_label", y="mol_pct_swing",
        order=disp_order,
        color=COLOR_BOX, width=0.5, showfliers=False,
        boxprops=dict(edgecolor=BAR_EDGE, linewidth=BAR_LW),
        medianprops=dict(color=BAR_EDGE, linewidth=1.2),
        whiskerprops=dict(color=BAR_EDGE, linewidth=BAR_LW),
        capprops=dict(color=BAR_EDGE, linewidth=BAR_LW),
        ax=ax_c,
    )
    strip_data = pd.concat([
        g.sample(min(len(g), 200), random_state=42)
        for _, g in plot_df.groupby("endpoint_label")
    ], ignore_index=True)
    sns.stripplot(
        data=strip_data,
        x="endpoint_label", y="mol_pct_swing",
        order=disp_order,
        color=COLOR_STRIP, size=2.5, alpha=0.5, jitter=True, ax=ax_c,
    )

    p95 = consistency_df["mol_pct_swing"].quantile(0.95)
    ax_c.set_ylim(bottom=-p95 * 0.04, top=p95 * 1.3)
    ax_c.set_xlabel("")
    ax_c.set_ylabel("Error range across resonance forms\n(% of baseline)", fontsize=10)
    ax_c.set_title("C", fontsize=16, fontweight="bold", loc="left")
    ax_c.set_xticklabels(disp_order, rotation=30, ha="right", fontsize=10)
    ax_c.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5, color="lightgray")
    ax_c.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_c.spines[spine].set_visible(False)

    fig.savefig(model_dir / "resonance_sensitivity_panel.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved resonance_sensitivity_panel.png")
    plt.close()


# ── Combined figures ──────────────────────────────────────────────────────────

def _generate_combined_figures(output_dir: Path, dpi: int) -> None:
    """Load both models' consistency summaries and produce comparison figures."""
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

    # Comparison CSV
    merge_cols = ["endpoint", "baseline_rmse", "rms_minrd", "rms_maxrd",
                  "rms_resrange", "pct_swing", "improve_pct", "worsen_pct"]
    comparison_df = (
        xgb[merge_cols].rename(columns={c: f"{c}_xgboost" for c in merge_cols if c != "endpoint"})
        .merge(
            chemprop[merge_cols].rename(columns={c: f"{c}_chemprop" for c in merge_cols if c != "endpoint"}),
            on="endpoint",
        )
    )
    comparison_df.to_csv(combined_dir / "consistency_comparison.csv", index=False)
    logger.info("Saved consistency_comparison.csv")

    # Grouped bar charts
    data_by_model = {"xgboost": xgb, "chemprop": chemprop}
    for metric, ylabel, fname in [
        ("pct_swing", "% RMSE swing (Max−Min)", "pct_swing_comparison"),
        ("rms_resrange", "RMS resonance prediction range", "rms_resrange_comparison"),
        ("worsen_pct", "Worst-case RMSE worsening (%)", "worsen_pct_comparison"),
    ]:
        plot_model_comparison_bars(
            data_by_model, "endpoint", metric, ylabel,
            f"{ylabel} — XGBoost vs Chemprop",
            combined_dir / f"{fname}.png", dpi=dpi,
        )
        logger.info(f"Saved {fname}.png")

    logger.info(f"Combined figures saved to {combined_dir}")


# -----------------------------
# Main
# -----------------------------
@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.15-zalte-resonance-variants"
    ),
    dpi: int = typer.Option(DEFAULT_DPI),
    model: str = typer.Option("xgboost", help="Model architecture: xgboost or chemprop"),
    combined: bool = typer.Option(False, help="Generate combined XGBoost+Chemprop comparison figures"),
):
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

    # 1. Load Core Datasets
    logger.info("Loading core datasets...")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    cluster_folds = cluster_folds[cluster_folds["repeat"] == 0]

    res_dir = PROCESSED_DATA_DIR / "2.14-zalte-resonance-generation"
    resonance_groups: dict[float, dict[str, list[str]]] = {}
    for ph in [7.4, 6.5]:
        rdf = pd.read_csv(res_dir / f"resonance_structures_ph{ph}.csv")
        resonance_groups[ph] = {
            canonicalize_smiles(row["parent_smi"]): [
                canonicalize_smiles(s) for s in json.loads(row["resonance_smis"])
            ]
            for _, row in rdf.iterrows()
        }

    name_to_orig_smi = dict(zip(df["Molecule Name"], df["SMILES"]))
    all_smiles = df["SMILES"].tolist()

    # -----------------------------
    # Fingerprint Similarity Analysis (model-agnostic)
    # ECFP4 is used here for fingerprint instability measurement only.
    # It is always computed regardless of model choice.
    # -----------------------------
    logger.info("Computing intra-group fingerprint distances...")
    fp_sim_rows = []
    fp_dists_per_mol = {}

    for parent, res_list in resonance_groups[7.4].items():
        if len(res_list) > 1:
            dists = compute_tanimoto_distances_within_group(res_list)
            if dists:
                fp_dists_per_mol[parent] = {"max_dist": max(dists), "mean_dist": np.mean(dists)}
                for d in dists:
                    fp_sim_rows.append({
                        "variant_type": "resonance",
                        "group_key": parent,
                        "tanimoto_distance": d
                    })

    rng = np.random.default_rng(42)
    n_random_pairs = min(10000, len(fp_sim_rows))
    random_indices = rng.integers(0, len(all_smiles), size=(n_random_pairs, 2))

    for i, j in random_indices:
        if i == j:
            continue
        dists = compute_tanimoto_distances_within_group([all_smiles[i], all_smiles[j]])
        if dists:
            fp_sim_rows.append({
                "variant_type": "random",
                "group_key": "random",
                "tanimoto_distance": dists[0]
            })

    fp_sim_df = pd.DataFrame(fp_sim_rows)
    fp_sim_df.to_csv(output_dir / "fingerprint_similarity.csv", index=False)

    # Fingerprint distances figure (model-agnostic, saved to output_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"resonance": "coral", "random": "gray"}
    for vtype in ["resonance", "random"]:
        dists = fp_sim_df[fp_sim_df["variant_type"] == vtype]["tanimoto_distance"].values
        if len(dists) > 0:
            ax.hist(dists, bins=50, density=True, alpha=0.6, color=colors[vtype],
                    label=f"{vtype.capitalize()} (median={np.median(dists):.3f})")
    ax.set_xlabel("ECFP4 Tanimoto Distance")
    ax.set_ylabel("Density")
    ax.set_title("Intra-group Fingerprint Distances (Resonance vs. Random Pairs)")
    ax.legend()
    ax.grid(False)
    fig.savefig(output_dir / "fingerprint_distances.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Skip-training check
    # -----------------------------
    if (model_dir / "consistency_metrics.csv").exists():
        logger.info(f"Existing {model} outputs found — regenerating figures only")
        consistency_df = pd.read_csv(model_dir / "consistency_metrics.csv")
        summary_df = pd.read_csv(model_dir / "consistency_summary.csv")
        ep_metrics = {row["endpoint"]: row.to_dict() for _, row in summary_df.iterrows()}
        _generate_figures(consistency_df, ep_metrics, model_dir, dpi)
        logger.info("Done.")
        return

    # -----------------------------
    # Predictions
    # -----------------------------
    logger.info(f"Training {model} models...")
    oof_predictions = {}

    optuna_cache = INTERIM_DATA_DIR / "optuna_cache"
    chemprop_cache = INTERIM_DATA_DIR / "chemprop_pred_cache"

    for ep in tqdm(ENDPOINTS, desc="Evaluating Endpoints"):
        ph = ENDPOINT_PH[ep]
        mask = df[ep].notna()
        ep_df = df[mask].copy()
        if len(ep_df) < 50:
            continue

        orig_smiles = ep_df["SMILES"].tolist()
        names = ep_df["Molecule Name"].values
        y = ep_df[ep].values

        if ep in LOG_TRANSFORM_ENDPOINTS:
            y = clip_and_log_transform(y)

        # Protonated SMILES for training molecules (used by both models)
        train_smiles_prot = protonate_at_ph(orig_smiles, ph)

        # XGBoost: compute ECFP4 + RDKit features (training features only)
        if model == "xgboost":
            X_fp = compute_bit_fp(train_smiles_prot)
            X_desc = compute_rdkit_descriptors(train_smiles_prot)
            var_mask = X_desc.var(axis=0) > 0
            X_desc = X_desc[:, var_mask]
            scaler = StandardScaler()
            X_desc = scaler.fit_transform(X_desc)
            X = np.hstack([X_fp, X_desc])

        # Use endpoint-specific fold assignments
        ep_folds = cluster_folds[cluster_folds["endpoint"] == ep]
        fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
        fold_ids = np.array([fold_map.get(n, -1) for n in names])
        unique_folds = sorted(set(fold_ids[fold_ids >= 0]))

        for fold_id in unique_folds:
            te = fold_ids == fold_id
            tr = (fold_ids >= 0) & ~te

            if model == "xgboost":
                model_obj, _, _ = tune_xgboost(
                    X[tr], y[tr], cache_dir=optuna_cache,
                    cache_key=f"{ep}_cluster_fold{fold_id}",
                )
                # Predict on each test molecule using all its resonance forms
                for name, smi_orig, true in zip(names[te], np.array(orig_smiles)[te], y[te]):
                    smi_canon = canonicalize_smiles(smi_orig)
                    res_forms = resonance_groups[ph].get(smi_canon, [smi_canon])

                    res_desc = compute_rdkit_descriptors(res_forms)
                    Xr = np.hstack([
                        compute_bit_fp(res_forms),
                        scaler.transform(res_desc[:, var_mask])
                    ])

                    preds = model_obj.predict(Xr)
                    if name not in oof_predictions:
                        oof_predictions[name] = {}
                    oof_predictions[name][ep] = {
                        "preds": preds.tolist(),
                        "true": float(true),
                    }
            else:
                # Chemprop: train once per fold, predict on all resonance forms in one shot.
                # Mirrors XGBoost: one model per fold, inference on the full augmented test set.
                train_prot_fold = [train_smiles_prot[i] for i in np.where(tr)[0]]

                # Build flat list of all resonance forms across test molecules, tracking slices.
                all_res_smiles: list[str] = []
                res_slices: dict[str, tuple[int, int]] = {}
                mol_trues: dict[str, float] = {}
                for name, smi_orig, true in zip(names[te], np.array(orig_smiles)[te], y[te]):
                    smi_canon = canonicalize_smiles(smi_orig)
                    forms_prot = [protonate_at_ph([s], ph)[0]
                                  for s in resonance_groups[ph].get(smi_canon, [smi_canon])]
                    res_slices[name] = (len(all_res_smiles), len(all_res_smiles) + len(forms_prot))
                    all_res_smiles.extend(forms_prot)
                    mol_trues[name] = float(true)

                # One train_chemprop call — model trained once, predicts on all resonance forms.
                all_preds = train_chemprop(
                    train_prot_fold, y[tr], all_res_smiles,
                    cache_dir=chemprop_cache,
                    cache_key=f"2.15_{ep}_cluster_fold{fold_id}_allres",
                    checkpoint_dir=model_dir / "models",
                )

                for name, (s, e) in res_slices.items():
                    if name not in oof_predictions:
                        oof_predictions[name] = {}
                    oof_predictions[name][ep] = {
                        "preds": all_preds[s:e].tolist(),
                        "true": mol_trues[name],
                    }

    # -----------------------------
    # RIGR-aligned metrics
    # Methodology: Zalte et al. 2025 JCIM
    #   RMS Resonance Range  = sqrt(mean((max_pred - min_pred)^2))
    #   RMS MaxRD            = sqrt(mean(max_i|pred_i - true|^2))  [worst-case RMSE]
    #   RMS MinRD            = sqrt(mean(min_i|pred_i - true|^2))  [best-case RMSE]
    #   Baseline RMSE        = RMSE using canonical (index-0) prediction
    #   % RMSE swing         = (MaxRD - MinRD) / Baseline * 100
    # -----------------------------
    logger.info("Computing RIGR-aligned metrics...")

    rigr_rows = []   # per-molecule (multi-form only), for distribution plots
    ep_metrics = {}  # per-endpoint aggregated

    for ep in ENDPOINTS:
        mol_rows = []
        for name, ep_dict in oof_predictions.items():
            if ep not in ep_dict:
                continue
            preds = np.array(ep_dict[ep]["preds"])
            true = ep_dict[ep]["true"]
            maes = np.abs(preds - true)

            orig_smi = name_to_orig_smi.get(name)
            smi_canon = canonicalize_smiles(orig_smi) if orig_smi else None
            fp_stats = fp_dists_per_mol.get(smi_canon, {"max_dist": np.nan, "mean_dist": np.nan})

            mol_rows.append({
                "baseline_err": float(abs(preds[0] - true)),
                "max_rd":       float(maes.max()),
                "min_rd":       float(maes.min()),
                "res_range":    float(preds.max() - preds.min()) if len(preds) > 1 else 0.0,
                "num_forms":    int(len(preds)),
            })

            if len(preds) > 1:
                baseline_err = float(abs(preds[0] - true))
                mol_pct = (float(maes.max()) - float(maes.min())) / (baseline_err + 1e-6) * 100
                rigr_rows.append({
                    "molecule_name":  name,
                    "endpoint":       ep,
                    "resonance_range": float(preds.max() - preds.min()),
                    "max_rd":         float(maes.max()),
                    "min_rd":         float(maes.min()),
                    "baseline_err":   baseline_err,
                    "num_forms":      int(len(preds)),
                    "max_fp_dist":    fp_stats["max_dist"],
                    "mol_pct_swing":  mol_pct,
                })

        if not mol_rows:
            continue

        n_total = len(mol_rows)
        n_multi = sum(1 for r in mol_rows if r["num_forms"] > 1)

        base_errs = np.array([r["baseline_err"] for r in mol_rows])
        max_rds   = np.array([r["max_rd"]       for r in mol_rows])
        min_rds   = np.array([r["min_rd"]        for r in mol_rows])
        ranges    = np.array([r["res_range"] for r in mol_rows if r["num_forms"] > 1])

        baseline_rmse = np.sqrt(np.mean(base_errs ** 2))
        rms_maxrd     = np.sqrt(np.mean(max_rds ** 2))
        rms_minrd     = np.sqrt(np.mean(min_rds ** 2))
        rms_resrange  = float(np.sqrt(np.mean(ranges ** 2))) if len(ranges) > 0 else np.nan
        pct_swing     = (rms_maxrd - rms_minrd) / baseline_rmse * 100
        improve_pct   = (baseline_rmse - rms_minrd) / baseline_rmse * 100
        worsen_pct    = (rms_maxrd - baseline_rmse) / baseline_rmse * 100

        ep_metrics[ep] = {
            "endpoint":      ep,
            "n_total":       n_total,
            "n_multi":       n_multi,
            "coverage_pct":  round(n_multi / n_total * 100, 1),
            "baseline_rmse": round(float(baseline_rmse), 4),
            "rms_minrd":     round(float(rms_minrd),     4),
            "rms_maxrd":     round(float(rms_maxrd),     4),
            "rms_resrange":  round(float(rms_resrange),  4) if not np.isnan(rms_resrange) else np.nan,
            "improve_pct":   round(float(improve_pct),   1),
            "worsen_pct":    round(float(worsen_pct),    1),
            "pct_swing":     round(float(pct_swing),     1),
        }

    consistency_df = pd.DataFrame(rigr_rows)
    consistency_df.to_csv(model_dir / "consistency_metrics.csv", index=False)

    summary = pd.DataFrame(list(ep_metrics.values())).sort_values("pct_swing", ascending=False)
    summary.to_csv(model_dir / "consistency_summary.csv", index=False)

    # -----------------------------
    # Plotting Suite
    # -----------------------------
    logger.info("Generating plots...")
    _generate_figures(consistency_df, ep_metrics, model_dir, dpi)

    logger.info("Done.")

if __name__ == "__main__":
    app()
