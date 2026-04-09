#!/usr/bin/env python
"""Molecular variant consistency analysis: Resonance Structures."""

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

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.tuning import tune_xgboost
from polaris_generalization.visualization import DEFAULT_DPI, set_style

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


# -----------------------------
# Main
# -----------------------------
@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.15-zalte-resonance-variants"
    ),
    dpi: int = typer.Option(DEFAULT_DPI),
):
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Fingerprint Similarity Analysis
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
        if i == j: continue
        dists = compute_tanimoto_distances_within_group([all_smiles[i], all_smiles[j]])
        if dists:
            fp_sim_rows.append({
                "variant_type": "random",
                "group_key": "random",
                "tanimoto_distance": dists[0]
            })

    fp_sim_df = pd.DataFrame(fp_sim_rows)
    fp_sim_df.to_csv(output_dir / "fingerprint_similarity.csv", index=False)

    # -----------------------------
    # Predictions & Cache
    # -----------------------------
    cache_path = output_dir / "raw_predictions_cache.json"

    if cache_path.exists():
        logger.info(f"Loading predictions cache from {cache_path}...")
        with open(cache_path) as f:
            oof_predictions = json.load(f)
    else:
        logger.info("No cache found. Training models...")
        oof_predictions = {}
        for ep in tqdm(ENDPOINTS, desc="Evaluating Endpoints"):
            mask = df[ep].notna()
            ep_df = df[mask].copy()
            if len(ep_df) < 50: continue

            orig_smiles = ep_df["SMILES"].tolist()
            names = ep_df["Molecule Name"].values
            y = ep_df[ep].values

            if ep in LOG_TRANSFORM_ENDPOINTS:
                y = clip_and_log_transform(y)

            train_smiles = protonate_at_ph(orig_smiles, ENDPOINT_PH[ep])
            X_fp = compute_bit_fp(train_smiles)
            X_desc = compute_rdkit_descriptors(train_smiles)
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

                cache_dir = INTERIM_DATA_DIR / "optuna_cache"
                model, _, _ = tune_xgboost(X[tr], y[tr], cache_dir=cache_dir, cache_key=f"{ep}_cluster_fold{fold_id}")

                for name, smi_orig, true in zip(names[te], np.array(orig_smiles)[te], y[te]):
                    smi_canon = canonicalize_smiles(smi_orig)
                    res_forms = resonance_groups[ENDPOINT_PH[ep]].get(smi_canon, [smi_canon])

                    res_desc = compute_rdkit_descriptors(res_forms)
                    Xr = np.hstack([
                        compute_bit_fp(res_forms),
                        scaler.transform(res_desc[:, var_mask])
                    ])

                    preds = model.predict(Xr)
                    if name not in oof_predictions: oof_predictions[name] = {}
                    oof_predictions[name][ep] = {
                        "preds": preds.tolist(),
                        "true": float(true),
                    }

        with open(cache_path, "w") as f:
            json.dump(oof_predictions, f, indent=4)

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
    consistency_df.to_csv(output_dir / "consistency_metrics.csv", index=False)

    summary = pd.DataFrame(list(ep_metrics.values())).sort_values("pct_swing", ascending=False)
    summary.to_csv(output_dir / "consistency_summary.csv", index=False)

    # Endpoint order: sorted by total pct_swing descending (shared across all panels)
    ep_order = summary["endpoint"].tolist()

    # -----------------------------
    # Plotting Suite
    # -----------------------------
    logger.info("Generating plots...")

    # 1. Fingerprint Distances (resonance vs random pairs) — standalone figure
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

    # 2. Three-panel resonance sensitivity figure
    import matplotlib.gridspec as gridspec

    # Colors
    COLOR_A       = "#2F7DBF"   # steel blue
    COLOR_IMPROV  = "#2D7D3A"   # forest green
    COLOR_WORSEN  = "#C0392B"   # crimson
    COLOR_BOX     = "#E07B54"   # warm coral
    COLOR_STRIP   = "#7B2D1E"   # dark rust
    BAR_EDGE      = "black"
    BAR_LW        = 0.6

    # Shortened display labels
    disp = [ENDPOINT_LABELS.get(ep, ep) for ep in ep_order]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2], hspace=0.28, wspace=0.18)

    y_pos = np.arange(len(ep_order))

    # --- Panel A: Coverage (top-left) ---
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
    ax_a.set_xlim(0, max(coverages) * 1.35)  # headroom for labels
    ax_a.set_title("A", fontsize=16, fontweight="bold", loc="left")
    ax_a.invert_yaxis()
    ax_a.grid(False)
    for spine in ["top", "right"]:
        ax_a.spines[spine].set_visible(False)

    # --- Panel B: Diverging improvement/worsening (top-right) ---
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
    ax_b.legend(loc="lower right", fontsize=8, framealpha=0.95,
                edgecolor="lightgray")
    ax_b.invert_yaxis()
    ax_b.grid(False)
    for spine in ["top", "right"]:
        ax_b.spines[spine].set_visible(False)

    # --- Panel C: Per-molecule severity (bottom, full width) ---
    ax_c = fig.add_subplot(gs[1, :])

    # Map endpoint names in df to display names for x-axis
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

    fig.savefig(output_dir / "resonance_sensitivity_panel.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Done.")

if __name__ == "__main__":
    app()
