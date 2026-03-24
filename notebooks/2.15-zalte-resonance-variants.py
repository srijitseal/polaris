#!/usr/bin/env python
"""Molecular variant consistency analysis: Resonance Structures."""

import json
from collections import Counter
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
from xgboost import XGBRegressor

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

RDLogger.DisableLog("rdApp.*")

app = typer.Typer()

ENDPOINTS = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Papp A>B",
    "Caco-2 Permeability Efflux",
    "MPPB",
    "MBPB",
    "MGMB",
]

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
            res = dimorphite_protonate(
                smi, ph_min=ph - 0.5, ph_max=ph + 0.5, max_variants=1
            )
            out.append(res[0] if res else smi)
        except Exception:
            out.append(smi)
    return out


def compute_bit_fp(smiles_list, nbits=2048, radius=2):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius, nBits=nbits, useChirality=True
            )
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
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048, useChirality=True)
        if m
        else None
        for m in mols
    ]

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

    resonance_df = pd.read_csv(
        PROCESSED_DATA_DIR
        / "2.14-zalte-resonance-generation"
        / "resonance_structures.csv"
    )

    # Dictionary for mapping Name -> Original SMILES later
    name_to_orig_smi = dict(zip(df["Molecule Name"], df["SMILES"]))
    all_smiles = df["SMILES"].tolist()

    # Build resonance groups (canonicalized)
    resonance_groups = {}
    for _, row in resonance_df.iterrows():
        parent = canonicalize_smiles(row["parent_smi"])
        res_list = [canonicalize_smiles(s) for s in json.loads(row["resonance_smis"])]
        resonance_groups[parent] = res_list

    # -----------------------------
    # NEW: Fingerprint Similarity Analysis
    # -----------------------------
    logger.info("Computing intra-group fingerprint distances...")
    fp_sim_rows = []
    fp_dists_per_mol = {}  # Store max/mean per molecule for the scatter plot later

    for parent, res_list in resonance_groups.items():
        if len(res_list) > 1:
            dists = compute_tanimoto_distances_within_group(res_list)
            if dists:
                fp_dists_per_mol[parent] = {
                    "max_dist": max(dists),
                    "mean_dist": np.mean(dists),
                }
                for d in dists:
                    fp_sim_rows.append(
                        {
                            "variant_type": "resonance",
                            "group_key": parent,
                            "tanimoto_distance": d,
                        }
                    )

    # Random Baseline
    rng = np.random.default_rng(42)
    n_random_pairs = min(10000, len(fp_sim_rows))
    random_indices = rng.integers(0, len(all_smiles), size=(n_random_pairs, 2))

    for i, j in random_indices:
        if i == j:
            continue
        dists = compute_tanimoto_distances_within_group([all_smiles[i], all_smiles[j]])
        if dists:
            fp_sim_rows.append(
                {
                    "variant_type": "random",
                    "group_key": "random",
                    "tanimoto_distance": dists[0],
                }
            )

    fp_sim_df = pd.DataFrame(fp_sim_rows)
    fp_sim_df.to_csv(output_dir / "fingerprint_similarity.csv", index=False)

    # -----------------------------
    # Predictions & Cache
    # -----------------------------
    cache_path = output_dir / "raw_predictions_cache.json"

    if cache_path.exists():
        logger.info(f"Loading predictions cache from {cache_path}...")
        with open(cache_path, "r") as f:
            oof_predictions = json.load(f)
    else:
        logger.info("No cache found. Training models...")
        oof_predictions = {}
        for ep in tqdm(ENDPOINTS, desc="Evaluating Endpoints"):
            mask = df[ep].notna()
            ep_df = df[mask].copy()
            if len(ep_df) < 50:
                continue

            orig_smiles = ep_df["SMILES"].tolist()
            names = ep_df["Molecule Name"].values
            y = ep_df[ep].values

            if ep in LOG_TRANSFORM_ENDPOINTS:
                y = clip_and_log_transform(y)

            train_smiles = protonate_at_ph(orig_smiles, ENDPOINT_PH[ep])
            X_fp = compute_bit_fp(train_smiles)
            X_desc = compute_rdkit_descriptors(train_smiles)

            scaler = StandardScaler()
            X_desc = scaler.fit_transform(X_desc)
            X = np.hstack([X_fp, X_desc])

            fold_map = dict(zip(cluster_folds["Molecule Name"], cluster_folds["fold"]))
            fold_ids = np.array([fold_map.get(n, -1) for n in names])

            for fold in set(fold_ids):
                if fold < 0:
                    continue
                tr, te = fold_ids != fold, fold_ids == fold

                model = XGBRegressor(random_state=42, verbosity=0)
                model.fit(X[tr], y[tr])

                for name, smi_orig, true in zip(
                    names[te], np.array(orig_smiles)[te], y[te]
                ):
                    smi_canon = canonicalize_smiles(smi_orig)
                    res_forms = resonance_groups.get(smi_canon, [smi_canon])

                    Xr = np.hstack(
                        [
                            compute_bit_fp(res_forms),
                            scaler.transform(compute_rdkit_descriptors(res_forms)),
                        ]
                    )

                    preds = model.predict(Xr)
                    if name not in oof_predictions:
                        oof_predictions[name] = {}
                    oof_predictions[name][ep] = {
                        "preds": preds.tolist(),
                        "true": float(true),
                    }

        with open(cache_path, "w") as f:
            json.dump(oof_predictions, f, indent=4)

    # -----------------------------
    # Metrics
    # -----------------------------
    logger.info("Calculating metrics...")
    rows = []
    for name, ep_dict in oof_predictions.items():
        # Look up the canonical SMILES dynamically to match with our FP dictionaries
        orig_smi = name_to_orig_smi.get(name)
        smi_canon = canonicalize_smiles(orig_smi) if orig_smi else None

        for ep, d in ep_dict.items():
            preds = np.array(d["preds"])
            true = d["true"]

            if len(preds) > 1:
                maes = np.abs(preds - true)
                pred_mean = np.mean(preds)
                pred_std = np.std(preds)
                pred_cv = (
                    pred_std / abs(pred_mean) if abs(pred_mean) > 1e-10 else np.nan
                )

                # Fetch pre-calculated FP distances for this molecule
                fp_stats = fp_dists_per_mol.get(
                    smi_canon, {"max_dist": np.nan, "mean_dist": np.nan}
                )

                rows.append(
                    {
                        "molecule_name": name,
                        "endpoint": ep,
                        "resonance_range": preds.max() - preds.min(),
                        "pred_cv": pred_cv,
                        "max_mae": maes.max(),
                        "min_mae": maes.min(),
                        "num_resonance_forms": len(preds),
                        "max_fp_dist": fp_stats["max_dist"],
                        "mean_fp_dist": fp_stats["mean_dist"],
                    }
                )

    consistency_df = pd.DataFrame(rows)
    consistency_df.to_csv(output_dir / "consistency_metrics.csv", index=False)

    summary = (
        consistency_df.groupby("endpoint")[
            ["resonance_range", "pred_cv", "max_mae", "min_mae", "max_fp_dist"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(output_dir / "consistency_summary.csv", index=False)

    # -----------------------------
    # Plotting Suite
    # -----------------------------
    logger.info("Generating plots...")

    # 1. Fingerprint Distances
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"resonance": "coral", "random": "gray"}
    for vtype in ["resonance", "random"]:
        dists = fp_sim_df[fp_sim_df["variant_type"] == vtype][
            "tanimoto_distance"
        ].values
        if len(dists) > 0:
            ax.hist(
                dists,
                bins=50,
                density=True,
                alpha=0.6,
                color=colors[vtype],
                label=f"{vtype.capitalize()} (median={np.median(dists):.3f})",
            )
    ax.set_xlabel("ECFP4 Tanimoto Distance")
    ax.set_ylabel("Density")
    ax.set_title("Intra-group Fingerprint Distances (Resonance vs. Random)")
    ax.legend()
    fig.savefig(output_dir / "fingerprint_distances.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # 2. Prediction Consistency Boxplots (Grid of CVs)
    active_endpoints = sorted(consistency_df["endpoint"].unique())
    n_ep = len(active_endpoints)
    nrows = (n_ep + 2) // 3
    ncols = min(n_ep, 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for ax_idx, ep in enumerate(active_endpoints):
        ax = axes[ax_idx]
        ep_data = consistency_df[consistency_df["endpoint"] == ep]["pred_cv"].dropna()

        if len(ep_data) == 0:
            ax.set_visible(False)
            continue

        sns.boxplot(y=ep_data, ax=ax, color="coral", width=0.4, showfliers=False)

        ax.set_title(ep, fontsize=11, fontweight="bold")
        if ax_idx % ncols == 0:
            ax.set_ylabel("Prediction CV (Spread / Mean)")
        else:
            ax.set_ylabel("")
        ax.set_xticks([])  # Remove empty x-ticks

    # Hide unused axes if endpoints aren't a multiple of 3
    for ax_idx in range(n_ep, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Prediction CV Across Resonance Structures by Endpoint", fontsize=14, y=1.02
    )
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_consistency.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # 3. Consistency Heatmap (CV)
    fig, ax = plt.subplots(figsize=(4, 6))
    heatmap_data = summary.set_index("endpoint")[["pred_cv"]]
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean Prediction CV"},
    )
    ax.set_title("Mean Prediction CV\n(Resonance Forms)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_dir / "consistency_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    # 4. FP Distance vs Prediction Spread Scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=consistency_df,
        x="max_fp_dist",
        y="resonance_range",
        hue="endpoint",
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Max Fingerprint Distance (Within Resonance Group)")
    ax.set_ylabel("Prediction Range (Max - Min)")
    ax.set_title("Fingerprint Shift vs. Prediction Instability")
    fig.tight_layout()
    fig.savefig(output_dir / "spread_scatter.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info("Done.")


if __name__ == "__main__":
    app()
