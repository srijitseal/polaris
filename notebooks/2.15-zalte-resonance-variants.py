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
from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import \
    MolecularDescriptorCalculator
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor

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

    # 1. Define Cache Path
    cache_path = output_dir / "raw_predictions_cache.json"

    # 2. Check if Cache Exists
    if cache_path.exists():
        logger.info(
            f"Found existing predictions cache at {cache_path}. Loading data and skipping model training..."
        )
        with open(cache_path, "r") as f:
            oof_predictions = json.load(f)

    else:
        logger.info(
            "No cache found. Commencing model training and feature extraction..."
        )

        # Load data
        df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
        cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
        cluster_folds = cluster_folds[cluster_folds["repeat"] == 0]

        # Load resonance structures
        resonance_df = pd.read_csv(
            PROCESSED_DATA_DIR
            / "2.14-zalte-resonance-generation"
            / "resonance_structures.csv"
        )

        # Build resonance groups (canonicalized)
        resonance_groups = {}
        for _, row in resonance_df.iterrows():
            parent = canonicalize_smiles(row["parent_smi"])
            res_list = [
                canonicalize_smiles(s) for s in json.loads(row["resonance_smis"])
            ]
            resonance_groups[parent] = res_list

        # Train + evaluate
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

                tr = fold_ids != fold
                te = fold_ids == fold

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

        # Save Cache
        with open(cache_path, "w") as f:
            json.dump(oof_predictions, f, indent=4)
        logger.info(f"Successfully cached raw predictions to {cache_path}")

    # -----------------------------
    # Metrics - FILTERED to only those WITH resonance
    # -----------------------------
    logger.info("Calculating consistency metrics...")
    rows = []
    for name, ep_dict in oof_predictions.items():
        for ep, d in ep_dict.items():
            preds = np.array(d["preds"])
            true = d["true"]

            if len(preds) > 1:
                maes = np.abs(preds - true)
                rows.append(
                    {
                        "molecule_name": name,
                        "endpoint": ep,
                        "resonance_range": preds.max() - preds.min(),
                        "max_mae": maes.max(),
                        "min_mae": maes.min(),
                        "num_resonance_forms": len(preds),
                    }
                )

    if not rows:
        logger.warning("No molecules with multiple resonance forms found. Exiting.")
        return

    consistency_df = pd.DataFrame(rows)
    consistency_df.to_csv(output_dir / "consistency_metrics.csv", index=False)

    summary = (
        consistency_df.groupby("endpoint")[["resonance_range", "max_mae", "min_mae"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(output_dir / "consistency_summary.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=consistency_df, x="endpoint", y="resonance_range", color="coral")
    plt.title("Prediction Variance Across Valid Resonance Structures")
    plt.ylabel("Prediction Range (Max - Min)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_consistency.png", dpi=dpi)
    plt.close()

    logger.info("Done.")


if __name__ == "__main__":
    app()
