#!/usr/bin/env python
"""Molecular variant consistency analysis: Resonance Structures.

Evaluates model robustness against resonance artifacts. The model is trained 
on standard (canonical) SMILES. During cross-validation testing, the test set 
is augmented with all valid resonance structures. We report the Resonance Range, 
Max MAE (worst-case resonance prediction), and Min MAE (best-case prediction)
using bit-based ECFP4 fingerprints with chirality.

Usage:
    pixi run -e cheminformatics python notebooks/2.14-zalte-resonance-variants.py

Outputs:
    data/processed/2.14-zalte-resonance-variants/resonance_distribution.csv
    data/processed/2.14-zalte-resonance-variants/consistency_metrics.csv
    data/processed/2.14-zalte-resonance-variants/consistency_summary.csv
    data/processed/2.14-zalte-resonance-variants/fingerprint_distances.png
    data/processed/2.14-zalte-resonance-variants/prediction_consistency.png
"""

from itertools import combinations
from collections import Counter
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

ENDPOINT_PH = {ep: 7.4 for ep in ENDPOINTS}
ENDPOINT_PH["Caco-2 Permeability Papp A>B"] = 6.5
ENDPOINT_PH["Caco-2 Permeability Efflux"] = 6.5

DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
DESC_CALC = MolecularDescriptorCalculator(DESCRIPTOR_NAMES)

VARIANT_COLORS = {"resonance": "coral", "random": "gray"}
VARIANT_ORDER = ["resonance", "random"]


def clip_and_log_transform(x: np.ndarray) -> np.ndarray:
    """Log-transform matching competition evaluation."""
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


def generate_resonance_structures(smiles: str) -> list[str]:
    """Generates unique resonance structures for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [smiles]

    suppl = Chem.ResonanceMolSupplier(mol)
    unique_smiles = set()
    for res_mol in suppl:
        if res_mol is not None:
            # isomericSmiles=True tracks stereocenters through resonance
            unique_smiles.add(Chem.MolToSmiles(res_mol, isomericSmiles=True))
    
    res_list = list(unique_smiles)
    return res_list if res_list else [smiles]


def compute_bit_fp(smiles_list: list[str], nbits: int = 2048, radius: int = 2) -> np.ndarray:
    """Compute BIT-BASED ECFP4 fingerprints with chirality."""
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # Generates standard bit vector, but accounts for stereochemistry
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True)
            fps.append(np.array(fp, dtype=np.uint8))
        else:
            fps.append(np.zeros(nbits, dtype=np.uint8))
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


def compute_tanimoto_distances_within_group(smiles_list: list[str], nbits: int = 2048, radius: int = 2) -> list[float]:
    """Compute standard Tanimoto distances within a group using bit-based vectors."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    fps = []
    for mol in mols:
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits, useChirality=True))
        else:
            fps.append(None)

    distances = []
    for i, j in combinations(range(len(fps)), 2):
        if fps[i] is not None and fps[j] is not None:
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1.0 - sim)
    return distances


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.14-zalte-resonance-variants", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    smiles_all = df["SMILES"].tolist()
    names_all = df["Molecule Name"].values

    logger.info("Loading cluster CV folds (repeat 0)")
    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    cluster_folds = cluster_folds[cluster_folds["repeat"] == 0]

    # ── 2. Identify Resonance Groups & Log Distribution ───────────────
    logger.info("Generating resonance groups for the entire dataset...")
    resonance_groups: dict[str, list[str]] = {}
    resonance_counts = []
    
    for idx, smi in enumerate(smiles_all):
        res_forms = generate_resonance_structures(smi)
        count = len(res_forms)
        resonance_counts.append(count)
        if count >= 2:
            resonance_groups[smi] = res_forms

    # Calculate and log the distribution of resonance structures
    dist_counter = Counter(resonance_counts)
    dist_rows = [{"n_resonance_structures": k, "n_molecules": v, "percentage": (v/len(df))*100} 
                 for k, v in sorted(dist_counter.items())]
    dist_df = pd.DataFrame(dist_rows)
    dist_df.to_csv(output_dir / "resonance_distribution.csv", index=False)
    
    logger.info("\n--- Resonance Structure Distribution ---")
    logger.info(dist_df.to_string(index=False, float_format="%.1f%%"))
    logger.info("----------------------------------------\n")

    # ── 3. Fingerprint similarity within groups ───────────────────────
    logger.info("Computing intra-group Tanimoto distances (Bit-based, Chiral)")
    fp_sim_rows = []
    all_distances = []
    
    for canonical_smi, res_list in resonance_groups.items():
        dists = compute_tanimoto_distances_within_group(res_list)
        all_distances.extend(dists)
        for d in dists:
            fp_sim_rows.append({"variant_type": "resonance", "tanimoto_distance": d})

    # Random baseline
    rng = np.random.default_rng(42)
    n_random_pairs = min(len(all_distances), 5000) if all_distances else 0
    random_indices = rng.integers(0, len(smiles_all), size=(n_random_pairs, 2))
    
    for i, j in random_indices:
        if i != j:
            dists = compute_tanimoto_distances_within_group([smiles_all[i], smiles_all[j]])
            if dists:
                fp_sim_rows.append({"variant_type": "random", "tanimoto_distance": dists[0]})

    if fp_sim_rows:
        pd.DataFrame(fp_sim_rows).to_csv(output_dir / "fingerprint_similarity.csv", index=False)

    # ── 4. Train and collect predictions across resonance forms ───────
    logger.info("Training on standard SMILES, testing on all resonance variations...")
    oof_predictions: dict[str, dict[str, dict]] = {}

    for ep in ENDPOINTS:
        ph = ENDPOINT_PH[ep]
        mask = df[ep].notna().values
        ep_df = df[mask].copy()
        
        if len(ep_df) < 50:
            continue

        ep_names = ep_df["Molecule Name"].values
        ep_smiles = ep_df["SMILES"].tolist()
        raw_values = ep_df[ep].values
        y_all = clip_and_log_transform(raw_values) if ep in LOG_TRANSFORM_ENDPOINTS else raw_values

        # Generate features for standard canonical training set
        prot_smiles = protonate_at_ph(ep_smiles, ph)
        ecfp = compute_bit_fp(prot_smiles)
        desc = compute_rdkit_descriptors(prot_smiles)
        
        variance = desc.var(axis=0)
        valid_desc_mask = variance > 0
        desc = desc[:, valid_desc_mask]
        
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc)
        X_all = np.hstack([ecfp, desc_scaled])

        # Setup CV Folds
        ep_folds = cluster_folds[cluster_folds["endpoint"] == ep]
        fold_map = dict(zip(ep_folds["Molecule Name"], ep_folds["fold"]))
        fold_ids = np.array([fold_map.get(n, -1) for n in ep_names])
        unique_folds = sorted(set(fold_ids[fold_ids >= 0]))

        logger.info(f"  {ep}: {len(ep_df)} molecules, {len(unique_folds)} folds")

        for fold_id in unique_folds:
            test_mask = fold_ids == fold_id
            train_mask = (fold_ids >= 0) & ~test_mask

            # Train only on canonical representations
            X_tr, y_tr = X_all[train_mask], y_all[train_mask]
            model = XGBRegressor(random_state=42, verbosity=0)
            model.fit(X_tr, y_tr)

            # Evaluate test set
            test_names = ep_names[test_mask]
            test_smiles_original = [ep_smiles[i] for i, m in enumerate(test_mask) if m]
            test_y_true = y_all[test_mask]

            for name, smi, true_val in zip(test_names, test_smiles_original, test_y_true):
                if name not in oof_predictions:
                    oof_predictions[name] = {}
                
                # Fetch all resonance structures for the test molecule
                res_forms_to_test = resonance_groups.get(smi, [smi])
                
                # If there's only 1 form, skip appending to save processing/memory
                if len(res_forms_to_test) < 2:
                    continue

                res_forms_prot = protonate_at_ph(res_forms_to_test, ph)
                
                # Generate features for all test resonance variants
                res_ecfp = compute_bit_fp(res_forms_prot)
                res_desc = compute_rdkit_descriptors(res_forms_prot)[:, valid_desc_mask]
                res_desc_scaled = scaler.transform(res_desc) 
                
                X_res_test = np.hstack([res_ecfp, res_desc_scaled])
                
                # Predict all forms
                res_preds = model.predict(X_res_test)
                
                oof_predictions[name][ep] = {
                    "preds": res_preds.tolist(), 
                    "true": float(true_val),
                    "n_forms": len(res_preds)
                }

    # ── 5. Compute MAE and Range Metrics ──────────────────────────────
    logger.info("Computing Max/Min MAE and Resonance Range metrics...")
    consistency_rows = []

    for name, ep_dict in oof_predictions.items():
        for ep, data in ep_dict.items():
            preds_arr = np.array(data["preds"])
            true_val = data["true"]
            
            # Absolute errors for all resonance forms of this molecule
            maes = np.abs(preds_arr - true_val)
            
            max_mae = np.max(maes)
            min_mae = np.min(maes)
            pred_range = np.ptp(preds_arr)  # ptp is peak-to-peak (max - min)
            pred_mean = np.mean(preds_arr)

            consistency_rows.append({
                "molecule_name": name,
                "endpoint": ep,
                "n_resonance_forms": data["n_forms"],
                "true_val": true_val,
                "pred_mean": pred_mean,
                "resonance_range": pred_range,
                "max_mae": max_mae,
                "min_mae": min_mae,
            })

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(output_dir / "consistency_metrics.csv", index=False)

    # ── 6. Aggregate consistency summary ──────────────────────────────
    summary_rows = []
    for ep, grp in consistency_df.groupby("endpoint"):
        summary_rows.append({
            "endpoint": ep,
            "n_test_molecules_with_resonance": len(grp),
            "mean_resonance_range": grp["resonance_range"].mean(),
            "mean_max_mae": grp["max_mae"].mean(),
            "mean_min_mae": grp["min_mae"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "consistency_summary.csv", index=False)
    
    logger.info("\n--- Model Error Bounds on Resonance Variations (Test Set) ---")
    for _, row in summary_df.iterrows():
        logger.info(f"  {row['endpoint']:>30s} | Range={row['mean_resonance_range']:.3f} | Min MAE={row['mean_min_mae']:.3f} | Max MAE={row['mean_max_mae']:.3f}")

    # ── 7. Visualizations ─────────────────────────────────────────────
    # Plot: Range across endpoints
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=consistency_df, x="endpoint", y="resonance_range", color="coral", width=0.5, ax=ax, fliersize=3)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Resonance Range (Max Pred - Min Pred)")
    ax.set_title("Test Set Prediction Spread Across Resonance Forms", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "prediction_consistency.png", dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"All outputs successfully saved to {output_dir}")

if __name__ == "__main__":
    app()