#!/usr/bin/env python
"""
2.14-zalte-resonance-generation.py

Generates resonance structures for a list of SMILES and saves the results for downstream analysis.

Usage:
    python 2.14-zalte-resonance-generation.py

Input: Uses canonical dataset and cluster CV folds.
Output: CSV with columns 'parent_smi', 'resonance_smis' (as a JSON list), 'num_resonance'.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, ResonanceMolSupplier
from tqdm import tqdm

# Disable RDKit warnings and errors from printing to the console
RDLogger.DisableLog("rdApp.*")


def generate_resonance_structure_smis_rdkit(
    parent_smi: str, max_structs: int = 50, keep_stereo: bool = False
):
    """
    Generate chemically reasonable resonance structures for a molecule using RDKit.

    Parameters
    ----------
    parent_smi : str
        The canonical SMILES of the parent molecule.
    max_structs : int
        Maximum number of forms per enumeration strategy to generate.
    keep_stereo : bool
        Whether to preserve stereochemistry in SMILES.

    Returns
    -------
    tuple
        (parent_smi, list_of_resonance_smis, num_resonance)
    """

    mol = Chem.MolFromSmiles(parent_smi)
    if mol is None:
        return parent_smi, [parent_smi], 1

    try:
        # Sanitize molecule
        Chem.SanitizeMol(mol)

        # 1. Calculate parent properties for validation
        parent_abs_charge = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
        parent_radicals = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())

        # 2. Generate exact canonical SMILES for the parent
        canon_parent_smi = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=keep_stereo
        )

        def is_valid_resonance(res_mol):
            """Basic chemical validity, charge separation, and multiplicity check."""
            # RDKit Sanitize check (catches impossible valences, bad aromaticity)
            sanitize_flag = Chem.SanitizeMol(res_mol, catchErrors=True)
            if sanitize_flag != Chem.SanitizeFlags.SANITIZE_NONE:
                return False

            # Dynamic charge separation check
            total_abs_charge = sum(abs(a.GetFormalCharge()) for a in res_mol.GetAtoms())
            if total_abs_charge > parent_abs_charge + 2:
                return False

            # Multiplicity / Radical check
            total_radicals = sum(a.GetNumRadicalElectrons() for a in res_mol.GetAtoms())
            if total_radicals != parent_radicals:
                return False

            return True

        # Initialize our final list and seen set with the parent SMILES guaranteed at index 0
        final_list = [canon_parent_smi]
        seen_smiles = {canon_parent_smi}

        # --- Strategy 1: default RDKit enumeration ---
        suppl1 = ResonanceMolSupplier(mol, flags=0, maxStructs=max_structs)
        for res_mol in suppl1:
            if res_mol is None:
                continue
            if is_valid_resonance(res_mol):
                smi = Chem.MolToSmiles(
                    res_mol, canonical=True, isomericSmiles=keep_stereo
                )
                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    final_list.append(smi)

        # --- Strategy 2: allow charge separation ---
        suppl2 = ResonanceMolSupplier(
            mol, flags=Chem.ALLOW_CHARGE_SEPARATION, maxStructs=max_structs
        )
        for res_mol in suppl2:
            if res_mol is None:
                continue
            if is_valid_resonance(res_mol):
                smi = Chem.MolToSmiles(
                    res_mol, canonical=True, isomericSmiles=keep_stereo
                )
                if smi not in seen_smiles:
                    seen_smiles.add(smi)
                    final_list.append(smi)

        return parent_smi, final_list, len(final_list)

    except Exception:
        return parent_smi, [parent_smi], 1


def visualize_high_resonance_examples(df_out, n_samples=5, max_cols=3, output_dir=None):
    """
    Sample molecules with highest number of resonance forms and visualize them.
    """

    # Sort by number of resonance forms (descending)
    df_sorted = df_out.sort_values("num_resonance", ascending=False)

    # Filter to molecules with >1 resonance form
    df_sorted = df_sorted[df_sorted["num_resonance"] > 1]

    # Take top candidates, then sample
    top_df = df_sorted.head(2000)  # restrict pool
    sample_df = top_df.sample(min(n_samples, len(top_df)), random_state=42)

    for i, row in enumerate(sample_df.itertuples()):
        parent = row.parent_smi
        res_smis = json.loads(row.resonance_smis)

        mols = []
        legends = []

        for j, smi in enumerate(res_smis):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols.append(mol)
                legends.append(f"{j+1}")

        if not mols:
            continue

        img = Draw.MolsToGridImage(
            mols, molsPerRow=max_cols, subImgSize=(250, 250), legends=legends
        )

        if output_dir is not None:
            save_path = output_dir / f"resonance_example_{i}.png"
            img.save(save_path)
        else:
            pass


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed/2.14-zalte-resonance-generation")
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = interim_dir / "expansion_tx.parquet"
    folds_path = interim_dir / "cluster_cv_folds.parquet"
    output_path = processed_dir / "resonance_structures.csv"

    # Load data
    df = pd.read_parquet(input_path)
    folds = pd.read_parquet(folds_path)

    folds = folds[folds["repeat"] == 0]

    # Get test set molecules
    test_names = folds["Molecule Name"].unique()
    test_df = df[df["Molecule Name"].isin(test_names)]

    test_smis = test_df[["Molecule Name", "SMILES"]].drop_duplicates()

    results = []

    for row in tqdm(
        test_smis.itertuples(index=False),
        total=len(test_smis),
        desc="Generating resonance structures",
    ):
        name = row[0]
        smi = row[1]

        parent, res_smis, n = generate_resonance_structure_smis_rdkit(smi)

        results.append(
            {
                "molecule_name": name,
                "parent_smi": parent,
                "resonance_smis": json.dumps(res_smis),
                "num_resonance": n,
            }
        )

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)

    print(f"Saved resonance structures to {output_path}")

    # -----------------------------
    # Visualize examples with high number of resonance forms
    # -----------------------------
    visualize_high_resonance_examples(df_out, output_dir=processed_dir)

    # -----------------------------
    # Analysis plot
    # -----------------------------
    counts = df_out["num_resonance"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    counts.plot(kind="bar", logy=True)

    plt.xlabel("Number of Resonance Structures")
    plt.ylabel("Number of Molecules")
    # add a text giving the average number of resonance structures
    avg_resonance = df_out["num_resonance"].mean()
    plt.text(
        0.9,
        0.9,
        f"Avg: {avg_resonance:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
    )
    plt.tight_layout()
    plt.savefig(processed_dir / "resonance_structures_distribution.png")
    plt.close()


# -----------------------------
if __name__ == "__main__":
    main()
