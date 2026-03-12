#!/usr/bin/env python
"""Dataset loading pipeline for the Expansion Tx ADMET dataset.

Load ML-ready train + test CSVs, validate counts and SMILES, add computed
columns (mol_index, split, n_endpoints, valid_smiles), and save a canonical
parquet file for downstream analysis notebooks.

Usage:
    pixi run -e cheminformatics python notebooks/1.01-seal-dataset-loader.py

Outputs:
    data/interim/expansion_tx.parquet
    data/interim/molecule_names.csv
"""

import re
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from rdkit import Chem

from polaris_generalization.config import INTERIM_DATA_DIR, RAW_DATA_DIR

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


@app.command()
def main(
    output_dir: Path = typer.Option(INTERIM_DATA_DIR, help="Output directory"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading train and test CSVs")
    train = pd.read_csv(RAW_DATA_DIR / "expansion_data_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "expansion_data_test.csv")
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)
    logger.info(f"Loaded {len(train)} train + {len(test)} test = {len(df)} total")

    # ── 2. Validate ───────────────────────────────────────────────────
    logger.info("Validating dataset")

    # Counts
    assert len(train) == 5326, f"Expected 5326 train, got {len(train)}"
    assert len(test) == 2282, f"Expected 2282 test, got {len(test)}"
    assert len(df) == 7608, f"Expected 7608 total, got {len(df)}"
    logger.info("  Counts OK: 5,326 train + 2,282 test = 7,608 total")

    # No overlap
    overlap = set(train["Molecule Name"]) & set(test["Molecule Name"])
    assert len(overlap) == 0, f"Train/test overlap: {len(overlap)} molecules"
    logger.info("  No train/test overlap")

    # Endpoint columns present
    for ep in ENDPOINTS:
        assert ep in df.columns, f"Missing endpoint column: {ep}"
    logger.info(f"  All {len(ENDPOINTS)} endpoint columns present")

    # Molecule name format
    name_pattern = re.compile(r"^E-\d{7}$")
    bad_names = df["Molecule Name"][~df["Molecule Name"].str.match(r"^E-\d{7}$")]
    if len(bad_names) > 0:
        logger.warning(f"  {len(bad_names)} molecule names don't match E-XXXXXXX pattern")
        logger.warning(f"  Examples: {bad_names.head().tolist()}")
    else:
        logger.info("  All molecule names match E-XXXXXXX pattern")

    # SMILES validation
    valid_smiles = []
    n_failed = 0
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            valid_smiles.append(False)
            n_failed += 1
        else:
            valid_smiles.append(True)
    df["valid_smiles"] = valid_smiles
    if n_failed > 0:
        logger.warning(f"  {n_failed} SMILES failed to parse")
    else:
        logger.info(f"  All {len(df)} SMILES parse successfully")

    # ── 3. Add computed columns ───────────────────────────────────────
    logger.info("Adding computed columns")

    # Ordinal index from molecule name
    df["mol_index"] = df["Molecule Name"].str.extract(r"E-(\d+)")[0].astype(int)

    # Endpoint coverage per molecule
    df["n_endpoints"] = df[ENDPOINTS].notna().sum(axis=1)

    logger.info(f"  mol_index range: {df['mol_index'].min()} — {df['mol_index'].max()}")
    logger.info(f"  n_endpoints range: {df['n_endpoints'].min()} — {df['n_endpoints'].max()}")

    # ── 4. Ordinal ordering validation ────────────────────────────────
    train_median = df[df["split"] == "train"]["mol_index"].median()
    test_median = df[df["split"] == "test"]["mol_index"].median()
    logger.info(f"  Ordinal ordering — train median: {train_median:.0f}, test median: {test_median:.0f}")
    if test_median > train_median:
        logger.info("  Confirmed: test molecules have later ordinal indices than train")
    else:
        logger.warning("  Unexpected: test median index is not greater than train")

    # ── 5. Save ───────────────────────────────────────────────────────
    parquet_path = output_dir / "expansion_tx.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"Saved {parquet_path} ({len(df)} rows, {len(df.columns)} cols)")

    names_path = output_dir / "molecule_names.csv"
    df[["Molecule Name", "split", "mol_index"]].to_csv(names_path, index=False)
    logger.info(f"Saved {names_path}")

    # Summary
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Dtypes:\n{df.dtypes.to_string()}")


if __name__ == "__main__":
    app()
