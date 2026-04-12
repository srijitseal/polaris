# ABOUTME: Compute max Tanimoto similarity of ExpansionRX compounds to ChEMBL 36.
# ABOUTME: Demonstrates structural novelty of RNA-targeting molecules vs public chemical matter.
"""ChEMBL Tanimoto similarity analysis.

Compute maximum Tanimoto similarity of each ExpansionRX compound to any
compound in ChEMBL 36, demonstrating structural novelty relative to public
chemical matter. Two comparisons: (1) all ChEMBL compounds, (2) ChEMBL
compounds with ADME assay data.

Setup (first time):
    1. pixi install
    2. pixi run download
    3. First run downloads ChEMBL 36 SQLite (~4GB) via chembl_downloader.
       Cached at ~/.data/chembl/ and only downloaded once.
    4. ChEMBL fingerprints are cached at data/interim/chembl36_*.npz
       Delete these to force recomputation.

Usage:
    pixi run -e cheminformatics python notebooks/0.03-araripe-chembl-tanimoto.py

Outputs:
    data/processed/0.03-araripe-chembl-tanimoto/max_tanimoto_all_chembl.png
    data/processed/0.03-araripe-chembl-tanimoto/max_tanimoto_comparison.png
    data/processed/0.03-araripe-chembl-tanimoto/summary_statistics.csv
    data/interim/chembl36_morgan2048_fps.npz  (cached ChEMBL fingerprints)
    data/interim/chembl36_adme_morgan2048_fps.npz  (cached ADME subset FPs)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()

CHEMBL_VERSION = "36"
FP_NBITS = 2048
FP_RADIUS = 2


def compute_morgan_fps(smiles_list: list[str]) -> list:
    """Compute Morgan fingerprints from SMILES. Returns list with None for failed parses."""
    fps = []
    failed = 0
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            failed += 1
        else:
            fps.append(
                AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS, useChirality=True)
            )
    if failed > 0:
        logger.warning(f"{failed} SMILES failed to parse")
    return fps


def fps_to_numpy(fps: list) -> np.ndarray:
    """Convert list of ExplicitBitVect to numpy array (n_mols, n_bits)."""
    arr = np.zeros((len(fps), FP_NBITS), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def numpy_to_fps(arr: np.ndarray) -> list:
    """Convert numpy array back to list of ExplicitBitVect."""
    fps = []
    for row in tqdm(arr, desc="Reconstructing fingerprints"):
        bv = DataStructs.ExplicitBitVect(FP_NBITS)
        on_bits = np.where(row)[0].tolist()
        bv.SetBitsFromList(on_bits)
        fps.append(bv)
    return fps


def query_chembl_smiles(subset: str = "all") -> list[str]:
    """Extract canonical SMILES from ChEMBL via SQL.

    Args:
        subset: "all" for all compounds, "adme" for compounds with ADME assay data.
    """
    import chembl_downloader

    if subset == "all":
        sql = """
            SELECT DISTINCT canonical_smiles
            FROM compound_structures
            WHERE canonical_smiles IS NOT NULL
        """
        logger.info("Querying all ChEMBL compound SMILES...")
    elif subset == "adme":
        sql = """
            SELECT DISTINCT cs.canonical_smiles
            FROM compound_structures cs
            JOIN activities act ON cs.molregno = act.molregno
            JOIN assays a ON act.assay_id = a.assay_id
            WHERE a.assay_type = 'A'
              AND cs.canonical_smiles IS NOT NULL
        """
        logger.info("Querying ChEMBL ADME-assayed compound SMILES...")
    else:
        raise ValueError(f"Unknown subset: {subset}")

    df = chembl_downloader.query(sql, version=CHEMBL_VERSION)
    smiles = df["canonical_smiles"].tolist()
    logger.info(f"Retrieved {len(smiles):,} unique SMILES ({subset})")
    return smiles


def get_or_compute_chembl_fps(cache_path: Path, subset: str = "all") -> list:
    """Load cached ChEMBL fingerprints or compute and cache them."""
    if cache_path.exists():
        logger.info(f"Loading cached fingerprints from {cache_path}")
        data = np.load(cache_path)
        arr = data["fps"]
        logger.info(f"Loaded {arr.shape[0]:,} fingerprints")
        return numpy_to_fps(arr)

    smiles = query_chembl_smiles(subset=subset)

    logger.info(f"Computing Morgan fingerprints for {len(smiles):,} ChEMBL compounds...")
    fps = []
    failed = 0
    for smi in tqdm(smiles, desc=f"ChEMBL {subset} FPs"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failed += 1
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_NBITS, useChirality=True))
    logger.info(f"Computed {len(fps):,} fingerprints ({failed} SMILES failed to parse)")

    logger.info(f"Caching fingerprints to {cache_path}")
    arr = fps_to_numpy(fps)
    np.savez_compressed(cache_path, fps=arr)
    logger.info(f"Saved {cache_path} ({cache_path.stat().st_size / 1e6:.0f} MB)")

    return fps


def compute_max_tanimoto(query_fps: list, ref_fps: list, label: str = "") -> np.ndarray:
    """Compute max Tanimoto similarity of each query FP to any reference FP."""
    max_sims = np.zeros(len(query_fps))
    for i, qfp in enumerate(tqdm(query_fps, desc=f"Max Tanimoto ({label})")):
        sims = DataStructs.BulkTanimotoSimilarity(qfp, ref_fps)
        max_sims[i] = max(sims)
    return max_sims


def plot_comparison_publication(
    max_sim_all: np.ndarray,
    max_sim_adme: np.ndarray,
    n_all: int,
    n_adme: int,
    output_dir: Path,
    dpi: int,
) -> None:
    """Publication-quality comparison figure (no title, relative frequency, large fonts)."""
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 51)

    # Relative frequency (weights-based)
    weights_all = np.ones_like(max_sim_all) / len(max_sim_all)
    weights_adme = np.ones_like(max_sim_adme) / len(max_sim_adme)

    ax.hist(max_sim_all, bins=bins, weights=weights_all, alpha=0.6,
            color="steelblue", label=f"All ChEMBL 36 (n={n_all:,})", edgecolor="white")
    ax.hist(max_sim_adme, bins=bins, weights=weights_adme, alpha=0.6,
            color="coral", label=f"ADME subset (n={n_adme:,})", edgecolor="white")

    ax.set_xlabel("Maximum Tanimoto similarity")
    ax.set_ylabel("Relative frequency")

    # Median lines with legend entries
    med_all = np.median(max_sim_all)
    med_adme = np.median(max_sim_adme)
    ax.axvline(med_all, color="steelblue", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Median = {med_all:.2f}")
    ax.axvline(med_adme, color="coral", linestyle="--", linewidth=1.5, alpha=0.8,
               label=f"Median = {med_adme:.2f}")

    # Activity-relevant threshold
    ax.axvline(0.7, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
               label="Activity-relevant threshold (0.7)")

    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "max_tanimoto_comparison_v2.png", dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    logger.info("Saved max_tanimoto_comparison_v2.png")
    plt.close("all")


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "0.03-araripe-chembl-tanimoto", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    figures_only: bool = typer.Option(False, help="Regenerate figures from precomputed data"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figures-only mode: load precomputed similarities ──────────────
    if figures_only:
        sim_path = output_dir / "max_similarities.npz"
        if not sim_path.exists():
            logger.error(f"Cannot find {sim_path}. Run without --figures-only first.")
            raise typer.Exit(1)
        data = np.load(sim_path)
        max_sim_all = data["max_sim_all"]
        max_sim_adme = data["max_sim_adme"]
        n_all = int(data["n_all"])
        n_adme = int(data["n_adme"])
        logger.info(f"Loaded precomputed similarities from {sim_path}")
        plot_comparison_publication(max_sim_all, max_sim_adme, n_all, n_adme, output_dir, dpi)
        return

    # ── 1. Load ExpansionRX data and compute fingerprints ─────────────
    logger.info("Loading ExpansionRX datasets")
    train = pd.read_csv(RAW_DATA_DIR / "expansion_data_train.csv")
    test = pd.read_csv(RAW_DATA_DIR / "expansion_data_test.csv")
    train["split"] = "train"
    test["split"] = "test"
    df = pd.concat([train, test], ignore_index=True)
    logger.info(f"Combined: {len(df)} molecules")

    logger.info(f"Computing Morgan fingerprints ({FP_NBITS}-bit, radius {FP_RADIUS}, chirality-aware)")
    query_fps = compute_morgan_fps(df["SMILES"].tolist())

    # Filter out failed parses
    valid_mask = [fp is not None for fp in query_fps]
    if not all(valid_mask):
        n_invalid = sum(1 for v in valid_mask if not v)
        logger.warning(f"Dropping {n_invalid} molecules with invalid SMILES")
        df = df[valid_mask].reset_index(drop=True)
        query_fps = [fp for fp in query_fps if fp is not None]
    logger.info(f"Valid query molecules: {len(query_fps)}")

    # ── 2. Get or compute ChEMBL fingerprints ─────────────────────────
    all_chembl_fps = get_or_compute_chembl_fps(
        INTERIM_DATA_DIR / "chembl36_morgan2048_fps.npz", subset="all"
    )
    adme_chembl_fps = get_or_compute_chembl_fps(
        INTERIM_DATA_DIR / "chembl36_adme_morgan2048_fps.npz", subset="adme"
    )

    # ── 3. Compute max Tanimoto similarities ──────────────────────────
    logger.info("Computing max Tanimoto similarities...")
    max_sim_all = compute_max_tanimoto(query_fps, all_chembl_fps, label="all ChEMBL")
    max_sim_adme = compute_max_tanimoto(query_fps, adme_chembl_fps, label="ADME subset")

    # ── 4. Plot: max Tanimoto distribution (all ChEMBL) ───────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 101)
    ax.hist(max_sim_all, bins=bins, edgecolor="white", color="steelblue", alpha=0.8)
    ax.set_xlabel("Max Tanimoto similarity to ChEMBL")
    ax.set_ylabel("Count")
    ax.set_title(f"ExpansionRX vs ChEMBL {CHEMBL_VERSION} (N={len(max_sim_all):,}, all {len(all_chembl_fps):,} compounds)")

    # Threshold lines
    for thresh, ls, label in [(0.4, ":", "0.4"), (0.7, "--", "0.7"), (0.85, "-", "0.85")]:
        ax.axvline(thresh, color="gray", linestyle=ls, alpha=0.6, label=f"Tc={label}")
    ax.legend(fontsize=9)

    text = (
        f"median={np.median(max_sim_all):.3f}\n"
        f"mean={np.mean(max_sim_all):.3f}\n"
        f"<0.4: {(max_sim_all < 0.4).mean():.1%}\n"
        f"<0.7: {(max_sim_all < 0.7).mean():.1%}"
    )
    ax.text(
        0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top",
        fontsize=9, bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )
    fig.tight_layout()
    fig.savefig(output_dir / "max_tanimoto_all_chembl.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved max_tanimoto_all_chembl.png")
    plt.close("all")

    # ── 5. Plot: comparison (all ChEMBL vs ADME subset) ───────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(max_sim_all, bins=bins, density=True, alpha=0.6, color="steelblue", label=f"All ChEMBL (n={len(all_chembl_fps):,})", edgecolor="white")
    ax.hist(max_sim_adme, bins=bins, density=True, alpha=0.6, color="coral", label=f"ADME subset (n={len(adme_chembl_fps):,})", edgecolor="white")
    ax.set_xlabel("Max Tanimoto similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"ExpansionRX max similarity to ChEMBL {CHEMBL_VERSION}: all vs ADME subset")

    for vals, color, label, y_frac in [(max_sim_all, "steelblue", "all", 0.92), (max_sim_adme, "coral", "ADME", 0.82)]:
        med = np.median(vals)
        ax.axvline(med, color=color, linestyle="--", alpha=0.7)
        ax.text(med + 0.01, ax.get_ylim()[1] * y_frac, f"{label} med={med:.3f}", color=color, fontsize=9)

    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "max_tanimoto_comparison.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved max_tanimoto_comparison.png")
    plt.close("all")

    # ── 6. Save similarities for figures-only mode ──────────────────────
    np.savez_compressed(
        output_dir / "max_similarities.npz",
        max_sim_all=max_sim_all,
        max_sim_adme=max_sim_adme,
        n_all=len(all_chembl_fps),
        n_adme=len(adme_chembl_fps),
    )
    logger.info("Saved max_similarities.npz for --figures-only mode")

    # ── 7. Publication-quality comparison figure ───────────────────────
    plot_comparison_publication(max_sim_all, max_sim_adme, len(all_chembl_fps), len(adme_chembl_fps), output_dir, dpi)

    # ── 8. Summary statistics ─────────────────────────────────────────
    stats = []
    for label, vals in [("all_chembl", max_sim_all), ("adme_subset", max_sim_adme)]:
        stats.append({
            "reference_set": label,
            "n_reference": len(all_chembl_fps) if label == "all_chembl" else len(adme_chembl_fps),
            "n_query": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "q1": float(np.percentile(vals, 25)),
            "q3": float(np.percentile(vals, 75)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "frac_below_0.4": float((vals < 0.4).mean()),
            "frac_below_0.7": float((vals < 0.7).mean()),
            "frac_below_0.85": float((vals < 0.85).mean()),
        })
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(output_dir / "summary_statistics.csv", index=False)
    logger.info("Saved summary_statistics.csv")
    logger.info(f"\n{stats_df.to_string(index=False)}")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
