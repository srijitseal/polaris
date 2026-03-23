#!/usr/bin/env python
"""Scaffold boundary analysis: formalizing why scaffold splits are arbitrary.

Hypothesis: Activity similarity is a continuous function of fingerprint distance.
Scaffold membership (same vs. different Murcko framework) provides no additional
information about activity difference beyond what fingerprint distance already
captures. Scaffold splits therefore create an artificial partition that ignores
the underlying probability landscape of activity-relevant similarity.

Three analyses:
1. Scaffold boundary violations — fraction of molecules whose overall 1-NN
   has a different scaffold (i.e., the most informative training molecule
   would be on the wrong side of a scaffold split boundary).
2. Cross-scaffold proximity — distribution of 1-NN distances to the nearest
   molecule with a different scaffold, compared to overall 1-NN.
3. Activity concordance conditioned on scaffold — at a given fingerprint
   distance, do same-scaffold pairs have more similar activities than
   different-scaffold pairs? Mann-Whitney U per distance bin, per endpoint.

Usage:
    pixi run -e cheminformatics python notebooks/2.15-araripe-scaffold-boundary.py

Outputs:
    data/processed/2.15-araripe-scaffold-boundary/boundary_violations.csv
    data/processed/2.15-araripe-scaffold-boundary/proximity_stats.csv
    data/processed/2.15-araripe-scaffold-boundary/activity_concordance.csv
    data/processed/2.15-araripe-scaffold-boundary/mann_whitney_tests.csv
    data/processed/2.15-araripe-scaffold-boundary/cross_scaffold_proximity.png
    data/processed/2.15-araripe-scaffold-boundary/activity_concordance.png
    data/processed/2.15-araripe-scaffold-boundary/boundary_violation_summary.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

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

# Distance bins for activity concordance analysis
BIN_WIDTH = 0.05
BIN_EDGES = np.arange(0.0, 1.0 + BIN_WIDTH, BIN_WIDTH)
MIN_PAIRS_PER_BIN = 30


def clip_and_log_transform(x: np.ndarray) -> np.ndarray:
    """Log-transform matching competition evaluation: log10(clip(x, 1e-10) + 1)."""
    return np.log10(np.clip(x, 1e-10, None) + 1)


def get_scaffold(smi: str) -> str:
    """Get generic Murcko scaffold SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(core)
    except Exception:
        return smi


def compute_scaffold_ids(smiles: list[str]) -> tuple[np.ndarray, dict[int, str]]:
    """Assign integer scaffold IDs to each molecule.

    Returns:
        scaffold_ids: array of integer IDs, one per molecule
        id_to_scaffold: mapping from ID to scaffold SMILES
    """
    scaffolds = [get_scaffold(s) for s in smiles]
    unique_scaffolds = sorted(set(scaffolds))
    scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
    id_to_scaffold = {i: s for s, i in scaffold_to_id.items()}
    scaffold_ids = np.array([scaffold_to_id[s] for s in scaffolds])
    return scaffold_ids, id_to_scaffold


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.15-araripe-scaffold-boundary",
        help="Output directory",
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading canonical dataset")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    n_total = len(df)
    logger.info(f"Loaded {n_total} molecules")

    logger.info("Loading precomputed distance matrix")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz")
    dist_condensed = npz["condensed"]
    # Load molecule names from CSV (avoids pickle dependency in npz)
    names_df = pd.read_csv(INTERIM_DATA_DIR / "molecule_names.csv")
    dist_mol_names = names_df["Molecule Name"].values
    name_to_dist_idx = {str(n): i for i, n in enumerate(dist_mol_names)}

    # Convert to square form once
    logger.info("Converting to square distance matrix")
    dist_square = squareform(dist_condensed)
    n_dist = dist_square.shape[0]
    logger.info(f"Distance matrix: {n_dist} x {n_dist}")

    # ── 2. Compute scaffold IDs ───────────────────────────────────────
    logger.info("Computing Murcko scaffolds")
    smiles_list = df["SMILES"].tolist()
    mol_names = df["Molecule Name"].values

    # Map molecules to distance matrix ordering
    mol_to_dist = np.array([name_to_dist_idx[n] for n in mol_names])

    scaffold_ids, id_to_scaffold = compute_scaffold_ids(smiles_list)
    n_scaffolds = len(id_to_scaffold)

    # Scaffold group sizes
    _, counts = np.unique(scaffold_ids, return_counts=True)
    n_singletons = int(np.sum(counts == 1))
    logger.info(
        f"  {n_scaffolds} unique scaffolds, "
        f"{n_singletons} singletons ({100 * n_singletons / n_scaffolds:.1f}%)"
    )

    # Reorder scaffold_ids to match distance matrix ordering
    scaffold_ids_dist = scaffold_ids[np.argsort(mol_to_dist)]

    # ── 3. Analysis 1: Scaffold boundary violations ───────────────────
    logger.info("Computing scaffold boundary violations")

    # For each molecule, find:
    #   - overall 1-NN distance
    #   - cross-scaffold 1-NN distance (nearest neighbor with different scaffold)
    #   - within-scaffold 1-NN distance (nearest neighbor with same scaffold)
    overall_1nn = np.full(n_dist, np.inf)
    cross_scaffold_1nn = np.full(n_dist, np.inf)
    within_scaffold_1nn = np.full(n_dist, np.inf)
    nn_is_different_scaffold = np.zeros(n_dist, dtype=bool)

    for i in range(n_dist):
        dists = dist_square[i].copy()
        dists[i] = np.inf  # exclude self

        # Overall 1-NN
        nn_idx = np.argmin(dists)
        overall_1nn[i] = dists[nn_idx]
        nn_is_different_scaffold[i] = scaffold_ids_dist[nn_idx] != scaffold_ids_dist[i]

        # Cross-scaffold 1-NN
        same_mask = scaffold_ids_dist == scaffold_ids_dist[i]
        cross_dists = dists.copy()
        cross_dists[same_mask] = np.inf
        cross_scaffold_1nn[i] = cross_dists.min()

        # Within-scaffold 1-NN (inf for singletons)
        within_dists = dists.copy()
        within_dists[~same_mask] = np.inf
        within_scaffold_1nn[i] = within_dists.min()

    # Boundary violation rate: fraction whose overall 1-NN has a different scaffold
    violation_rate = nn_is_different_scaffold.mean()
    logger.info(f"  Scaffold boundary violation rate: {violation_rate:.3f}")
    logger.info(
        f"  {nn_is_different_scaffold.sum()}/{n_dist} molecules have their "
        f"nearest neighbor in a different scaffold group"
    )

    # For non-singletons: fraction where cross-scaffold NN is closer than within-scaffold NN
    is_singleton_dist = np.array([
        np.sum(scaffold_ids_dist == scaffold_ids_dist[i]) == 1
        for i in range(n_dist)
    ])
    non_singleton_mask = ~is_singleton_dist
    n_non_singleton = non_singleton_mask.sum()

    if n_non_singleton > 0:
        cross_closer = (
            cross_scaffold_1nn[non_singleton_mask]
            < within_scaffold_1nn[non_singleton_mask]
        )
        cross_closer_rate = cross_closer.mean()
        logger.info(
            f"  Among non-singletons ({n_non_singleton}): "
            f"{cross_closer_rate:.3f} have a closer neighbor in a different scaffold"
        )
    else:
        cross_closer_rate = np.nan

    # Distance statistics
    proximity_rows = []
    for label, arr in [
        ("overall_1nn", overall_1nn),
        ("cross_scaffold_1nn", cross_scaffold_1nn),
        ("within_scaffold_1nn_nonsingletons", within_scaffold_1nn[non_singleton_mask]),
    ]:
        finite = arr[np.isfinite(arr)]
        proximity_rows.append({
            "metric": label,
            "n": len(finite),
            "mean": float(np.mean(finite)) if len(finite) > 0 else np.nan,
            "median": float(np.median(finite)) if len(finite) > 0 else np.nan,
            "q25": float(np.percentile(finite, 25)) if len(finite) > 0 else np.nan,
            "q75": float(np.percentile(finite, 75)) if len(finite) > 0 else np.nan,
        })

    proximity_df = pd.DataFrame(proximity_rows)
    proximity_df.to_csv(output_dir / "proximity_stats.csv", index=False)
    logger.info("Saved proximity_stats.csv")
    for _, row in proximity_df.iterrows():
        logger.info(
            f"    {row['metric']}: median={row['median']:.3f}, "
            f"mean={row['mean']:.3f} (n={row['n']})"
        )

    # Save boundary violation summary
    violation_summary = pd.DataFrame([{
        "n_molecules": n_dist,
        "n_scaffolds": n_scaffolds,
        "n_singletons": n_singletons,
        "pct_singleton_scaffolds": 100 * n_singletons / n_scaffolds,
        "boundary_violation_rate": violation_rate,
        "n_violations": int(nn_is_different_scaffold.sum()),
        "n_non_singletons": int(n_non_singleton),
        "cross_closer_rate_non_singletons": float(cross_closer_rate),
        "median_overall_1nn": float(np.median(overall_1nn)),
        "median_cross_scaffold_1nn": float(np.median(cross_scaffold_1nn)),
    }])
    violation_summary.to_csv(output_dir / "boundary_violations.csv", index=False)
    logger.info("Saved boundary_violations.csv")

    # ── 4. Figure A: Cross-scaffold proximity ─────────────────────────
    logger.info("Plotting cross-scaffold proximity distributions")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Overall vs cross-scaffold 1-NN
    ax = axes[0]
    bins = np.linspace(0, 1, 60)
    ax.hist(
        overall_1nn, bins=bins, density=True, alpha=0.6,
        color="steelblue", label=f"Overall 1-NN (med={np.median(overall_1nn):.3f})",
        edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        cross_scaffold_1nn, bins=bins, density=True, alpha=0.6,
        color="coral", label=f"Cross-scaffold 1-NN (med={np.median(cross_scaffold_1nn):.3f})",
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title("Overall vs. cross-scaffold nearest neighbor")
    ax.legend(fontsize=8)

    # Panel B: Within-scaffold vs cross-scaffold 1-NN (non-singletons only)
    ax = axes[1]
    within_finite = within_scaffold_1nn[non_singleton_mask]
    cross_finite = cross_scaffold_1nn[non_singleton_mask]
    ax.hist(
        within_finite[np.isfinite(within_finite)], bins=bins, density=True,
        alpha=0.6, color="forestgreen",
        label=f"Within-scaffold 1-NN (med={np.median(within_finite[np.isfinite(within_finite)]):.3f})",
        edgecolor="white", linewidth=0.5,
    )
    ax.hist(
        cross_finite, bins=bins, density=True, alpha=0.6,
        color="coral",
        label=f"Cross-scaffold 1-NN (med={np.median(cross_finite):.3f})",
        edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel("Tanimoto distance")
    ax.set_ylabel("Density")
    ax.set_title("Non-singleton molecules: within vs. cross-scaffold 1-NN")
    ax.legend(fontsize=8)

    # Panel C: Boundary violation rate at distance thresholds
    ax = axes[2]
    thresholds = np.arange(0.05, 1.01, 0.05)
    violation_at_threshold = []
    for t in thresholds:
        # Fraction of molecules with a different-scaffold neighbor within distance t
        frac = np.mean(cross_scaffold_1nn <= t)
        violation_at_threshold.append(frac)

    ax.plot(thresholds, violation_at_threshold, "o-", color="coral", markersize=4)
    ax.axhline(y=violation_rate, color="gray", linestyle="--", alpha=0.5,
               label=f"Overall violation rate: {violation_rate:.3f}")
    ax.set_xlabel("Distance threshold")
    ax.set_ylabel("Fraction with cross-scaffold neighbor")
    ax.set_title("Cumulative cross-scaffold proximity")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"Scaffold boundary analysis: {violation_rate:.1%} of molecules have their "
        f"nearest neighbor in a different scaffold group",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cross_scaffold_proximity.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved cross_scaffold_proximity.png")
    plt.close("all")

    # ── 5. Analysis 2: Activity concordance ───────────────────────────
    logger.info("Computing activity concordance conditioned on scaffold membership")

    concordance_rows = []
    mw_test_rows = []

    for ep in ENDPOINTS:
        mask = df[ep].notna().values
        ep_names = mol_names[mask]
        n_ep = len(ep_names)

        if n_ep < 50:
            logger.warning(f"Skipping {ep}: only {n_ep} molecules")
            continue

        # Map to distance matrix indices
        ep_dist_idx = np.array([name_to_dist_idx[n] for n in ep_names])

        # Sub-distance matrix
        D_ep = dist_square[np.ix_(ep_dist_idx, ep_dist_idx)]

        # Scaffold IDs for this endpoint's molecules
        scaf_ep = scaffold_ids_dist[ep_dist_idx]

        # Activity values (log-transformed where appropriate)
        raw_y = df.loc[mask, ep].values
        if ep in LOG_TRANSFORM_ENDPOINTS:
            y_ep = clip_and_log_transform(raw_y)
        else:
            y_ep = raw_y.copy()

        # Upper-triangle mask (unique pairs only)
        triu = np.triu(np.ones((n_ep, n_ep), dtype=bool), k=1)

        logger.info(f"  {ep} ({n_ep} molecules, {int(triu.sum())} pairs)")

        for bin_lo, bin_hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            # Find pairs in this distance bin
            in_bin = triu & (D_ep >= bin_lo) & (D_ep < bin_hi)
            pi, pj = np.where(in_bin)

            if len(pi) < MIN_PAIRS_PER_BIN:
                continue

            # Activity differences
            dy = np.abs(y_ep[pi] - y_ep[pj])

            # Same vs different scaffold
            is_same = scaf_ep[pi] == scaf_ep[pj]
            same_dy = dy[is_same]
            diff_dy = dy[~is_same]

            bin_center = (bin_lo + bin_hi) / 2

            concordance_rows.append({
                "endpoint": ep,
                "bin_lo": bin_lo,
                "bin_hi": bin_hi,
                "bin_center": bin_center,
                "n_same": len(same_dy),
                "n_diff": len(diff_dy),
                "n_total": len(dy),
                "mean_dy_same": float(np.mean(same_dy)) if len(same_dy) > 0 else np.nan,
                "mean_dy_diff": float(np.mean(diff_dy)) if len(diff_dy) > 0 else np.nan,
                "mean_dy_all": float(np.mean(dy)),
                "median_dy_same": float(np.median(same_dy)) if len(same_dy) > 0 else np.nan,
                "median_dy_diff": float(np.median(diff_dy)) if len(diff_dy) > 0 else np.nan,
                "median_dy_all": float(np.median(dy)),
                "frac_same": len(same_dy) / len(dy) if len(dy) > 0 else np.nan,
            })

            # Mann-Whitney U test (only if both groups have enough observations)
            if len(same_dy) >= 10 and len(diff_dy) >= 10:
                u_stat, p_val = mannwhitneyu(same_dy, diff_dy, alternative="two-sided")
                mw_test_rows.append({
                    "endpoint": ep,
                    "bin_center": bin_center,
                    "bin_lo": bin_lo,
                    "bin_hi": bin_hi,
                    "n_same": len(same_dy),
                    "n_diff": len(diff_dy),
                    "u_statistic": u_stat,
                    "p_value": p_val,
                    "significant_005": p_val < 0.05,
                    "mean_dy_same": float(np.mean(same_dy)),
                    "mean_dy_diff": float(np.mean(diff_dy)),
                    "effect_direction": "same < diff" if np.mean(same_dy) < np.mean(diff_dy) else "same >= diff",
                })

        # Free memory
        del D_ep, triu

    concordance_df = pd.DataFrame(concordance_rows)
    concordance_df.to_csv(output_dir / "activity_concordance.csv", index=False)
    logger.info(f"Saved activity_concordance.csv ({len(concordance_df)} rows)")

    mw_df = pd.DataFrame(mw_test_rows)

    # Benjamini-Hochberg FDR correction across all tests
    if len(mw_df) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            mw_df["p_value"].values, alpha=0.05, method="fdr_bh"
        )
        mw_df["p_value_bh"] = pvals_corrected
        mw_df["significant_bh_005"] = reject
        # Keep uncorrected column for reference
        mw_df["significant_uncorrected_005"] = mw_df["significant_005"]

    mw_df.to_csv(output_dir / "mann_whitney_tests.csv", index=False)
    logger.info(f"Saved mann_whitney_tests.csv ({len(mw_df)} rows)")

    # Summary: how many bins show significant differences?
    if len(mw_df) > 0:
        n_tests = len(mw_df)

        n_sig_uncorrected = mw_df["significant_005"].sum()
        n_sig_bh = mw_df["significant_bh_005"].sum()
        expected_by_chance = 0.05 * n_tests
        logger.info(
            f"  Mann-Whitney tests: {n_sig_uncorrected}/{n_tests} significant "
            f"uncorrected at p<0.05 (expected by chance: {expected_by_chance:.1f})"
        )
        logger.info(
            f"  After Benjamini-Hochberg FDR correction: "
            f"{n_sig_bh}/{n_tests} significant at q<0.05"
        )
        # Among BH-significant: direction of effect
        if n_sig_bh > 0:
            sig_subset = mw_df[mw_df["significant_bh_005"]]
            n_same_lower = (sig_subset["effect_direction"] == "same < diff").sum()
            logger.info(
                f"  Among BH-significant: {n_same_lower}/{n_sig_bh} have "
                f"same-scaffold |dy| < different-scaffold |dy|"
            )

    # ── 6. Figure B: Activity concordance curves ──────────────────────
    logger.info("Plotting activity concordance curves")

    # Use well-covered endpoints for the main figure
    plot_endpoints = [ep for ep in ENDPOINTS if ep in concordance_df["endpoint"].unique()]
    n_panels = len(plot_endpoints)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.ravel() if n_panels > 1 else [axes]

    for ax_idx, ep in enumerate(plot_endpoints):
        ax = axes[ax_idx]
        ep_data = concordance_df[concordance_df["endpoint"] == ep].copy()

        # Plot same-scaffold and different-scaffold curves
        has_same = ep_data["n_same"] >= 10
        has_diff = ep_data["n_diff"] >= 10

        if has_same.any():
            same_data = ep_data[has_same]
            ax.plot(
                same_data["bin_center"], same_data["mean_dy_same"],
                "o-", color="coral", markersize=4, linewidth=1.5,
                label="Same scaffold", alpha=0.8,
            )

        if has_diff.any():
            diff_data = ep_data[has_diff]
            ax.plot(
                diff_data["bin_center"], diff_data["mean_dy_diff"],
                "s-", color="steelblue", markersize=4, linewidth=1.5,
                label="Different scaffold", alpha=0.8,
            )

        ax.set_xlabel("Tanimoto distance")
        ax.set_ylabel("Mean |Δactivity|")
        ax.set_title(ep, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.set_xlim(0, 1)

    # Hide unused panels
    for ax_idx in range(n_panels, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(
        "Activity difference vs. fingerprint distance, conditioned on scaffold membership",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "activity_concordance.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved activity_concordance.png")
    plt.close("all")

    # ── 7. Figure C: Boundary violation summary ───────────────────────
    logger.info("Plotting boundary violation summary")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Scatter of cross-scaffold vs within-scaffold 1-NN (non-singletons)
    ax = axes[0]
    within_ns = within_scaffold_1nn[non_singleton_mask]
    cross_ns = cross_scaffold_1nn[non_singleton_mask]
    valid = np.isfinite(within_ns) & np.isfinite(cross_ns)

    ax.scatter(
        within_ns[valid], cross_ns[valid],
        s=3, alpha=0.15, color="steelblue", rasterized=True,
    )
    lim = max(within_ns[valid].max(), cross_ns[valid].max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="gray", alpha=0.5, label="y = x")
    ax.set_xlabel("Within-scaffold 1-NN distance")
    ax.set_ylabel("Cross-scaffold 1-NN distance")
    ax.set_title(
        f"Non-singleton molecules (n={valid.sum()})\n"
        f"{cross_closer_rate:.1%} have closer cross-scaffold neighbor"
    )
    ax.legend(fontsize=8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    # Panel B: Fraction of same-scaffold pairs by distance bin (averaged across endpoints)
    ax = axes[1]
    if len(concordance_df) > 0:
        frac_by_bin = concordance_df.groupby("bin_center")["frac_same"].mean()
        ax.bar(
            frac_by_bin.index, frac_by_bin.values,
            width=BIN_WIDTH * 0.8, color="coral", alpha=0.7, edgecolor="white",
        )
        ax.set_xlabel("Tanimoto distance")
        ax.set_ylabel("Fraction same-scaffold pairs")
        ax.set_title("Same-scaffold pair prevalence by distance\n(averaged across endpoints)")
        ax.set_xlim(0, 1)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "boundary_violation_summary.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved boundary_violation_summary.png")
    plt.close("all")

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
