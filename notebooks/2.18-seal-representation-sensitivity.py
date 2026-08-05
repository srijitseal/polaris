#!/usr/bin/env python
"""Benchmark how sensitively ECFP4 and CheMeleon respond to minimal structural edits.

Tests the claim that circular fingerprints "amplify" small structural changes because
of their discreteness (one decorating atom flips several bits), while a pretrained
graph representation is smoother (the same edit perturbs the embedding less relative
to typical pairwise distances).

Two matched-pair sets from 2.13 are used as controlled perturbations:
    stereoisomer        - identical constitution, differing only in stereochemistry
    scaffold_decoration - same Murcko scaffold, differing by peripheral substituents

For each pair the distance is measured in three representations (ECFP4 Tanimoto,
CheMeleon cosine, CheMeleon Euclidean) and expressed on two scales:

    ratio-to-background  median pair distance / median random-pair distance.
                         Interpretable but metric-dependent - a monotone rescaling
                         of a representation's distance changes it.
    neighbour rank       percentile rank of the partner among all N molecules in
                         that representation. Scale-free: invariant to any monotone
                         transform of the distance, so it is the fair comparison.

Also reported: blind-spot rate (pairs at exactly zero distance) and, for scaffold
decorations, perturbation size as a function of heavy-atom difference.

Two further checks back the claims made in the manuscript from this notebook:

    metric dependence      ratio-to-background recomputed under monotone transforms
                           of the ECFP4 distance, showing the ratio carries no
                           representation-level information.
    consistency robustness 2.13's per-group consistency ratios re-aggregated by
                           median as well as mean, since the ratio distribution is
                           right-skewed and the two disagree in direction.

Usage:
    pixi run -e cheminformatics python notebooks/2.18-seal-representation-sensitivity.py

Outputs (under data/processed/2.18-seal-representation-sensitivity/):
    sensitivity_summary.csv          one row per (representation, variant type)
    pair_distances.csv               per-pair distances and neighbour ranks
    decoration_size_effect.csv       perturbation size vs heavy-atom difference
    metric_dependence.csv            ratio-to-background under monotone transforms
    consistency_ratio_robustness.csv mean vs median consistency ratio, both models
    representation_sensitivity.png
    decoration_size_effect.png
    consistency_ratio_robustness.png
"""

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from loguru import logger
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import (
    DEFAULT_DPI,
    MODEL_COLORS,
    MODEL_LABELS,
    set_style,
)

RDLogger.DisableLog("rdApp.*")

app = typer.Typer()

VARIANT_TYPES = ["stereoisomer", "scaffold_decoration"]
VARIANT_LABELS = {
    "stereoisomer": "Stereoisomer pairs",
    "scaffold_decoration": "Scaffold decoration pairs",
}
REPRESENTATIONS = ["ECFP4 Tanimoto", "CheMeleon cosine", "CheMeleon Euclidean"]
REP_COLORS = {
    "ECFP4 Tanimoto": MODEL_COLORS["xgboost"],
    "CheMeleon cosine": MODEL_COLORS["chemeleon"],
    "CheMeleon Euclidean": "darkred",
}
MODELS = ["xgboost", "chemeleon"]
ENDPOINTS = [
    "LogD", "KSOL", "HLM CLint", "MLM CLint",
    "Caco-2 Permeability Papp A>B", "Caco-2 Permeability Efflux",
    "MPPB", "MBPB", "MGMB",
]
MONOTONE_TRANSFORMS = {
    "sqrt(d)": lambda x: np.sqrt(x),
    "d (as reported)": lambda x: x,
    "d^1.5": lambda x: x**1.5,
    "d^2": lambda x: x**2,
    "d^3": lambda x: x**3,
}
N_BACKGROUND_PAIRS = 200_000
NBITS = 2048
RADIUS = 2
EMBEDDING_CACHE = INTERIM_DATA_DIR / "chemeleon_pretrained_embeddings.npz"


# ── Representations ───────────────────────────────────────────────────────────

def compute_ecfp4_matrix(smiles_list: list[str]) -> np.ndarray:
    """Chirality-aware ECFP4 bit matrix, shape (n, NBITS), dtype uint8."""
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append(np.zeros(NBITS, dtype=np.uint8))
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=NBITS, useChirality=True)
            rows.append(np.array(fp, dtype=np.uint8))
    return np.vstack(rows)


def compute_chemeleon_embeddings(smiles_list: list[str], cache: Path = EMBEDDING_CACHE) -> np.ndarray:
    """Pretrained CheMeleon molecule embeddings (no fine-tuning), shape (n, d).

    The message-passing block is loaded from the frozen foundation-model weights and
    run in eval mode; molecule vectors are the mean aggregation of atom states. Cached
    to `cache` keyed on the SMILES list so repeat runs are instant.
    """
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        if list(blob["smiles"]) == list(smiles_list):
            logger.info(f"Loaded cached CheMeleon embeddings from {cache}")
            return blob["embeddings"]
        logger.info("Cached embeddings do not match current molecule set; recomputing")

    import torch
    from chemprop import data, featurizers
    from chemprop import nn as chemprop_nn

    from polaris_generalization.chemprop_utils import _ensure_chemeleon_weights

    weights_path = _ensure_chemeleon_weights()
    ckpt = torch.load(weights_path, weights_only=True)
    mp = chemprop_nn.BondMessagePassing(**ckpt["hyper_parameters"])
    mp.load_state_dict(ckpt["state_dict"])
    mp.eval()
    agg = chemprop_nn.MeanAggregation()

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(
        [data.MoleculeDatapoint.from_smi(smi) for smi in smiles_list], featurizer
    )
    loader = data.build_dataloader(dset, shuffle=False, batch_size=128, num_workers=0)

    logger.info(f"Embedding {len(smiles_list)} molecules with pretrained CheMeleon")
    chunks = []
    with torch.no_grad():
        for batch in loader:
            bmg = batch.bmg
            chunks.append(agg(mp(bmg), bmg.batch).numpy())
    embeddings = np.vstack(chunks).astype(np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, smiles=np.array(smiles_list, dtype=object), embeddings=embeddings)
    logger.info(f"Saved embeddings to {cache}")
    return embeddings


def featurizer_sees_chirality() -> tuple[bool, int, int]:
    """Does the Chemprop atom featurizer distinguish enantiomeric atoms at all?"""
    from chemprop import featurizers

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    left = featurizer.atom_featurizer(Chem.MolFromSmiles("C[C@H](N)C(=O)O").GetAtomWithIdx(1))
    right = featurizer.atom_featurizer(Chem.MolFromSmiles("C[C@@H](N)C(=O)O").GetAtomWithIdx(1))
    n_diff = int(np.sum(np.asarray(left) != np.asarray(right)))
    return n_diff > 0, n_diff, len(left)


# ── Distance kernels (all vectorised: query rows against the full library) ─────

def tanimoto_rows(bits: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    """Tanimoto distance from each query molecule to every molecule, shape (q, n)."""
    x = bits.astype(np.float32)
    popcount = x.sum(axis=1)
    intersect = x[query_idx] @ x.T
    union = popcount[query_idx][:, None] + popcount[None, :] - intersect
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, intersect / union, 1.0)
    return (1.0 - sim).astype(np.float32)


def cosine_rows(emb: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return (1.0 - unit[query_idx] @ unit.T).astype(np.float32)


def euclidean_rows(emb: np.ndarray, query_idx: np.ndarray) -> np.ndarray:
    sq = (emb**2).sum(axis=1)
    d2 = sq[query_idx][:, None] + sq[None, :] - 2.0 * (emb[query_idx] @ emb.T)
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def background_distances(kernel, rng: np.random.Generator, n_mol: int) -> np.ndarray:
    """Distances for random molecule pairs, defining the representation's own scale."""
    queries = rng.choice(n_mol, size=2000, replace=False)
    rows = kernel(queries)
    partners = rng.integers(0, n_mol, size=(len(queries), N_BACKGROUND_PAIRS // len(queries)))
    vals = np.take_along_axis(rows, partners, axis=1).ravel()
    self_hits = np.repeat(queries, partners.shape[1]) == partners.ravel()
    return vals[~self_hits]


# ── Analysis ──────────────────────────────────────────────────────────────────

def load_pairs(variant_groups: pd.DataFrame, idx_of: dict[str, int]) -> dict[str, list[tuple[int, int]]]:
    pairs: dict[str, list[tuple[int, int]]] = {}
    for vtype, group in variant_groups.groupby("variant_type"):
        collected = []
        for _, members in group.groupby("group_id"):
            idx = sorted({idx_of[s] for s in members["smiles"] if s in idx_of})
            collected += list(combinations(idx, 2))
        pairs[vtype] = collected
    return pairs


def heavy_atom_counts(smiles_list: list[str]) -> np.ndarray:
    counts = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        counts.append(mol.GetNumHeavyAtoms() if mol else 0)
    return np.array(counts)


def bit_flip_counts(bits: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    return (bits[i] != bits[j]).sum(axis=1)


def metric_dependence_table(pair_d: np.ndarray, background: np.ndarray,
                            cosine_ratio: float, euclidean_ratio: float) -> pd.DataFrame:
    """Ratio-to-background under monotone transforms of the SAME ECFP4 distance.

    Monotone transforms leave the representation, its neighbour ordering, and anything
    a model consumes untouched. If they move the ratio across the CheMeleon values,
    the ratio cannot be evidence about a representation's smoothness.
    """
    rows = [
        {"distance": f"ECFP4 Tanimoto, {name}",
         "ratio_to_background": round(float(np.median(f(pair_d)) / np.median(f(background))), 4)}
        for name, f in MONOTONE_TRANSFORMS.items()
    ]
    rows.append({"distance": "CheMeleon cosine", "ratio_to_background": round(cosine_ratio, 4)})
    rows.append({"distance": "CheMeleon Euclidean", "ratio_to_background": round(euclidean_ratio, 4)})
    return pd.DataFrame(rows)


def consistency_robustness_table(processed_dir: Path) -> pd.DataFrame:
    """Re-aggregate 2.13's per-group consistency ratios by median as well as mean.

    A consistency ratio is a ratio of two positive quantities and is heavily
    right-skewed; the mean is dominated by a minority of groups with near-zero
    measured variation. The median and the fraction of groups above 1 describe the
    typical group instead.
    """
    rows = []
    for model in MODELS:
        path = processed_dir / "2.13-seal-molecular-variants" / model / "consistency_metrics.csv"
        if not path.exists():
            logger.warning(f"Missing {path}; skipping {model}")
            continue
        metrics = pd.read_csv(path)
        for vtype in VARIANT_TYPES:
            for endpoint in ENDPOINTS:
                sub = metrics[
                    (metrics["variant_type"] == vtype) & (metrics["endpoint"] == endpoint)
                ]
                ratios = sub["consistency_ratio"].dropna().to_numpy()
                if len(ratios) == 0:
                    continue
                rows.append({
                    "model": model,
                    "variant_type": vtype,
                    "endpoint": endpoint,
                    "n_groups": len(ratios),
                    "ratio_mean": round(float(np.mean(ratios)), 3),
                    "ratio_median": round(float(np.median(ratios)), 3),
                    "pct_groups_above_1": round(100.0 * float(np.mean(ratios > 1.0)), 1),
                    "ratio_p90": round(float(np.percentile(ratios, 90)), 3),
                    "mean_over_median": (
                        round(float(np.mean(ratios) / np.median(ratios)), 2)
                        if np.median(ratios) > 0
                        else np.nan
                    ),
                })
    return pd.DataFrame(rows)


def prediction_cv_table(processed_dir: Path) -> pd.DataFrame:
    """Decoration-group prediction CV relative to random-group CV, mean and median."""
    rows = []
    for model in MODELS:
        path = processed_dir / "2.13-seal-molecular-variants" / model / "consistency_summary.csv"
        if not path.exists():
            continue
        summary = pd.read_csv(path)
        deco = summary[summary["variant_type"] == "scaffold_decoration"].set_index("endpoint")
        rand = summary[summary["variant_type"] == "random"].set_index("endpoint")
        for endpoint in ENDPOINTS:
            if endpoint not in deco.index or endpoint not in rand.index:
                continue
            rows.append({
                "model": model,
                "endpoint": endpoint,
                "decoration_cv_median": round(float(deco.loc[endpoint, "pred_cv_median"]), 4),
                "random_cv_median": round(float(rand.loc[endpoint, "pred_cv_median"]), 4),
                "fold_lower_median": round(
                    float(rand.loc[endpoint, "pred_cv_median"] / deco.loc[endpoint, "pred_cv_median"]), 2
                ),
                "fold_lower_mean": round(
                    float(rand.loc[endpoint, "pred_cv_mean"] / deco.loc[endpoint, "pred_cv_mean"]), 2
                ),
            })
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_sensitivity(pair_df: pd.DataFrame, bg: dict[str, np.ndarray], output_dir: Path, dpi: int) -> None:
    set_style()
    fig, axes = plt.subplots(2, len(VARIANT_TYPES), figsize=(11, 8))

    for col, vtype in enumerate(VARIANT_TYPES):
        sub = pair_df[pair_df["variant_type"] == vtype]

        xmax = 0.3 if vtype == "stereoisomer" else 1.2

        ax = axes[0, col]
        for rep in REPRESENTATIONS:
            vals = sub[f"{rep} | relative"].to_numpy()
            bins = np.linspace(0, xmax, 41)
            weights = np.ones_like(vals) / len(vals)
            ax.hist(vals, bins=bins, weights=weights, histtype="step", linewidth=1.8,
                    color=REP_COLORS[rep], label=rep)
        ax.set_xlabel(f"{VARIANT_LABELS[vtype]}:\ndistance / median random-pair distance")
        ax.set_ylabel("Relative frequency")
        ax.set_xlim(0, xmax)
        if col == 0:
            ax.legend(fontsize=8, frameon=False)

        ax = axes[1, col]
        for rep in REPRESENTATIONS:
            vals = np.sort(sub[f"{rep} | rank_pct"].to_numpy())
            ax.plot(vals, np.arange(1, len(vals) + 1) / len(vals), linewidth=1.8,
                    color=REP_COLORS[rep], label=rep)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlabel(f"{VARIANT_LABELS[vtype]}:\npartner's neighbour rank (percentile of library)")
        ax.set_ylabel("Cumulative fraction of pairs")
        ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(output_dir / "representation_sensitivity.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'representation_sensitivity.png'}")


def plot_size_effect(size_df: pd.DataFrame, pair_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    set_style()
    deco = pair_df[pair_df["variant_type"] == "scaffold_decoration"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    max_delta = 8
    groups = [deco[deco["heavy_atom_diff"] == d] for d in range(max_delta + 1)]
    keep = [(d, g) for d, g in zip(range(max_delta + 1), groups) if len(g) >= 20]

    ax = axes[0]
    ax.boxplot([g["ECFP4 bits flipped"] for _, g in keep], positions=[d for d, _ in keep],
               widths=0.6, showfliers=False,
               medianprops={"color": REP_COLORS["ECFP4 Tanimoto"], "linewidth": 2})
    ax.set_xlabel("Heavy-atom difference within pair")
    ax.set_ylabel("ECFP4 bits flipped (of 2,048)")
    ax.set_xticks([d for d, _ in keep])
    ax.set_xticklabels([str(d) for d, _ in keep])

    ax = axes[1]
    for rep in REPRESENTATIONS:
        med = [np.median(g[f"{rep} | relative"]) for _, g in keep]
        ax.plot([d for d, _ in keep], med, marker="o", linewidth=1.8,
                color=REP_COLORS[rep], label=rep)
    ax.set_xlabel("Heavy-atom difference within pair")
    ax.set_ylabel("Median distance / median random-pair distance")
    ax.set_xticks([d for d, _ in keep])
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "decoration_size_effect.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'decoration_size_effect.png'}")


def plot_consistency_robustness(robust_df: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    """Mean vs median consistency ratio per endpoint, both models, decorations."""
    set_style()
    deco = robust_df[robust_df["variant_type"] == "scaffold_decoration"]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11, 4.6), sharey=True)

    for ax, model in zip(np.atleast_1d(axes), MODELS):
        sub = deco[deco["model"] == model].set_index("endpoint").reindex(ENDPOINTS).dropna(how="all")
        y = np.arange(len(sub))
        ax.barh(y - 0.2, sub["ratio_mean"], height=0.38, color=MODEL_COLORS[model],
                label="Mean across groups")
        ax.barh(y + 0.2, sub["ratio_median"], height=0.38, color=MODEL_COLORS[model],
                alpha=0.45, label="Median across groups")
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(sub.index)
        ax.invert_yaxis()
        ax.set_xlabel(f"{MODEL_LABELS[model]}: consistency ratio\n(scaffold decoration groups)")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("Endpoint")

    fig.tight_layout()
    fig.savefig(output_dir / "consistency_ratio_robustness.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'consistency_ratio_robustness.png'}")


@app.command()
def main(
    dpi: int = typer.Option(DEFAULT_DPI, help="Figure resolution"),
    seed: int = typer.Option(42, help="Random seed for the background pair sample"),
) -> None:
    """Compare ECFP4 and CheMeleon sensitivity to minimal structural perturbations."""
    output_dir = PROCESSED_DATA_DIR / "2.18-seal-representation-sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading canonical dataset and 2.13 variant groups")
    dataset = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    variant_groups = pd.read_csv(
        PROCESSED_DATA_DIR / "2.13-seal-molecular-variants" / "variant_groups.csv"
    )

    smiles_all = sorted(set(dataset["SMILES"]) | set(variant_groups["smiles"]))
    idx_of = {s: i for i, s in enumerate(smiles_all)}
    logger.info(f"{len(smiles_all)} unique molecules")

    pairs = load_pairs(variant_groups, idx_of)
    for vtype in VARIANT_TYPES:
        logger.info(f"{vtype}: {len(pairs[vtype])} pairs")

    has_chirality, n_diff, n_feat = featurizer_sees_chirality()
    logger.info(
        f"Chemprop atom featurizer distinguishes enantiomeric atoms: {has_chirality} "
        f"({n_diff}/{n_feat} features differ)"
    )

    bits = compute_ecfp4_matrix(smiles_all)
    embeddings = compute_chemeleon_embeddings(smiles_all).astype(np.float32)
    heavy = heavy_atom_counts(smiles_all)

    kernels = {
        "ECFP4 Tanimoto": lambda q: tanimoto_rows(bits, q),
        "CheMeleon cosine": lambda q: cosine_rows(embeddings, q),
        "CheMeleon Euclidean": lambda q: euclidean_rows(embeddings, q),
    }

    # Background scale per representation
    background = {}
    for rep, kernel in kernels.items():
        background[rep] = background_distances(kernel, np.random.default_rng(seed), len(smiles_all))
        logger.info(f"{rep}: background median {np.median(background[rep]):.4f}")

    # Per-pair distances and scale-free neighbour ranks
    records = []
    for vtype in VARIANT_TYPES:
        plist = pairs[vtype]
        left = np.array([p[0] for p in plist])
        right = np.array([p[1] for p in plist])
        frame = pd.DataFrame({
            "variant_type": vtype,
            "smiles_a": [smiles_all[i] for i in left],
            "smiles_b": [smiles_all[j] for j in right],
            "heavy_atom_diff": np.abs(heavy[left] - heavy[right]),
            "ECFP4 bits flipped": bit_flip_counts(bits, plist),
        })
        queries = np.unique(np.concatenate([left, right]))
        qpos = {q: k for k, q in enumerate(queries)}
        left_pos = np.array([qpos[i] for i in left])
        for rep, kernel in kernels.items():
            rows = kernel(queries)
            d_ab = rows[left_pos, right]
            # Neighbour rank: how many of the 7,608 molecules sit strictly closer to A
            # than B does. Scale-free, so comparable across representations.
            rows_sorted = np.sort(rows, axis=1)
            n_closer = np.array([
                np.searchsorted(rows_sorted[p], d, side="left") for p, d in zip(left_pos, d_ab)
            ])
            ranks = 100.0 * n_closer / (len(smiles_all) - 1)
            frame[f"{rep} | distance"] = d_ab
            frame[f"{rep} | relative"] = d_ab / np.median(background[rep])
            frame[f"{rep} | rank_pct"] = ranks
        records.append(frame)

    pair_df = pd.concat(records, ignore_index=True)
    pair_df.to_csv(output_dir / "pair_distances.csv", index=False)
    logger.info(f"Saved pair_distances.csv ({len(pair_df)} rows)")

    # Summary table
    summary_rows = []
    for rep in REPRESENTATIONS:
        bgv = background[rep]
        for vtype in VARIANT_TYPES:
            sub = pair_df[pair_df["variant_type"] == vtype]
            d = sub[f"{rep} | distance"].to_numpy()
            summary_rows.append({
                "representation": rep,
                "variant_type": vtype,
                "n_pairs": len(d),
                "median_distance": round(float(np.median(d)), 5),
                "background_median": round(float(np.median(bgv)), 5),
                "ratio_to_background": round(float(np.median(d) / np.median(bgv)), 4),
                "median_neighbour_rank_pct": round(float(np.median(sub[f"{rep} | rank_pct"])), 4),
                "pct_partner_is_nearest_neighbour": round(
                    100.0 * float(np.mean(sub[f"{rep} | rank_pct"] == 0.0)), 2
                ),
                "pct_zero_distance": round(100.0 * float(np.mean(d <= 1e-9)), 2),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "sensitivity_summary.csv", index=False)
    logger.info("Saved sensitivity_summary.csv")

    # Perturbation size vs number of atoms changed
    deco = pair_df[pair_df["variant_type"] == "scaffold_decoration"]
    size_rows = []
    for delta, group in deco.groupby("heavy_atom_diff"):
        if len(group) < 20:
            continue
        row = {"heavy_atom_diff": int(delta), "n_pairs": len(group),
               "median_bits_flipped": float(np.median(group["ECFP4 bits flipped"]))}
        for rep in REPRESENTATIONS:
            row[f"{rep} | median_relative"] = round(float(np.median(group[f"{rep} | relative"])), 4)
        size_rows.append(row)
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv(output_dir / "decoration_size_effect.csv", index=False)
    logger.info("Saved decoration_size_effect.csv")

    # Is "smoother under cosine" a fact about the representation, or about the metric?
    deco_summary = summary_df[summary_df["variant_type"] == "scaffold_decoration"].set_index(
        "representation"
    )["ratio_to_background"]
    metric_df = metric_dependence_table(
        deco[f"{REPRESENTATIONS[0]} | distance"].to_numpy(),
        background[REPRESENTATIONS[0]],
        float(deco_summary["CheMeleon cosine"]),
        float(deco_summary["CheMeleon Euclidean"]),
    )
    metric_df.to_csv(output_dir / "metric_dependence.csv", index=False)
    logger.info("Saved metric_dependence.csv")

    # Do 2.13's amplification ratios survive a median aggregation?
    robust_df = consistency_robustness_table(PROCESSED_DATA_DIR)
    cv_df = prediction_cv_table(PROCESSED_DATA_DIR)
    robust_df.to_csv(output_dir / "consistency_ratio_robustness.csv", index=False)
    cv_df.to_csv(output_dir / "prediction_cv_comparison.csv", index=False)
    logger.info("Saved consistency_ratio_robustness.csv and prediction_cv_comparison.csv")

    plot_sensitivity(pair_df, background, output_dir, dpi)
    plot_size_effect(size_df, pair_df, output_dir, dpi)
    if not robust_df.empty:
        plot_consistency_robustness(robust_df, output_dir, dpi)

    print("\nRepresentation sensitivity to minimal structural edits:")
    print(summary_df.to_string(index=False))
    print("\nScaffold decoration: perturbation size vs heavy-atom difference:")
    print(size_df.to_string(index=False))

    print("\nRatio-to-background under monotone transforms of the same ECFP4 distance:")
    print(metric_df.to_string(index=False))

    if not robust_df.empty:
        deco_robust = robust_df[robust_df["variant_type"] == "scaffold_decoration"]
        print("\nConsistency ratio, mean vs median aggregation (scaffold decorations):")
        print(deco_robust.to_string(index=False))
        for model in MODELS:
            sub = deco_robust[deco_robust["model"] == model]
            if len(sub):
                print(
                    f"  {MODEL_LABELS[model]}: mean > 1 in {(sub['ratio_mean'] > 1).sum()}/{len(sub)} "
                    f"endpoints, median > 1 in {(sub['ratio_median'] > 1).sum()}/{len(sub)}"
                )
        print("\nDecoration vs random prediction CV:")
        print(cv_df.to_string(index=False))
    print(
        f"\nChemprop atom featurizer distinguishes enantiomeric atoms: {has_chirality} "
        f"({n_diff}/{n_feat} features differ)"
    )


if __name__ == "__main__":
    app()
