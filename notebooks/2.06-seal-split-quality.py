#!/usr/bin/env python
"""Split quality diagnostics across all three splitting strategies.

Compares cluster-split, time-split, and target-split on LogD using four
checks: fold sizes, target distributions, test-to-train distances, and
structural overlap. Includes UMAP visualization.

Usage:
    pixi run -e cheminformatics python notebooks/2.06-seal-split-quality.py

Outputs:
    data/processed/2.06-seal-split-quality/fold_sizes_comparison.png
    data/processed/2.06-seal-split-quality/target_distributions_by_strategy.png
    data/processed/2.06-seal-split-quality/distance_distributions_by_strategy.png
    data/processed/2.06-seal-split-quality/structural_overlap.png
    data/processed/2.06-seal-split-quality/umap_by_strategy.png
    data/processed/2.06-seal-split-quality/split_quality_summary.csv
    data/processed/2.06-seal-split-quality/label_drift_over_time.png
    data/processed/2.06-seal-split-quality/label_drift_over_time.csv
    data/processed/2.06-seal-split-quality/extrapolation_regimes.png
    data/processed/2.06-seal-split-quality/extrapolation_regimes.csv
    data/processed/2.06-seal-split-quality/label_drift_and_regimes_combined.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
import umap
from loguru import logger
from scipy import stats
from scipy.spatial.distance import squareform

from polaris_generalization.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR
from polaris_generalization.visualization import DEFAULT_DPI, set_style

app = typer.Typer()

ENDPOINT = "LogD"

# Endpoints with temporal coverage (matches the 2.04 time-split set; RLM is
# absent from the canonical dataset). Used for the per-assay drift checks.
DRIFT_ENDPOINTS = [
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

# Cap on points drawn in the swarm overlay per temporal window: a random
# subsample keeps the swarm legible and fast for the larger assays (the box
# summary underneath still uses every molecule).
SWARM_MAX_PER_WINDOW = 120

# An |rho| this large is treated as an appreciable temporal trend. With several
# thousand molecules per assay, p-values alone flag trivial effects, so drift is
# reported as significant only when it is both reliable (p<0.05) and appreciable.
DRIFT_RHO_THRESHOLD = 0.1


def log_transform(values: np.ndarray, endpoint: str) -> np.ndarray:
    """Modeling log transform: log10(clip(x, 1e-10) + 1) for all but LogD."""
    if endpoint == "LogD":
        return values
    return np.log10(np.clip(values, 1e-10, None) + 1.0)


def fold_ids_for(folds_df: pd.DataFrame, endpoint: str, names: np.ndarray) -> np.ndarray:
    """Map a folds parquet to a fold-id array aligned to `names` (repeat 0)."""
    sub = folds_df[folds_df["endpoint"] == endpoint]
    if "repeat" in sub.columns:
        sub = sub[sub["repeat"] == 0]
    fold_map = dict(zip(sub["Molecule Name"], sub["fold"]))
    return np.array([fold_map.get(n, -99) for n in names])


def get_train_test_masks(fold_ids: np.ndarray, fold_id: int, strategy: str) -> tuple[np.ndarray, np.ndarray]:
    """Get train and test boolean masks for a given fold."""
    test_mask = fold_ids == fold_id
    if strategy == "cluster":
        train_mask = (fold_ids != fold_id) & (fold_ids >= 0)
    else:
        # Expanding window: train = all folds before test
        train_mask = np.zeros(len(fold_ids), dtype=bool)
        for k in range(-1, fold_id):
            train_mask |= (fold_ids == k)
    return train_mask, test_mask


def plot_distance_distributions(strategy_nn1: dict, colors: dict, output_dir: Path, dpi: int) -> None:
    """Plot pooled test-to-train 1-NN distance distributions by strategy."""
    strategy_labels = {"cluster": "Cluster-based", "time": "Time-based", "target": "Target-value"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 51)

    # Plot from narrowest to widest for visual clarity
    for strat_name in ["target", "time", "cluster"]:
        nn1 = strategy_nn1[strat_name]
        weights = np.ones_like(nn1) / len(nn1)
        ax.hist(nn1, bins=bins, weights=weights, histtype="stepfilled", alpha=0.35,
                color=colors[strat_name], edgecolor=colors[strat_name], linewidth=1.2,
                label=strategy_labels[strat_name])

    # Vertical median lines
    for strat_name in ["target", "time", "cluster"]:
        med = np.median(strategy_nn1[strat_name])
        ax.axvline(med, color=colors[strat_name], linestyle="--", linewidth=1.5, alpha=0.9)

    ax.set_xlabel("Test-to-train 1-NN Jaccard distance")
    ax.set_ylabel("Relative frequency")
    ax.set_xlim(0, 1)
    ax.legend(frameon=True)

    fig.tight_layout()
    fig.savefig(output_dir / "distance_distributions_by_strategy.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved distance_distributions_by_strategy.png")
    plt.close("all")


def draw_drift_panel(ax, ep: str, row: pd.Series, windows: list, rng: np.random.Generator) -> None:
    """Draw one label-drift panel: box over every molecule plus a swarm of a
    capped random subsample. Coral if the assay drifts appreciably, else gray."""
    face = "coral" if row["drift_significant"] else "lightgray"
    order = list(range(1, len(windows) + 1))

    box_frames, swarm_frames = [], []
    for i, w in enumerate(windows):
        box_frames.append(pd.DataFrame({"window": i + 1, "value": w}))
        sample = w if len(w) <= SWARM_MAX_PER_WINDOW else rng.choice(
            w, SWARM_MAX_PER_WINDOW, replace=False
        )
        swarm_frames.append(pd.DataFrame({"window": i + 1, "value": sample}))
    box_df = pd.concat(box_frames, ignore_index=True)
    swarm_df = pd.concat(swarm_frames, ignore_index=True)

    sns.boxplot(
        data=box_df, x="window", y="value", order=order, ax=ax,
        color=face, showfliers=False, width=0.6, boxprops={"alpha": 0.6},
    )
    sns.swarmplot(
        data=swarm_df, x="window", y="value", order=order, ax=ax,
        color="0.25", size=1.3, alpha=0.65,
    )
    ax.set_xlabel("Temporal window (early → late)")
    ax.set_ylabel(ep if ep == "LogD" else f"log10({ep} + 1)", fontsize=9)
    ax.set_title(
        f"{ep}  (ρ={row['spearman_rho']:+.2f}{'*' if row['drift_significant'] else ''})",
        fontsize=10,
    )


def draw_regime_panel(ax, points_df: pd.DataFrame, color_fn, size: int, title: str) -> None:
    """Draw one extrapolation-regime scatter: x = structural novelty (median
    test-to-train 1-NN), y = value extrapolation (% test beyond training range)."""
    for _, r in points_df.iterrows():
        ax.scatter(r["median_1nn"], r["value_novelty_frac"] * 100,
                   color=color_fn(r["label"]), s=size, edgecolor="white", zorder=3)
        ax.annotate(r["label"], (r["median_1nn"], r["value_novelty_frac"] * 100),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Median test-to-train 1-NN Jaccard distance\n(structural novelty: unseen series)")
    ax.set_ylabel("% test beyond training value range\n(value extrapolation: unseen labels)")
    ax.set_title(title, fontsize=11)


def analyze_label_drift_over_time(
    df: pd.DataFrame, time_folds: pd.DataFrame, output_dir: Path, dpi: int
) -> tuple[pd.DataFrame, dict]:
    """Per-assay label drift across temporal windows (Engkvist review point).

    Temporal windows are the exact time-split folds defined in 2.04 and loaded
    from time_cv_folds.parquet (fold -1 = earliest train-only chunk, folds 0..k
    = successive expanding-window test chunks). For each endpoint, reports the
    monotonic trend (Spearman rho of label vs ordinal index) plus the
    first-vs-last-window distribution shift (KS). Matches the studies Engkvist
    referenced: some assays drift, others do not.
    """
    rows = []
    panel_windows = {}
    for ep in DRIFT_ENDPOINTS:
        mask = df[ep].notna().values
        sub = df[mask]
        names = sub["Molecule Name"].values
        midx = sub["mol_index"].values
        vals = log_transform(sub[ep].values.astype(float), ep)
        fold_ids = fold_ids_for(time_folds, ep, names)

        # Order windows earliest → latest using the saved time-split folds.
        fold_vals = sorted(f for f in set(fold_ids.tolist()) if f >= -1)
        windows = [vals[fold_ids == w] for w in fold_vals]
        rho, p = stats.spearmanr(midx, vals)
        ks_stat, ks_p = stats.ks_2samp(windows[0], windows[-1])
        significant = bool((p < 0.05) and (abs(rho) >= DRIFT_RHO_THRESHOLD))

        rows.append({
            "endpoint": ep,
            "n": int(mask.sum()),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "median_first_window": float(np.median(windows[0])),
            "median_last_window": float(np.median(windows[-1])),
            "delta_median": float(np.median(windows[-1]) - np.median(windows[0])),
            "ks_first_last": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "drift_significant": significant,
        })
        panel_windows[ep] = windows

    drift_df = pd.DataFrame(rows)

    # Figure: per-endpoint value distributions across temporal windows, shown as
    # a box (every molecule) with a swarm of a capped random subsample on top.
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, ep in zip(axes.ravel(), DRIFT_ENDPOINTS):
        row = drift_df[drift_df["endpoint"] == ep].iloc[0]
        draw_drift_panel(ax, ep, row, panel_windows[ep], rng)

    fig.tight_layout()
    fig.savefig(output_dir / "label_drift_over_time.png", dpi=dpi, bbox_inches="tight")
    drift_df.to_csv(output_dir / "label_drift_over_time.csv", index=False)
    logger.info("Saved label_drift_over_time.png / .csv")
    for _, r in drift_df.iterrows():
        logger.info(
            f"  {r['endpoint']}: rho={r['spearman_rho']:+.3f} (p={r['spearman_p']:.1e}), "
            f"d_median={r['delta_median']:+.3f}, drift={'YES' if r['drift_significant'] else 'no'}"
        )
    plt.close("all")
    return drift_df, panel_windows


def _structural_and_value_novelty(
    fold_ids: np.ndarray, D_sub: np.ndarray, values: np.ndarray, strategy: str
) -> tuple[float, float]:
    """Pool folds and return (median test-to-train 1-NN, fraction of test beyond
    the training value range). The first axis is structural novelty (unseen
    series); the second is value extrapolation (unseen labels)."""
    n_folds = int(max(f for f in fold_ids if f >= 0)) + 1
    nn1_pool, n_outside, n_test = [], 0, 0
    for k in range(n_folds):
        train_mask, test_mask = get_train_test_masks(fold_ids, k, strategy)
        test_idx, train_idx = np.where(test_mask)[0], np.where(train_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        nn1_pool.append(D_sub[np.ix_(test_idx, train_idx)].min(axis=1))
        lo, hi = values[train_mask].min(), values[train_mask].max()
        n_outside += int(np.sum((values[test_mask] < lo) | (values[test_mask] > hi)))
        n_test += int(test_mask.sum())
    median_1nn = float(np.median(np.concatenate(nn1_pool)))
    value_frac = n_outside / n_test if n_test else 0.0
    return median_1nn, value_frac


def analyze_extrapolation_regimes(
    df: pd.DataFrame,
    dist_square: np.ndarray,
    time_folds: pd.DataFrame,
    strategies: dict,
    logd_values: np.ndarray,
    logd_Dsub: np.ndarray,
    colors: dict,
    output_dir: Path,
    dpi: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Separate value extrapolation from structural extrapolation (Rodriguez point).

    Panel A: each assay under the time-split, placed on (structural novelty,
    value extrapolation). Panel B: the three strategies for LogD, anchoring the
    two axes (cluster = unseen structure / values seen before; target = unseen
    values; time = mixed).
    """
    # Panel A: per-endpoint time-split.
    a_rows = []
    for ep in DRIFT_ENDPOINTS:
        mask = df[ep].notna().values
        names = df.loc[mask, "Molecule Name"].values
        vals = df.loc[mask, ep].values.astype(float)
        idx = np.where(mask)[0]
        D_sub = dist_square[np.ix_(idx, idx)]
        fold_ids = fold_ids_for(time_folds, ep, names)
        median_1nn, value_frac = _structural_and_value_novelty(fold_ids, D_sub, vals, "time")
        a_rows.append({
            "group": "time-split (per assay)",
            "label": ep,
            "median_1nn": median_1nn,
            "value_novelty_frac": value_frac,
        })
    a_df = pd.DataFrame(a_rows)

    # Panel B: three strategies for LogD.
    b_rows = []
    for strat_name, strat in strategies.items():
        median_1nn, value_frac = _structural_and_value_novelty(
            strat["fold_ids"], logd_Dsub, logd_values, strat_name
        )
        b_rows.append({
            "group": "strategy (LogD)",
            "label": strat_name,
            "median_1nn": median_1nn,
            "value_novelty_frac": value_frac,
        })
    b_df = pd.DataFrame(b_rows)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    draw_regime_panel(axes[0], a_df, lambda _: "coral", 60, "Time-split per assay")
    draw_regime_panel(axes[1], b_df, lambda label: colors[label], 120, "Strategies (LogD)")

    fig.tight_layout()
    fig.savefig(output_dir / "extrapolation_regimes.png", dpi=dpi, bbox_inches="tight")
    regimes_df = pd.concat([a_df, b_df], ignore_index=True)
    regimes_df.to_csv(output_dir / "extrapolation_regimes.csv", index=False)
    logger.info("Saved extrapolation_regimes.png / .csv")
    for _, r in regimes_df.iterrows():
        logger.info(
            f"  [{r['group']}] {r['label']}: median_1nn={r['median_1nn']:.3f}, "
            f"value_novelty={r['value_novelty_frac']:.1%}"
        )
    plt.close("all")
    return regimes_df, a_df, b_df


def plot_combined_si_figure(
    drift_df: pd.DataFrame,
    panel_windows: dict,
    a_df: pd.DataFrame,
    b_df: pd.DataFrame,
    colors: dict,
    output_dir: Path,
    dpi: int,
) -> None:
    """Stack both analyses into one SI figure: per-assay label drift (top, a)
    over the value-vs-structural extrapolation map (bottom, b)."""
    rng = np.random.default_rng(42)
    fig = plt.figure(figsize=(15, 18), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[3, 1])

    top_axes = subfigs[0].subplots(3, 3)
    for ax, ep in zip(top_axes.ravel(), DRIFT_ENDPOINTS):
        row = drift_df[drift_df["endpoint"] == ep].iloc[0]
        draw_drift_panel(ax, ep, row, panel_windows[ep], rng)
    subfigs[0].suptitle("(a) Label drift over time", x=0.01, ha="left",
                        fontsize=14, fontweight="bold")

    bot_axes = subfigs[1].subplots(1, 2)
    draw_regime_panel(bot_axes[0], a_df, lambda _: "coral", 60, "Time-split per assay")
    draw_regime_panel(bot_axes[1], b_df, lambda label: colors[label], 120, "Strategies (LogD)")
    subfigs[1].suptitle("(b) Value vs structural extrapolation", x=0.01, ha="left",
                        fontsize=14, fontweight="bold")

    fig.savefig(output_dir / "label_drift_and_regimes_combined.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved label_drift_and_regimes_combined.png")
    plt.close("all")


@app.command()
def main(
    output_dir: Path = typer.Option(
        PROCESSED_DATA_DIR / "2.06-seal-split-quality", help="Output directory"
    ),
    dpi: int = typer.Option(DEFAULT_DPI, help="DPI for saved figures"),
    overlap_threshold: float = typer.Option(0.1, help="1-NN distance threshold for structural overlap"),
    figures_only: bool = typer.Option(False, help="Regenerate figures from existing data without rerunning analysis"),
) -> None:
    set_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {"cluster": "steelblue", "time": "coral", "target": "forestgreen"}

    # ── Figures-only mode ─────────────────────────────────────────────
    if figures_only:
        logger.info("Figures-only mode: loading existing data")
        nn1_data = np.load(output_dir / "strategy_nn1.npz")
        strategy_nn1 = {k: nn1_data[k] for k in nn1_data}
        plot_distance_distributions(strategy_nn1, colors, output_dir, dpi)
        logger.info("Figures regenerated (no data recomputed)")
        return

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info("Loading data")
    df = pd.read_parquet(INTERIM_DATA_DIR / "expansion_tx.parquet")
    npz = np.load(INTERIM_DATA_DIR / "tanimoto_distance_matrix.npz", allow_pickle=True)
    dist_square = squareform(npz["condensed"])

    cluster_folds = pd.read_parquet(INTERIM_DATA_DIR / "cluster_cv_folds.parquet")
    time_folds = pd.read_parquet(INTERIM_DATA_DIR / "time_cv_folds.parquet")
    target_folds = pd.read_parquet(INTERIM_DATA_DIR / "target_cv_folds.parquet")

    # Filter to LogD
    mask = df[ENDPOINT].notna().values
    ep_df = df[mask].copy()
    ep_names = ep_df["Molecule Name"].values
    ep_values = ep_df[ENDPOINT].values
    idx_in_full = np.where(mask)[0]
    D_sub = dist_square[np.ix_(idx_in_full, idx_in_full)]
    n_mol = len(ep_df)
    logger.info(f"{ENDPOINT}: {n_mol} molecules")

    # Build fold ID arrays per strategy
    strategies = {}
    for name, folds_df, repeat_filter in [
        ("cluster", cluster_folds, True),
        ("time", time_folds, True),
        ("target", target_folds, True),
    ]:
        sub = folds_df[folds_df["endpoint"] == ENDPOINT]
        if repeat_filter and "repeat" in sub.columns:
            sub = sub[sub["repeat"] == 0]
        fold_map = dict(zip(sub["Molecule Name"], sub["fold"]))
        fold_ids = np.array([fold_map.get(n, -99) for n in ep_names])
        n_folds = max(f for f in fold_ids if f >= 0) + 1
        strategies[name] = {"fold_ids": fold_ids, "n_folds": n_folds}
        logger.info(f"  {name}: {n_folds} folds, assigned={np.sum(fold_ids >= -1)}")

    # ── 2. Check 1: Fold sizes ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x_offset = 0
    tick_positions = []
    tick_labels = []

    for strat_name, strat in strategies.items():
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]
        for fold_id in range(n_folds):
            _, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)
            size = test_mask.sum()
            ax.bar(x_offset, size, color=colors[strat_name], edgecolor="white", width=0.8)
            tick_positions.append(x_offset)
            tick_labels.append(f"{strat_name[0].upper()}{fold_id}")
            x_offset += 1
        x_offset += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
    ax.set_ylabel("Test fold size")
    ax.set_title(f"Fold sizes across strategies ({ENDPOINT})")
    ax.axhline(y=n_mol * 0.1, color="red", linestyle="--", alpha=0.3, label="10% threshold")
    ax.axhline(y=n_mol * 0.4, color="red", linestyle="--", alpha=0.3, label="40% threshold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in colors.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "fold_sizes_comparison.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved fold_sizes_comparison.png")
    plt.close("all")

    # ── 3. Check 2: Target distribution per fold ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (strat_name, strat) in zip(axes, strategies.items()):
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]

        box_data = []
        box_labels = []
        for fold_id in range(n_folds):
            train_mask, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)
            box_data.append(ep_values[test_mask])
            box_labels.append(f"F{fold_id} test")

        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(colors[strat_name])
            patch.set_alpha(0.6)
        ax.set_ylabel(ENDPOINT)
        ax.set_title(f"{strat_name.capitalize()}-split")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

    fig.suptitle(f"Target distributions per fold ({ENDPOINT})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "target_distributions_by_strategy.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved target_distributions_by_strategy.png")
    plt.close("all")

    # ── 4. Check 3: Test-to-train 1-NN distances ─────────────────────
    summary_rows = []
    strategy_nn1 = {}

    for strat_name, strat in strategies.items():
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]
        fold_nn1s = []

        for fold_id in range(n_folds):
            train_mask, test_mask = get_train_test_masks(fold_ids, fold_id, strat_name)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(train_mask)[0]

            if len(test_idx) == 0 or len(train_idx) == 0:
                continue

            nn1 = D_sub[np.ix_(test_idx, train_idx)].min(axis=1)
            med = np.median(nn1)
            overlap_frac = np.mean(nn1 < overlap_threshold)

            # KS test on target values
            ks_stat, ks_p = stats.ks_2samp(ep_values[train_mask], ep_values[test_mask])

            summary_rows.append({
                "strategy": strat_name,
                "fold": fold_id,
                "test_size": int(test_mask.sum()),
                "train_size": int(train_mask.sum()),
                "median_1nn": float(med),
                "overlap_frac": float(overlap_frac),
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_p),
            })

            fold_nn1s.append(nn1)

        strategy_nn1[strat_name] = np.concatenate(fold_nn1s)

    # Save pooled distances for figures-only mode
    np.savez(output_dir / "strategy_nn1.npz",
             cluster=strategy_nn1["cluster"],
             time=strategy_nn1["time"],
             target=strategy_nn1["target"])

    plot_distance_distributions(strategy_nn1, colors, output_dir, dpi)

    # ── 5. Check 4: Structural overlap ────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x_offset = 0
    tick_positions = []
    tick_labels_overlap = []

    for strat_name in strategies:
        strat_rows = summary_df[summary_df["strategy"] == strat_name]
        for _, row in strat_rows.iterrows():
            ax.bar(x_offset, row["overlap_frac"] * 100, color=colors[strat_name], edgecolor="white", width=0.8)
            tick_positions.append(x_offset)
            tick_labels_overlap.append(f"{strat_name[0].upper()}{int(row['fold'])}")
            x_offset += 1
        x_offset += 0.5

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_overlap, fontsize=8, rotation=45)
    ax.set_ylabel(f"% test molecules with 1-NN < {overlap_threshold}")
    ax.set_title(f"Structural overlap: near-duplicates in train ({ENDPOINT})")

    legend_elements = [Patch(facecolor=c, label=n) for n, c in colors.items()]
    ax.legend(handles=legend_elements)

    fig.tight_layout()
    fig.savefig(output_dir / "structural_overlap.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved structural_overlap.png")
    plt.close("all")

    # Log summary
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['strategy']} fold {row['fold']}: "
            f"test={row['test_size']}, med_1nn={row['median_1nn']:.3f}, "
            f"overlap={row['overlap_frac']:.1%}, KS={row['ks_statistic']:.3f}"
        )

    # ── 6. UMAP embedding ────────────────────────────────────────────
    logger.info("Computing UMAP embedding")
    reducer = umap.UMAP(metric="precomputed", n_neighbors=30, min_dist=0.3, random_state=42)
    embedding = reducer.fit_transform(D_sub)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (strat_name, strat) in zip(axes, strategies.items()):
        fold_ids = strat["fold_ids"]
        n_folds = strat["n_folds"]
        fold_colors_map = plt.cm.Set2(np.linspace(0, 1, max(n_folds, 5)))

        # Plot train-only as gray first
        train_only = fold_ids == -1
        if train_only.any():
            ax.scatter(embedding[train_only, 0], embedding[train_only, 1],
                       c="lightgray", s=3, alpha=0.3, label="train-only", rasterized=True)

        for fold_id in range(n_folds):
            fold_mask = fold_ids == fold_id
            if fold_mask.any():
                ax.scatter(embedding[fold_mask, 0], embedding[fold_mask, 1],
                           c=[fold_colors_map[fold_id]], s=5, alpha=0.5,
                           label=f"Fold {fold_id}", rasterized=True)

        ax.set_title(f"{strat_name.capitalize()}-split")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.legend(fontsize=7, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"UMAP of fold assignments ({ENDPOINT})", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "umap_by_strategy.png", dpi=dpi, bbox_inches="tight")
    logger.info("Saved umap_by_strategy.png")
    plt.close("all")

    # ── 7. Save summary ──────────────────────────────────────────────
    summary_df.to_csv(output_dir / "split_quality_summary.csv", index=False)
    logger.info(f"Saved split_quality_summary.csv ({len(summary_df)} rows)")

    # ── 8. Label drift over time, per assay (Engkvist review point) ───
    logger.info("Analyzing per-assay label drift over time")
    drift_df, panel_windows = analyze_label_drift_over_time(df, time_folds, output_dir, dpi)

    # ── 9. Value vs structural extrapolation regimes (Rodriguez point)─
    logger.info("Mapping value vs structural extrapolation regimes")
    _, a_df, b_df = analyze_extrapolation_regimes(
        df, dist_square, time_folds, strategies, ep_values, D_sub, colors, output_dir, dpi
    )

    # ── 10. Combined SI figure (drift on top, regimes on bottom) ──────
    logger.info("Building combined SI figure")
    plot_combined_si_figure(drift_df, panel_windows, a_df, b_df, colors, output_dir, dpi)

    logger.info(f"All outputs saved to {output_dir}")


if __name__ == "__main__":
    app()
