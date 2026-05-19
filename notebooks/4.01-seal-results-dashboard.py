#!/usr/bin/env python
"""Interactive manuscript companion for the Polaris generalization framework.

Follows the narrative structure of the manuscript, presenting each section's
key figures, tables, and findings as an interactive dashboard.

Usage:
    pixi install
    pixi run -e cheminformatics streamlit run notebooks/4.01-seal-results-dashboard.py

Audit without Streamlit:
    pixi run python notebooks/4.01-seal-results-dashboard.py --check
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from polaris_generalization.config import PROCESSED_DATA_DIR

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

app = typer.Typer(add_completion=False)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
TABLE_SUFFIXES = {".csv", ".tsv", ".parquet"}
TEXT_SUFFIXES = {".md", ".txt"}
JSON_SUFFIXES = {".json"}
ARRAY_SUFFIXES = {".npy", ".npz"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | TABLE_SUFFIXES | TEXT_SUFFIXES | JSON_SUFFIXES | ARRAY_SUFFIXES
SYSTEM_FILENAMES = {".DS_Store"}
MODEL_NAMES = {"xgboost", "chemeleon", "combined"}

SECTIONS = [
    "Framework Overview",
    "Dataset Characterization",
    "Chemical Space & Splitting",
    "FM1: Performance over Distance",
    "FM1b: IID vs OOD Series",
    "FM2: Scaffold vs Random",
    "FM2b: Split Variance",
    "FM3: Activity Cliffs",
    "FM4: Molecular Variants",
    "FM4b: Resonance Forms",
    "Baseline Performance",
    "Explorer",
]

MODEL_SECTIONS = {
    "FM1: Performance over Distance",
    "FM1b: IID vs OOD Series",
    "FM2: Scaffold vs Random",
    "FM2b: Split Variance",
    "FM3: Activity Cliffs",
    "FM4: Molecular Variants",
    "FM4b: Resonance Forms",
    "Baseline Performance",
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def require_streamlit() -> Any:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run `pixi install`, then launch with "
            "`pixi run -e cheminformatics streamlit run notebooks/4.01-seal-results-dashboard.py`."
        )
    return st


def show_figure(rel_path: str, caption: str = "", width: str = "stretch") -> None:
    s = require_streamlit()
    path = PROCESSED_DATA_DIR / rel_path
    if path.exists():
        s.image(str(path), caption=caption or rel_path, width=width)
    else:
        s.warning(f"Figure not found: {rel_path}")


def show_table(rel_path: str, caption: str = "") -> None:
    s = require_streamlit()
    path = PROCESSED_DATA_DIR / rel_path
    if not path.exists():
        s.warning(f"Table not found: {rel_path}")
        return
    try:
        df = load_dataframe(str(path))
        if caption:
            s.caption(caption)
        s.dataframe(df, width="stretch", hide_index=True)
    except Exception as exc:
        s.error(f"Could not load {rel_path}: {exc}")


def model_key(model_label: str) -> str:
    return {"Combined": "combined", "XGBoost": "xgboost", "CheMeleon": "chemeleon"}[model_label]


def section_header(title: str, narrative: str) -> None:
    s = require_streamlit()
    s.header(title)
    s.info(narrative)


# ---------------------------------------------------------------------------
# Data loaders (unchanged)
# ---------------------------------------------------------------------------

def load_dataframe(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table suffix: {suffix}")


def load_text(path: str, max_chars: int = 200_000) -> str:
    text = Path(path).read_text(errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated in dashboard preview]"
    return text


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def render_framework_overview() -> None:
    s = require_streamlit()
    section_header(
        "Generalization Evaluation Framework",
        "This dashboard accompanies the manuscript on model generalization for molecular "
        "property prediction. The framework evaluates two architectures (XGBoost and CheMeleon) "
        "on a real-world ADMET dataset of 7,608 molecules across 9 endpoints, identifying four "
        "failure modes: extrapolation, interpolation, representation, and evaluation failure.",
    )

    cols = s.columns(5)
    cols[0].metric("Molecules", "7,608")
    cols[1].metric("Endpoints", "9")
    cols[2].metric("Failure Modes", "4")
    cols[3].metric("Architectures", "2")
    cols[4].metric("Butina Clusters", "135")

    show_figure("3.02-seal-framework-figure/framework_overview.png",
                "Figure 1. Generalization evaluation framework overview.")

    s.markdown("""
**Splitting strategies map to deployment scenarios:**

| Discovery phase | Deployment scenario | Recommended split | Failure mode |
|---|---|---|---|
| Hit identification | Novel scaffolds | Cluster-based | Extrapolation |
| Lead optimization | Value-range extrapolation | Target-value | Value extrapolation |
| Prospective deployment | Temporal drift | Time-based | Temporal drift |
| SAR fine-tuning | Local structural changes | Activity cliff + variant checks | Interpolation / representation |
""")

    s.markdown("""
**Practical recommendations (Box 1):**
1. Start from the deployment scenario
2. Characterize the dataset before splitting
3. Use distance-aware splits with multiple repeats
4. Verify the split creates the intended shift
5. Probe specific failure modes
6. Treat distance as an AD diagnostic, not a complete uncertainty model
7. Consider complementary representations matched to the deployment scenario
""")


def render_dataset_characterization() -> None:
    s = require_streamlit()
    section_header(
        "Dataset Characterization",
        "Assay cascades create coupled data-size and diversity gaps. Coverage ranges from "
        "96% (LogD, KSOL) to 6% (MGMB). 93.5% pass Lipinski; median MW 383 Da. "
        "Median similarity to ChEMBL is 0.44 (all) / 0.39 (ADME subset). "
        "Test-to-train 1-NN median distance is 0.37, nearly double the within-train 0.20.",
    )

    s.subheader("Endpoint Summary (Table 2)")
    show_table("0.01-seal-dataset-exploration/summary_statistics.csv",
               "Coverage, KS statistic, and median 1-NN distance per endpoint.")

    col1, col2 = s.columns(2)
    with col1:
        show_figure("0.01-seal-dataset-exploration/endpoint_distributions.png",
                     "Endpoint value distributions.")
    with col2:
        show_figure("0.01-seal-dataset-exploration/endpoint_missingness.png",
                     "Endpoint missingness patterns.")

    s.subheader("Physicochemical Properties")
    col1, col2 = s.columns(2)
    with col1:
        show_figure("0.01-seal-dataset-exploration/physicochemical_properties.png",
                     "Physicochemical property distributions.")
    with col2:
        show_table("0.01-seal-dataset-exploration/physicochemical_summary.csv")

    s.subheader("Structural Novelty vs ChEMBL")
    show_figure("0.03-araripe-chembl-tanimoto/max_tanimoto_comparison.png",
                "Maximum Tanimoto similarity to ChEMBL and ADME subset (Figure S).")
    show_table("0.03-araripe-chembl-tanimoto/summary_statistics.csv")

    s.subheader("Train vs Test Distance Gap (Figure 2)")
    show_figure("0.02-seal-ecfp-distance-exploration/train_vs_test_distances.png",
                "Figure 2. Test-to-train vs within-train 1-NN distance distributions.")
    with s.expander("Distance statistics"):
        show_table("0.02-seal-ecfp-distance-exploration/distance_stats.csv")

    with s.expander("Molecule coverage and ordinal ordering"):
        col1, col2 = s.columns(2)
        with col1:
            show_figure("0.01-seal-dataset-exploration/molecule_coverage.png")
        with col2:
            show_figure("0.01-seal-dataset-exploration/ordinal_ordering.png")


def render_chemical_space() -> None:
    s = require_streamlit()
    section_header(
        "Chemical Space & Splitting Strategies",
        "Butina clustering (Jaccard 0.7) identifies 135 clusters; three dominant clusters "
        "(2,572 / 1,301 / 885 molecules) contain 62.5%. Three splitting strategies produce "
        "distinct distance distributions: cluster (0.37–0.44), time (0.30–0.39), "
        "target-value (0.24–0.29).",
    )

    s.subheader("Cluster Structure (Figure 3)")
    show_figure("2.01-seal-chemical-space-analysis/cluster_size_distribution.png",
                "Figure 3. Butina cluster size distribution.")
    show_table("2.01-seal-chemical-space-analysis/cluster_stats.csv",
               "Cluster statistics.")

    s.subheader("Splitting Strategy Distances (Figure 4)")
    show_figure("2.06-seal-split-quality/distance_distributions_by_strategy.png",
                "Figure 4. Test-to-train 1-NN distances by splitting strategy.")

    col1, col2 = s.columns(2)
    with col1:
        show_figure("2.06-seal-split-quality/fold_sizes_comparison.png", "Fold sizes.")
    with col2:
        show_figure("2.06-seal-split-quality/structural_overlap.png", "Structural overlap.")

    show_table("2.06-seal-split-quality/split_quality_summary.csv",
               "Split quality diagnostics.")

    with s.expander("UMAP visualization"):
        show_figure("2.06-seal-split-quality/umap_by_strategy.png",
                     "UMAP embeddings colored by fold assignment.")

    with s.expander("Target distributions by strategy"):
        show_figure("2.06-seal-split-quality/target_distributions_by_strategy.png")

    with s.expander("Top cluster examples"):
        show_figure("2.01-seal-chemical-space-analysis/top_clusters_grid.png",
                     "Representative structures from the five largest clusters.")


def render_performance_distance(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 1: Performance Degrades with Distance",
        "Both models show systematic RMSE degradation with structural distance. "
        "XGBoost MA-RAE: 0.67 (cluster), 0.77 (time), 6.12 (target). "
        "CheMeleon: 0.63, 0.73, 5.26. The strategy ranking is preserved across architectures.",
    )

    s.subheader("Performance-over-Distance Curves (Figure 5)")
    if mk == "combined":
        show_figure("2.07-seal-performance-distance/combined/performance_over_distance_combined.png",
                     "Figure 5. RMSE vs 1-NN distance, XGBoost (top) and CheMeleon (bottom).")
    else:
        show_figure(f"2.07-seal-performance-distance/{mk}/performance_over_distance.png",
                     f"Performance over distance ({model}).")

    s.subheader("Per-Endpoint RAE by Strategy (Table 3)")
    if mk == "combined":
        show_table("2.07-seal-performance-distance/combined/metrics_comparison.csv",
                   "Table 3. RAE under three splitting strategies for both models.")
    else:
        show_table(f"2.07-seal-performance-distance/{mk}/per_fold_performance.csv",
                   f"Per-fold performance ({model}).")

    with s.expander("Degradation ratios"):
        if mk == "combined":
            show_figure("2.07-seal-performance-distance/combined/degradation_ratio_comparison.png",
                         "Farthest/closest bin RMSE ratio by endpoint.")
        show_figure("2.07-seal-performance-distance/combined/performance_over_target_distance_combined.png",
                     "Performance over target-value distance.")


def render_iid_ood(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 1b: IID vs OOD Series Degradation",
        "Training on the largest Butina cluster and testing on the second-largest: "
        "R² goes negative on 5/8 endpoints (XGBoost), 6/8 (CheMeleon). "
        "KSOL median-SE increases 6.6× (XGB) to 47× (CheMeleon). "
        "Within-series performance is fundamentally unreliable for cross-series prediction.",
    )

    s.subheader("Squared Error Distributions (Figure 6)")
    if mk == "combined":
        show_figure("2.09-seal-iid-vs-ood-series/combined/squared_error_distributions_combined.png",
                     "Figure 6. IID vs OOD squared error, XGBoost and CheMeleon.")
    else:
        show_figure(f"2.09-seal-iid-vs-ood-series/{mk}/squared_error_distributions.png",
                     f"Squared error distributions ({model}).")

    s.subheader("IID vs OOD Metrics (Table 4)")
    if mk == "combined":
        for m in ["xgboost", "chemeleon"]:
            s.markdown(f"**{m.title()}**")
            show_table(f"2.09-seal-iid-vs-ood-series/{m}/summary_metrics.csv")
    else:
        show_table(f"2.09-seal-iid-vs-ood-series/{mk}/summary_metrics.csv",
                   f"IID vs OOD summary ({model}).")

    with s.expander("Split characterization"):
        show_table("2.09-seal-iid-vs-ood-series/split_summary.csv",
                   "IID/OOD split sizes and distance summary.")
        p = PROCESSED_DATA_DIR / "2.09-seal-iid-vs-ood-series/distance_characterization.png"
        if p.exists():
            show_figure("2.09-seal-iid-vs-ood-series/distance_characterization.png")

    with s.expander("Per-endpoint metric breakdowns"):
        for metric in ["r2", "rae", "mae", "spearman"]:
            p = PROCESSED_DATA_DIR / f"2.09-seal-iid-vs-ood-series/{mk}/{metric}_by_endpoint.png"
            if p.exists():
                show_figure(f"2.09-seal-iid-vs-ood-series/{mk}/{metric}_by_endpoint.png",
                             f"{metric.upper()} by endpoint ({model}).")


def render_scaffold_vs_random(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 2: Scaffold Splits ≈ Random Splits",
        "Bemis-Murcko scaffold splits produce MA-RAE 0.508 (XGB), marginally worse than "
        "random (0.474) but far more optimistic than cluster-based (0.674). "
        "67.5% of 3,337 scaffolds are singletons; 56.4% of molecules have their nearest "
        "neighbor in a different scaffold group.",
    )

    s.subheader("Split Comparison")
    if mk == "combined":
        show_figure("2.11-seal-scaffold-vs-random/combined/scaffold_vs_random_rae_combined.png",
                     "Figure S2. RAE comparison: random vs scaffold vs cluster.")
    else:
        show_figure(f"2.11-seal-scaffold-vs-random/{mk}/metric_comparison.png",
                     f"Metric comparison ({model}).")

    eff_mk = mk if mk != "combined" else "xgboost"
    show_table(f"2.11-seal-scaffold-vs-random/{eff_mk}/aggregated_metrics.csv",
               f"Aggregated metrics ({eff_mk}).")

    s.subheader("Scaffold Boundary Analysis")
    col1, col2 = s.columns(2)
    with col1:
        show_figure("2.16-araripe-scaffold-boundary/boundary_violation_summary.png",
                     "Scaffold boundary violation rates.")
    with col2:
        show_figure("2.16-araripe-scaffold-boundary/cross_scaffold_proximity.png",
                     "Cross-scaffold vs within-scaffold proximity.")

    s.subheader("Activity Concordance (Figure S7)")
    show_figure("2.16-araripe-scaffold-boundary/activity_concordance.png",
                "Mann-Whitney tests: same-scaffold vs different-scaffold activity differences.")
    show_table("2.16-araripe-scaffold-boundary/mann_whitney_tests.csv",
               "Mann-Whitney U test results per endpoint and distance bin.")

    with s.expander("Proximity statistics"):
        show_table("2.16-araripe-scaffold-boundary/proximity_stats.csv")

    with s.expander("Distance distributions"):
        show_figure(f"2.11-seal-scaffold-vs-random/{eff_mk}/distance_distributions.png",
                     "1-NN distance distributions by split type.")


def render_split_variance(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 2b: Random CV is Precise at the Wrong Level",
        "Random CV RAE std 0.003–0.012 vs cluster std 0.007–0.059. "
        "The gap (~0.20 MA-RAE) is ~33× the within-random std and ~7× the "
        "within-cluster std (XGBoost); ~29× and ~7× under CheMeleon. "
        "Mann-Whitney U=25, p=0.0079 for all 9 endpoints. Splitting strategy matters "
        "far more than model architecture.",
    )

    eff_mk = mk if mk != "combined" else "xgboost"

    s.subheader("RAE Distributions (Figure S3)")
    if mk == "combined":
        show_figure("2.12-seal-split-variance/combined/split_variance_rae_combined.png",
                     "Figure S3. RAE distributions for random vs cluster repeats.")
    else:
        show_figure(f"2.12-seal-split-variance/{mk}/rae_distributions.png",
                     f"RAE distributions ({model}).")

    s.subheader("Variance Summary (Table S3)")
    show_table(f"2.12-seal-split-variance/{eff_mk}/variance_summary.csv",
               f"Per-endpoint variance statistics ({eff_mk}).")

    col1, col2 = s.columns(2)
    with col1:
        show_figure(f"2.12-seal-split-variance/{eff_mk}/ma_rae_distribution.png",
                     f"MA-RAE distribution ({eff_mk}).")
    with col2:
        show_figure(f"2.12-seal-split-variance/{eff_mk}/variance_heatmap.png",
                     f"Variance heatmap ({eff_mk}).")

    with s.expander("Single-split danger"):
        show_figure(f"2.12-seal-split-variance/{eff_mk}/single_split_danger.png",
                     "R² can span 0.32–0.58 for MGMB across cluster repeats.")

    with s.expander("Mann-Whitney tests"):
        show_table(f"2.12-seal-split-variance/{eff_mk}/mannwhitney_rae_tests.csv",
                   "Mann-Whitney U test: random vs cluster RAE distributions.")


def render_activity_cliffs(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 3: Activity Cliffs",
        "Activity cliff molecules (Tanimoto > 0.85, Δactivity ≥ 1.0 log unit) "
        "comprise 0–4.6% of compounds per endpoint but show systematically higher "
        "prediction error across all 7 endpoints with sufficient cliff populations. "
        "CheMeleon lifts the non-cliff baseline but does not close the cliff penalty.",
    )

    s.subheader("Cliff vs Non-Cliff RAE (Figure 7)")
    show_figure("2.10-seal-activity-cliff-eval/combined/rae_cliff_comparison.png",
                "Figure 7. Cliff vs non-cliff RAE under cluster-split CV.")

    eff_mk = mk if mk != "combined" else "xgboost"
    s.subheader("Cliff Summary (Table S4)")
    show_table(f"2.10-seal-activity-cliff-eval/{eff_mk}/summary_metrics.csv",
               f"Cliff vs non-cliff performance ({eff_mk}).")

    s.subheader("Threshold Sensitivity (Figure S4)")
    if mk == "combined":
        show_figure("2.10-seal-activity-cliff-eval/combined/cliff_sensitivity_combined.png",
                     "Figure S4. Cliff prevalence and RAE gap across similarity thresholds.")
    else:
        show_figure(f"2.10-seal-activity-cliff-eval/{mk}/cliff_sensitivity_combined.png",
                     f"Cliff sensitivity ({model}).")

    with s.expander("Cliff characterization"):
        p = PROCESSED_DATA_DIR / "2.10-seal-activity-cliff-eval/cliff_characterization.png"
        if p.exists():
            show_figure("2.10-seal-activity-cliff-eval/cliff_characterization.png")
        p2 = PROCESSED_DATA_DIR / "2.10-seal-activity-cliff-eval/cliff_stats.csv"
        if p2.exists():
            show_table("2.10-seal-activity-cliff-eval/cliff_stats.csv")

    with s.expander("Cliff sensitivity data"):
        show_table(f"2.10-seal-activity-cliff-eval/{eff_mk}/cliff_sensitivity.csv",
                   "Cliff statistics at each similarity threshold.")


def render_molecular_variants(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 4: Representation Failures",
        "Stereoisomers (548 groups, 15.1%): median Tanimoto distance 0.067; "
        "9% invisible to fingerprint. Consistency ratio < 1 on 8/9 (XGB), 9/9 (CheMeleon) "
        "— models under-predict stereochemical variation. "
        "Scaffold decorations (1,050 groups, 50.4%): consistency ratio > 1 on 8/9 (XGB), "
        "9/9 (CheMeleon) — models amplify substituent effects.",
    )

    eff_mk = mk if mk != "combined" else "xgboost"

    s.subheader("Fingerprint Distances (Figure 8)")
    show_figure(f"2.13-seal-molecular-variants/{eff_mk}/fingerprint_distances.png",
                "Figure 8. Intra-group Tanimoto distances: stereo (0.07), scaffold (0.35), random (0.85).")

    s.subheader("Prediction Consistency (Table 5)")
    show_table("2.13-seal-molecular-variants/combined/consistency_comparison.csv",
               "Table 5. Prediction CV and consistency ratios by variant type.")

    s.subheader("Prediction CV (Figure S5)")
    show_figure("2.13-seal-molecular-variants/combined/pred_cv_all_variants_combined.png",
                "Figure S5. Per-endpoint prediction CV for stereoisomer, scaffold, and random groups.")

    col1, col2 = s.columns(2)
    with col1:
        s.markdown("**Stereoisomer consistency ratios**")
        show_figure("2.13-seal-molecular-variants/combined/consistency_ratio_stereoisomer_comparison.png")
    with col2:
        s.markdown("**Scaffold decoration consistency ratios**")
        show_figure("2.13-seal-molecular-variants/combined/consistency_ratio_scaffold_decoration_comparison.png")

    with s.expander("Spread scatter (Figure S6)"):
        show_figure("2.13-seal-molecular-variants/combined/spread_scatter_combined.png",
                     "Figure S6. Predicted vs true range within variant groups.")

    with s.expander("Fingerprint similarity data"):
        show_table("2.13-seal-molecular-variants/fingerprint_similarity.csv",
                   "Pairwise Tanimoto distances within variant groups.")

    with s.expander(f"Prediction consistency ({eff_mk})"):
        show_figure(f"2.13-seal-molecular-variants/{eff_mk}/prediction_consistency.png",
                     f"Prediction consistency heatmap ({eff_mk}).")
        show_figure(f"2.13-seal-molecular-variants/{eff_mk}/consistency_heatmap.png",
                     f"Consistency ratio heatmap ({eff_mk}).")


def render_resonance(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Failure Mode 4b: Resonance Form Ambiguity",
        "Resonance forms are chemically identical but fingerprint-distant (median Tanimoto "
        "0.46). RMSE variation reaches 27.4% for LogD (XGBoost), 20.6% (CheMeleon). "
        "Worsening meets or exceeds improvement across all 9 endpoints in both models. "
        "The canonical form is already reasonable — deviating is asymmetrically risky.",
    )

    s.subheader("Resonance Sensitivity Panel (Figure 9)")
    show_figure("2.15-zalte-resonance-variants/combined/resonance_sensitivity_panel_combined.png",
                "Figure 9. (A) Fraction with >1 resonance form; (B) RMSE change; (C) per-molecule error range.")

    s.subheader("Fingerprint Distances")
    col1, col2 = s.columns(2)
    with col1:
        show_figure("2.15-zalte-resonance-variants/fingerprint_distances.png",
                     "Figure S8. Resonance-form Tanimoto distances vs controls.")
    with col2:
        show_table("2.15-zalte-resonance-variants/fingerprint_similarity.csv",
                   "Pairwise Tanimoto distances for resonance forms.")

    s.subheader("Sensitivity Summary (Table S5)")
    show_table("2.15-zalte-resonance-variants/combined/consistency_comparison.csv",
               "RMSE variation by endpoint, XGBoost and CheMeleon.")

    with s.expander("RMSE range distributions"):
        show_figure("2.15-zalte-resonance-variants/resonance_range_distribution.png",
                     "Per-molecule prediction error range across resonance forms.")
        show_figure("2.15-zalte-resonance-variants/rmse_range_dumbbell.png",
                     "RMSE improvement vs worsening dumbbell chart.")

    with s.expander("Model comparison breakdowns"):
        col1, col2 = s.columns(2)
        with col1:
            show_figure("2.15-zalte-resonance-variants/combined/worsen_pct_comparison.png",
                         "Worsening % by endpoint.")
        with col2:
            show_figure("2.15-zalte-resonance-variants/combined/pct_swing_comparison.png",
                         "Total RMSE variation % by endpoint.")

    eff_mk = mk if mk != "combined" else "xgboost"
    report_path = PROCESSED_DATA_DIR / "2.15-zalte-resonance-variants/resonance_sensitivity_report.md"
    if report_path.exists():
        with s.expander("Full sensitivity report"):
            s.markdown(load_text(str(report_path)))


def render_baseline(model: str) -> None:
    s = require_streamlit()
    mk = model_key(model)
    section_header(
        "Baseline Performance",
        "On the competition split (time-based), XGBoost achieves MA-RAE 0.776 while "
        "CheMeleon improves to 0.684. The foundation model lifts baselines on 8/9 endpoints.",
    )

    s.subheader("Combined Baseline Panel")
    show_figure("2.08-seal-baseline-performance/combined/baseline_combined_panel.png",
                "Figure S1. Competition-split predictions, XGBoost vs CheMeleon.")

    s.subheader("Model Comparison (Table S1)")
    show_table("2.08-seal-baseline-performance/combined/metrics_comparison.csv",
               "Per-endpoint metrics on the competition split.")

    col1, col2 = s.columns(2)
    with col1:
        show_figure("2.08-seal-baseline-performance/combined/rae_comparison.png", "RAE comparison.")
    with col2:
        show_figure("2.08-seal-baseline-performance/combined/r2_comparison.png", "R² comparison.")

    with s.expander("Additional metrics"):
        col1, col2 = s.columns(2)
        with col1:
            show_figure("2.08-seal-baseline-performance/combined/mae_comparison.png", "MAE comparison.")
        with col2:
            show_figure("2.08-seal-baseline-performance/combined/spearman_comparison.png", "Spearman comparison.")


# ---------------------------------------------------------------------------
# Explorer (legacy file browser)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResultFile:
    path: Path
    rel_path: str
    analysis: str
    model: str
    kind: str
    suffix: str
    size_bytes: int
    modified: datetime


ANALYSIS_DESCRIPTIONS = {
    "0.01-seal-dataset-exploration": "Dataset coverage, missingness, endpoint distributions.",
    "0.02-seal-ecfp-distance-exploration": "ECFP4/Jaccard distance distributions.",
    "0.03-araripe-chembl-tanimoto": "Similarity to ChEMBL compounds.",
    "2.01-seal-chemical-space-analysis": "Butina cluster structure and size distribution.",
    "2.06-seal-split-quality": "Cross-strategy split diagnostics.",
    "2.07-seal-performance-distance": "Performance-over-distance curves.",
    "2.08-seal-baseline-performance": "Competition-split baseline performance.",
    "2.09-seal-iid-vs-ood-series": "IID vs OOD chemical-series tests.",
    "2.10-seal-activity-cliff-eval": "Activity-cliff prevalence and errors.",
    "2.11-seal-scaffold-vs-random": "Random, scaffold, and cluster split comparison.",
    "2.12-seal-split-variance": "Repeat-to-repeat variance.",
    "2.13-seal-molecular-variants": "Stereoisomer and scaffold-decoration consistency.",
    "2.15-zalte-resonance-variants": "Prediction sensitivity to resonance forms.",
    "2.16-araripe-scaffold-boundary": "Murcko scaffold-boundary leakage.",
    "3.02-seal-framework-figure": "Framework overview figure.",
}


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if path.name in SYSTEM_FILENAMES:
        return "system"
    if suffix in IMAGE_SUFFIXES:
        return "figure"
    if suffix in TABLE_SUFFIXES:
        return "table"
    if suffix in TEXT_SUFFIXES:
        return "text"
    if suffix in JSON_SUFFIXES:
        return "json"
    if suffix in ARRAY_SUFFIXES:
        return "array"
    return "unsupported"


def infer_model(rel_path: Path) -> str:
    for part in rel_path.parts:
        if part in MODEL_NAMES:
            return part
    if "pred_cache" in rel_path.parts:
        return "cache"
    return "model-agnostic"


def iter_result_files(results_dir: Path, include_system: bool = False) -> list[ResultFile]:
    files: list[ResultFile] = []
    for path in sorted(results_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(results_dir)
        kind = classify_file(path)
        if kind == "system" and not include_system:
            continue
        stat = path.stat()
        files.append(
            ResultFile(
                path=path,
                rel_path=rel_path.as_posix(),
                analysis=rel_path.parts[0] if rel_path.parts else "(root)",
                model=infer_model(rel_path),
                kind=kind,
                suffix=path.suffix.lower() or "(none)",
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime),
            )
        )
    return files


def inventory_dataframe(results_dir: Path, include_system: bool = False) -> pd.DataFrame:
    rows = [
        {
            "analysis": f.analysis,
            "model": f.model,
            "kind": f.kind,
            "suffix": f.suffix,
            "size_mb": f.size_bytes / 1024 / 1024,
            "modified": f.modified,
            "relative_path": f.rel_path,
            "absolute_path": str(f.path),
        }
        for f in iter_result_files(results_dir, include_system=include_system)
    ]
    return pd.DataFrame(rows)


def render_explorer(results_dir: Path) -> None:
    s = require_streamlit()
    s.header("File Explorer")
    s.caption("Browse all files under data/processed/.")

    df = inventory_dataframe(results_dir)
    if df.empty:
        s.warning(f"No files found under {results_dir}")
        return

    col1, col2, col3, col4 = s.columns(4)
    with col1:
        analyses = s.multiselect("Analysis", sorted(df["analysis"].unique()))
    with col2:
        models = s.multiselect("Model", sorted(df["model"].unique()))
    with col3:
        kinds = s.multiselect("Kind", sorted(df["kind"].unique()))
    with col4:
        search = s.text_input("Search path")

    filtered = df.copy()
    if analyses:
        filtered = filtered[filtered["analysis"].isin(analyses)]
    if models:
        filtered = filtered[filtered["model"].isin(models)]
    if kinds:
        filtered = filtered[filtered["kind"].isin(kinds)]
    if search:
        filtered = filtered[filtered["relative_path"].str.lower().str.contains(search.lower(), regex=False)]
    filtered = filtered.sort_values(["analysis", "model", "kind", "relative_path"])

    non_system = filtered[filtered["kind"] != "system"]
    cols = s.columns(5)
    cols[0].metric("Analyses", non_system["analysis"].nunique())
    cols[1].metric("Files", f"{len(non_system):,}")
    cols[2].metric("Figures", f"{(non_system['kind'] == 'figure').sum():,}")
    cols[3].metric("Tables", f"{(non_system['kind'] == 'table').sum():,}")
    cols[4].metric("Size", f"{non_system['size_mb'].sum():.1f} MB")

    s.dataframe(filtered[["analysis", "model", "kind", "suffix", "size_mb", "relative_path"]],
                width="stretch", hide_index=True)

    if not filtered.empty:
        selected = s.selectbox("Open file", filtered["relative_path"].tolist())
        row = filtered[filtered["relative_path"] == selected].iloc[0]
        s.markdown(f"**`{row['relative_path']}`** ({row['kind']}, {row['size_mb']:.3f} MB)")

        if row["kind"] == "figure":
            s.image(row["absolute_path"], caption=row["relative_path"], width="stretch")
        elif row["kind"] == "table":
            try:
                data = load_dataframe(row["absolute_path"])
                s.dataframe(data, width="stretch", hide_index=True)
            except Exception as exc:
                s.error(f"Could not load: {exc}")
        elif row["kind"] == "json":
            try:
                s.json(json.loads(Path(row["absolute_path"]).read_text()))
            except Exception as exc:
                s.error(str(exc))
        elif row["kind"] == "text":
            text = load_text(row["absolute_path"])
            if row["suffix"] == ".md":
                s.markdown(text)
            else:
                s.code(text)
        elif row["kind"] == "array":
            try:
                p = Path(row["absolute_path"])
                if p.suffix.lower() == ".npz":
                    with np.load(p, allow_pickle=False) as data:
                        for key in data.files:
                            arr = data[key]
                            s.write(f"**{key}**: shape={arr.shape}, dtype={arr.dtype}")
                else:
                    arr = np.load(p, mmap_mode="r", allow_pickle=False)
                    s.write(f"shape={arr.shape}, dtype={arr.dtype}")
            except Exception as exc:
                s.error(str(exc))


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def render_dashboard(results_dir: Path) -> None:
    s = require_streamlit()
    s.set_page_config(
        page_title="Polaris Generalization Framework",
        layout="wide",
        page_icon="⭐",
    )

    s.sidebar.title("Polaris Framework")
    s.sidebar.caption("Interactive manuscript companion")
    section = s.sidebar.radio("Section", SECTIONS, label_visibility="collapsed")

    model = "Combined"
    if section in MODEL_SECTIONS:
        model = s.sidebar.radio("Model", ["Combined", "XGBoost", "CheMeleon"])

    s.sidebar.markdown("---")
    s.sidebar.caption(
        "Launch: `pixi run -e cheminformatics streamlit run "
        "notebooks/4.01-seal-results-dashboard.py`"
    )

    dispatch = {
        "Framework Overview": lambda: render_framework_overview(),
        "Dataset Characterization": lambda: render_dataset_characterization(),
        "Chemical Space & Splitting": lambda: render_chemical_space(),
        "FM1: Performance over Distance": lambda: render_performance_distance(model),
        "FM1b: IID vs OOD Series": lambda: render_iid_ood(model),
        "FM2: Scaffold vs Random": lambda: render_scaffold_vs_random(model),
        "FM2b: Split Variance": lambda: render_split_variance(model),
        "FM3: Activity Cliffs": lambda: render_activity_cliffs(model),
        "FM4: Molecular Variants": lambda: render_molecular_variants(model),
        "FM4b: Resonance Forms": lambda: render_resonance(model),
        "Baseline Performance": lambda: render_baseline(model),
        "Explorer": lambda: render_explorer(results_dir),
    }

    dispatch[section]()


# ---------------------------------------------------------------------------
# CLI audit mode (unchanged)
# ---------------------------------------------------------------------------

def run_inventory_check(results_dir: Path) -> None:
    df = inventory_dataframe(results_dir, include_system=True)
    if df.empty:
        typer.echo(f"No files found under {results_dir}")
        raise typer.Exit(1)

    non_system = df[df["kind"] != "system"]
    unsupported = non_system[non_system["kind"] == "unsupported"]
    typer.echo(f"Results directory: {results_dir}")
    typer.echo(f"Total files: {len(df):,}")
    typer.echo(f"Non-system result files: {len(non_system):,}")
    typer.echo("Counts by kind:")
    for kind, count in non_system["kind"].value_counts().sort_index().items():
        typer.echo(f"  {kind}: {count:,}")
    typer.echo("Counts by suffix:")
    for suffix, count in non_system["suffix"].value_counts().sort_index().items():
        typer.echo(f"  {suffix}: {count:,}")
    if unsupported.empty:
        typer.echo("OK: every non-system result file has a renderer.")
    else:
        typer.echo("ERROR: unsupported result files found:")
        for rel_path in unsupported["relative_path"].tolist():
            typer.echo(f"  {rel_path}")
        raise typer.Exit(2)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    results_dir: Path = typer.Option(PROCESSED_DATA_DIR, help="Processed results directory to browse."),
    check: bool = typer.Option(False, "--check", help="Audit renderer coverage without launching Streamlit."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if check:
        run_inventory_check(results_dir)
        return
    render_dashboard(results_dir)


if __name__ == "__main__":
    app()
