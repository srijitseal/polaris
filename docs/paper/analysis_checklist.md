# Analysis Checklist

Status tracking for all planned analyses. This is the source of truth for analysis status.

## Phase 0: Exploration

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `0.01-seal-dataset-exploration.py` | Initial Expansion Tx dataset exploration: load raw + ML-ready CSVs, count molecules/endpoints, check missingness, endpoint coverage, CRO distribution, physicochemical property characterization |
| DONE | `0.02-seal-ecfp-distance-exploration.py` | ECFP fingerprint and distance exploration: compute ECFP4 fingerprints, all-pairwise Tanimoto distances, 1-NN and 5-NN distance distributions |

## Phase 1: Data Loading

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `1.01-seal-dataset-loader.py` | Dataset loading pipeline: load raw and ML-ready CSVs, validate molecule counts, parse SMILES, handle out-of-range modifiers, ordinal ordering validation, save to interim/ |

## Phase 2: Analysis

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `2.01-seal-chemical-space-analysis.py` | Chemical space characterization (paper Fig 2): 1-NN and all-pairwise ECFP4 Tanimoto distance distributions, Butina clustering (cutoff 0.7), cluster size distribution, representative molecules from top 5 clusters |
| DONE | `2.02-seal-target-distribution.py` | Target/endpoint distribution analysis: per-endpoint distribution plots, coverage heatmap, train/test KS shift tests |
| DONE | `2.03-seal-cluster-split.py` | Cluster-based splitting: EKM via KernelPCA + MiniBatchKMeans, greedy assignment to 5 folds, per-endpoint fold assignments, 5 repeats |
| DONE | `2.04-seal-time-split.py` | Time-split using ordinal ordering: extract ordinal index from molecule names, expanding-window CV across 4 folds |
| DONE | `2.05-seal-target-split.py` | Target distribution split: molecules sorted by endpoint value, expanding-window CV across 4 folds |
| DONE | `2.06-seal-split-quality.py` | Split quality diagnostics (paper Fig 3): fold size balance, target distributions, test-to-train 1-NN distance distributions, structural overlap, UMAP embeddings per strategy |
| DONE | `2.07-seal-performance-distance.py` | Performance-over-distance curves (paper Fig 4, Table 2): Optuna TPE-tuned XGBoost on ECFP4 + RDKit 2D + dimorphite_dl. Sliding window bins, all 9 endpoints × 3 strategies |
| DONE | `2.08-seal-baseline-performance.py` | Baseline on competition split (paper Fig S1, Table S1): Optuna TPE-tuned XGBoost on ECFP4 + RDKit 2D + dimorphite_dl. Bootstrap 95% CIs (1000 resamples) |
| DONE | `2.09-seal-iid-vs-ood-series.py` | IID vs OOD on chemical series (paper Fig 5, Table 3): Optuna TPE-tuned XGBoost. Train on largest Butina cluster (temporal 80/20), OOD test on second cluster |
| DONE | `2.10-seal-activity-cliff-eval.py` | Activity cliff evaluation (paper Fig S4, Table S4): Optuna TPE-tuned XGBoost under cluster-split CV. Cliffs = Tanimoto sim > 0.85 + top-quartile Δactivity |
| DONE | `2.11-seal-scaffold-vs-random.py` | Scaffold vs random split comparison (paper Fig S2, Table S2): Optuna TPE-tuned XGBoost, 5-fold CV. Naive Bemis-Murcko scaffold vs random vs cluster-based |
| DONE | `2.12-seal-split-variance.py` | Split variance study (paper Fig S3, Table S3): Optuna TPE-tuned XGBoost, 5 random + 5 cluster repeats of 5-fold CV. Mann-Whitney U tests |
| DONE | `2.13-seal-molecular-variants.py` | Molecular variant consistency (paper Fig 6, Table 4, Figs S5–S6): Optuna TPE-tuned XGBoost under cluster-split CV. Stereoisomer and scaffold decoration groups with random-pair baseline |
| DONE | `2.14-zalte-resonance-generation.py` | Resonance form generation: enumerate tautomeric and resonance forms using RDKit |
| DONE | `2.15-zalte-resonance-variants.py` | Resonance variant analysis (Figs S7–S8): Optuna TPE-tuned XGBoost under cluster-split CV. Resonance form prediction consistency and fingerprint instability |
| DONE | `2.16-araripe-scaffold-boundary.py` | Scaffold boundary analysis (outline §6b): scaffold boundary violation rate, cross-scaffold proximity, activity concordance conditioned on scaffold |

## Phase 3: Figures

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `3.01-seal-paper-figures.py` | Publication figures: assemble all main (Fig 1–6) and supplementary (Fig S1–S8) at publication quality. Sources: Fig 1 (NB 0.02), Fig 2 (NB 2.01), Fig 3 (NB 2.06), Fig 4 (NB 2.07), Fig 5 (NB 2.09), Fig 6 (NB 2.13), Fig S1 (NB 2.08), Fig S2 (NB 2.11), Fig S3 (NB 2.12), Fig S4 (NB 2.10), Fig S5–S6 (NB 2.13), Fig S7–S8 (NB 2.15) |
