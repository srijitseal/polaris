# Analysis Checklist

Status tracking for all planned analyses. This is the source of truth for analysis status.

## Phase 0: Exploration

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `0.01-seal-dataset-exploration.py` | Initial Expansion Tx dataset exploration: load raw + ML-ready CSVs, count molecules/endpoints, check missingness, endpoint coverage, CRO distribution, physicochemical property characterization |
| DONE | `0.02-seal-ecfp-distance-exploration.py` | ECFP fingerprint and distance exploration: compute ECFP4 fingerprints, all-pairwise Tanimoto distances, 1-NN and 5-NN distance distributions |
| DONE | `0.03-araripe-chembl-tanimoto.py` | ChEMBL Tanimoto similarity (issue #24): max Tanimoto of each ExpansionRX compound to ChEMBL 36 (all 2.85M compounds + 308K ADME subset). Median max Tc 0.444 (all) / 0.388 (ADME); 98.2% below Tc 0.7. Confirms structural novelty of RNA-targeting chemical space |

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
| DONE | `2.06-seal-split-quality.py` | Split quality diagnostics (paper Fig 3): fold size balance, target distributions, test-to-train 1-NN distance distributions, structural overlap, UMAP embeddings per strategy. Adds (a) per-assay temporal label drift — 8/9 assays drift, only MBPB stable (Engkvist review); (b) value-vs-structural extrapolation regimes — time-split ≤0.9% beyond range vs target-split 91.5% (Rodríguez review); (c) adversarial-validation covariate-shift AUROC per strategy (random 0.51, target 0.86, time 0.98, cluster 1.00) reusing the 2.03/2.04/2.05 folds |
| DONE (XGB + CheMeleon) | `2.07-seal-performance-distance.py` | Performance-over-distance curves (paper Fig 4, Table 2): both Optuna TPE-tuned XGBoost (ECFP4 + RDKit 2D + dimorphite_dl) and CheMeleon foundation-model fine-tune. Sliding window bins, all 9 endpoints × 3 strategies. Outputs in `xgboost/`, `chemeleon/`, `combined/` |
| DONE (XGB + CheMeleon) | `2.08-seal-baseline-performance.py` | Baseline on competition split (paper Fig S1, Table S1): XGBoost (Optuna TPE) and CheMeleon. Bootstrap 95% CIs (1000 resamples). MA-RAE 0.776 (XGB) → 0.684 (CheMeleon) |
| DONE (XGB + CheMeleon) | `2.09-seal-iid-vs-ood-series.py` | IID vs OOD on chemical series (paper Fig 5, Table 3): Train on largest Butina cluster (temporal 80/20), OOD test on second cluster. R² negative on 5/8 endpoints (XGB) and 6/8 (CheMeleon). Adds amine-pKa LogD diagnostic: OOD series is 85% amino-pyridine vs 97% aliphatic amine in IID, with a systematic LogD over-prediction bias (−0.4 to −1.5 log units vs +0.09 IID) consistent with an ionization-regime shift the IID-trained model cannot transfer |
| DONE (XGB + CheMeleon) | `2.10-seal-activity-cliff-eval.py` | Activity cliff evaluation (paper Fig S4, Table S4): cluster-split CV. Cliffs = Tanimoto sim > 0.85 + absolute ≥ 1.0 log units. Cliff RAE > non-cliff RAE for all 7 endpoints with sufficient cliffs in both models |
| DONE (XGB + CheMeleon) | `2.11-seal-scaffold-vs-random.py` | Scaffold vs random split comparison (paper Fig S2, Table S2): 5-fold CV. Naive Bemis-Murcko scaffold vs random vs cluster-based. MA-RAE: random 0.474 / scaffold 0.508 / cluster 0.674 (XGB); 0.430 / 0.471 / 0.623 (CheMeleon) |
| DONE (XGB + CheMeleon) | `2.12-seal-split-variance.py` | Split variance study (paper Fig S3, Table S3): 5 random + 5 cluster repeats of 5-fold CV. Mann-Whitney U=25, p=0.0079 for all 9 endpoints in both models |
| DONE (XGB + CheMeleon) | `2.13-seal-molecular-variants.py` | Molecular variant consistency (paper Fig 6, Table 4, Figs S5–S6): cluster-split CV. Stereoisomer and scaffold decoration groups with random-pair baseline. Stereo consistency ratio <1 for 8/9 (XGB) and 9/9 (CheMeleon) endpoints |
| DONE | `2.14-zalte-resonance-generation.py` | Resonance form generation: protonate-enumerate-reprotonate-deduplicate pipeline using RDKit `ResonanceMolSupplier` at endpoint-specific pH (7.4 or 6.5). 19.6–53.4% of molecules have >1 form at pH 7.4, 1.8% at pH 6.5. Representation-level — no model run |
| DONE (XGB + CheMeleon) | `2.15-zalte-resonance-variants.py` | Resonance sensitivity analysis (paper Fig 7, Fig S8): cluster-split CV. RIGR-aligned metrics (Zalte et al. 2025 JCIM). Total RMSE swing 0.4–27.4% across endpoints, worsening meets or exceeds improvement (XGB and CheMeleon). Three-panel main figure: prevalence, improvement/worsening, per-molecule error range |
| DONE | `2.16-araripe-scaffold-boundary.py` | Scaffold boundary analysis (outline §6b): scaffold boundary violation rate (56.4%), cross-scaffold proximity, activity concordance conditioned on scaffold. Pure dataset analysis, no model run |

## Phase 3: Figures

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `3.01-seal-paper-figures.py` | Publication figures: assemble all main (Fig 1–7) and supplementary (Fig S1–S8) at publication quality from the per-model `xgboost/` outputs. Plus Fig S9: per-notebook `combined/` panels showing XGBoost vs CheMeleon side by side. Sources: Fig 1 (NB 0.02), Fig 2 (NB 2.01), Fig 3 (NB 2.06), Fig 4 (NB 2.07/xgboost), Fig 5 (NB 2.09/xgboost), Fig 6 (NB 2.13/xgboost), Fig 7 (NB 2.15/xgboost), Fig S1 (NB 2.08/xgboost), Fig S2 (NB 2.11/xgboost), Fig S3 (NB 2.12/xgboost), Fig S4 (NB 2.10/xgboost), Fig S5–S6 (NB 2.13/xgboost), Fig S7 (NB 2.16), Fig S8 (NB 2.15), Fig S9 (combined/ across modeling notebooks) |

## Phase 4: Cross-dataset / external validation

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `4.01-seal-cross-dataset-characterization.py` | Cross-dataset characterization case study (ExpansionRx vs public Biogen ADME, Fang et al. 2023). Same representation (ECFP4, Tanimoto, Butina cutoff 0.7); compares Butina cluster concentration, intra-set 1-NN distance, and random-split test-to-train distance. Biogen is structurally diffuse — largest cluster 2.1% vs 33.8%, top-10 clusters 14% vs 84%, median 1-NN 0.55 vs 0.19, random-split distance 0.57 vs 0.20 — so series-based IID/OOD splits are not constructible and a random split is already hard. Demonstrates that splitting strategy must follow dataset characterization. Biogen data lives in `data/external/`, fetched by `scripts/download_data.sh`. (Replaces the removed `4.01-seal-results-dashboard.py`.) |
