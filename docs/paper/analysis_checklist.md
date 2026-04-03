# Analysis Checklist

Status tracking for all planned analyses. This is the source of truth for analysis status.

## Phase 0: Exploration

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `0.01-seal-dataset-exploration.py` | Initial Expansion Tx dataset exploration: load raw + ML-ready CSVs, count molecules/endpoints, check missingness, endpoint coverage per molecule, CRO distribution (if annotated), physicochemical property characterization (RNA-binding vs protein modulator hypothesis) |
| DONE | `0.02-seal-ecfp-distance-exploration.py` | ECFP fingerprint and distance exploration: compute 2048-bit ECFP4 fingerprints, all-pairwise Tanimoto distances, 1-NN and 5-NN distance distributions, compare to Cas's Biogen HLM case study |

## Phase 1: Data Loading

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `1.01-seal-dataset-loader.py` | Dataset loading pipeline: load raw and ML-ready CSVs, validate molecule counts (7,618 raw / 5,326 train / 2,282 test), parse SMILES, handle out-of-range modifiers in raw data, CRO annotation (if available), ordinal ordering validation from molecule names (E-XXXXXXX), save to interim/ |

## Phase 2: Analysis

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `2.01-seal-chemical-space-analysis.py` | Chemical space characterization (paper Fig 2): 1-NN ECFP4 Tanimoto distance distribution (diversity measure), all-pairwise distance distribution, Butina clustering (cutoff 0.7) → 135 clusters, top 3 (2,572 / 1,301 / 885) contain 62.5%, 33 singletons (24.4%), median size 4. Cluster size distribution plot, representative molecules from top 5 clusters (6 random each) |
| DONE | `2.02-seal-target-distribution.py` | Target/endpoint distribution analysis: distribution plot for each of 9 endpoints, total counts per endpoint, coverage heatmap (which molecules have which endpoints). Coverage: LogD/KSOL 96%, MLM CLint 75%, HLM CLint 60%, Caco-2 50%, MPPB 23%, MBPB 19%, MGMB 6%. All endpoints show significant train/test KS shift (p < 0.01), especially Caco-2 Efflux (KS=0.37) and Caco-2 Papp (KS=0.28) |
| DONE | `2.03-seal-cluster-split.py` | Cluster-based splitting: Empirical Kernel Map (EKM) via KernelPCA (n_components=50) on Tanimoto similarity matrix, MiniBatchKMeans (k=20, batch_size=min(1024,n), n_init=3), greedy assignment of 20 clusters to 5 folds for balance (max/min ratio < 1.12). Per-endpoint fold assignments using only non-null molecules. 5 repeats × 5 folds = 25 train/test splits per endpoint |
| DONE | `2.04-seal-time-split.py` | Time-split using ordinal ordering: extract ordinal index from molecule names (E-XXXXXXX), sort by index, divide into 5 equal groups, expanding-window CV across 4 folds. Pooled median 1-NN distance 0.350 (lower than cluster 0.410). Per-fold medians non-monotonic (0.386, 0.351, 0.302, 0.364) — chemical series exploration non-linear in time |
| DONE | `2.05-seal-target-split.py` | Target distribution split: molecules sorted by endpoint value, divided into 5 equal groups, expanding-window CV across 4 folds. Low structural distances (LogD median 1-NN: 0.245–0.286) but maximal target shift (KS 0.956–0.988). Tests value extrapolation capability |
| DONE | `2.06-seal-split-quality.py` | Split quality diagnostics (paper Fig 3): validate all three strategies — fold size balance, target distribution per fold, test-to-train 1-NN distance distributions, structural overlap (fraction with neighbor < Jaccard 0.1). UMAP embeddings per strategy. Distance distributions illustrated for LogD: cluster median 0.369–0.440, time 0.301–0.386, target 0.245–0.286 |
| DONE | `2.07-seal-performance-distance.py` | Performance-over-distance curves (paper Fig 4, Table 2): Optuna TPE-tuned XGBoost on ECFP4 + full RDKit 2D descriptors + dimorphite_dl protonation, log-transform. Sliding window bins (Q1-1.5×IQR to Q3+1.5×IQR, binwidth/5, step/20, min 25 molecules). All 9 endpoints × 3 strategies (cluster, time, target). Results: issue #10 |
| DONE | `2.08-seal-baseline-performance.py` | Baseline on competition split (paper Fig S1, Table S1): Optuna TPE-tuned XGBoost on ECFP4 + RDKit 2D + dimorphite_dl. Trained on 5,326, evaluated on 2,282. Bootstrap 95% CIs (1000 resamples). Results: issue #8 |
| DONE | `2.09-seal-iid-vs-ood-series.py` | IID vs OOD on chemical series (paper Fig 5, Table 3): Optuna TPE-tuned XGBoost. Train on largest Butina cluster (temporal 80/20 split), OOD test on second cluster. MGMB excluded (sparse). Results: issue #11 |
| DONE | `2.10-seal-activity-cliff-eval.py` | Activity cliff evaluation (paper Fig S4, Table S4): Optuna TPE-tuned XGBoost under cluster-split CV (5 folds). Cliffs = Tanimoto sim > 0.85 + top-quartile Δactivity in log₁₀(x+1) space. Results: issue #12 |
| DONE | `2.11-seal-scaffold-vs-random.py` | Scaffold vs random split comparison (paper Fig S2, Table S2): Optuna TPE-tuned XGBoost, 5-fold CV. Naive Bemis-Murcko scaffold vs random (seed=42) vs cluster-based. Scaffold group statistics + histogram. Results: issue #13 |
| DONE | `2.12-seal-split-variance.py` | Split variance study (paper Fig S3, Table S3): Optuna TPE-tuned XGBoost, 5 random 5-fold repeats (seeds 0–4) + 5 cluster repeats (~450 models). Mann-Whitney U tests for non-overlap. Results: issue #14 |
| DONE | `2.13-seal-molecular-variants.py` | Molecular variant consistency (paper Fig 6, Table 4, Figs S5–S6): Optuna TPE-tuned XGBoost under cluster-split CV (5 folds). Stereoisomer and scaffold decoration groups with random-pair baseline. Results: issue #15 |
| DONE | `2.14-zalte-resonance-generation.py` | Resonance form generation: enumerate tautomeric and resonance forms for molecules in the dataset using RDKit |
| DONE | `2.15-zalte-resonance-variants.py` | Resonance variant analysis: investigate tautomeric and resonance-form consistency in molecular representations and predictions |
| DONE | `2.16-araripe-scaffold-boundary.py` | Scaffold boundary analysis (outline §6b, issue #16): (1) scaffold boundary violation rate — 56.2% (4,276/7,608) of molecules have their nearest neighbor in a different scaffold group; (2) cross-scaffold proximity — overall vs cross-scaffold 1-NN distributions nearly overlap (medians 0.188 vs 0.233), 36.8% of non-singletons have closer cross-scaffold neighbor; (3) activity concordance conditioned on scaffold — ~29M pairs per endpoint, 94/133 Mann-Whitney tests significant, 84/94 show same-scaffold pairs have smaller |Δy| at given distance. Strong endpoint-specific effects: Caco-2 Efflux and HLM/MLM CLint most scaffold-dependent, MBPB mostly non-significant, KSOL reverses at high distance. Scaffold identity carries real biological signal but scaffold split procedure fails to leverage it: no tunability, 67.5% singleton scaffolds, 56.2% boundary violation rate, endpoint-agnostic boundary |

## Phase 3: Figures

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `3.01-seal-paper-figures.py` | Publication figures: assemble all main (Fig 1–6) and supplementary (Fig S1–S6) figures at publication quality, consistent styling, export as PDF/SVG. Current source: Fig 1 (NB 0.02), Fig 2 (NB 2.01), Fig 3 (NB 2.06), Fig 4 (NB 2.07), Fig 5 (NB 2.09), Fig 6 (NB 2.13), Fig S1 (NB 2.08), Fig S2 (NB 2.11), Fig S3 (NB 2.12), Fig S4 (NB 2.10), Fig S5–S6 (NB 2.13) |
