# Analysis Checklist

Status tracking for all planned analyses. This is the source of truth for analysis status.

## Phase 0: Exploration

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `0.01-seal-dataset-exploration.py` | Initial Expansion Tx dataset exploration: load raw + ML-ready CSVs, count molecules/endpoints, check missingness, endpoint coverage per molecule, CRO distribution (if annotated), physicochemical property characterization (RNA-binding vs protein modulator hypothesis) |
| TODO | `0.02-seal-ecfp-distance-exploration.py` | ECFP fingerprint and distance exploration: compute 2048-bit ECFP4 fingerprints, all-pairwise Tanimoto distances, 1-NN and 5-NN distance distributions, compare to Cas's Biogen HLM case study |

## Phase 1: Data Loading

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `1.01-seal-dataset-loader.py` | Dataset loading pipeline: load raw and ML-ready CSVs, validate molecule counts (7,618 raw / 5,326 train / 2,282 test), parse SMILES, handle out-of-range modifiers in raw data, CRO annotation (if available), ordinal ordering validation from molecule names (E-XXXXXXX), save to interim/ |

## Phase 2: Analysis

| Status | Notebook | Description |
|--------|----------|-------------|
| DONE | `2.01-seal-chemical-space-analysis.py` | Chemical space characterization (paper Fig 2): 1-NN ECFP4 Tanimoto distance distribution (diversity measure), all-pairwise distance distribution, Butina clustering (cutoff 0.7), cluster size distribution, visualize 6 random molecules from each of the 5 largest clusters |
| DONE | `2.02-seal-target-distribution.py` | Target/endpoint distribution analysis (paper Fig 3): distribution plot for each of 10 endpoints, total counts per endpoint, coverage heatmap (which molecules have which endpoints), verify range covers expected deployment distribution |
| TODO | `2.03-seal-cluster-split.py` | Cluster-based splitting (paper Fig 4): Empirical Kernel Map (EKM) to project ECFP4 into kernel space, Mini Batch K-Means clustering, each cluster/subset = one CV fold, verify roughly equal fold sizes, implement 5-fold CV, test with 5x5 CV using stochasticity from EKM + Mini Batch K-Means |
| TODO | `2.04-seal-time-split.py` | Time-split using ordinal ordering (paper Fig 4): extract ordinal index from molecule names (E-XXXXXXX), split earlier molecules as train / later as test, implement as CV-compatible (rolling origin or expanding window), compare split characteristics to cluster-based split |
| TODO | `2.05-seal-target-split.py` | Target distribution split (paper Fig 4): Rolling Window CV for regression endpoints, for classification: create folds with larger class imbalance in train than test, evaluate compatibility with repeated CV |
| TODO | `2.06-seal-split-quality.py` | Split quality diagnostics (paper Fig 5): check split sizes (no excessively large/small test sets), target distribution per fold (no drastically different distributions), test-to-train distance distributions per fold, train-test structural overlap check, optional UMAP embedding visualization |
| TODO | `2.07-seal-performance-distance.py` | Performance-over-distance curves (paper Fig 6): for each CV split compute test-to-train 1-NN distance, define sliding window bins (min=Q1-1.5*IQR, max=Q3+1.5*IQR, binwidth=(max-min)/5, stepsize=binwidth/20), compute performance per bin per split (skip bins <25 samples), combine into figure with confidence intervals, create two curves: ECFP4 Tanimoto distance + target space distance, train default RF/MLP on ECFP fingerprints |
| TODO | `2.08-seal-iid-vs-ood-series.py` | IID vs OOD on chemical series — the "hero" example (paper Fig 7): take two large clusters (chemical series), time-split the largest into train + IID validation, use smaller cluster as OOD test, train default random forest on ECFP, compare squared error intra-series (IID) vs inter-series (OOD), demonstrate that chemical series + time-split uniquely enable this analysis |
| TODO | `2.10-seal-activity-cliff-eval.py` | Activity cliff evaluation (paper Fig S3, ref: MoleculeACE): identify activity cliffs (structurally similar molecules with large activity differences), set aside cliff pairs, evaluate model performance specifically on cliffs vs non-cliffs, demonstrate that smoothly interpolating models fail on cliffs |
| TODO | `2.11-seal-scaffold-vs-random.py` | Scaffold vs random split comparison (paper Fig S1, ref: Greg Landrum's blog): implement naive Bemis-Murcko scaffold split, compare performance to random split — show they are similar without careful considerations, compare to cluster-based split to demonstrate when scaffold splits fail, quantify similarity |
| TODO | `2.12-seal-split-variance.py` | Split variance study (paper Fig S2, ref: Pat's blog): run multiple random seeds / fold assignments, show variance in performance across splits, demonstrate single-split evaluation is misleading, motivate need for CV with confidence intervals |
| TODO | `2.13-seal-molecular-variants.py` | Molecular variant consistency (paper Fig S4, ref: Srijit's LinkedIn): identify groups of related molecules (stereoisomers, tautomers, Kekule structures, different protonation states, scaffold decorations), compute prediction variance within each group, test whether models learn chemistry vs memorize fingerprint artifacts |

## Phase 3: Figures

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `3.01-seal-paper-figures.py` | Publication figures: assemble all main (Fig 1-7) and supplementary (Fig S1-S4) figures at publication quality, consistent styling, export as PDF/SVG |
