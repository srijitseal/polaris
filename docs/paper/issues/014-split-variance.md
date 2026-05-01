# Random CV reproducibility is structural — between-strategy gap dominates within-strategy variance

## Summary

Random 5-fold CV is highly reproducible (RAE std 0.002–0.012 across 5 repeats), but this tightness is a **structural artefact** of sampling design: each fold is an unbiased draw from the same joint distribution over chemistry and labels, so different seeds produce near-identical train/test marginals and near-identical error estimates. Cluster-based CV deliberately breaks this symmetry by sampling disjoint structural groups (RAE std 0.007–0.064) at a substantially harder performance level (MA-RAE ~0.67 vs ~0.47). The cluster/random std ratio varies per endpoint from 0.56× (KSOL, where cluster is tighter) to 14× (Caco-2 Efflux), median ~5× — "cluster is always noisier than random" is not uniformly true. The invariant is that the between-strategy MA-RAE gap (~0.20) dwarfs within-strategy spread by ~10×. The practical consequence: the choice of splitting strategy changes the reported answer far more than the number of repeats, and random CV's reproducibility should not be read as deployment confidence.

## Method

1. **Random splits**: Generate 5 independent 5-fold CV assignments (seeds 0–4) per endpoint
2. **Cluster splits**: Use all 5 precomputed repeats from `cluster_cv_folds.parquet` (stochastic EKM + MiniBatchKMeans)
3. **Train and evaluate**: Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) on ECFP4 + RDKit 2D descriptors per repeat × endpoint × fold (~450 models total)
4. **Aggregate**: Per-repeat mean metrics (across 5 folds), then compute variance statistics across repeats

## Key Findings

### Performance variance across repeats — RAE

| Endpoint | Random RAE (mean±std) | Random range | Cluster RAE (mean±std) | Cluster range |
|----------|----------------------|-------------|----------------------|--------------|
| LogD | 0.345±0.005 | 0.013 | 0.520±0.009 | 0.026 |
| KSOL | 0.516±0.012 | 0.033 | 0.715±0.007 | 0.018 |
| HLM CLint | 0.597±0.007 | 0.019 | 0.820±0.027 | 0.074 |
| MLM CLint | 0.547±0.004 | 0.010 | 0.735±0.016 | 0.044 |
| Caco-2 Papp A>B | 0.489±0.003 | 0.009 | 0.686±0.023 | 0.069 |
| Caco-2 Efflux | 0.435±0.003 | 0.007 | 0.643±0.036 | 0.092 |
| MPPB | 0.461±0.002 | 0.006 | 0.653±0.010 | 0.030 |
| MBPB | 0.402±0.003 | 0.009 | 0.618±0.022 | 0.063 |
| MGMB | 0.454±0.007 | 0.019 | 0.629±0.064 | 0.182 |

Random RAE std: 0.002–0.012; random RAE range: 0.006–0.033. Cluster RAE std: 0.007–0.064; cluster RAE range: 0.018–0.182. Cluster splits show wider variance on 8 of 9 endpoints; KSOL is the exception (cluster range 0.018 < random 0.033).

### Performance variance across repeats — R²

| Endpoint | Random R² (mean±std) | Random range | Cluster R² (mean±std) | Cluster range |
|----------|---------------------|-------------|----------------------|--------------|
| LogD | 0.864±0.004 | 0.010 | 0.702±0.009 | 0.027 |
| KSOL | 0.649±0.008 | 0.024 | 0.418±0.010 | 0.031 |
| HLM CLint | 0.599±0.009 | 0.026 | 0.273±0.046 | 0.131 |
| MLM CLint | 0.674±0.003 | 0.010 | 0.419±0.029 | 0.078 |
| Caco-2 Papp A>B | 0.696±0.004 | 0.013 | 0.454±0.025 | 0.069 |
| Caco-2 Efflux | 0.745±0.003 | 0.008 | 0.504±0.045 | 0.117 |
| MPPB | 0.740±0.003 | 0.007 | 0.528±0.019 | 0.056 |
| MBPB | 0.778±0.004 | 0.009 | 0.552±0.031 | 0.096 |
| MGMB | 0.697±0.005 | 0.015 | 0.485±0.100 | 0.279 |

Cluster-split R² ranges are dramatic: MGMB spans 0.299–0.577 across 5 repeats — a single split could report either "poor" or "moderate" performance depending on the random seed.

### MA-RAE distribution

| Strategy | Mean MA-RAE | Std | Range |
|----------|-------------|-----|-------|
| Random (5 repeats) | 0.472 | ~0.003 | ~0.008 |
| Cluster (5 repeats) | 0.669 | ~0.010 | ~0.025 |

Random MA-RAE is tightly clustered — 5 repeats look almost identical. Cluster MA-RAE shows meaningful spread, confirming that cluster-based evaluation is sensitive to which structural groups are held out. The gap between strategies (~0.20 MA-RAE) dwarfs the within-strategy variance by ~10×.

### Cluster/random std ratio per endpoint

| Endpoint | Cluster RAE std | Random RAE std | Ratio (C/R) |
|----------|----------------|----------------|-------------|
| Caco-2 Efflux | 0.0365 | 0.0026 | 13.9× |
| MGMB | 0.0637 | 0.0067 | 9.6× |
| Caco-2 Papp A>B | 0.0230 | 0.0030 | 7.6× |
| MBPB | 0.0222 | 0.0034 | 6.6× |
| MPPB | 0.0104 | 0.0019 | 5.5× |
| MLM CLint | 0.0156 | 0.0038 | 4.2× |
| HLM CLint | 0.0270 | 0.0069 | 3.9× |
| LogD | 0.0085 | 0.0051 | 1.7× |
| **KSOL** | **0.0066** | **0.0118** | **0.56× (cluster tighter)** |

Median ratio ~5×, spanning 0.56×–14×.

### Statistical tests (Mann-Whitney U)

| Endpoint | U statistic | p-value | Cluster median RAE | Random median RAE |
|---|---|---|---|---|
| LogD | 25.0 | 0.0079 | 0.517 | 0.342 |
| KSOL | 25.0 | 0.0079 | 0.718 | 0.517 |
| HLM CLint | 25.0 | 0.0079 | 0.832 | 0.593 |
| MLM CLint | 25.0 | 0.0079 | 0.741 | 0.546 |
| Caco-2 Papp A>B | 25.0 | 0.0079 | 0.688 | 0.489 |
| Caco-2 Efflux | 25.0 | 0.0079 | 0.666 | 0.435 |
| MPPB | 25.0 | 0.0079 | 0.654 | 0.462 |
| MBPB | 25.0 | 0.0079 | 0.613 | 0.403 |
| MGMB | 25.0 | 0.0079 | 0.613 | 0.451 |

Mann-Whitney U tests confirm that cluster and random RAE distributions are entirely non-overlapping for all 9 endpoints (U = 25.0, p = 0.0079). The maximum U statistic (25 = 5 × 5, all cluster values exceed all random values) indicates complete separation. p = 0.0079 is the minimum achievable p-value for a two-sided test with n1 = n2 = 5. Note that KSOL, despite having a cluster std tighter than random, still shows complete separation in RAE medians — the strategy effect is on the mean, not just the variance.

### Interpretation

The results reveal a subtlety beyond "single splits are noisy." Random CV is *not* noisy — 5 repeats produce nearly identical metrics. Two points worth separating:

1. **Random CV's reproducibility is structural, not informative.** By construction, random sampling produces folds with near-identical marginals over both the chemistry (domain) and the labels (co-domain). Each fold is an unbiased draw from the same joint, so different seeds produce near-identical metrics. Cluster and series-based splits deliberately bias the distribution in the held-out set (along chemical space and label space respectively), so different seeds produce qualitatively different "hard" subsets. This is a consequence of sampling design and should not be surprising.

2. **The consequence is not obvious.** Practitioners routinely report tight random-CV confidence intervals as evidence of reliable performance estimation. Concretely, random CV delivers precise estimates (MA-RAE ~0.47, std ~0.003) of a quantity — performance on near-duplicates that leak across fold boundaries — that does not correspond to deployment. The between-strategy MA-RAE gap (~0.20) is ~10× the within-strategy spread, so picking random over cluster changes the reported answer far more than running more repeats does. A single cluster-split can report R²=0.30 or R²=0.58 for MGMB — only the mean across repeats (0.49) with its CI gives the full story.

The practical takeaways:

1. **Splitting strategy matters far more than number of repeats.** Five random repeats with tight CIs give false confidence; five cluster repeats with wide CIs give an honest picture.
2. **Report confidence intervals from distance-aware splits.** Narrow random-CV intervals are a property of the sampling, not of the model's generalization.

## Plots

- `data/processed/2.12-seal-split-variance/rae_distributions.png` — Per-endpoint RAE distributions across repeats (main Fig S2)
<!-- Paste: rae_distributions.png -->
- `data/processed/2.12-seal-split-variance/r2_distributions.png` — Per-endpoint R² distributions across repeats
<!-- Paste: r2_distributions.png -->
- `data/processed/2.12-seal-split-variance/ma_rae_distribution.png` — MA-RAE distribution by strategy
<!-- Paste: ma_rae_distribution.png -->
- `data/processed/2.12-seal-split-variance/variance_heatmap.png` — Coefficient of variation heatmap
<!-- Paste: variance_heatmap.png -->
- `data/processed/2.12-seal-split-variance/single_split_danger.png` — Cherry-picking danger illustration
<!-- Paste: single_split_danger.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.12-seal-split-variance.py
```

## Source

`notebooks/2.12-seal-split-variance.py`
