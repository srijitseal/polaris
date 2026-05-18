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
| LogD | 0.346±0.005 | 0.012 | 0.519±0.011 | 0.033 |
| KSOL | 0.516±0.012 | 0.032 | 0.711±0.007 | 0.017 |
| HLM CLint | 0.596±0.008 | 0.025 | 0.825±0.028 | 0.081 |
| MLM CLint | 0.546±0.004 | 0.012 | 0.734±0.014 | 0.040 |
| Caco-2 Papp A>B | 0.490±0.004 | 0.012 | 0.687±0.025 | 0.071 |
| Caco-2 Efflux | 0.435±0.003 | 0.007 | 0.644±0.037 | 0.096 |
| MPPB | 0.461±0.003 | 0.010 | 0.652±0.011 | 0.029 |
| MBPB | 0.404±0.003 | 0.009 | 0.617±0.026 | 0.077 |
| MGMB | 0.454±0.005 | 0.015 | 0.630±0.059 | 0.174 |

Random RAE std: 0.003–0.012; random RAE range: 0.007–0.032. Cluster RAE std: 0.007–0.059; cluster RAE range: 0.017–0.174. Cluster splits show wider variance on 8 of 9 endpoints; KSOL is the exception (cluster std 0.007 < random 0.012).

### Performance variance across repeats — R²

| Endpoint | Random R² (mean±std) | Random range | Cluster R² (mean±std) | Cluster range |
|----------|---------------------|-------------|----------------------|--------------|
| LogD | 0.863±0.003 | 0.009 | 0.705±0.012 | 0.036 |
| KSOL | 0.649±0.008 | 0.022 | 0.423±0.007 | 0.018 |
| HLM CLint | 0.600±0.011 | 0.033 | 0.265±0.045 | 0.119 |
| MLM CLint | 0.675±0.004 | 0.012 | 0.419±0.028 | 0.082 |
| Caco-2 Papp A>B | 0.695±0.005 | 0.016 | 0.452±0.027 | 0.072 |
| Caco-2 Efflux | 0.745±0.003 | 0.007 | 0.506±0.042 | 0.116 |
| MPPB | 0.739±0.002 | 0.007 | 0.528±0.015 | 0.041 |
| MBPB | 0.776±0.004 | 0.010 | 0.554±0.032 | 0.099 |
| MGMB | 0.694±0.003 | 0.010 | 0.483±0.091 | 0.255 |

Cluster-split R² ranges are dramatic: MGMB spans 0.321–0.576 across 5 repeats — a single split could report either "poor" or "moderate" performance depending on the random seed.

### MA-RAE distribution

| Strategy | Mean MA-RAE | Std | Range |
|----------|-------------|-----|-------|
| Random (5 repeats) | 0.472 | ~0.003 | ~0.007 |
| Cluster (5 repeats) | 0.669 | ~0.012 | ~0.030 |

Random MA-RAE is tightly clustered — 5 repeats look almost identical. Cluster MA-RAE shows meaningful spread, confirming that cluster-based evaluation is sensitive to which structural groups are held out. The gap between strategies (~0.20 MA-RAE) dwarfs the within-strategy variance.

### Cluster/random std ratio per endpoint

| Endpoint | Cluster RAE std | Random RAE std | Ratio (C/R) |
|----------|----------------|----------------|-------------|
| Caco-2 Efflux | 0.0372 | 0.0028 | 13.5× |
| MGMB | 0.0591 | 0.0052 | 11.3× |
| MBPB | 0.0257 | 0.0035 | 7.4× |
| Caco-2 Papp A>B | 0.0246 | 0.0041 | 6.0× |
| MLM CLint | 0.0144 | 0.0041 | 3.5× |
| HLM CLint | 0.0281 | 0.0084 | 3.4× |
| MPPB | 0.0105 | 0.0035 | 3.1× |
| LogD | 0.0108 | 0.0046 | 2.3× |
| **KSOL** | **0.0069** | **0.0115** | **0.6× (cluster tighter)** |

Median ratio 3.5×, spanning 0.6×–13.5×.

### Statistical tests (Mann-Whitney U)

| Endpoint | U statistic | p-value | Cluster median RAE | Random median RAE |
|---|---|---|---|---|
| LogD | 25.0 | 0.0079 | 0.518 | 0.343 |
| KSOL | 25.0 | 0.0079 | 0.708 | 0.518 |
| HLM CLint | 25.0 | 0.0079 | 0.827 | 0.593 |
| MLM CLint | 25.0 | 0.0079 | 0.739 | 0.545 |
| Caco-2 Papp A>B | 25.0 | 0.0079 | 0.689 | 0.489 |
| Caco-2 Efflux | 25.0 | 0.0079 | 0.669 | 0.434 |
| MPPB | 25.0 | 0.0079 | 0.654 | 0.462 |
| MBPB | 25.0 | 0.0079 | 0.611 | 0.403 |
| MGMB | 25.0 | 0.0079 | 0.620 | 0.452 |

Mann-Whitney U tests confirm that cluster and random RAE distributions are entirely non-overlapping for all 9 endpoints (U = 25.0, p = 0.0079). The maximum U statistic (25 = 5 × 5, all cluster values exceed all random values) indicates complete separation. p = 0.0079 is the minimum achievable p-value for a two-sided test with n1 = n2 = 5. Note that KSOL, despite having a cluster std tighter than random, still shows complete separation in RAE medians — the strategy effect is on the mean, not just the variance.

### Interpretation

The results reveal a subtlety beyond "single splits are noisy." Random CV is *not* noisy — 5 repeats produce nearly identical metrics. Two points worth separating:

1. **Random CV's reproducibility is structural, not informative.** By construction, random sampling produces folds with near-identical marginals over both the chemistry (domain) and the labels (co-domain). Each fold is an unbiased draw from the same joint, so different seeds produce near-identical metrics. Cluster and series-based splits deliberately bias the distribution in the held-out set (along chemical space and label space respectively), so different seeds produce qualitatively different "hard" subsets. This is a consequence of sampling design and should not be surprising.

2. **The consequence is not obvious.** Practitioners routinely report tight random-CV confidence intervals as evidence of reliable performance estimation. Concretely, random CV delivers precise estimates (MA-RAE ~0.47, std ~0.003) of a quantity — performance on near-duplicates that leak across fold boundaries — that does not correspond to deployment. The between-strategy MA-RAE gap (~0.20) is roughly 8× the within-cluster std and ~50× the within-random std, so picking random over cluster changes the reported answer far more than running more repeats does. A single cluster-split can report R²=0.32 or R²=0.58 for MGMB — only the mean across repeats (0.48) with its CI gives the full story.

The practical takeaways:

1. **Splitting strategy matters far more than number of repeats.** Five random repeats with tight CIs give false confidence; five cluster repeats with wide CIs give an honest picture.
2. **Report confidence intervals from distance-aware splits.** Narrow random-CV intervals are a property of the sampling, not of the model's generalization.

## Plots

- `data/processed/2.12-seal-split-variance/xgboost/rae_distributions.png` — Per-endpoint RAE distributions across repeats (main Fig S3 candidate)
<!-- Paste: rae_distributions.png -->
- `data/processed/2.12-seal-split-variance/xgboost/r2_distributions.png` — Per-endpoint R² distributions across repeats
<!-- Paste: r2_distributions.png -->
- `data/processed/2.12-seal-split-variance/xgboost/ma_rae_distribution.png` — MA-RAE distribution by strategy
<!-- Paste: ma_rae_distribution.png -->
- `data/processed/2.12-seal-split-variance/xgboost/variance_heatmap.png` — Coefficient of variation heatmap
<!-- Paste: variance_heatmap.png -->
- `data/processed/2.12-seal-split-variance/xgboost/single_split_danger.png` — Cherry-picking danger illustration
- Combined XGBoost vs CheMeleon panels: `data/processed/2.12-seal-split-variance/combined_<strategy>_<metric>.png`
<!-- Paste: single_split_danger.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.12-seal-split-variance.py
```

## Source

`notebooks/2.12-seal-split-variance.py`

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran with CheMeleon (5 random + 5 cluster repeats × 5-fold CV per endpoint, fine-tuned per fold). **Trend matches XGBoost verbatim, including at the level of statistical tests.**

### Mann-Whitney U test (random vs cluster RAE distributions)

For all 9 endpoints, **U = 25, p = 0.0079** in both XGBoost and CheMeleon — the floor for n₁=n₂=5. Random and cluster RAE distributions are entirely non-overlapping in both models.

### Between-strategy gap dominates within-strategy spread

| | XGBoost | CheMeleon |
|---|---------|-----------|
| MA-RAE random | 0.472 | 0.430 |
| MA-RAE cluster | 0.669 | 0.622 |
| Gap | **0.197** | **0.192** |
| Median within-random std | 0.004 | 0.005 |
| Median within-cluster std | 0.025 | 0.020 |
| Gap / within-random std | 48× | 41× |
| Gap / within-cluster std | 8.0× | 9.6× |

### Per-endpoint cluster/random std ratio

| Endpoint | XGB ratio | CM ratio |
|----------|-----------|----------|
| Caco-2 Efflux | 13.5× | 10.7× |
| Caco-2 Papp | 6.0× | 2.3× |
| HLM CLint | 3.4× | 6.0× |
| KSOL | 0.6× | 2.5× |
| LogD | 2.3× | 5.4× |
| MBPB | 7.4× | 3.5× |
| MGMB | 11.3× | 4.9× |
| MLM CLint | 3.5× | 7.7× |
| MPPB | 3.1× | 1.3× |

Within-strategy variance ratios are heterogeneous across endpoints in both models, and the sign of the random-vs-cluster reproducibility comparison can flip per endpoint (XGBoost KSOL ratio 0.6× → CheMeleon 2.5×). The *between*-strategy gap remains an order of magnitude larger than within-strategy spread regardless. Random-CV tightness is a structural property of sampling design, not informative about generalization, in both model classes.

### Source

- `data/processed/2.12-seal-split-variance/chemeleon/variance_summary.csv`
- `data/processed/2.12-seal-split-variance/chemeleon/mannwhitney_rae_tests.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/14#issuecomment-4381145964
- Reproduce: `pixi run -e cheminformatics python notebooks/2.12-seal-split-variance.py --model chemeleon`
