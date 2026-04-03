# Random splits give false confidence — splitting strategy matters more than number of repeats

## Summary

Random 5-fold CV produces remarkably tight variance (RAE std 0.002–0.012) but at an optimistic level (mean RAE ~0.47). Cluster-based CV reveals the true deployment-relevant uncertainty (RAE std 0.007–0.064, range up to 0.182) at a harder level (mean RAE ~0.67). The key insight is not "single splits are noisy" — random CV is actually very stable — but that random CV gives **precise estimates of the wrong thing**. The splitting strategy matters far more than the number of repeats: distance-aware splits with repeated CV provide honest performance estimates with calibrated confidence intervals.

## Method

1. **Random splits**: Generate 5 independent 5-fold CV assignments (seeds 0–4) per endpoint
2. **Cluster splits**: Use all 5 precomputed repeats from `cluster_cv_folds.parquet` (stochastic EKM + MiniBatchKMeans)
3. **Train and evaluate**: Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) on ECFP4 + RDKit 2D descriptors per repeat x endpoint x fold (~450 models total)
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

Random splits show tight variance (range 0.006–0.033). Cluster splits show wider ranges (0.018–0.182), with MGMB having the most extreme variance (RAE range 0.568–0.750).

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
| Random (5 repeats) | ~0.47 | ~0.003 | ~0.008 |
| Cluster (5 repeats) | ~0.67 | ~0.010 | ~0.025 |

Random MA-RAE is tightly clustered — 5 repeats look almost identical. Cluster MA-RAE shows meaningful spread, confirming that cluster-based evaluation is sensitive to which structural groups are held out. The gap between strategies (~0.20 MA-RAE) dwarfs the within-strategy variance.

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

Mann-Whitney U tests confirm that cluster and random RAE distributions are entirely non-overlapping for all 9 endpoints (U = 25.0, p = 0.0079). The maximum U statistic (25 = 5 x 5, all cluster values exceed all random values) indicates complete separation. p = 0.0079 is the minimum achievable p-value for a two-sided test with n1 = n2 = 5.

### Interpretation

The results reveal a subtlety beyond "single splits are noisy." Random CV is *not* noisy — 5 repeats produce nearly identical metrics. The problem is that random CV gives **precise estimates of the wrong thing**: performance on structurally similar test molecules that leaked across fold boundaries.

Cluster-based CV is genuinely variable because different EKM + KMeans seeds create different structural boundaries. Some repeats hold out "easier" chemical groups (structurally closer to training data), others hold out "harder" groups. This variance is *informative* — it reflects real uncertainty about how the model will perform on novel chemistry.

The practical takeaway is twofold:
1. **Choosing the right splitting strategy matters far more than running more repeats.** Five random repeats with tight CIs give false confidence; five cluster repeats with wide CIs give an honest picture.
2. **Report confidence intervals from distance-aware splits.** A single cluster-split could report R²=0.30 or R²=0.58 for MGMB — only the mean across repeats (0.49) with its CI gives the full story.

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
