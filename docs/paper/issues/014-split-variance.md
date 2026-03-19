# Random splits give false confidence — splitting strategy matters more than number of repeats

## Summary

Random 5-fold CV produces remarkably tight variance (RAE std 0.002–0.012) but at an optimistic level (mean RAE ~0.48). Cluster-based CV reveals the true deployment-relevant uncertainty (RAE std 0.020–0.060, range up to 0.185) at a harder level (mean RAE ~0.71). The key insight is not "single splits are noisy" — random CV is actually very stable — but that random CV gives **precise estimates of the wrong thing**. The splitting strategy matters far more than the number of repeats: distance-aware splits with repeated CV provide honest performance estimates with calibrated confidence intervals.

## Method

1. **Random splits**: Generate 20 independent 5-fold CV assignments (seeds 0–19) per endpoint
2. **Cluster splits**: Use all 5 precomputed repeats from `cluster_cv_folds.parquet` (stochastic EKM + MiniBatchKMeans)
3. **Train and evaluate**: Default XGBoost on ECFP4 + RDKit 2D descriptors per repeat × endpoint × fold (~1,125 models total)
4. **Aggregate**: Per-repeat mean metrics (across 5 folds), then compute variance statistics across repeats

## Key Findings

### Performance variance across repeats — RAE

| Endpoint | Random RAE (mean±std) | Random range | Cluster RAE (mean±std) | Cluster range |
|----------|----------------------|-------------|----------------------|--------------|
| LogD | 0.359±0.003 | — | 0.566±0.016 | — |
| KSOL | 0.512±0.004 | — | 0.741±0.012 | — |
| HLM CLint | 0.607±0.005 | — | 0.869±0.034 | — |
| MLM CLint | 0.552±0.004 | — | 0.786±0.024 | — |
| Caco-2 Papp A>B | 0.475±0.003 | — | 0.708±0.034 | — |
| Caco-2 Efflux | 0.425±0.005 | — | 0.686±0.049 | — |
| MPPB | 0.492±0.007 | — | 0.731±0.024 | — |
| MBPB | 0.424±0.007 | — | 0.670±0.016 | — |
| MGMB | 0.481±0.011 | — | 0.685±0.083 | — |

Random splits show tight variance (range 0.009–0.055). Cluster splits show 3–17× wider ranges (0.057–0.185), with MLM CLint having the most extreme variance (RAE range 0.718–0.903).

### Performance variance across repeats — R²

| Endpoint | Random R² (mean±std) | Random range | Cluster R² (mean±std) | Cluster range |
|----------|---------------------|-------------|----------------------|--------------|
| LogD | 0.849±0.003 | 0.013 | 0.648±0.020 | 0.057 |
| KSOL | 0.634±0.005 | 0.021 | 0.356±0.014 | 0.044 |
| HLM CLint | 0.584±0.007 | 0.025 | 0.191±0.058 | 0.164 |
| MLM CLint | 0.661±0.005 | 0.020 | 0.339±0.045 | 0.139 |
| Caco-2 Papp A>B | 0.698±0.003 | 0.015 | 0.401±0.041 | 0.126 |
| Caco-2 Efflux | 0.746±0.008 | 0.034 | 0.423±0.065 | 0.179 |
| MPPB | 0.696±0.008 | 0.031 | 0.409±0.038 | 0.098 |
| MBPB | 0.747±0.010 | 0.033 | 0.467±0.028 | 0.084 |
| MGMB | 0.646±0.017 | 0.083 | 0.377±0.142 | 0.414 |

Cluster-split R² ranges are dramatic: MGMB spans 0.117–0.531 across 5 repeats — a single split could report either "poor" or "moderate" performance depending on the random seed.

### MA-RAE distribution

| Strategy | Mean MA-RAE | Std | Range |
|----------|-------------|-----|-------|
| Random (20 repeats) | ~0.48 | ~0.003 | ~0.01 |
| Cluster (5 repeats) | ~0.71 | ~0.03 | ~0.08 |

Random MA-RAE is tightly clustered — 20 repeats look almost identical. Cluster MA-RAE shows meaningful spread, confirming that cluster-based evaluation is sensitive to which structural groups are held out. The gap between strategies (~0.23 MA-RAE) dwarfs the within-strategy variance.

### Interpretation

The results reveal a subtlety beyond "single splits are noisy." Random CV is *not* noisy — 20 repeats produce nearly identical metrics. The problem is that random CV gives **precise estimates of the wrong thing**: performance on structurally similar test molecules that leaked across fold boundaries.

Cluster-based CV is genuinely variable because different EKM + KMeans seeds create different structural boundaries. Some repeats hold out "easier" chemical groups (structurally closer to training data), others hold out "harder" groups. This variance is *informative* — it reflects real uncertainty about how the model will perform on novel chemistry.

The practical takeaway is twofold:
1. **Choosing the right splitting strategy matters far more than running more repeats.** Twenty random repeats with tight CIs give false confidence; five cluster repeats with wide CIs give an honest picture.
2. **Report confidence intervals from distance-aware splits.** A single cluster-split could report R²=0.12 or R²=0.53 for MGMB — only the mean across repeats (0.38) with its CI gives the full story.

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
