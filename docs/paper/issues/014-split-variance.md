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
| LogD | 0.358±0.002 | 0.009 | 0.570±0.023 | 0.061 |
| KSOL | 0.513±0.004 | 0.013 | 0.729±0.020 | 0.057 |
| HLM CLint | 0.604±0.005 | 0.018 | 0.873±0.031 | 0.081 |
| MLM CLint | 0.550±0.003 | 0.011 | 0.795±0.060 | 0.185 |
| Caco-2 Papp A>B | 0.473±0.004 | 0.014 | 0.712±0.029 | 0.082 |
| Caco-2 Efflux | 0.424±0.004 | 0.019 | 0.654±0.047 | 0.136 |
| MPPB | 0.493±0.007 | 0.021 | 0.757±0.034 | 0.101 |
| MBPB | 0.422±0.005 | 0.019 | 0.651±0.038 | 0.108 |
| MGMB | 0.483±0.012 | 0.055 | 0.691±0.051 | 0.133 |

Random splits show tight variance (range 0.009–0.055). Cluster splits show 3–17× wider ranges (0.057–0.185), with MLM CLint having the most extreme variance (RAE range 0.718–0.903).

### Performance variance across repeats — R²

| Endpoint | Random R² (mean±std) | Random range | Cluster R² (mean±std) | Cluster range |
|----------|---------------------|-------------|----------------------|--------------|
| LogD | 0.849±0.002 | 0.010 | 0.644±0.026 | 0.072 |
| KSOL | 0.632±0.005 | 0.018 | 0.357±0.022 | 0.064 |
| HLM CLint | 0.586±0.006 | 0.026 | 0.193±0.056 | 0.147 |
| MLM CLint | 0.663±0.004 | 0.016 | 0.320±0.118 | 0.345 |
| Caco-2 Papp A>B | 0.700±0.005 | 0.018 | 0.395±0.044 | 0.117 |
| Caco-2 Efflux | 0.746±0.006 | 0.028 | 0.483±0.064 | 0.187 |
| MPPB | 0.694±0.010 | 0.034 | 0.380±0.042 | 0.117 |
| MBPB | 0.748±0.009 | 0.040 | 0.498±0.043 | 0.122 |
| MGMB | 0.647±0.018 | 0.086 | 0.398±0.075 | 0.187 |

Cluster-split R² ranges are dramatic: MLM CLint spans 0.109–0.454 across 5 repeats — a single split could report either "poor" or "moderate" performance depending on the random seed.

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
2. **Report confidence intervals from distance-aware splits.** A single cluster-split could report R²=0.11 or R²=0.45 for MLM CLint — only the mean across repeats (0.32) with its CI gives the full story.

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
