# Naive scaffold splits are no harder than random splits — cluster-based splitting produces genuine distribution shift

## Summary

Naive Bemis-Murcko scaffold splits produce nearly identical XGBoost performance and test-to-train distance distributions as random splits across all 9 Expansion Tx endpoints. Cluster-based splitting (from 2.03) produces notably worse metrics and larger structural distances, confirming that it creates genuine distribution shift. This supports Greg Landrum's observation that scaffold splits are widely used but without careful considerations they're no better than random.

## Method

1. **Three splitting strategies** applied per endpoint (molecules with non-null values only):
   - **Scaffold split**: Compute generic Murcko scaffolds via RDKit, group molecules by scaffold, sort scaffolds by frequency (largest first), greedily assign each scaffold group to the fold with fewest molecules (5 folds)
   - **Random split**: 5-fold random assignment (seed=42)
   - **Cluster split**: Precomputed cluster-based folds from 2.03 (repeat 0, 5 folds)
2. **Train and evaluate**: Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters per endpoint per fold; see ADR-002) on ECFP4 + RDKit 2D descriptors (protonated at assay pH), log-transformed targets (except LogD)
3. **Competition metrics**: MAE, R², Spearman ρ, Kendall τ, RAE per fold per endpoint
4. **Distance characterization**: Pooled test-to-train 1-NN Tanimoto distances per strategy

## Key Findings

### Performance comparison (MA-RAE)

| Strategy | MA-RAE |
|----------|--------|
| Random | 0.474 |
| Scaffold | 0.510 |
| Cluster | 0.672 |

Scaffold and random splits produce similar metrics across all endpoints. Cluster-based splitting consistently produces worse performance — higher MAE, higher RAE, lower R², lower rank correlations — reflecting genuine distribution shift.

### Per-endpoint RAE (mean across folds)

| Endpoint | Scaffold | Random | Cluster |
|----------|----------|--------|---------|
| LogD | 0.376 | 0.350 | 0.504 |
| KSOL | 0.557 | 0.529 | 0.705 |
| HLM CLint | 0.640 | 0.589 | 0.827 |
| MLM CLint | 0.584 | 0.548 | 0.739 |
| Caco-2 Papp A>B | 0.522 | 0.492 | 0.686 |
| Caco-2 Efflux | 0.455 | 0.436 | 0.614 |
| MPPB | 0.515 | 0.463 | 0.650 |
| MBPB | 0.436 | 0.399 | 0.594 |
| MGMB | 0.502 | 0.456 | 0.726 |

Scaffold–random RAE differences are 0.02–0.06 (small). Cluster–random differences are 0.13–0.27 (large, consistent).

### Per-endpoint R² (mean across folds)

| Endpoint | Scaffold | Random | Cluster |
|----------|----------|--------|---------|
| LogD | 0.841 | 0.859 | 0.719 |
| KSOL | 0.603 | 0.637 | 0.429 |
| HLM CLint | 0.552 | 0.608 | 0.254 |
| MLM CLint | 0.634 | 0.673 | 0.409 |
| Caco-2 Papp A>B | 0.662 | 0.692 | 0.456 |
| Caco-2 Efflux | 0.727 | 0.746 | 0.537 |
| MPPB | 0.686 | 0.740 | 0.545 |
| MBPB | 0.749 | 0.783 | 0.595 |
| MGMB | 0.649 | 0.705 | 0.347 |

### Distance distributions

| Strategy | Median 1-NN | Mean 1-NN | Q75 | Q90 |
|----------|-------------|-----------|-----|-----|
| Random | 0.203 | 0.210 | 0.264 | 0.328 |
| Scaffold | 0.246 | 0.248 | 0.314 | 0.387 |
| Cluster | 0.424 | 0.431 | 0.524 | 0.619 |

Scaffold and random 1-NN distributions overlap substantially (medians 0.246 vs 0.203). The cluster-based split produces a right-shifted distribution with ~2× larger median distance (0.424), confirming genuine structural separation.

### Statistical tests (KS)

| Comparison | KS statistic D | p-value | n_1 | n_2 |
|---|---|---|---|---|
| scaffold vs random | 0.2028 | < 10^-10 | 36,003 | 36,003 |
| cluster vs random | 0.6617 | < 10^-10 | 36,003 | 36,003 |
| cluster vs scaffold | 0.5356 | < 10^-10 | 36,003 | 36,003 |

Two-sample Kolmogorov-Smirnov tests on pooled 1-NN distance arrays confirm that while the scaffold vs random difference is statistically significant (D = 0.203), it is modest compared to the cluster vs random difference (D = 0.662). All three comparisons are highly significant (p ≈ 0), but the effect sizes tell the real story: scaffold splits create only a small shift from random, while cluster-based splits create a qualitatively different distribution.

### Interpretation

The naive scaffold split fails to create distribution shift because:
1. Many scaffolds appear only once — these are randomly scattered across folds regardless of strategy
2. Large scaffold groups keep structurally identical molecules together, but structurally *similar* molecules with different scaffolds still leak across folds
3. The greedy balancing ensures roughly equal fold sizes but not structural separation

The cluster-based approach succeeds because it groups molecules by fingerprint similarity in kernel space, ensuring that test molecules are genuinely distant from training molecules.

**Practical takeaway**: Scaffold splits are a poor proxy for real-world deployment conditions. Researchers should use distance-aware splitting (cluster-based, scaffold with distance constraints, or similar) to honestly evaluate model generalization.

## Plots

- `data/processed/2.11-seal-scaffold-vs-random/metric_comparison.png` — Combined metric comparison (main Fig S1)
<!-- Paste: metric_comparison.png -->
- `data/processed/2.11-seal-scaffold-vs-random/distance_distributions.png` — Overlaid 1-NN distance histograms
<!-- Paste: distance_distributions.png -->
- `data/processed/2.11-seal-scaffold-vs-random/strategy_summary.png` — MA-RAE by strategy
<!-- Paste: strategy_summary.png -->
- `data/processed/2.11-seal-scaffold-vs-random/metric_mae.png` — MAE by endpoint
- `data/processed/2.11-seal-scaffold-vs-random/metric_r2.png` — R² by endpoint
- `data/processed/2.11-seal-scaffold-vs-random/metric_rae.png` — RAE by endpoint
- `data/processed/2.11-seal-scaffold-vs-random/metric_spearman_r.png` — Spearman ρ by endpoint
- `data/processed/2.11-seal-scaffold-vs-random/metric_kendall_tau.png` — Kendall τ by endpoint
- `data/processed/2.11-seal-scaffold-vs-random/scaffold_group_sizes.png` — Scaffold group size distribution

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.11-seal-scaffold-vs-random.py
```

## Source

`notebooks/2.11-seal-scaffold-vs-random.py`
