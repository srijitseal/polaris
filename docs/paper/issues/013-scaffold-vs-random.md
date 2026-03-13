# Naive scaffold splits are no harder than random splits — cluster-based splitting produces genuine distribution shift

## Summary

Naive Bemis-Murcko scaffold splits produce nearly identical XGBoost performance and test-to-train distance distributions as random splits across all 9 Expansion Tx endpoints. Cluster-based splitting (from 2.03) produces notably worse metrics and larger structural distances, confirming that it creates genuine distribution shift. This supports Greg Landrum's observation that scaffold splits are widely used but without careful considerations they're no better than random.

## Method

1. **Three splitting strategies** applied per endpoint (molecules with non-null values only):
   - **Scaffold split**: Compute generic Murcko scaffolds via RDKit, group molecules by scaffold, sort scaffolds by frequency (largest first), greedily assign each scaffold group to the fold with fewest molecules (5 folds)
   - **Random split**: 5-fold random assignment (seed=42)
   - **Cluster split**: Precomputed cluster-based folds from 2.03 (repeat 0, 5 folds)
2. **Train and evaluate**: Default XGBoost on ECFP4 + RDKit 2D descriptors (protonated at assay pH), log-transformed targets (except LogD)
3. **Competition metrics**: MAE, R², Spearman ρ, Kendall τ, RAE per fold per endpoint
4. **Distance characterization**: Pooled test-to-train 1-NN Tanimoto distances per strategy

## Key Findings

### Performance comparison (MA-RAE)

| Strategy | MA-RAE |
|----------|--------|
| Random | 0.479 |
| Scaffold | 0.536 |
| Cluster | 0.723 |

Scaffold and random splits produce similar metrics across all endpoints. Cluster-based splitting consistently produces worse performance — higher MAE, higher RAE, lower R², lower rank correlations — reflecting genuine distribution shift.

### Per-endpoint RAE (mean across folds)

| Endpoint | Scaffold | Random | Cluster |
|----------|----------|--------|---------|
| LogD | 0.399 | 0.358 | 0.560 |
| KSOL | 0.566 | 0.507 | 0.763 |
| HLM CLint | 0.662 | 0.609 | 0.843 |
| MLM CLint | 0.598 | 0.553 | 0.786 |
| Caco-2 Papp A>B | 0.517 | 0.467 | 0.757 |
| Caco-2 Efflux | 0.450 | 0.423 | 0.651 |
| MPPB | 0.573 | 0.490 | 0.756 |
| MBPB | 0.491 | 0.419 | 0.642 |
| MGMB | 0.572 | 0.481 | 0.745 |

Scaffold–random RAE differences are 0.04–0.09 (small). Cluster–random differences are 0.17–0.29 (large, consistent).

### Per-endpoint R² (mean across folds)

| Endpoint | Scaffold | Random | Cluster |
|----------|----------|--------|---------|
| LogD | 0.818 | 0.847 | 0.648 |
| KSOL | 0.565 | 0.641 | 0.327 |
| HLM CLint | 0.524 | 0.583 | 0.248 |
| MLM CLint | 0.613 | 0.663 | 0.316 |
| Caco-2 Papp A>B | 0.645 | 0.703 | 0.331 |
| Caco-2 Efflux | 0.718 | 0.750 | 0.493 |
| MPPB | 0.612 | 0.698 | 0.374 |
| MBPB | 0.680 | 0.745 | 0.504 |
| MGMB | 0.564 | 0.666 | 0.338 |

### Distance distributions

| Strategy | Median 1-NN | Mean 1-NN | Q75 | Q90 |
|----------|-------------|-----------|-----|-----|
| Random | 0.203 | 0.201 | 0.264 | 0.328 |
| Scaffold | 0.246 | 0.243 | 0.314 | 0.387 |
| Cluster | 0.419 | 0.427 | 0.521 | 0.613 |

Scaffold and random 1-NN distributions overlap substantially (medians 0.246 vs 0.203). The cluster-based split produces a right-shifted distribution with ~2× larger median distance (0.419), confirming genuine structural separation.

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

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.11-seal-scaffold-vs-random.py
```

## Source

`notebooks/2.11-seal-scaffold-vs-random.py`
