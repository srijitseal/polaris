# Performance degrades with distance from training data across all splitting strategies

## Summary

Training XGBoost (fixed architecture) on ECFP4 + full RDKit 2D descriptors (~200, with dimorphite_dl protonation at assay pH) across all 9 endpoints and 3 splitting strategies (cluster, time, target). Targets are log-transformed for non-LogD endpoints, matching the competition protocol. Evaluated with competition metrics: MAE, R², Spearman ρ, RAE, MA-RAE.

## Key Findings

### MA-RAE by splitting strategy

| Strategy | MA-RAE | Interpretation |
|----------|--------|----------------|
| Cluster | **0.732** | Standard 5-fold CV, ~80:20 per fold |
| Time | 0.839 | Expanding window (50:50 → 80:20), temporal ordering |
| Target | 6.909 | By design creates impossible extrapolation tasks |

### Per-endpoint metrics (cluster-split, mean across 5 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.482 | 0.680 | 0.828 | 0.542 |
| Caco-2 Efflux | 0.248 | 0.475 | 0.673 | 0.666 |
| MBPB | 0.242 | 0.482 | 0.702 | 0.695 |
| KSOL | 0.432 | 0.390 | 0.534 | 0.728 |
| Caco-2 Papp | 0.269 | 0.391 | 0.606 | 0.742 |
| MLM CLint | 0.413 | 0.387 | 0.623 | 0.753 |
| MPPB | 0.276 | 0.399 | 0.682 | 0.764 |
| MGMB | 0.229 | 0.194 | 0.619 | 0.853 |
| HLM CLint | 0.395 | 0.219 | 0.522 | 0.843 |

### Per-endpoint metrics (time-split, mean across 4 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.528 | 0.602 | 0.772 | 0.613 |
| MBPB | 0.262 | 0.349 | 0.665 | 0.790 |
| Caco-2 Papp | 0.288 | 0.264 | 0.575 | 0.804 |
| KSOL | 0.482 | 0.240 | 0.487 | 0.836 |
| HLM CLint | 0.387 | 0.255 | 0.541 | 0.849 |
| MPPB | 0.294 | 0.224 | 0.532 | 0.866 |
| Caco-2 Efflux | 0.284 | 0.221 | 0.589 | 0.779 |
| MLM CLint | 0.468 | 0.093 | 0.460 | 0.913 |
| MGMB | 0.259 | -0.149 | 0.511 | 1.101 |

### Structural distance degradation

| Endpoint | Cluster | Time | Target |
|----------|---------|------|--------|
| LogD | **3.98x** | 2.44x | 1.18x |
| HLM CLint | **3.19x** | 1.88x | 1.09x |
| MLM CLint | **1.95x** | 2.54x | 1.11x |
| KSOL | **1.90x** | 1.52x | 1.42x |
| MBPB | **1.83x** | 1.54x | 0.75x |
| MPPB | **1.71x** | 2.06x | 0.95x |
| MGMB | **1.51x** | 1.72x | 0.91x |
| Caco-2 Papp | **1.13x** | 0.80x | 1.12x |
| Caco-2 Efflux | 1.05x | 1.85x | 0.35x |

### Interpretation

- **Cluster-split** (MA-RAE 0.732) gives the best overall performance since each fold sees structurally diverse training data. R² ranges from 0.19 (MGMB) to 0.68 (LogD). Structural degradation is 2-4x for LogD and HLM CLint, 1.5-2x for most others.
- **Time-split** (MA-RAE 0.839) is worse because early folds have small training sets (expanding window starts at 50:50). R² drops below 0 for MGMB where training data is smallest (86 molecules at fold 0).
- **Target-split** (MA-RAE 6.909) produces catastrophically negative R² across all endpoints. This is by design — folds are ordered by target value, so test molecules systematically fall outside the training distribution in target space. Spearman correlations collapse to near-zero (~0.01–0.07), confirming the model cannot rank molecules it hasn't seen in the relevant value range.
- **Caco-2 endpoints show minimal structural degradation** (1.1x for cluster) — suggesting less structural diversity between clusters or better generalization for permeability.
- **Comparison to 2.08 baseline** (MA-RAE 0.821 on competition split): cluster-split CV gives a more optimistic estimate (0.732) than the held-out test set, which is expected — the competition split likely contains more structurally novel molecules.

## Model

- XGBRegressor: fixed config (n_estimators=1000, max_depth=6, lr=0.1, subsample=0.8, colsample_bytree=0.4, min_child_weight=5, gamma=1.0, reg_alpha=0.1, reg_lambda=1.5, tree_method="hist")
- Features: 2048-bit ECFP4 + ~192 RDKit 2D descriptors (scaled), computed on dimorphite_dl-protonated SMILES at assay pH (7.4 for most, 6.5 for Caco-2)
- Targets: log10(clip(x, 1e-10) + 1) for all endpoints except LogD

## Plots

- `data/processed/2.07-seal-performance-distance/overall_r2.png` — R² by endpoint and strategy
<!-- Paste: overall_r2.png -->
- `data/processed/2.07-seal-performance-distance/overall_mae.png` — MAE by endpoint and strategy
<!-- Paste: overall_mae.png -->
- `data/processed/2.07-seal-performance-distance/overall_spearman_r.png` — Spearman ρ by endpoint and strategy
<!-- Paste: overall_spearman_r.png -->
- `data/processed/2.07-seal-performance-distance/overall_rae.png` — RAE by endpoint and strategy
<!-- Paste: overall_rae.png -->
- `data/processed/2.07-seal-performance-distance/performance_over_distance.png` — RMSE vs structural distance
<!-- Paste: performance_over_distance.png -->
- `data/processed/2.07-seal-performance-distance/performance_over_target_distance.png` — RMSE vs target-space distance
<!-- Paste: performance_over_target_distance.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py
```

## Source

`notebooks/2.07-seal-performance-distance.py`
