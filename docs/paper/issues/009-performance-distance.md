# Performance degrades with distance from training data across all splitting strategies

## Summary

Training XGBoost (fixed architecture) on ECFP4 + full RDKit 2D descriptors (~200, with dimorphite_dl protonation at assay pH) across all 9 endpoints and 3 splitting strategies (cluster, time, target). Targets are log-transformed for non-LogD endpoints, matching the competition protocol. Evaluated with competition metrics: MAE, R², Spearman ρ, RAE, MA-RAE.

## Key Findings

### MA-RAE by splitting strategy

| Strategy | MA-RAE | Interpretation |
|----------|--------|----------------|
| Cluster | **0.726** | Standard 5-fold CV, ~80:20 per fold |
| Time | 0.837 | Expanding window (50:50 → 80:20), temporal ordering |
| Target | 6.927 | By design creates impossible extrapolation tasks |

### Per-endpoint metrics (cluster-split, mean across 5 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.502 | 0.659 | 0.824 | 0.557 |
| Caco-2 Efflux | 0.237 | 0.483 | 0.654 | 0.672 |
| MBPB | 0.246 | 0.453 | 0.696 | 0.700 |
| KSOL | 0.422 | 0.383 | 0.556 | 0.743 |
| MPPB | 0.279 | 0.377 | 0.632 | 0.771 |
| MLM CLint | 0.418 | 0.370 | 0.592 | 0.750 |
| Caco-2 Papp | 0.279 | 0.367 | 0.664 | 0.756 |
| MGMB | 0.219 | 0.363 | 0.607 | 0.755 |
| HLM CLint | 0.379 | 0.273 | 0.562 | 0.830 |

### Per-endpoint metrics (time-split, mean across 4 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.525 | 0.608 | 0.777 | 0.609 |
| MBPB | 0.262 | 0.348 | 0.664 | 0.790 |
| Caco-2 Papp | 0.282 | 0.290 | 0.595 | 0.789 |
| KSOL | 0.476 | 0.271 | 0.493 | 0.823 |
| HLM CLint | 0.388 | 0.256 | 0.548 | 0.850 |
| MPPB | 0.293 | 0.227 | 0.533 | 0.864 |
| Caco-2 Efflux | 0.288 | 0.198 | 0.590 | 0.788 |
| MLM CLint | 0.466 | 0.106 | 0.473 | 0.909 |
| MGMB | 0.260 | -0.156 | 0.506 | 1.107 |

### Structural distance degradation

| Endpoint | Cluster | Time | Target |
|----------|---------|------|--------|
| LogD | **2.89x** | 2.71x | 1.25x |
| MGMB | **2.89x** | 1.76x | 0.92x |
| HLM CLint | **2.75x** | 1.63x | 1.12x |
| MPPB | **2.33x** | 2.07x | 0.94x |
| MLM CLint | **2.21x** | 2.62x | 1.14x |
| KSOL | **1.94x** | 1.53x | 1.43x |
| MBPB | **1.93x** | 1.67x | 0.70x |
| Caco-2 Papp | **1.06x** | 1.00x | 1.04x |
| Caco-2 Efflux | 1.03x | 1.11x | 0.36x |

### Interpretation

- **Cluster-split** (MA-RAE 0.726) gives the best overall performance since each fold sees structurally diverse training data. R² ranges from 0.27 (HLM CLint) to 0.66 (LogD). Structural degradation is 2-3x for most endpoints.
- **Time-split** (MA-RAE 0.837) is worse because early folds have small training sets (expanding window starts at 50:50). R² drops below 0 for MGMB where training data is smallest (86 molecules at fold 0).
- **Target-split** (MA-RAE 6.927) produces catastrophically negative R² across all endpoints. This is by design — folds are ordered by target value, so test molecules systematically fall outside the training distribution in target space. Spearman correlations collapse to near-zero (~0.01–0.07), confirming the model cannot rank molecules it hasn't seen in the relevant value range.
- **Caco-2 endpoints show minimal structural degradation** (1.0–1.1x for cluster) — suggesting less structural diversity between clusters or better generalization for permeability.
- **Comparison to 2.08 baseline** (MA-RAE 0.820 on competition split): cluster-split CV gives a more optimistic estimate (0.726) than the held-out test set, which is expected — the competition split likely contains more structurally novel molecules.

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
