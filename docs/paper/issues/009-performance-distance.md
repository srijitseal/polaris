# Performance degrades with distance from training data across all splitting strategies

## Summary

Training XGBoost (Optuna TPE-tuned, per [ADR-002](../../decisions/002-optuna-hyperparameter-tuning.md)) on ECFP4 + full RDKit 2D descriptors (~200, with dimorphite_dl protonation at assay pH) across all 9 endpoints and 3 splitting strategies (cluster, time, target). Targets are log-transformed for non-LogD endpoints, matching the competition protocol. Evaluated with competition metrics: MAE, R², Spearman ρ, RAE, MA-RAE.

## Key Findings

### MA-RAE by splitting strategy

| Strategy | MA-RAE | Interpretation |
|----------|--------|----------------|
| Cluster | **0.673** | Standard 5-fold CV, ~80:20 per fold |
| Time | 0.771 | Expanding window (50:50 → 80:20), temporal ordering |
| Target | 6.118 | By design creates impossible extrapolation tasks |

### Per-endpoint metrics (cluster-split, mean across 5 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.448 | 0.716 | 0.844 | 0.504 |
| MBPB | 0.205 | 0.598 | 0.764 | 0.593 |
| MPPB | 0.235 | 0.541 | 0.740 | 0.651 |
| Caco-2 Efflux | 0.229 | 0.534 | 0.712 | 0.617 |
| Caco-2 Papp | 0.249 | 0.454 | 0.657 | 0.686 |
| KSOL | 0.419 | 0.427 | 0.564 | 0.705 |
| MLM CLint | 0.405 | 0.405 | 0.644 | 0.742 |
| MGMB | 0.195 | 0.357 | 0.703 | 0.729 |
| HLM CLint | 0.387 | 0.251 | 0.554 | 0.826 |

### Per-endpoint metrics (time-split, mean across 4 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.499 | 0.638 | 0.796 | 0.580 |
| MBPB | 0.222 | 0.486 | 0.732 | 0.672 |
| Caco-2 Papp | 0.265 | 0.359 | 0.636 | 0.740 |
| MPPB | 0.267 | 0.341 | 0.610 | 0.785 |
| HLM CLint | 0.374 | 0.304 | 0.581 | 0.820 |
| Caco-2 Efflux | 0.274 | 0.279 | 0.632 | 0.750 |
| KSOL | 0.464 | 0.278 | 0.519 | 0.808 |
| MGMB | 0.220 | 0.176 | 0.563 | 0.906 |
| MLM CLint | 0.450 | 0.171 | 0.506 | 0.879 |

### Structural distance degradation

| Endpoint | Cluster | Time | Target |
|----------|---------|------|--------|
| LogD | **3.60x** | 2.51x | 1.25x |
| HLM CLint | **3.04x** | 1.50x | 1.12x |
| MPPB | **2.14x** | 2.09x | 1.09x |
| MLM CLint | **2.08x** | 2.44x | 1.23x |
| KSOL | **1.98x** | 1.83x | 1.44x |
| MBPB | **1.90x** | 1.76x | 1.09x |
| MGMB | **1.87x** | 2.38x | 0.83x |
| Caco-2 Papp | 0.98x | 0.86x | 1.21x |
| Caco-2 Efflux | 0.95x | 2.03x | 0.40x |

### Interpretation

- **Cluster-split** (MA-RAE 0.673) gives the best overall performance since each fold sees structurally diverse training data. R² ranges from 0.25 (HLM CLint) to 0.72 (LogD). Structural degradation is 2-4x for LogD and HLM CLint, 1.9-2.1x for most others.
- **Time-split** (MA-RAE 0.771) is worse because early folds have small training sets (expanding window starts at 50:50). R² drops below 0 for MGMB where training data is smallest (86 molecules at fold 0).
- **Target-split** (MA-RAE 6.118) produces catastrophically negative R² across all endpoints. This is by design — folds are ordered by target value, so test molecules systematically fall outside the training distribution in target space. Spearman correlations collapse to near-zero (~0.01-0.07), confirming the model cannot rank molecules it hasn't seen in the relevant value range.
- **Caco-2 endpoints show minimal structural degradation** (0.95-0.98x for cluster) — suggesting less structural diversity between clusters or better generalization for permeability.
- **Tuning impact**: Optuna TPE improved cluster MA-RAE from 0.732 to 0.673 (-8.1%), with the largest R² gains in MPPB (+0.14) and MBPB (+0.12). Two endpoints (MGMB, HLM CLint) showed marginal R² decreases, likely due to high variance from small dataset sizes.

## Model

- XGBRegressor: **Optuna TPE** (30 trials, 3-fold CV, MAE scoring) — 9 hyperparameters tuned per endpoint per fold (see [ADR-002](../../decisions/002-optuna-hyperparameter-tuning.md))
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Cached in `data/interim/optuna_cache/`
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
