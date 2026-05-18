# Performance degrades with distance from training data across all splitting strategies

## Summary

Training XGBoost (Optuna TPE-tuned, per [ADR-002](../../decisions/002-optuna-hyperparameter-tuning.md)) on ECFP4 + full RDKit 2D descriptors (~200, with dimorphite_dl protonation at assay pH) across all 9 endpoints and 3 splitting strategies (cluster, time, target). Targets are log-transformed for non-LogD endpoints, matching the competition protocol. Evaluated with competition metrics: MAE, R², Spearman ρ, RAE, MA-RAE.

## Key Findings

### MA-RAE by splitting strategy

| Strategy | MA-RAE | Interpretation |
|----------|--------|----------------|
| Cluster | **0.673** | Standard 5-fold CV, ~80:20 per fold |
| Time | 0.772 | Expanding window (50:50 → 80:20), temporal ordering |
| Target | 6.106 | By design creates impossible extrapolation tasks |

### Per-endpoint metrics (cluster-split, mean across 5 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.446 | 0.722 | 0.847 | 0.502 |
| MBPB | 0.206 | 0.595 | 0.755 | 0.599 |
| Caco-2 Efflux | 0.227 | 0.545 | 0.717 | 0.611 |
| MPPB | 0.236 | 0.541 | 0.739 | 0.653 |
| Caco-2 Papp | 0.251 | 0.444 | 0.649 | 0.690 |
| KSOL | 0.418 | 0.437 | 0.563 | 0.703 |
| MGMB | 0.193 | 0.355 | 0.680 | 0.720 |
| MLM CLint | 0.407 | 0.401 | 0.644 | 0.745 |
| HLM CLint | 0.389 | 0.237 | 0.544 | 0.831 |

### Per-endpoint metrics (time-split, mean across 4 folds)

| Endpoint | MAE | R² | Spearman ρ | RAE |
|----------|-----|-----|-----------|-----|
| LogD | 0.503 | 0.636 | 0.794 | 0.584 |
| MBPB | 0.222 | 0.486 | 0.731 | 0.670 |
| Caco-2 Papp | 0.264 | 0.366 | 0.631 | 0.738 |
| Caco-2 Efflux | 0.273 | 0.275 | 0.633 | 0.750 |
| MPPB | 0.265 | 0.351 | 0.620 | 0.778 |
| KSOL | 0.459 | 0.300 | 0.530 | 0.798 |
| HLM CLint | 0.373 | 0.309 | 0.587 | 0.818 |
| MLM CLint | 0.455 | 0.155 | 0.501 | 0.888 |
| MGMB | 0.225 | 0.170 | 0.573 | 0.924 |

### Structural distance degradation (RMSE farthest / closest bin)

| Endpoint | Cluster | Time | Target |
|----------|---------|------|--------|
| LogD | **3.60x** | 2.59x | 1.26x |
| HLM CLint | **3.04x** | 1.56x | 1.20x |
| MPPB | **2.14x** | 2.20x | 1.10x |
| MLM CLint | **2.08x** | 2.53x | 1.22x |
| KSOL | **2.01x** | 1.81x | 1.45x |
| MGMB | **1.87x** | 2.53x | 0.83x |
| MBPB | **1.90x** | 1.92x | 1.08x |
| Caco-2 Efflux | 0.95x | 2.19x | 0.40x |
| Caco-2 Papp | 0.98x | 0.92x | 1.23x |

### Interpretation

- **Cluster-split** (MA-RAE 0.673) gives the best overall performance since each fold sees structurally diverse training data. R² ranges from 0.24 (HLM CLint) to 0.72 (LogD). Structural degradation is 3–3.6× for LogD and HLM CLint, 1.9–2.1× for most others.
- **Time-split** (MA-RAE 0.772) is worse because early folds have small training sets (expanding window). R² is lowest for MLM CLint (0.155) and MGMB (0.170).
- **Target-split** (MA-RAE 6.106) produces catastrophically negative R² across all endpoints (mean R² ≈ −81; KSOL R² ≈ −590 due to outlier folds). This is by design — folds are ordered by target value, so test molecules systematically fall outside the training distribution. Spearman correlations collapse to near-zero (mean ~0.08), confirming the model cannot rank molecules it hasn't seen in the relevant value range.
- **Caco-2 endpoints show minimal cluster-split degradation under XGBoost** (Papp 0.96×, Efflux 0.97×) — note this is a representation effect: under CheMeleon (see appendix below) Caco-2 Papp degrades 2.91× and Efflux 1.60×, so the flat curve is not a property of the endpoints themselves but of the ECFP4 + RDKit 2D feature mix.

## Model

- XGBRegressor: **Optuna TPE** (30 trials, 3-fold CV, MAE scoring) — 9 hyperparameters tuned per endpoint per fold (see [ADR-002](../../decisions/002-optuna-hyperparameter-tuning.md))
- Parameters tuned: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Cached in `data/interim/optuna_cache/`
- Features: 2048-bit ECFP4 + ~192 RDKit 2D descriptors (scaled), computed on dimorphite_dl-protonated SMILES at assay pH (7.4 for most, 6.5 for Caco-2)
- Targets: log10(clip(x, 1e-10) + 1) for all endpoints except LogD

## Plots

- `data/processed/2.07-seal-performance-distance/xgboost/overall_r2.png` — R² by endpoint and strategy
- `data/processed/2.07-seal-performance-distance/xgboost/overall_mae.png` — MAE by endpoint and strategy
- `data/processed/2.07-seal-performance-distance/xgboost/overall_spearman_r.png` — Spearman ρ by endpoint and strategy
- `data/processed/2.07-seal-performance-distance/xgboost/overall_rae.png` — RAE by endpoint and strategy
- `data/processed/2.07-seal-performance-distance/xgboost/performance_over_distance.png` — RMSE vs structural distance
- `data/processed/2.07-seal-performance-distance/xgboost/performance_over_target_distance.png` — RMSE vs target-space distance
- Combined XGBoost vs CheMeleon panels: `data/processed/2.07-seal-performance-distance/combined/`

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py
```

## Source

`notebooks/2.07-seal-performance-distance.py`

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran the three splitting strategies (cluster, time, target) with CheMeleon (pre-trained BondMessagePassing weights, 2×1024 FFN, fine-tuned per fold). **Trends match XGBoost — cluster < time ≪ target ranking preserved; cluster-split structural degradation persists across endpoints.**

### MA-RAE by strategy

| Strategy | XGBoost | CheMeleon | Δ |
|----------|---------|-----------|---|
| Cluster | 0.673 | **0.625** | −0.048 |
| Time | 0.772 | **0.728** | −0.044 |
| Target | 6.106 | 5.259 | −0.847 |

### Cluster-split per-endpoint (mean across 5 folds)

| Endpoint | R² XGB | R² CM | RAE XGB | RAE CM |
|----------|--------|-------|---------|--------|
| LogD | 0.722 | 0.830 | 0.502 | 0.376 |
| MBPB | 0.595 | 0.681 | 0.599 | 0.502 |
| MPPB | 0.541 | 0.589 | 0.653 | 0.597 |
| Caco-2 Efflux | 0.545 | 0.520 | 0.611 | 0.612 |
| Caco-2 Papp | 0.444 | 0.260 | 0.690 | 0.776 |
| KSOL | 0.437 | 0.457 | 0.703 | 0.600 |
| MGMB | 0.355 | 0.408 | 0.720 | 0.646 |
| MLM CLint | 0.401 | 0.427 | 0.745 | 0.726 |
| HLM CLint | 0.237 | 0.297 | 0.831 | 0.793 |

### Structural degradation (RMSE farthest / RMSE closest bin)

| Endpoint | XGB | CM |
|----------|-----|-----|
| LogD | 3.60× | 3.54× |
| HLM CLint | 3.04× | **4.30×** |
| MPPB | 2.14× | 2.63× |
| MLM CLint | 2.08× | 1.92× |
| KSOL | 2.01× | 1.34× |
| MBPB | 1.90× | 2.54× |
| MGMB | 1.87× | 2.32× |
| Caco-2 Papp | 0.98× | **2.91×** |
| Caco-2 Efflux | 0.95× | **1.60×** |

### Trend match — important nuance

The XGBoost write-up explained Caco-2 robustness (0.95–0.98×) as *"passive permeability is governed by global properties (size, lipophilicity, H-bonding) captured by RDKit 2D descriptors."* **That mechanistic claim does not survive under CheMeleon** — Caco-2 Papp degrades 2.91× and Caco-2 Efflux 1.60× when the model has no access to the global descriptors. The flat XGBoost curve was a feature of the input representation, not a property of the endpoints. For all other endpoints the qualitative picture is unchanged: 1.3–4.3× degradation under cluster-split, and the worst-degraded endpoints (LogD, HLM CLint) remain the worst-degraded under CheMeleon.

### Source

- `data/processed/2.07-seal-performance-distance/chemeleon/per_fold_performance.csv`
- `data/processed/2.07-seal-performance-distance/combined/metrics_comparison.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/10#issuecomment-4376596586
- Reproduce: `pixi run -e cheminformatics python notebooks/2.07-seal-performance-distance.py --model chemeleon`
