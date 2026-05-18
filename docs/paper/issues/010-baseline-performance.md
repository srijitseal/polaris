# XGBoost + ECFP4 + RDKit descriptors baseline achieves MA-RAE 0.776 on competition split

## Summary

Training XGBoost on ECFP4 fingerprints + RDKit 2D descriptors with per-endpoint hyperparameter tuning (Optuna TPE, 30 trials, 3-fold CV) on the original competition train/test split (5,326 / 2,282). Molecules are protonated at assay-relevant pH (7.4 for most, 6.5 for Caco-2) using dimorphite_dl before feature computation. Targets are log-transformed (`log10(clip(x, 1e-10) + 1)`) for all endpoints except LogD, matching the competition evaluation protocol. See ADR-002 (`docs/decisions/002-optuna-hyperparameter-tuning.md`) for the tuning protocol.

## Key Findings

### Per-endpoint results (XGBoost + Optuna TPE + full RDKit 2D + protonation)

| Endpoint | pH | R² | R² 95% CI | Spearman ρ | Spearman 95% CI | RAE | RAE 95% CI | n_train | n_test |
|----------|-----|-----|-----------|-----------|-----------------|-----|------------|---------|--------|
| LogD | 7.4 | 0.592 | [0.561, 0.621] | 0.758 | [0.737, 0.777] | 0.628 | [0.606, 0.650] | 5,039 | 2,270 |
| KSOL | 7.4 | 0.295 | [0.236, 0.344] | 0.500 | [0.467, 0.530] | 0.826 | [0.802, 0.852] | 5,128 | 2,170 |
| HLM CLint | 7.4 | 0.231 | [0.149, 0.310] | 0.562 | [0.509, 0.612] | 0.855 | [0.806, 0.903] | 3,759 | 782 |
| MLM CLint | 7.4 | 0.189 | [0.116, 0.258] | 0.464 | [0.420, 0.507] | 0.920 | [0.880, 0.960] | 4,522 | 1,170 |
| Caco-2 Papp A>B | 6.5 | 0.192 | [0.142, 0.243] | 0.512 | [0.475, 0.549] | 0.859 | [0.831, 0.888] | 2,157 | 1,616 |
| Caco-2 Efflux | 6.5 | 0.186 | [0.142, 0.231] | 0.645 | [0.616, 0.677] | 0.775 | [0.740, 0.810] | 2,161 | 1,616 |
| MPPB | 7.4 | 0.223 | [0.100, 0.326] | 0.616 | [0.549, 0.673] | 0.879 | [0.819, 0.938] | 1,302 | 454 |
| MBPB | 7.4 | 0.570 | [0.509, 0.625] | 0.778 | [0.731, 0.817] | 0.582 | [0.538, 0.625] | 975 | 451 |
| MGMB | 7.4 | 0.452 | [0.353, 0.546] | 0.687 | [0.593, 0.766] | 0.659 | [0.589, 0.733] | 222 | 209 |

**MA-RAE = 0.776** across the 9 endpoints.

### Key insight: log-transform is essential

The competition evaluates on log-transformed values for all endpoints except LogD. Training on raw-scale targets and only log-transforming at evaluation time produces poor R2 because the model optimizes for raw-scale MSE, which is dominated by high-value outliers. Training on log-scale targets aligns the model's loss function with the evaluation metric.

### Optuna TPE hyperparameter characteristics

Per-endpoint best parameters are saved in `data/processed/2.08-seal-baseline-performance/xgboost/best_params.csv`. Tuned values typically fall in: lower learning rates, shallower trees, and higher n_estimators — consistent with a regularized ensemble that generalizes better to the test set than untuned defaults. (Earlier comparisons against HalvingRandomSearchCV and a 16-descriptor variant were exploratory; current artifacts only retain the final Optuna + full RDKit 2D + dimorphite_dl configuration.)

### Bootstrap confidence intervals

To quantify uncertainty from the single train/test split, we computed 95% bootstrap confidence intervals using 1,000 resamples of the test-set predictions. Each resample draws n_test samples with replacement from the (y_true, y_pred) pairs and recomputes all metrics (R², MAE, Spearman, RAE). The 2.5th and 97.5th percentiles define the 95% CI.

CI widths vary substantially with test-set size. Endpoints with large test sets have tight CIs: LogD (n=2,270, R² width 0.060), Caco-2 endpoints (n=1,616, widths ~0.10), and KSOL (n=2,170, width 0.108). Endpoints with small test sets have wide CIs: MPPB (n=454, R² width 0.226), MGMB (n=209, width 0.193), and HLM CLint (n=782, width 0.161). Wide CIs on MPPB and MGMB mean point estimates are less reliable — a 0.05 R² improvement on these endpoints is not statistically meaningful on this split.

## Models

- XGBRegressor: `tree_method="hist"`, per-endpoint tuning via Optuna TPE (30 trials, 3-fold CV, scoring=neg_MAE, 9 hyperparameters)
- Tuned hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Features: 2048-bit ECFP4 + full RDKit 2D descriptor suite (~200, scaled), computed on dimorphite_dl-protonated SMILES at assay pH
- Protonation: dimorphite_dl with pH +/- 0.5 window, max 1 variant per molecule
- Caching: Optuna results cached in `data/interim/optuna_cache/` keyed by endpoint + split

## Plots

- `data/processed/2.08-seal-baseline-performance/xgboost/r2_by_endpoint.png` — R² bar chart
- `data/processed/2.08-seal-baseline-performance/xgboost/mae_by_endpoint.png` — MAE bar chart
- `data/processed/2.08-seal-baseline-performance/xgboost/spearman_by_endpoint.png` — Spearman ρ bar chart
- `data/processed/2.08-seal-baseline-performance/xgboost/scatter_predictions.png` — 3×3 scatter grid
- Combined XGBoost vs CheMeleon panels: `data/processed/2.08-seal-baseline-performance/combined/`

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py
```

## Source

`notebooks/2.08-seal-baseline-performance.py`

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran the baseline with CheMeleon (pre-trained BondMessagePassing weights, 2×1024 FFN, fine-tuned per endpoint, single model — no ensemble). Same competition split, same protonation, same log-transform protocol. **Trend matches XGBoost; CheMeleon is strictly better on 8/9 endpoints.**

### MA-RAE

| Model | MA-RAE |
|-------|--------|
| XGBoost (Optuna TPE, ECFP4 + RDKit 2D) | 0.776 |
| **CheMeleon (fine-tuned)** | **0.684** |

Δ = −0.092 (−11.9% relative).

### Per-endpoint R² and RAE

| Endpoint | R² XGB | R² CM | ΔR² | RAE XGB | RAE CM | ΔRAE |
|----------|--------|-------|-----|---------|--------|------|
| LogD | 0.592 | 0.720 | +0.128 | 0.628 | 0.489 | −0.138 |
| KSOL | 0.295 | 0.462 | +0.166 | 0.826 | 0.593 | −0.233 |
| HLM CLint | 0.231 | 0.328 | +0.096 | 0.855 | 0.799 | −0.056 |
| MLM CLint | 0.189 | 0.170 | −0.019 | 0.920 | 0.936 | +0.016 |
| Caco-2 Papp A>B | 0.192 | 0.206 | +0.014 | 0.859 | 0.830 | −0.029 |
| Caco-2 Efflux | 0.186 | 0.207 | +0.021 | 0.775 | 0.748 | −0.027 |
| MPPB | 0.223 | 0.533 | +0.310 | 0.879 | 0.674 | −0.205 |
| MBPB | 0.570 | 0.628 | +0.058 | 0.582 | 0.504 | −0.078 |
| MGMB | 0.452 | 0.463 | +0.010 | 0.659 | 0.580 | −0.079 |

Largest gains: MPPB (+0.31 R²), KSOL (+0.17), LogD (+0.13). Only regression: MLM CLint (−0.02 R²) — within the noise of a single split. Endpoint-difficulty ranking is preserved: foundation model lifts the floor without changing which endpoints are hard or easy.

### Source

- `data/processed/2.08-seal-baseline-performance/chemeleon/overall_metrics.csv`
- `data/processed/2.08-seal-baseline-performance/combined/metrics_comparison.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/8#issuecomment-4376590294
- Reproduce: `pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py --model chemeleon`
