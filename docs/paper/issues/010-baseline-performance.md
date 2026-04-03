# XGBoost + ECFP4 + RDKit descriptors baseline achieves MA-RAE 0.773 on competition split

## Summary

Training XGBoost on ECFP4 fingerprints + RDKit 2D descriptors with per-endpoint hyperparameter tuning (Optuna TPE, 30 trials, 3-fold CV) on the original competition train/test split (5,326 / 2,282). Molecules are protonated at assay-relevant pH (7.4 for most, 6.5 for Caco-2) using dimorphite_dl before feature computation. Applying the competition's log-transform (`log10(clip(x, 1e-10) + 1)`) to all endpoints except LogD -- for both training and evaluation -- was critical. See ADR-002 (`docs/decisions/002-optuna-hyperparameter-tuning.md`) for the switch from HalvingRandomSearchCV to Optuna TPE.

## Key Findings

### Evolution of baseline performance (MA-RAE)

| Configuration | MA-RAE |
|--------------|--------|
| RF (100 trees, default params, raw-scale training, log-eval only) | 0.892 |
| RF (500 trees, tuned params, log-scale training + eval) | 0.868 |
| XGBoost (HalvingRandomSearchCV, ECFP4 + 16 physchem, log-scale train + eval) | 0.826 |
| XGBoost (HalvingRandomSearchCV, ECFP4 + full RDKit 2D, dimorphite_dl protonation) | 0.821 |
| **XGBoost (Optuna TPE 30 trials, ECFP4 + full RDKit 2D, dimorphite_dl protonation)** | **0.773** |

### Per-endpoint results (final: XGBoost + Optuna TPE + full RDKit 2D + protonation)

| Endpoint | pH | R2 | R2 95% CI | Spearman rho | Spearman 95% CI | RAE | RAE 95% CI | n_train | n_test |
|----------|-----|-----|-----------|-----------|-----------------|-----|------------|---------|--------|
| LogD | 7.4 | 0.583 | [0.551, 0.613] | 0.753 | [0.732, 0.773] | 0.630 | [0.608, 0.652] | 5,039 | 2,270 |
| KSOL | 7.4 | 0.256 | [0.193, 0.310] | 0.489 | [0.455, 0.520] | 0.850 | [0.826, 0.876] | 5,128 | 2,170 |
| HLM CLint | 7.4 | 0.255 | [0.171, 0.332] | 0.575 | [0.524, 0.625] | 0.847 | [0.800, 0.895] | 3,759 | 782 |
| MLM CLint | 7.4 | 0.177 | [0.102, 0.247] | 0.452 | [0.407, 0.497] | 0.925 | [0.884, 0.964] | 4,522 | 1,170 |
| Caco-2 Papp A>B | 6.5 | 0.230 | [0.183, 0.276] | 0.527 | [0.492, 0.561] | 0.840 | [0.812, 0.867] | 2,157 | 1,616 |
| Caco-2 Efflux | 6.5 | 0.191 | [0.147, 0.235] | 0.646 | [0.616, 0.677] | 0.775 | [0.740, 0.810] | 2,161 | 1,616 |
| MPPB | 7.4 | 0.223 | [0.094, 0.328] | 0.613 | [0.550, 0.665] | 0.869 | [0.812, 0.928] | 1,302 | 454 |
| MBPB | 7.4 | 0.560 | [0.494, 0.615] | 0.782 | [0.734, 0.821] | 0.592 | [0.548, 0.636] | 975 | 451 |
| MGMB | 7.4 | 0.502 | [0.402, 0.592] | 0.741 | [0.651, 0.813] | 0.633 | [0.566, 0.703] | 222 | 209 |

### HalvingRandomSearchCV vs Optuna TPE comparison

Optuna TPE improved MA-RAE by 0.048 (0.821 to 0.773), a 5.8% relative improvement. R2 improved across 8 of 9 endpoints (only MLM CLint regressed by 0.017). Biggest RAE improvements: MGMB (-0.147), MBPB (-0.085), HLM CLint (-0.066) -- endpoints with fewer training samples (222-975) where HPO matters most. KSOL (+0.008) and MLM CLint (+0.005) showed minimal RAE change, having the most training data (4,500-5,100) and likely near their performance ceiling.

| Endpoint | Old RAE | New RAE | Delta RAE | Old R2 | New R2 | Delta R2 |
|----------|---------|---------|-----------|--------|--------|----------|
| LogD | 0.664 | 0.630 | -0.034 | 0.523 | 0.583 | +0.060 |
| KSOL | 0.842 | 0.850 | +0.008 | 0.245 | 0.256 | +0.011 |
| HLM CLint | 0.913 | 0.847 | -0.066 | 0.139 | 0.255 | +0.116 |
| MLM CLint | 0.920 | 0.925 | +0.005 | 0.194 | 0.177 | -0.017 |
| Caco-2 Papp A>B | 0.876 | 0.840 | -0.036 | 0.163 | 0.230 | +0.067 |
| Caco-2 Efflux | 0.808 | 0.775 | -0.033 | 0.121 | 0.191 | +0.070 |
| MPPB | 0.912 | 0.869 | -0.043 | 0.142 | 0.223 | +0.081 |
| MBPB | 0.677 | 0.592 | -0.085 | 0.429 | 0.560 | +0.131 |
| MGMB | 0.780 | 0.633 | -0.147 | 0.347 | 0.502 | +0.155 |
| **MA-RAE** | **0.821** | **0.773** | **-0.048** | | | |

### Key insight: log-transform is essential

The competition evaluates on log-transformed values for all endpoints except LogD. Training on raw-scale targets and only log-transforming at evaluation time produces poor R2 because the model optimizes for raw-scale MSE, which is dominated by high-value outliers. Training on log-scale targets aligns the model's loss function with the evaluation metric.

### Effect of protonation + full descriptor suite

Adding dimorphite_dl protonation and expanding from 16 hand-picked to ~200 RDKit 2D descriptors produced mixed per-endpoint effects: Caco-2 Efflux improved from R2=-0.21 to +0.13 and MPPB from -0.07 to +0.14, but HLM CLint and Caco-2 Papp regressed. Overall MA-RAE improved marginally (0.826 to 0.820). The full descriptor suite introduces noisy features that XGBoost's built-in feature selection doesn't fully compensate for.

### Optuna TPE hyperparameter characteristics

Optuna found notably different hyperparameters than HalvingRandomSearchCV -- generally lower learning rates (0.026-0.159), shallower trees for most endpoints (max_depth 3 for 6 of 9), and higher n_estimators (301-997). The combination of more trees with lower learning rate and shallower depth is consistent with a regularized ensemble that generalizes better to the test set.

### Bootstrap confidence intervals

To quantify uncertainty from the single train/test split, we computed 95% bootstrap confidence intervals using 1,000 resamples of the test-set predictions. Each resample draws n_test samples with replacement from the (y_true, y_pred) pairs and recomputes all metrics (R2, MAE, Spearman, RAE). The 2.5th and 97.5th percentiles define the 95% CI.

CI widths vary substantially with test-set size. Endpoints with large test sets have tight CIs: LogD (n=2,270, R2 width 0.062), KSOL (n=2,170, width 0.117), and Caco-2 endpoints (n=1,616, widths 0.088-0.092). Endpoints with small test sets have wide CIs: MPPB (n=454, R2 width 0.234), MGMB (n=209, width 0.190), and HLM CLint (n=782, width 0.162). The wide CIs on MPPB and MGMB mean their point estimates are less reliable -- a 0.05 improvement on these endpoints may not be statistically meaningful on this split.

## Models

- XGBRegressor: `tree_method="hist"`, per-endpoint tuning via Optuna TPE (30 trials, 3-fold CV, scoring=neg_MAE, 9 hyperparameters)
- Tuned hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Features: 2048-bit ECFP4 + full RDKit 2D descriptor suite (~200, scaled), computed on dimorphite_dl-protonated SMILES at assay pH
- Protonation: dimorphite_dl with pH +/- 0.5 window, max 1 variant per molecule
- Caching: Optuna results cached in `data/interim/optuna_cache/` keyed by endpoint + split

## Plots

- `data/processed/2.08-seal-baseline-performance/r2_by_endpoint.png` -- R2 bar chart
<!-- Paste: r2_by_endpoint.png -->
- `data/processed/2.08-seal-baseline-performance/mae_by_endpoint.png` -- MAE bar chart
<!-- Paste: mae_by_endpoint.png -->
- `data/processed/2.08-seal-baseline-performance/spearman_by_endpoint.png` -- Spearman rho bar chart
<!-- Paste: spearman_by_endpoint.png -->
- `data/processed/2.08-seal-baseline-performance/scatter_predictions.png` -- 3x3 scatter grid
<!-- Paste: scatter_predictions.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py
```

## Source

`notebooks/2.08-seal-baseline-performance.py`
