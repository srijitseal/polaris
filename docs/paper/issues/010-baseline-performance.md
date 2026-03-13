# XGBoost + ECFP4 + RDKit descriptors baseline achieves MA-RAE 0.826 on competition split

## Summary

Training XGBoost on ECFP4 fingerprints + RDKit 2D descriptors with per-endpoint hyperparameter tuning (HalvingRandomSearchCV) on the original competition train/test split (5,326 / 2,282). Molecules are protonated at assay-relevant pH (7.4 for most, 6.5 for Caco-2) using dimorphite_dl before feature computation. Applying the competition's log-transform (`log10(clip(x, 1e-10) + 1)`) to all endpoints except LogD — for both training and evaluation — was critical.

## Key Findings

### Evolution of baseline performance (MA-RAE)

| Configuration | MA-RAE |
|--------------|--------|
| RF (100 trees, default params, raw-scale training, log-eval only) | 0.892 |
| RF (500 trees, tuned params, log-scale training + eval) | 0.868 |
| XGBoost (tuned, ECFP4 + 16 physchem, log-scale train + eval) | 0.826 |
| **XGBoost (tuned, ECFP4 + full RDKit 2D, dimorphite_dl protonation)** | **0.820** |

### Per-endpoint results (final: XGBoost + full RDKit 2D + protonation)

| Endpoint | pH | R² | Spearman ρ | RAE | n_train | n_test |
|----------|-----|-----|-----------|-----|---------|--------|
| LogD | 7.4 | 0.534 | 0.719 | 0.663 | 5,039 | 2,270 |
| KSOL | 7.4 | 0.199 | 0.457 | 0.878 | 5,128 | 2,170 |
| HLM CLint | 7.4 | 0.170 | 0.546 | 0.889 | 3,759 | 782 |
| MLM CLint | 7.4 | 0.210 | 0.496 | 0.912 | 4,522 | 1,170 |
| Caco-2 Papp A>B | 6.5 | 0.105 | 0.437 | 0.903 | 2,157 | 1,616 |
| Caco-2 Efflux | 6.5 | 0.128 | 0.600 | 0.799 | 2,161 | 1,616 |
| MPPB | 7.4 | 0.140 | 0.506 | 0.913 | 1,302 | 454 |
| MBPB | 7.4 | 0.398 | 0.696 | 0.702 | 975 | 451 |
| MGMB | 7.4 | 0.347 | 0.732 | 0.780 | 222 | 209 |

### Key insight: log-transform is essential

The competition evaluates on log-transformed values for all endpoints except LogD. Training on raw-scale targets and only log-transforming at evaluation time produces poor R² because the model optimizes for raw-scale MSE, which is dominated by high-value outliers. Training on log-scale targets aligns the model's loss function with the evaluation metric.

### Effect of protonation + full descriptor suite

Adding dimorphite_dl protonation and expanding from 16 hand-picked to ~200 RDKit 2D descriptors produced mixed per-endpoint effects: Caco-2 Efflux improved from R²=-0.21 to +0.13 and MPPB from -0.07 to +0.14, but HLM CLint and Caco-2 Papp regressed. Overall MA-RAE improved marginally (0.826 → 0.820). The full descriptor suite introduces noisy features that XGBoost's built-in feature selection doesn't fully compensate for.

## Models

- XGBRegressor: `tree_method="hist"`, per-endpoint tuning via `HalvingRandomSearchCV` (50 candidates, factor=3, 3-fold CV, scoring=neg_MAE)
- Features: 2048-bit ECFP4 + full RDKit 2D descriptor suite (~200, scaled), computed on dimorphite_dl-protonated SMILES at assay pH
- Protonation: dimorphite_dl with pH ± 0.5 window, max 1 variant per molecule

## Plots

- `data/processed/2.08-seal-baseline-performance/r2_by_endpoint.png` — R² bar chart
<!-- Paste: r2_by_endpoint.png -->
- `data/processed/2.08-seal-baseline-performance/mae_by_endpoint.png` — MAE bar chart
<!-- Paste: mae_by_endpoint.png -->
- `data/processed/2.08-seal-baseline-performance/spearman_by_endpoint.png` — Spearman ρ bar chart
<!-- Paste: spearman_by_endpoint.png -->
- `data/processed/2.08-seal-baseline-performance/scatter_predictions.png` — 3×3 scatter grid
<!-- Paste: scatter_predictions.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py
```

## Source

`notebooks/2.08-seal-baseline-performance.py`
