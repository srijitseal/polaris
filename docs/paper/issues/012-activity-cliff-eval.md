# Activity cliff molecules degrade model performance for 7 of 9 endpoints under cluster-split CV

## Summary

Activity cliffs — pairs of structurally similar molecules (Tanimoto similarity > 0.85) with large activity differences (top quartile) — comprise 6–10% of molecules per endpoint. Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; see [ADR-002](../decisions/002-optuna-hyperparameter-tuning.md)) trained with cluster-split CV shows higher error on cliff molecules than non-cliff molecules across 7 of 9 endpoints, with the largest gaps on Caco-2 and MPPB endpoints. The effect is moderate but consistent, confirming that smoothly interpolating models struggle where structure-activity relationships are discontinuous. Hyperparameter tuning improves absolute performance on both groups but does not close the cliff-vs-non-cliff gap.

## Method

1. **Identify activity cliffs** per endpoint using the precomputed ECFP4 Tanimoto distance matrix:
   - Find all pairs with Tanimoto similarity > 0.85 (distance < 0.15)
   - Compute activity differences on log-transformed scale (except LogD)
   - Define cliff pairs as those with activity difference in the top quartile among similar pairs
   - A molecule is a "cliff molecule" if it participates in at least one cliff pair
2. **Train and evaluate** using cluster-split CV (repeat 0, 5 folds) with Optuna TPE-tuned XGBoost on ECFP4 + RDKit 2D descriptors (30 trials, 3-fold inner CV, 9 hyperparameters tuned per endpoint per fold)
3. **Partition test predictions** into cliff vs non-cliff and compute competition metrics separately

## Key Findings

### Cliff prevalence

| Endpoint | n molecules | n cliff pairs | n cliff mols | % cliff | diff threshold |
|----------|------------|--------------|-------------|---------|---------------|
| LogD | 7,309 | 493 | 678 | 9.3% | 0.600 |
| KSOL | 7,298 | 478 | 661 | 9.1% | 0.341 |
| Caco-2 Papp A>B | 3,773 | 223 | 367 | 9.7% | 0.202 |
| Caco-2 Efflux | 3,777 | 223 | 323 | 8.6% | 0.278 |
| HLM CLint | 4,541 | 215 | 353 | 7.8% | 0.413 |
| MLM CLint | 5,692 | 299 | 426 | 7.5% | 0.496 |
| MPPB | 1,756 | 59 | 104 | 5.9% | 0.279 |
| MBPB | 1,426 | 47 | 86 | 6.0% | 0.164 |
| MGMB | 431 | 14 | 26 | 6.0% | 0.259 |

Cliff prevalence scales with endpoint coverage — well-covered endpoints (LogD, KSOL) have more similar pairs and thus more cliffs.

### Per-endpoint performance: cliff vs non-cliff

| Endpoint | Non-cliff RAE | Cliff RAE | Non-cliff R² | Cliff R² | Non-cliff Spearman | Cliff Spearman |
|----------|-------------|-----------|-------------|---------|-------------------|---------------|
| Caco-2 Papp A>B | 0.628 | 0.852 | 0.523 | 0.227 | 0.716 | 0.532 |
| MPPB | 0.616 | 0.804 | 0.591 | 0.273 | 0.757 | 0.545 |
| Caco-2 Efflux | 0.606 | 0.827 | 0.560 | 0.223 | 0.719 | 0.577 |
| KSOL | 0.696 | 0.834 | 0.442 | 0.249 | 0.564 | 0.428 |
| HLM CLint | 0.807 | 0.861 | 0.296 | 0.206 | 0.562 | 0.467 |
| MGMB | 0.543 | 0.694 | 0.611 | 0.311 | 0.792 | 0.589 |
| LogD | 0.490 | 0.557 | 0.737 | 0.637 | 0.857 | 0.812 |
| MLM CLint | 0.700 | 0.678 | 0.489 | 0.475 | 0.666 | 0.738 |
| MBPB | 0.571 | 0.492 | 0.613 | 0.736 | 0.790 | 0.863 |

Cliff RAE exceeds non-cliff RAE for 7 of 9 endpoints. The effect is most pronounced for Caco-2 Papp A>B (0.852 vs 0.628), Caco-2 Efflux (0.827 vs 0.606), and MPPB (0.804 vs 0.616). The two exceptions — MLM CLint and MBPB — show cliff molecules with *better* performance, possibly because cliff pairs in these endpoints have large value ranges that XGBoost handles via tree splitting.

R² drops on cliffs for 7 of 9 endpoints. Spearman rho is consistently lower on cliffs for 7 of 9 endpoints — ranking cliff molecules is generally harder than ranking non-cliff molecules.

### Interpretation

The moderate but consistent cliff penalty persists even with Optuna-tuned XGBoost, confirming that the effect is not an artifact of poor hyperparameter selection. Tuning improves absolute performance on both cliff and non-cliff molecules (e.g., MPPB cliff R² improved from 0.090 to 0.273; LogD non-cliff R² from 0.695 to 0.737) but does not close the gap between groups. This strengthens the conclusion that fingerprint-based models struggle at structure-activity discontinuities.

The effect is not catastrophic because:
1. Cliff molecules are the *minority* (~6–10%) in each endpoint
2. Cluster-split CV already introduces some distribution shift, raising the non-cliff baseline error
3. XGBoost's tree structure can partially capture discontinuities, unlike kernel methods

The practical takeaway: activity cliffs are a known failure mode that aggregate metrics obscure. Reporting cliff-specific performance would give practitioners a more honest view of where predictions can be trusted.

## Plots

- `data/processed/2.10-seal-activity-cliff-eval/cliff_characterization.png` — Cliff prevalence and pair counts per endpoint
<!-- Paste: cliff_characterization.png -->
- `data/processed/2.10-seal-activity-cliff-eval/squared_error_distributions.png` — 3x3 grid of cliff vs non-cliff SE histograms
<!-- Paste: squared_error_distributions.png -->
- `data/processed/2.10-seal-activity-cliff-eval/rae_by_endpoint.png` — RAE comparison
<!-- Paste: rae_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/r2_by_endpoint.png` — R² comparison
<!-- Paste: r2_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/spearman_by_endpoint.png` — Spearman rho comparison
<!-- Paste: spearman_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/mae_by_endpoint.png` — MAE comparison
<!-- Paste: mae_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/kendall_by_endpoint.png` — Kendall tau comparison
<!-- Paste: kendall_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/median_se_by_endpoint.png` — Median SE comparison
<!-- Paste: median_se_by_endpoint.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py
```

## Source

`notebooks/2.10-seal-activity-cliff-eval.py`
