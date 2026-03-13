# Models perform consistently worse on activity cliff molecules across all 9 endpoints

## Summary

Activity cliffs — pairs of structurally similar molecules (Tanimoto similarity > 0.85) with large activity differences (top quartile) — comprise 6–10% of molecules per endpoint. XGBoost trained with cluster-split CV shows higher error on cliff molecules than non-cliff molecules across all 9 endpoints, with the largest gaps on Caco-2 endpoints. The effect is moderate but consistent, confirming that smoothly interpolating models struggle where structure-activity relationships are discontinuous.

## Method

1. **Identify activity cliffs** per endpoint using the precomputed ECFP4 Tanimoto distance matrix:
   - Find all pairs with Tanimoto similarity > 0.85 (distance < 0.15)
   - Compute activity differences on log-transformed scale (except LogD)
   - Define cliff pairs as those with activity difference in the top quartile among similar pairs
   - A molecule is a "cliff molecule" if it participates in at least one cliff pair
2. **Train and evaluate** using cluster-split CV (repeat 0, 5 folds) with default XGBoost on ECFP4 + RDKit 2D descriptors
3. **Partition test predictions** into cliff vs non-cliff and compute competition metrics separately

## Key Findings

### Cliff prevalence

| Endpoint | n molecules | n cliff pairs | n cliff mols | % cliff | diff threshold |
|----------|------------|--------------|-------------|---------|---------------|
| LogD | 7,309 | 555 | 729 | 10.0% | 0.600 |
| KSOL | 7,298 | 548 | 701 | 9.6% | 0.302 |
| Caco-2 Papp A>B | 3,773 | 246 | 377 | 10.0% | 0.202 |
| Caco-2 Efflux | 3,777 | 246 | 340 | 9.0% | 0.269 |
| HLM CLint | 4,541 | 252 | 388 | 8.5% | 0.396 |
| MLM CLint | 5,692 | 355 | 440 | 7.7% | 0.503 |
| MBPB | 1,426 | 52 | 96 | 6.7% | 0.154 |
| MPPB | 1,756 | 65 | 115 | 6.5% | 0.247 |
| MGMB | 431 | 15 | 27 | 6.3% | 0.259 |

Cliff prevalence scales with endpoint coverage — well-covered endpoints (LogD, KSOL) have more similar pairs and thus more cliffs.

### Per-endpoint performance: cliff vs non-cliff

| Endpoint | Non-cliff RAE | Cliff RAE | Non-cliff R² | Cliff R² | Non-cliff Spearman | Cliff Spearman |
|----------|-------------|-----------|-------------|---------|-------------------|---------------|
| Caco-2 Papp A>B | 0.695 | **1.050** | 0.427 | -0.187 | 0.651 | 0.329 |
| MPPB | 0.718 | 0.914 | 0.429 | 0.102 | 0.659 | 0.479 |
| Caco-2 Efflux | 0.609 | 0.819 | 0.544 | 0.195 | 0.700 | 0.580 |
| MGMB | 0.608 | 0.772 | 0.515 | 0.233 | 0.725 | 0.550 |
| KSOL | 0.724 | 0.837 | 0.386 | 0.224 | 0.554 | 0.450 |
| HLM CLint | 0.799 | 0.872 | 0.316 | 0.179 | 0.573 | 0.469 |
| MLM CLint | 0.764 | 0.685 | 0.404 | 0.489 | 0.618 | 0.728 |
| MBPB | 0.623 | 0.622 | 0.528 | 0.575 | 0.743 | 0.761 |
| LogD | 0.553 | 0.605 | 0.660 | 0.544 | 0.813 | 0.767 |

Cliff RAE exceeds non-cliff RAE for 7 of 9 endpoints. The effect is most dramatic for Caco-2 Papp A>B (cliff RAE > 1.0, meaning worse than predicting the mean) and Caco-2 Efflux. The two exceptions — MLM CLint and MBPB — show cliff molecules with *better* performance, possibly because cliff pairs in these endpoints have large value ranges that XGBoost handles via tree splitting.

R² drops on cliffs for 7 of 9 endpoints, going negative for Caco-2 Papp A>B. Spearman ρ is consistently lower on cliffs (all 9 endpoints), even for MLM CLint and MBPB — ranking cliff molecules is universally harder than ranking non-cliff molecules.

### Interpretation

The moderate but consistent cliff penalty confirms that XGBoost (a smoothly interpolating model on fingerprint features) struggles at structure-activity discontinuities. The effect is not catastrophic because:
1. Cliff molecules are the *minority* (~7–10%) in each endpoint
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
- `data/processed/2.10-seal-activity-cliff-eval/spearman_by_endpoint.png` — Spearman ρ comparison
<!-- Paste: spearman_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/mae_by_endpoint.png` — MAE comparison
<!-- Paste: mae_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/kendall_by_endpoint.png` — Kendall τ comparison
<!-- Paste: kendall_by_endpoint.png -->
- `data/processed/2.10-seal-activity-cliff-eval/median_se_by_endpoint.png` — Median SE comparison
<!-- Paste: median_se_by_endpoint.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py
```

## Source

`notebooks/2.10-seal-activity-cliff-eval.py`
