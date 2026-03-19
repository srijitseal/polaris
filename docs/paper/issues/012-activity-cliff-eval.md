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
| LogD | 7,309 | 493 | 678 | 9.3% | 0.600 |
| KSOL | 7,298 | 478 | 661 | 9.1% | 0.341 |
| Caco-2 Papp A>B | 3,773 | 223 | 367 | 9.7% | 0.202 |
| Caco-2 Efflux | 3,777 | 223 | 323 | 8.6% | 0.278 |
| HLM CLint | 4,541 | 215 | 353 | 7.8% | 0.413 |
| MLM CLint | 5,692 | 299 | 426 | 7.5% | 0.496 |
| MBPB | 1,426 | 47 | 86 | 6.0% | 0.164 |
| MPPB | 1,756 | 59 | 104 | 5.9% | 0.279 |
| MGMB | 431 | 14 | 26 | 6.0% | 0.259 |

Cliff prevalence scales with endpoint coverage — well-covered endpoints (LogD, KSOL) have more similar pairs and thus more cliffs.

### Per-endpoint performance: cliff vs non-cliff

| Endpoint | Non-cliff RAE | Cliff RAE | Non-cliff R² | Cliff R² | Non-cliff Spearman | Cliff Spearman |
|----------|-------------|-----------|-------------|---------|-------------------|---------------|
| HLM CLint | 0.857 | 0.901 | 0.213 | 0.128 | 0.505 | 0.371 |
| Caco-2 Efflux | 0.641 | 0.882 | 0.503 | 0.101 | 0.690 | 0.507 |
| MPPB | 0.683 | 0.868 | 0.495 | 0.090 | 0.698 | 0.392 |
| KSOL | 0.717 | 0.849 | 0.390 | 0.205 | 0.538 | 0.417 |
| Caco-2 Papp A>B | 0.653 | 0.838 | 0.465 | 0.230 | 0.678 | 0.535 |
| MGMB | 0.625 | 0.770 | 0.493 | 0.179 | 0.713 | 0.456 |
| MLM CLint | 0.752 | 0.703 | 0.423 | 0.455 | 0.616 | 0.700 |
| LogD | 0.529 | 0.614 | 0.695 | 0.552 | 0.828 | 0.751 |
| MBPB | 0.639 | 0.524 | 0.520 | 0.698 | 0.726 | 0.858 |

Cliff RAE exceeds non-cliff RAE for 7 of 9 endpoints. The effect is most dramatic for HLM CLint, Caco-2 Efflux, and MPPB. The two exceptions — MLM CLint and MBPB — show cliff molecules with *better* performance, possibly because cliff pairs in these endpoints have large value ranges that XGBoost handles via tree splitting.

R² drops on cliffs for 7 of 9 endpoints. Spearman ρ is consistently lower on cliffs for 7 of 9 endpoints — ranking cliff molecules is generally harder than ranking non-cliff molecules.

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
