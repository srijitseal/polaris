# Activity cliff molecules (≥1.0 log unit, sim>0.85) degrade model performance across all endpoints with sufficient cliff populations

## Summary

Activity cliffs — pairs of structurally similar molecules (Tanimoto similarity > 0.85) with absolute activity differences ≥ 1.0 log units (approximately 10-fold change) — comprise 0–4.6% of molecules per endpoint. Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; see [ADR-002](../decisions/002-optuna-hyperparameter-tuning.md)) trained with cluster-split CV shows higher error on cliff molecules than non-cliff molecules across all 7 endpoints with sufficient cliff populations. MBPB has 0 cliff pairs and MPPB only 4 cliff molecules at this threshold, making them unsuitable for separate evaluation.

## Method

1. **Identify activity cliffs** per endpoint using the precomputed Morgan fingerprint Tanimoto distance matrix:
   - Find all pairs with Tanimoto similarity > 0.85 (distance < 0.15)
   - Compute activity differences on log-transformed scale (except LogD)
   - Define cliff pairs as those with absolute activity difference ≥ 1.0 log units
   - A molecule is a "cliff molecule" if it participates in at least one cliff pair
2. **Train and evaluate** using cluster-split CV (repeat 0, 5 folds) with Optuna TPE-tuned XGBoost on Morgan fingerprints + RDKit 2D descriptors (30 trials, 3-fold inner CV, 9 hyperparameters tuned per endpoint per fold)
3. **Partition test predictions** into cliff vs non-cliff and compute metrics separately

## Key Findings

### Cliff prevalence (absolute threshold ≥ 1.0 log units)

| Endpoint | n molecules | n cliff pairs | n cliff mols | % cliff | diff threshold |
|----------|------------|--------------|-------------|---------|---------------|
| LogD | 7,309 | 221 | 334 | 4.6% | 1.000 |
| KSOL | 7,298 | 178 | 253 | 3.5% | 1.000 |
| MLM CLint | 5,692 | 90 | 144 | 2.5% | 1.000 |
| HLM CLint | 4,541 | 20 | 37 | 0.8% | 1.000 |
| MGMB | 431 | 3 | 5 | 1.2% | 1.000 |
| Caco-2 Efflux | 3,777 | 13 | 22 | 0.6% | 1.000 |
| Caco-2 Papp A>B | 3,773 | 3 | 6 | 0.2% | 1.000 |
| MPPB | 1,756 | 2 | 4 | 0.2% | 1.000 |
| MBPB | 1,426 | 0 | 0 | 0.0% | 1.000 |

### Per-endpoint performance: cliff vs non-cliff

| Endpoint | Non-cliff RAE | Cliff RAE | Non-cliff R² | Cliff R² | Cliff Spearman |
|----------|-------------|-----------|-------------|---------|---------------|
| Caco-2 Efflux | 0.581 | 0.949 | 0.576 | 0.025 | 0.409 |
| Caco-2 Papp A>B | 0.643 | 0.907 | 0.507 | 0.070 | 0.257 |
| HLM CLint | 0.811 | 0.866 | 0.288 | 0.243 | 0.630 |
| MLM CLint | 0.696 | 0.843 | 0.495 | 0.268 | 0.582 |
| KSOL | 0.703 | 0.807 | 0.432 | 0.269 | 0.472 |
| MGMB | 0.548 | 0.760 | 0.608 | -0.204 | 0.872 |
| LogD | 0.491 | 0.593 | 0.735 | 0.590 | 0.768 |
| MPPB | -- | -- | -- | -- | -- |
| MBPB | -- | -- | -- | -- | -- |

Cliff RAE exceeds non-cliff RAE for all 7 endpoints with sufficient cliff molecules. The effect is most pronounced for Caco-2 Efflux (0.949 vs 0.581), Caco-2 Papp A>B (0.907 vs 0.643), and HLM CLint (0.866 vs 0.811).

### Change from previous version

Previous version used a relative top-quartile threshold, giving 6–10% cliff prevalence and 7/9 endpoints with cliff > non-cliff. The absolute 1.0 log unit threshold is more principled: it defines cliffs by a fixed magnitude of activity change (≈ 10-fold) rather than an endpoint-adaptive percentile. Cliff prevalence drops to 0–4.6%, but the cliff penalty is consistent across all analyzable endpoints.

## Plots

- `data/processed/2.10-seal-activity-cliff-eval/cliff_characterization.png` — Cliff prevalence and pair counts per endpoint
<!-- Paste: cliff_characterization.png -->
- `data/processed/2.10-seal-activity-cliff-eval/squared_error_distributions.png` — Cliff vs non-cliff SE distributions
<!-- Paste: squared_error_distributions.png -->
- `data/processed/2.10-seal-activity-cliff-eval/rae_by_endpoint.png` — RAE comparison
<!-- Paste: rae_by_endpoint.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py
```

## Source

`notebooks/2.10-seal-activity-cliff-eval.py`
