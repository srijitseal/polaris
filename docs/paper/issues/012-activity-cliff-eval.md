# Activity cliff molecules (≥1.0 log unit, sim>0.85) degrade model performance across all endpoints with sufficient cliff populations

## Summary

Activity cliffs — pairs of structurally similar molecules (Tanimoto similarity > 0.85) with absolute activity differences ≥ 1.0 log units (approximately 10-fold change) — comprise 0–4.6% of molecules per endpoint. Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; see [ADR-002](../../decisions/002-optuna-hyperparameter-tuning.md)) trained with cluster-split CV shows higher error on cliff molecules than non-cliff molecules across all 7 endpoints with sufficient cliff populations. MBPB has 0 cliff pairs and MPPB only 4 cliff molecules at this threshold, making them unsuitable for separate evaluation.

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
|----------|---------------|-----------|--------------|----------|----------------|
| Caco-2 Efflux | 0.585 | 0.962 | 0.574 | -0.006 | 0.104 |
| Caco-2 Papp A>B | 0.650 | 0.879 | 0.496 | 0.109 | 0.200 |
| HLM CLint | 0.810 | 0.865 | 0.286 | 0.240 | 0.602 |
| MLM CLint | 0.704 | 0.829 | 0.486 | 0.302 | 0.600 |
| KSOL | 0.698 | 0.807 | 0.438 | 0.294 | 0.502 |
| MGMB | 0.555 | 0.776 | 0.604 | -0.226 | 0.718 |
| LogD | 0.491 | 0.592 | 0.736 | 0.584 | 0.765 |
| MPPB | 0.622 | — | 0.580 | — | — |
| MBPB | 0.562 | — | 0.624 | — | — |

Cliff RAE exceeds non-cliff RAE for all 7 endpoints with sufficient cliff molecules. The effect is most pronounced for Caco-2 Efflux (0.962 vs 0.585), Caco-2 Papp A>B (0.879 vs 0.650), and MGMB (0.776 vs 0.555). Cliff R² is negative or near-zero for Caco-2 Efflux (−0.006) and MGMB (−0.226) — model performs no better than predicting the cliff-set mean.

### Change from previous version

Previous version used a relative top-quartile threshold, giving 6–10% cliff prevalence and 7/9 endpoints with cliff > non-cliff. The absolute 1.0 log unit threshold is more principled: it defines cliffs by a fixed magnitude of activity change (≈ 10-fold) rather than an endpoint-adaptive percentile. Cliff prevalence drops to 0–4.6%, but the cliff penalty is consistent across all analyzable endpoints.

## Plots

- `data/processed/2.10-seal-activity-cliff-eval/cliff_characterization.png` — cliff prevalence and pair counts per endpoint
- `data/processed/2.10-seal-activity-cliff-eval/xgboost/squared_error_distributions.png` — cliff vs non-cliff SE distributions
- `data/processed/2.10-seal-activity-cliff-eval/xgboost/rae_by_endpoint.png` — RAE comparison
- `data/processed/2.10-seal-activity-cliff-eval/xgboost/r2_by_endpoint.png`, `mae_by_endpoint.png`, `spearman_by_endpoint.png`, `kendall_by_endpoint.png`, `median_se_by_endpoint.png` — per-metric panels
- Combined XGBoost vs CheMeleon panels: `data/processed/2.10-seal-activity-cliff-eval/combined/`

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py main
```

## Source

`notebooks/2.10-seal-activity-cliff-eval.py`

## Follow-up: similarity-threshold sensitivity sweep

A `sensitivity` subcommand sweeps the Tanimoto cutoff across {0.70, 0.75, 0.80, 0.85, 0.90, 0.95} while holding |Δactivity| ≥ 1.0 log unit fixed, reusing the existing out-of-fold predictions (no retraining). Mean cliff prevalence falls ~12.7× from 0.70 to 0.85 and ~10.2× from 0.85 to 0.95 across endpoints. RAE gap (cliff − non-cliff) is near zero or negative below 0.80 for LogD/KSOL/MLM (the cliff label is not selective), becomes reliably positive at ≥ 0.85, and the cliff set becomes very sparse at stricter cutoffs: 3 of 9 endpoints have zero cliff pairs at 0.90, rising to 4 of 9 at 0.95. 0.85 on ECFP4 is therefore the empirical knee of the curve for this dataset rather than a universal convention: Stumpfe & Bajorath 2014's 0.85 applies to MACCS (ECFP4 equivalent ~0.56); van Tilborg et al. 2022 (MoleculeACE) use 0.9 on ECFP4 under a soft consensus across three similarity measures.

### Outputs saved

`data/processed/2.10-seal-activity-cliff-eval/`
- `cliff_sensitivity.csv` — 54 rows (9 endpoints × 6 thresholds)
- `cliff_sensitivity_pairs.parquet` — similar pairs (sim ≥ 0.70)
- `cliff_sensitivity_pct.png`, `cliff_sensitivity_rae_gap.png`, `cliff_sensitivity_joint.png`
- `cliff_sensitivity_combined.png` — combined 3-panel figure for manuscript Fig S5

### Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py sensitivity
```

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran with CheMeleon (cluster-split CV, 5 folds, fine-tuned per fold). Cliff definition unchanged: Tanimoto > 0.85 + |Δy| ≥ 1.0 log unit. Cliff/non-cliff partitions identical to XGBoost run (counts unchanged).

**Trend matches XGBoost. Cliff RAE > non-cliff RAE for all 7 endpoints with sufficient cliffs.**

### Cliff RAE vs non-cliff RAE

| Endpoint | n_cliff | RAE cliff XGB | RAE non-cliff XGB | RAE cliff CM | RAE non-cliff CM |
|----------|---------|---------------|-------------------|--------------|------------------|
| LogD | 334 | 0.592 | 0.491 | 0.422 | 0.360 |
| KSOL | 253 | 0.807 | 0.698 | 0.713 | 0.617 |
| HLM CLint | 37 | 0.865 | 0.810 | 0.939 | 0.792 |
| MLM CLint | 144 | 0.829 | 0.704 | 0.801 | 0.684 |
| Caco-2 Papp A>B | 6 | 0.879 | 0.650 | 0.942 | 0.677 |
| Caco-2 Efflux | 22 | 0.962 | 0.585 | 0.895 | 0.586 |
| MGMB | 5 | 0.776 | 0.555 | 0.944 | 0.501 |

MPPB (4 cliffs) and MBPB (0 cliffs) excluded from cliff evaluation in both models.

CheMeleon does not selectively help cliff molecules — non-cliff RAE drops by 0.05–0.13 from XGBoost, while the cliff/non-cliff RAE gap remains 0.06–0.44 (median ~0.13). Interpolation failure mode is preserved under a foundation model.

### Source

- `data/processed/2.10-seal-activity-cliff-eval/chemeleon/summary_metrics.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/12#issuecomment-4381083314
- Reproduce: `pixi run -e cheminformatics python notebooks/2.10-seal-activity-cliff-eval.py main --model chemeleon`
