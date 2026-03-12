# Endpoint coverage is highly variable and train/test distributions differ significantly

## Summary

Analysis of the 9 ADME endpoints in the Expansion Tx dataset reveals highly variable coverage (6–96%) and significant distribution shift between train and test for all endpoints (KS test p < 0.01). LogD and KSOL have near-complete coverage (96%), while MGMB has only 6%. The Caco-2 endpoints show the strongest train/test shift (KS=0.28–0.37), suggesting the competition split creates meaningful target-space distribution shift beyond just chemical-space shift.

## Key Findings

### Endpoint coverage

| Endpoint | Total | Coverage | Train % | Test % |
|----------|-------|----------|---------|--------|
| LogD | 7,309 | 96% | 95% | 99% |
| KSOL | 7,298 | 96% | 96% | 95% |
| MLM CLint | 5,692 | 75% | 85% | 51% |
| HLM CLint | 4,541 | 60% | 71% | 34% |
| Caco-2 Papp | 3,773 | 50% | 40% | 71% |
| Caco-2 Efflux | 3,777 | 50% | 41% | 71% |
| MPPB | 1,756 | 23% | 24% | 20% |
| MBPB | 1,426 | 19% | 18% | 20% |
| MGMB | 431 | 6% | 4% | 9% |

### Key observations

- **Asymmetric coverage**: Some endpoints have more data in test than train (Caco-2) or vice versa (HLM/MLM CLint), suggesting assay priorities shifted over time.
- **Target distribution shift**: All 9 endpoints show significant KS test differences between train and test. Caco-2 Efflux (KS=0.37) and Caco-2 Papp (KS=0.28) have the strongest shifts.
- **Sparse endpoints**: MGMB (6% coverage) may be too sparse for robust splitting. MPPB and MBPB (~20%) are borderline.
- **Implications for splitting**: Only LogD, KSOL, and MLM CLint have enough data for reliable cross-validation across all folds.

## Plots

- `data/processed/2.02-seal-target-distribution/endpoint_distributions.png` — 3x3 grid of train/test histograms
- `data/processed/2.02-seal-target-distribution/endpoint_counts.png` — coverage bar chart
- `data/processed/2.02-seal-target-distribution/coverage_heatmap.png` — endpoint coverage by molecule completeness

## Data artifacts

- `data/processed/2.02-seal-target-distribution/endpoint_stats.csv` — per-endpoint summary statistics (count, mean, median, Q1, Q3) by split

## Source

`notebooks/2.02-seal-target-distribution.py`
