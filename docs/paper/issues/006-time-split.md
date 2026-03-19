# Time-split produces less structural separation than cluster-split

## Summary

Expanding-window time-split using ordinal molecule indices produces 4-fold CV with balanced fold sizes but lower test-to-train distances than cluster-based splitting (pooled median 0.350 vs 0.408 for LogD). Per-fold median distances do not monotonically increase with time (0.386, 0.351, 0.301, 0.364), suggesting chemical series exploration is non-linear — temporally adjacent molecules are often structurally similar.

## Key Findings

### Time-split fold characteristics (LogD, 7,309 molecules)

| Fold | Test size | mol_index range | Median 1-NN |
|------|-----------|-----------------|-------------|
| 0 | 1,461 | 14,689–16,540 | 0.386 |
| 1 | 1,461 | 16,541–18,872 | 0.351 |
| 2 | 1,461 | 18,873–21,105 | 0.301 |
| 3 | 1,465 | 21,106–27,239 | 0.364 |

### Key observations

- **Cluster-split creates larger distribution shift**: Cluster-split pooled median (0.408) > time-split (0.350). Cluster-based splitting is more effective at separating structurally distinct molecules.
- **Non-monotonic distance over time**: Fold 2 has the lowest median distance (0.302), not fold 0. This suggests the drug discovery program cycled back to explore analogs of earlier series.
- **Complementary splitting strategies**: Time-split captures deployment-relevant temporal shift; cluster-split captures structural novelty. Both are informative but test different aspects of generalization.
- **Balanced fold sizes**: Expanding window produces equal-sized test windows (~1,461 per fold for LogD).

## Plots

- `data/processed/2.04-seal-time-split/fold_sizes_grid.png` — fold sizes per endpoint
- `data/processed/2.04-seal-time-split/fold_temporal_ranges.png` — mol_index ranges
- `data/processed/2.04-seal-time-split/fold_distance_distributions.png` — 1-NN per fold
- `data/processed/2.04-seal-time-split/time_vs_cluster_distances.png` — comparison

## Data artifacts

- `data/interim/time_cv_folds.parquet` — 36,003 rows (long format: Molecule Name, endpoint, repeat, fold)

## Source

`notebooks/2.04-seal-time-split.py`
