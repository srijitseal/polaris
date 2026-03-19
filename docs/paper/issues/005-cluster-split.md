# EKM + K-Means produces well-balanced per-endpoint cluster-based CV folds

## Summary

Cluster-based cross-validation splitting using Empirical Kernel Map (KernelPCA on Tanimoto similarity) + Mini Batch K-Means produces well-balanced 5-fold CV assignments for all 9 endpoints. Over-clustering (k=20) with greedy fold assignment achieves max/min fold size ratios of 1.02–1.14 across all endpoints, from LogD (7,309 molecules) to MGMB (431 molecules).

## Key Findings

### Fold balance (repeat 0)

| Endpoint | N molecules | Fold sizes | Max/min ratio |
|----------|-------------|------------|---------------|
| LogD | 7,309 | 1507/1467/1464/1449/1422 | 1.06 |
| KSOL | 7,298 | 1470/1476/1454/1454/1444 | 1.02 |
| HLM CLint | 4,541 | 898/933/901/901/908 | 1.04 |
| MLM CLint | 5,692 | 1204/1063/1205/1060/1160 | 1.14 |
| Caco-2 Papp | 3,773 | 773/746/738/761/755 | 1.05 |
| Caco-2 Efflux | 3,777 | 773/757/750/750/747 | 1.03 |
| MPPB | 1,756 | 368/354/341/345/348 | 1.08 |
| MBPB | 1,426 | 285/293/278/287/283 | 1.05 |
| MGMB | 431 | 90/85/85/86/85 | 1.06 |

### Key observations

- **Over-clustering is critical**: Direct k=5 K-Means produced ratios up to 6.6 due to uneven chemical space. Over-clustering (k=20) + greedy assignment solves this.
- **Per-endpoint splitting**: Each endpoint gets its own fold assignments based only on molecules with data for that endpoint. This avoids folds with very few measurements for sparse endpoints.
- **Stochasticity across repeats**: Different random seeds produce slightly different fold assignments, enabling 5x5 repeated CV (25 total train/test splits per endpoint).
- **Even MGMB works**: The sparsest endpoint (431 molecules) still gets balanced folds of ~86 each.

## Plots

- `data/processed/2.03-seal-cluster-split/fold_sizes_grid.png` — 3×3 grid of fold sizes per endpoint
- `data/processed/2.03-seal-cluster-split/fold_distance_distributions.png` — test-to-train 1-NN per fold (LogD)
- `data/processed/2.03-seal-cluster-split/fold_size_variation.png` — balance across repeats

## Data artifacts

- `data/interim/cluster_cv_folds.parquet` — 180,015 rows (long format: Molecule Name, endpoint, repeat, fold)
- `data/processed/2.03-seal-cluster-split/fold_stats.csv` — 225 rows (9 endpoints × 5 repeats × 5 folds)

## Source

`notebooks/2.03-seal-cluster-split.py`
