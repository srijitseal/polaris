# EKM + K-Means produces well-balanced per-endpoint cluster-based CV folds

## Summary

Cluster-based cross-validation splitting using Empirical Kernel Map (KernelPCA on Tanimoto similarity) + Mini Batch K-Means produces well-balanced 5-fold CV assignments for all 9 endpoints. Over-clustering (k=20) with greedy fold assignment achieves max/min fold size ratios of 1.02–1.11 across all endpoints, from LogD (7,309 molecules) to MGMB (431 molecules).

## Key Findings

### Fold balance (repeat 0)

| Endpoint | N molecules | Fold sizes | Max/min ratio |
|----------|-------------|------------|---------------|
| LogD | 7,309 | 1414/1524/1407/1405/1559 | 1.11 |
| KSOL | 7,298 | 1445/1441/1544/1439/1429 | 1.08 |
| HLM CLint | 4,541 | 895/914/878/931/923 | 1.06 |
| MLM CLint | 5,692 | 1138/1146/1137/1120/1151 | 1.03 |
| Caco-2 Papp | 3,773 | 787/740/744/754/748 | 1.06 |
| Caco-2 Efflux | 3,777 | 776/744/761/735/761 | 1.06 |
| MPPB | 1,756 | 348/350/356/353/349 | 1.02 |
| MBPB | 1,426 | 292/290/283/281/280 | 1.04 |
| MGMB | 431 | 84/88/88/85/86 | 1.05 |

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
