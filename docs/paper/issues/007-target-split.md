# Target-value splitting produces low structural distances — value proximity implies structural similarity

## Summary

Expanding-window CV sorted by endpoint value tests model extrapolation beyond the training target range. For LogD, test-to-train 1-NN distances are low (median 0.245–0.286), much lower than cluster-split (0.410) or time-split (0.350), confirming that molecules with similar target values tend to be structurally similar. CLint endpoints show extreme value ranges in later folds, creating challenging extrapolation scenarios.

## Key Findings

### Target-split fold characteristics (LogD, 7,309 molecules)

| Fold | Test size | Value range | Median 1-NN |
|------|-----------|-------------|-------------|
| 0 | 1,461 | 1.20–1.90 | 0.286 |
| 1 | 1,461 | 1.90–2.40 | 0.250 |
| 2 | 1,461 | 2.40–3.10 | 0.245 |
| 3 | 1,465 | 3.10–5.20 | 0.255 |

### Distance comparison across splitting strategies (LogD)

| Strategy | Pooled median 1-NN |
|----------|-------------------|
| Cluster-split | 0.410 |
| Time-split | 0.350 |
| Target-split | ~0.260 |

### Key observations

- **Low structural distances**: Target-split produces the lowest test-to-train structural distances of all three strategies. This is expected — molecules with similar endpoint values are often from the same chemical series.
- **Extrapolation challenge is in target space, not structure space**: The difficulty here is predicting values outside the training range, not predicting for structurally novel molecules.
- **Extreme tails in CLint**: HLM CLint fold 3 covers 58.5–2589.9 (44x range), MLM CLint fold 3 covers 639–10,355. These extreme values will be very challenging to predict.
- **Complementary to other strategies**: Cluster-split tests structural generalization, time-split tests temporal generalization, target-split tests value extrapolation. Together they cover three orthogonal axes of generalization.

## Plots

- `data/processed/2.05-seal-target-split/fold_sizes_grid.png` — fold sizes per endpoint
- `data/processed/2.05-seal-target-split/fold_target_ranges.png` — value ranges per fold
- `data/processed/2.05-seal-target-split/fold_target_distributions.png` — train vs test distributions
- `data/processed/2.05-seal-target-split/fold_distance_distributions.png` — 1-NN per fold

## Data artifacts

- `data/interim/target_cv_folds.parquet` — 36,003 rows (long format)

## Source

`notebooks/2.05-seal-target-split.py`
