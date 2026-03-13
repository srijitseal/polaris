# Split quality diagnostics confirm three complementary generalization axes

## Summary

Cross-strategy comparison of cluster-split, time-split, and target-split on LogD (7,309 molecules) validates that the three strategies produce well-formed folds testing distinct generalization axes. Cluster-split maximizes structural novelty, target-split maximizes value extrapolation, and time-split sits between them.

## Key Findings

### Fold sizes

All strategies produce reasonably balanced folds. Cluster-split (5 folds) ranges 1,367–1,549; time-split and target-split (4 folds each) range ~1,461–1,465. No fold falls below 10% or above 40% of the data.

### Structural distances (test-to-train 1-NN)

| Strategy | Median 1-NN range | Interpretation |
|----------|-------------------|----------------|
| Cluster-split | 0.363–0.471 | Highest — tests structural generalization |
| Time-split | 0.302–0.386 | Moderate — temporal drift includes some series continuity |
| Target-split | 0.245–0.286 | Lowest — similar values ≈ similar structures |

### Structural overlap (% test with 1-NN < 0.1)

| Strategy | Overlap range |
|----------|--------------|
| Cluster-split | 0.0–0.1% |
| Time-split | 0.9–1.5% |
| Target-split | 3.9–4.7% |

Cluster-split effectively eliminates near-duplicates between train and test. Target-split retains the most overlap, consistent with the observation that structurally similar molecules share similar property values.

### Target distribution shift (KS statistic)

| Strategy | KS range | Interpretation |
|----------|----------|----------------|
| Cluster-split | 0.035–0.107 | Minimal shift — folds sample broadly |
| Time-split | 0.040–0.093 | Minimal shift |
| Target-split | 0.742–1.000 | Near-maximal — by design |

### UMAP visualization

UMAP embedding colored by fold assignment shows:
- **Cluster-split**: spatially coherent fold regions (clusters map to contiguous UMAP areas)
- **Time-split**: folds interleave across chemical space (temporal order ≠ structural order)
- **Target-split**: folds form concentric-like bands (value gradient correlates with structure)

## Conclusions

The three strategies are complementary:
1. **Cluster-split** tests whether models generalize to structurally novel chemical series
2. **Time-split** tests whether models generalize to future compounds (temporal drift)
3. **Target-split** tests whether models extrapolate beyond the training value range

Together they cover structure, time, and target-value axes of generalization.

## Plots

- `data/processed/2.06-seal-split-quality/fold_sizes_comparison.png`
- `data/processed/2.06-seal-split-quality/target_distributions_by_strategy.png`
- `data/processed/2.06-seal-split-quality/distance_distributions_by_strategy.png`
- `data/processed/2.06-seal-split-quality/structural_overlap.png`
- `data/processed/2.06-seal-split-quality/umap_by_strategy.png`
- `data/processed/2.06-seal-split-quality/split_quality_summary.csv`

## Source

`notebooks/2.06-seal-split-quality.py`
