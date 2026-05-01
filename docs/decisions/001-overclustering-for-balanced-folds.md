# ADR-001: Over-cluster then greedily assign to folds for balanced CV splits

**Status**: Accepted

## Context

Cluster-based cross-validation requires assigning molecules to folds such that each fold contains structurally distinct molecules. The naive approach — running K-Means with k=number_of_folds (k=5) on EKM-projected ECFP4 fingerprints — produced highly imbalanced folds with max/min size ratios up to 6.6. This is because the chemical space is inherently uneven (3 dominant chemical series contain 63% of molecules), so k-means with few clusters tends to produce one or two very large clusters.

## Decision

Over-cluster with k=20 (more clusters than folds), then greedily assign clusters to folds to balance sizes. The greedy algorithm sorts clusters by size (largest first) and assigns each to the fold with the fewest molecules so far.

## Implementation

In `make_endpoint_folds()` (`notebooks/2.03-seal-cluster-split.py`):
1. KernelPCA (EKM) on Tanimoto similarity matrix → 50-d Euclidean space
2. MiniBatchKMeans with n_clusters=20
3. Sort clusters by size descending
4. Assign each cluster to the fold with current minimum total size

## Consequences

**Benefits:**
- All 9 endpoints achieve max/min fold size ratio < 1.14 across all 5 repeats (vs up to 6.6 before); the worst case is MLM CLint repeat 0 at 1.1368 (1205/1060)
- Works even for sparse endpoints (MGMB, 431 molecules → folds of ~86)
- Preserves structural separation — molecules in the same k-means cluster stay in the same fold

**Tradeoffs:**
- Structural separation between folds is slightly weaker than direct k=5 (some folds contain clusters that aren't adjacent in chemical space)
- The choice of k=20 is somewhat arbitrary; values between 10–50 would likely work similarly
