# Test molecules are nearly 2x farther from training data than within-train neighbors

## Summary

ECFP4 Tanimoto distance analysis of the Expansion Tx dataset (7,608 molecules) reveals moderate chemical diversity (1-NN median 0.20) with a clear distribution shift between train and test: test-to-train 1-NN median (0.375) is nearly double the within-train 1-NN median (0.203), confirming the competition split creates a meaningful generalization challenge.

## Key Findings

### Distance distribution summary

| Metric | Count | Mean | Median | Q1 | Q3 |
|--------|-------|------|--------|----|----|
| All pairwise | 28.9M | 0.824 | 0.851 | 0.802 | 0.879 |
| 1-NN | 7,608 | 0.207 | 0.200 | 0.157 | 0.250 |
| 5-NN | 7,608 | 0.322 | 0.311 | 0.250 | 0.377 |
| Test-to-train 1-NN | 2,282 | 0.387 | 0.375 | 0.254 | 0.516 |
| Within-train 1-NN | 5,326 | 0.195 | 0.203 | 0.153 | 0.255 |

### Key observations

- **Chemical diversity**: 1-NN median of 0.20 indicates many molecules have close neighbors — consistent with chemical series structure. The all-pairwise median of 0.85 confirms the dataset spans diverse scaffolds overall.
- **Distribution shift in competition split**: Test-to-train 1-NN median (0.375) vs within-train (0.203) — the test set is substantially farther from training data, creating a real OOD evaluation.
- **Long tail in test distances**: Test-to-train 1-NN extends to 0.73, with broad distribution from 0.1 to 0.7 — many test molecules have no close training analog.
- **Spike at zero in within-train**: Large spike at distance ~0 in within-train 1-NN indicates near-duplicate or very similar compounds within the training set (likely from same chemical series).

### Comparison to Biogen case study
- Similar to Cas's Biogen HLM analysis where the dataset was diverse "with lots of datapoints having far neighbors"
- The Expansion Tx dataset shows a cleaner separation between NN distances and bulk pairwise distances

## Plots

<!-- Paste: data/processed/0.02-seal-ecfp-distance-exploration/nn_distance_distributions.png -->

<!-- Paste: data/processed/0.02-seal-ecfp-distance-exploration/all_pairwise_distances.png -->

<!-- Paste: data/processed/0.02-seal-ecfp-distance-exploration/train_vs_test_distances.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/0.02-seal-ecfp-distance-exploration.py
```

Outputs: `data/processed/0.02-seal-ecfp-distance-exploration/`
Distance matrix saved for reuse: `data/interim/tanimoto_distance_matrix.npz`

## Conclusion

The competition train/test split creates a meaningful distribution shift (median test-to-train distance ~2x within-train). This validates the dataset for studying generalization: models evaluated with random splits would see much smaller train-test distances, inflating performance. The distance matrix is saved for reuse in downstream clustering and splitting notebooks.
