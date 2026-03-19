# Chirality-aware ECFP4 partially resolves stereoisomer blindness — prediction CV rises from 0.000 to 0.006–0.018

## Summary

After enabling `useChirality=True` in ECFP4 fingerprints (PR #16), stereoisomers now produce non-zero Tanimoto distances (median 0.067, up from exactly 0.000) and measurable prediction variation (CV 0.006–0.018, up from 0.000). The model can now *see* stereoisomers, but the signal remains weak — stereoisomer prediction CV is 10–20× lower than scaffold decorations (0.099–0.251) and random pairs (0.151–0.385). Consistency ratios < 1 for most endpoints indicate the model under-predicts the true biological variation between enantiomers. The scaffold decoration amplification finding persists unchanged.

## Method

1. **Identify variant groups**:
   - **Stereoisomers** (548 groups, 1,152 molecules, 15.1% of dataset): Strip stereochemistry (`isomericSmiles=False`), group by achiral canonical SMILES → enantiomers/diastereomers share a group. Most groups are pairs (size 2), max size 4.
   - **Scaffold decorations** (1,050 groups, 3,835 molecules, 50.4%): Compute Murcko scaffolds, group by scaffold SMILES, filter to groups of size 2–20. Median group size 2, mean 3.7.
2. **Fingerprint characterization**: Compute intra-group chirality-aware ECFP4 Tanimoto distances for each variant type; compare to 678 random-pair baseline distances.
3. **Out-of-fold predictions**: Train default XGBoost on chirality-aware ECFP4 + RDKit 2D descriptors using cluster-split CV repeat 0. Collect one out-of-fold prediction per molecule across 9 endpoints (~45 models total).
4. **Consistency metrics**: For each group × endpoint where ≥2 members have predictions, compute prediction std, range, CV, and consistency ratio (prediction spread / true spread). Compare to random groups of matched sizes.

## Key Findings

### Fingerprint distances: stereoisomers are now distinguishable

| Variant type | n pairs | Mean distance | Median distance |
|---|---|---|---|
| Stereoisomers | 679 | 0.087 | 0.067 |
| Scaffold decorations | 9,661 | 0.361 | 0.349 |
| Random pairs | 678 | 0.826 | 0.851 |

### Prediction consistency: stereoisomers show small but real variation

| Endpoint | Stereoisomer CV | Scaffold dec. CV | Random CV |
|---|---|---|---|
| LogD | 0.012 | 0.251 | 0.282 |
| KSOL | 0.010 | 0.126 | 0.172 |
| HLM CLint | 0.018 | 0.109 | 0.198 |
| MLM CLint | 0.012 | 0.099 | 0.151 |
| Caco-2 Papp A>B | 0.013 | 0.173 | 0.299 |
| Caco-2 Efflux | 0.013 | 0.164 | 0.279 |
| MPPB | 0.009 | 0.131 | 0.155 |
| MBPB | 0.006 | 0.169 | 0.338 |
| MGMB | 0.014 | 0.130 | 0.385 |

### Consistency ratio: stereoisomers still under-predicted

| Endpoint | Stereoisomer ratio | Scaffold dec. ratio | Random ratio |
|---|---|---|---|
| LogD | 0.236 | 1.027 | 1.225 |
| KSOL | 2.181 | 4.230 | 7.235 |
| HLM CLint | 0.688 | 1.658 | 1.493 |
| MLM CLint | 0.510 | 1.807 | 1.669 |
| Caco-2 Papp A>B | 1.879 | 2.570 | 1.516 |
| Caco-2 Efflux | 0.361 | 2.204 | 3.518 |
| MPPB | 0.392 | 1.579 | 3.184 |
| MBPB | 0.635 | 2.333 | 1.716 |
| MGMB | 1.220 | 3.227 | 62.277 |

### Interpretation

With chirality-aware fingerprints, the original "stereoisomer blindness" failure mode is partially resolved:

1. **Partial stereoisomer awareness**: The model can now distinguish stereoisomers (non-zero distances and prediction CV), but the signal is weak. Consistency ratios < 1 for most endpoints indicate the model under-reacts to stereochemical differences that genuinely affect ADME properties.

2. **Scaffold decoration amplification persists**: Scaffold decorations continue to show consistency ratios > 1, meaning the model amplifies substituent-level fingerprint changes beyond what the true activity warrants.

## Plots

- `data/processed/2.13-seal-molecular-variants/fingerprint_distances.png` — Intra-group Tanimoto distance distributions
<!-- Paste: fingerprint_distances.png -->
- `data/processed/2.13-seal-molecular-variants/prediction_consistency.png` — Per-endpoint boxplots of within-group prediction CV
<!-- Paste: prediction_consistency.png -->
- `data/processed/2.13-seal-molecular-variants/consistency_heatmap.png` — Heatmap of mean prediction CV
<!-- Paste: consistency_heatmap.png -->
- `data/processed/2.13-seal-molecular-variants/spread_scatter.png` — Predicted range vs true activity range
<!-- Paste: spread_scatter.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py
```

## Source

`notebooks/2.13-seal-molecular-variants.py`
