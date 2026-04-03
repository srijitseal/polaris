# Chirality-aware ECFP4 partially resolves stereoisomer blindness — prediction CV rises from 0.000 to 0.002–0.015

## Summary

After enabling `useChirality=True` in ECFP4 fingerprints (PR #16), stereoisomers now produce non-zero Tanimoto distances (median 0.067, up from exactly 0.000) and measurable prediction variation (CV 0.000–0.015, up from 0.000). With Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002), the model can now *see* stereoisomers, but the signal remains weak — stereoisomer prediction CV is 10–50× lower than scaffold decorations (0.074–0.442) and random pairs (0.128–0.404). Consistency ratios < 1 for most endpoints indicate the model under-predicts the true biological variation between enantiomers. Three endpoints (MPPB, MBPB, MGMB) show effectively zero stereoisomer CV, suggesting the tuned model learned near-identical predictions for these protein binding endpoints. The scaffold decoration amplification finding persists unchanged.

## Method

1. **Identify variant groups**:
   - **Stereoisomers** (548 groups, 1,152 molecules, 15.1% of dataset): Strip stereochemistry (`isomericSmiles=False`), group by achiral canonical SMILES → enantiomers/diastereomers share a group. Most groups are pairs (size 2), max size 4.
   - **Scaffold decorations** (1,050 groups, 3,835 molecules, 50.4%): Compute Murcko scaffolds, group by scaffold SMILES, filter to groups of size 2–20. Median group size 2, mean 3.7.
2. **Fingerprint characterization**: Compute intra-group chirality-aware ECFP4 Tanimoto distances for each variant type; compare to 678 random-pair baseline distances.
3. **Out-of-fold predictions**: Train Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) on chirality-aware ECFP4 + RDKit 2D descriptors using cluster-split CV repeat 0. Collect one out-of-fold prediction per molecule across 9 endpoints (~45 models total).
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
| LogD | 0.015 | 0.442 | 0.261 |
| KSOL | 0.005 | 0.097 | 0.155 |
| HLM CLint | 0.007 | 0.091 | 0.164 |
| MLM CLint | 0.003 | 0.074 | 0.128 |
| Caco-2 Papp A>B | 0.002 | 0.139 | 0.295 |
| Caco-2 Efflux | 0.004 | 0.127 | 0.269 |
| MPPB | 0.000 | 0.127 | 0.180 |
| MBPB | 0.000 | 0.241 | 0.392 |
| MGMB | 0.000 | 0.105 | 0.404 |

### Consistency ratio: stereoisomers still under-predicted

| Endpoint | Stereoisomer ratio | Scaffold dec. ratio | Random ratio |
|---|---|---|---|
| LogD | 0.158 | 0.907 | 1.081 |
| KSOL | 1.309 | 3.372 | 6.675 |
| HLM CLint | 0.597 | 1.195 | 1.310 |
| MLM CLint | 0.325 | 1.201 | 1.547 |
| Caco-2 Papp A>B | 0.481 | 2.041 | 1.610 |
| Caco-2 Efflux | 0.138 | 1.544 | 2.838 |
| MPPB | 0.005 | 1.167 | 1.425 |
| MBPB | 0.005 | 2.019 | 2.915 |
| MGMB | 0.000 | 2.257 | 62.833 |

### Interpretation

With chirality-aware fingerprints and Optuna TPE-tuned XGBoost, the original "stereoisomer blindness" failure mode is partially resolved:

1. **Partial stereoisomer awareness**: The tuned model can distinguish stereoisomers for clearance and lipophilicity endpoints (LogD CV 0.015, HLM CLint CV 0.007), but the signal is weaker than with default XGBoost. Three protein binding endpoints (MPPB, MBPB, MGMB) now show effectively zero stereoisomer CV, suggesting hyperparameter tuning produced models that collapse stereoisomer predictions for these endpoints. Consistency ratios < 1 (and near-zero for protein binding) indicate the model severely under-reacts to stereochemical differences.

2. **Scaffold decoration amplification persists**: Scaffold decorations continue to show consistency ratios > 1 for most endpoints, meaning the model amplifies substituent-level fingerprint changes beyond what the true activity warrants. LogD scaffold decoration CV (0.442) notably exceeds even random-pair CV (0.261), suggesting the tuned model is especially sensitive to scaffold variation for this endpoint.

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
