# Chirality-aware Morgan fingerprints partially resolve stereoisomer blindness — prediction CV 0.000–0.015, but variation confounds biology with assay noise

## Summary

After enabling `useChirality=True` in Morgan fingerprints, stereoisomers produce non-zero Tanimoto distances (median 0.067) and measurable prediction variation (CV 0.000–0.015). With Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002), the model can *see* stereoisomers, but the signal remains weak — stereoisomer prediction CV is 10–50× lower than scaffold decorations (0.074–0.442) and random pairs (0.128–0.404). Consistency ratios < 1 for most endpoints indicate the model under-predicts the total observed within-group variation.

**Important caveat**: The dataset contains no repeat measurements of the same molecule (0 duplicate SMILES in the ML-ready set). Therefore, the observed activity differences between stereoisomers reflect a mixture of true stereoselective biological effects and assay variability — these contributions cannot be fully separated without replicate data. Measured within-group activity differences are substantial (e.g., median |Δ| of 38 mL/min/kg for MLM CLint, 15 µM for KSOL, 0.10 units for LogD), but an unknown fraction may be attributable to assay noise.

## Method

1. **Identify variant groups**:
   - **Stereoisomers** (548 groups, 1,152 molecules, 15.1% of dataset): Strip stereochemistry (`isomericSmiles=False`), group by achiral canonical SMILES → enantiomers/diastereomers share a group. Most groups are pairs (size 2), max size 4.
   - **Scaffold decorations** (1,050 groups, 3,835 molecules, 50.4%): Compute Murcko scaffolds, group by scaffold SMILES, filter to groups of size 2–20.
2. **Stereoisomer activity differences**: Compute within-group pairwise absolute activity differences for all 9 endpoints (raw, untransformed values). Saved to `stereoisomer_activity_diffs.csv`.
3. **Fingerprint characterization**: Compute intra-group chirality-aware Morgan fingerprint Tanimoto distances for each variant type; compare to random-pair baseline.
4. **Out-of-fold predictions**: Train Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) on chirality-aware Morgan fingerprints + RDKit 2D descriptors using cluster-split CV repeat 0.
5. **Consistency metrics**: For each group × endpoint where ≥2 members have predictions, compute prediction std, range, CV, and consistency ratio.

## Key Findings

### Stereoisomer activity differences (raw values)

| Endpoint | Units | Groups w/ data | n pairs | Median |Δ| | Mean |Δ| | Max |Δ| |
|----------|-------|---------------|---------|-----------|---------|---------|
| LogD | log units | 536 | 667 | 0.10 | 0.13 | 1.80 |
| KSOL | µM | 536 | 667 | 15.0 | 33.6 | 287.8 |
| HLM CLint | mL/min/kg | 319 | 373 | 5.7 | 18.6 | 484.2 |
| MLM CLint | mL/min/kg | 394 | 487 | 38.4 | 163.2 | 4,179.4 |
| Caco-2 Papp A>B | 10⁻⁶ cm/s | 263 | 285 | 0.94 | 1.75 | 19.2 |
| Caco-2 Efflux | ratio | 263 | 285 | 0.71 | 4.74 | 206.4 |
| MPPB | % unbound | 76 | 80 | 1.45 | 3.08 | 56.6 |
| MBPB | % unbound | 67 | 71 | 0.43 | 0.93 | 7.5 |
| MGMB | % unbound | 12 | 14 | 2.26 | 3.77 | 14.0 |

These differences are in raw (untransformed) endpoint-native units. Without replicate measurements, they represent an upper bound on true stereoselective effects — the actual biological difference is likely smaller after accounting for typical ADMET assay CVs (2–3-fold for clearance and solubility assays).

### Prediction consistency

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

### Interpretation

1. **Stereoisomer awareness is partial**: The model distinguishes stereoisomers for clearance and lipophilicity endpoints (LogD CV 0.015, HLM CLint CV 0.007), but three protein binding endpoints (MPPB, MBPB, MGMB) show effectively zero stereoisomer CV.
2. **Observed variation confounds biology with noise**: Without replicate measurements, we cannot determine how much of the stereoisomer activity differences reflect true stereoselective effects vs assay variability. The consistency ratio < 1 indicates the model under-predicts *total observed* variation, but the relevant biological signal may be smaller.
3. **Scaffold decoration amplification persists**: Consistency ratios > 1 for most endpoints, with the model amplifying substituent-level fingerprint changes beyond what the measured activity warrants.

## Plots

- `data/processed/2.13-seal-molecular-variants/fingerprint_distances.png` — Intra-group Tanimoto distance distributions
<!-- Paste: fingerprint_distances.png -->
- `data/processed/2.13-seal-molecular-variants/prediction_consistency.png` — Per-endpoint prediction CV boxplots
<!-- Paste: prediction_consistency.png -->
- `data/processed/2.13-seal-molecular-variants/spread_scatter.png` — Predicted range vs true activity range
<!-- Paste: spread_scatter.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py
```

## Source

`notebooks/2.13-seal-molecular-variants.py`
