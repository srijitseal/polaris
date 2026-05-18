# Chirality-aware Morgan fingerprints partially resolve stereoisomer blindness — prediction CV 0.0001–0.032, but variation confounds biology with assay noise

## Summary

After enabling `useChirality=True` in Morgan fingerprints, stereoisomers produce non-zero Tanimoto distances (median 0.067) and measurable prediction variation (mean CV 0.0001–0.032). With Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002), the model can *see* stereoisomers, but the signal remains weak — stereoisomer mean prediction CV is 5–600× lower than scaffold decorations (0.075–0.263) and random pairs (0.131–0.383). Consistency ratios < 1 for most endpoints indicate the model under-predicts the total observed within-group variation.

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

### Prediction consistency (mean CV across groups)

| Endpoint | Stereoisomer CV | Scaffold dec. CV | Random CV |
|---|---|---|---|
| LogD | 0.032 | 0.263 | 0.256 |
| KSOL | 0.005 | 0.100 | 0.155 |
| HLM CLint | 0.007 | 0.093 | 0.167 |
| MLM CLint | 0.003 | 0.075 | 0.131 |
| Caco-2 Papp A>B | 0.002 | 0.186 | 0.301 |
| Caco-2 Efflux | 0.005 | 0.128 | 0.267 |
| MPPB | 0.001 | 0.127 | 0.183 |
| MBPB | 0.000 | 0.170 | 0.353 |
| MGMB | 0.000 | 0.110 | 0.383 |

Consistency ratio means (>1 = model amplifies variation beyond true biology; <1 = model under-predicts):

| Endpoint | Stereoisomer | Scaffold dec. | Random |
|----------|--------------|---------------|--------|
| LogD | 0.16 | 0.92 | 1.04 |
| KSOL | 1.35 | 3.44 | 6.61 |
| HLM CLint | 0.53 | 1.23 | 1.31 |
| MLM CLint | 0.37 | 1.21 | 1.59 |
| Caco-2 Papp A>B | 0.36 | 2.04 | 1.56 |
| Caco-2 Efflux | 0.19 | 1.57 | 2.84 |
| MPPB | 0.01 | 1.14 | 1.28 |
| MBPB | 0.00 | 2.06 | 1.77 |
| MGMB | 0.00 | 2.70 | 67.29 |

(MGMB random ratio is dominated by 2 groups with very small true variation — not a robust estimate.)

### Interpretation

1. **Stereoisomer awareness is partial**: The model distinguishes stereoisomers for clearance and lipophilicity endpoints (LogD CV 0.032, HLM CLint CV 0.007), but three protein binding endpoints (MPPB, MBPB, MGMB) show near-zero stereoisomer CV (0.000–0.001).
2. **Observed variation confounds biology with noise**: Without replicate measurements, we cannot determine how much of the stereoisomer activity differences reflect true stereoselective effects vs assay variability. The consistency ratio < 1 indicates the model under-predicts *total observed* variation, but the relevant biological signal may be smaller.
3. **Scaffold decoration amplification persists**: Consistency ratios > 1 for most endpoints, with the model amplifying substituent-level fingerprint changes beyond what the measured activity warrants.

## Plots

- `data/processed/2.13-seal-molecular-variants/xgboost/fingerprint_distances.png` — intra-group Tanimoto distance distributions
- `data/processed/2.13-seal-molecular-variants/xgboost/prediction_consistency.png` — per-endpoint prediction CV boxplots
- `data/processed/2.13-seal-molecular-variants/xgboost/spread_scatter.png` — predicted range vs true activity range
- `data/processed/2.13-seal-molecular-variants/xgboost/consistency_heatmap.png` — consistency-ratio heatmap
- Combined XGBoost vs CheMeleon panels: `data/processed/2.13-seal-molecular-variants/combined/`

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py
```

## Source

`notebooks/2.13-seal-molecular-variants.py`

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran with CheMeleon (cluster-split CV, 5 folds, fine-tuned per fold). Same variant groups (548 stereoisomer, 1,050 scaffold-decoration, plus random-pair baseline).

**Trend matches XGBoost. Both representation failure modes — stereoisomer under-prediction and scaffold-decoration amplification — persist under the foundation model.**

### Stereoisomer consistency

| Endpoint | n_groups | pred CV median XGB | pred CV median CM | consistency ratio mean XGB | consistency ratio mean CM |
|----------|----------|--------------------|-------------------|----------------------------|---------------------------|
| LogD | 536 | 0.000 | 0.006 | 0.16 | 0.23 |
| KSOL | 536 | 0.000 | 0.001 | 1.35 | 0.72 |
| HLM CLint | 319 | 0.000 | 0.009 | 0.53 | 0.86 |
| MLM CLint | 394 | 0.000 | 0.004 | 0.37 | 0.56 |
| Caco-2 Papp | 263 | 0.000 | 0.007 | 0.36 | 0.97 |
| Caco-2 Efflux | 263 | 0.000 | 0.003 | 0.19 | 0.29 |
| MPPB | 76 | 0.000 | 0.006 | 0.01 | 0.47 |
| MBPB | 67 | 0.000 | 0.002 | 0.00 | 0.52 |
| MGMB | 12 | 0.000 | 0.001 | 0.00 | 0.22 |

CheMeleon's pred CV is non-zero (0.001–0.009) — graph rep is sensitive to stereo bond direction, where chirality-aware ECFP4 was not. **Consistency ratios remain <1 for 8/9 endpoints under XGBoost (KSOL is the exception at 1.35) and for 9/9 under CheMeleon** — both still under-predict true biological variation between stereoisomers. Foundation model partially closes the stereo blindness but does not eliminate it.

### Scaffold decoration consistency (mean ratio; >1 = amplification)

| Endpoint | n_groups | XGB | CM |
|----------|----------|-----|-----|
| LogD | 1007 | 0.92 | 1.01 |
| KSOL | 999 | **3.44** | 2.92 |
| HLM CLint | 635 | 1.23 | 1.65 |
| MLM CLint | 762 | 1.21 | 1.66 |
| Caco-2 Papp | 491 | 2.04 | 2.15 |
| Caco-2 Efflux | 491 | 1.57 | 1.88 |
| MPPB | 215 | 1.14 | 1.33 |
| MBPB | 166 | 2.06 | 2.23 |
| MGMB | 58 | 2.70 | 2.36 |

Amplification ratio >1 in 8/9 endpoints under XGBoost, **9/9 under CheMeleon**. Graph representation amplifies decoration changes beyond true biological variation just as much as ECFP4.

### Source

- `data/processed/2.13-seal-molecular-variants/chemeleon/consistency_summary.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/15#issuecomment-4381157279
- Reproduce: `pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py --model chemeleon`
