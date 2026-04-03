# Resonance form enumeration causes large ECFP4 fingerprint shifts — prediction CV 0.05–0.96 across endpoints

## Summary

Tautomeric/resonance form enumeration produces chemically equivalent molecules that are far apart in ECFP4 fingerprint space (median Tanimoto distance 0.494 vs 0.849 for random pairs). This fingerprint instability propagates to model predictions: Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) under cluster-split CV yields prediction CVs of 0.05–0.96 (median) across endpoints, with LogD most affected (median CV 0.962). Resonance forms are an underappreciated source of representation instability in molecular ML.

## Method

1. **Resonance enumeration**: Enumerated tautomeric/resonance forms for all 7,608 molecules using RDKit (NB 2.14). 7,017 molecules (92.2%) had >1 form (median 4, max 9).
2. **Fingerprint characterization**: Computed ECFP4 Tanimoto distances between all resonance form pairs within each group. Compared to random molecule pairs as baseline.
3. **Out-of-fold predictions**: Trained Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) on chirality-aware ECFP4 + RDKit 2D descriptors using cluster-split CV (5 folds, repeat 0). For each test molecule, predicted all resonance forms and measured prediction spread (CV = std/|mean|, range = max−min).

## Key Findings

### Resonance forms are far apart in fingerprint space

| Metric | Value |
|--------|-------|
| Molecules with >1 resonance form | 7,017 / 7,608 (92.2%) |
| Median resonance forms per molecule | 4 |
| Intra-group Tanimoto distance (median) | 0.494 |
| Random pair Tanimoto distance (median) | 0.849 |
| Intra-group pairs measured | 49,398 |

Resonance forms — chemically identical molecules — have a median fingerprint distance of 0.494. This is comparable to the distance between *different scaffold decorations* (0.349, NB 2.13) and far larger than stereoisomers (0.067). A distance of 0.49 would place a molecule squarely in the "novel" regime on performance-over-distance curves.

### Prediction instability is endpoint-specific

| Endpoint | Resonance CV (median) | Random CV (median) | Resonance CV (mean) | Random CV (mean) | n resonance | n random |
|----------|----------------------|-------------------|--------------------|--------------------|-------------|----------|
| LogD | 0.962 | 0.227 | 5.336 | 0.406 | 6,724 | 1,000 |
| Caco-2 Papp A>B | 0.221 | 0.191 | 0.435 | 0.262 | 3,513 | 1,000 |
| Caco-2 Efflux | 0.128 | 0.267 | 0.139 | 0.289 | 3,516 | 1,000 |
| HLM CLint | 0.113 | 0.168 | 0.131 | 0.199 | 4,169 | 1,000 |
| MLM CLint | 0.083 | 0.093 | 0.092 | 0.133 | 5,214 | 1,000 |
| KSOL | 0.052 | 0.100 | 0.066 | 0.147 | 6,723 | 1,000 |
| MBPB | 0.088 | 0.230 | 0.129 | 0.277 | 1,304 | 1,000 |
| MGMB | 0.067 | 0.139 | 0.078 | 0.175 | 431 | 431 |
| MPPB | 0.076 | 0.153 | 0.094 | 0.191 | 1,613 | 1,000 |

For most endpoints, resonance CV is *lower* than random-pair CV — as expected, since resonance forms are more similar than arbitrary molecule pairs. But the CVs are still substantial (median 0.05–0.96) for molecules that are *chemically identical*. LogD is an extreme outlier (mean CV 5.34) because values span zero, inflating the CV denominator.

Resonance CV exceeds random CV for LogD (0.962 vs 0.227) and Caco-2 Papp (0.221 vs 0.191). For all other endpoints, random pairs show higher variation — resonance instability is real but bounded.

### Comparison to other molecular variants (NB 2.13)

| Variant type | Median ECFP4 distance | Prediction CV range (median) |
|-------------|----------------------|-------------------|
| Stereoisomers | 0.067 | 0.000–0.015 |
| Scaffold decorations | 0.349 | varies by endpoint |
| **Resonance forms** | **0.494** | **0.05–0.96** |
| Random pairs | 0.849 | 0.09–0.23 |

Resonance forms produce larger fingerprint shifts than scaffold decorations, despite being chemically identical. Tautomeric rearrangements change bond orders, aromaticity flags, and — for tautomers — migrate hydrogens between heavy atoms, altering the molecular graph that Morgan fingerprints encode.

## Plots

- `data/processed/2.15-zalte-resonance-variants/fingerprint_distances.png` — Intra-group Tanimoto distance distributions
<!-- Paste: fingerprint_distances.png -->
- `data/processed/2.15-zalte-resonance-variants/prediction_consistency.png` — Per-endpoint prediction CV boxplots
<!-- Paste: prediction_consistency.png -->
- `data/processed/2.15-zalte-resonance-variants/consistency_heatmap.png` — Heatmap of mean prediction CV
<!-- Paste: consistency_heatmap.png -->
- `data/processed/2.15-zalte-resonance-variants/spread_scatter.png` — Predicted range vs true activity range
<!-- Paste: spread_scatter.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.14-zalte-resonance-generation.py
pixi run -e cheminformatics python notebooks/2.15-zalte-resonance-variants.py
```

## Source

`notebooks/2.14-zalte-resonance-generation.py`, `notebooks/2.15-zalte-resonance-variants.py`
