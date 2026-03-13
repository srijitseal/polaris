# ECFP4 is completely blind to stereochemistry — predictions are identical for enantiomers

## Summary

ECFP4 fingerprints produce *identical* bit vectors for stereoisomers (Tanimoto distance = 0.000), making XGBoost predictions perfectly consistent within stereoisomer groups (prediction CV = 0.000 across all 9 endpoints) but also completely unable to distinguish enantiomers or diastereomers that may have different biological activity. Meanwhile, scaffold decoration groups (same Murcko scaffold, different substituents) show moderate fingerprint distances (median 0.351) and prediction CV (0.07–0.27), consistently lower than random pairs (median distance 0.850, CV 0.12–0.30). The result exposes a fundamental limitation: ECFP-based models don't predict *inconsistently* for stereoisomers — they can't see them at all.

## Method

1. **Identify variant groups**:
   - **Stereoisomers** (548 groups, 1,152 molecules, 15.1% of dataset): Strip stereochemistry (`isomericSmiles=False`), group by achiral canonical SMILES → enantiomers/diastereomers share a group. Most groups are pairs (size 2), max size 4.
   - **Scaffold decorations** (1,050 groups, 3,835 molecules, 50.4%): Compute Murcko scaffolds, group by scaffold SMILES, filter to groups of size 2–20. Median group size 2, mean 3.7.
2. **Fingerprint characterization**: Compute intra-group ECFP4 Tanimoto distances for each variant type; compare to 678 random-pair baseline distances.
3. **Out-of-fold predictions**: Train default XGBoost on ECFP4 + RDKit 2D descriptors using cluster-split CV repeat 0 (matching 2.10). Collect one out-of-fold prediction per molecule across 9 endpoints (~45 models total).
4. **Consistency metrics**: For each group × endpoint where ≥2 members have predictions, compute prediction std, range, CV, and consistency ratio (prediction spread / true spread). Compare to random groups of matched sizes.

## Key Findings

### Fingerprint distances reveal a blind spot

| Variant type | n pairs | Mean distance | Median distance |
|---|---|---|---|
| Stereoisomers | 679 | 0.000 | 0.000 |
| Scaffold decorations | 9,661 | 0.360 | 0.351 |
| Random pairs | 678 | 0.826 | 0.850 |

Stereoisomer distances are exactly zero — ECFP4 (Morgan fingerprints without `useChirality=True`) strips all stereochemical information. This isn't "near-identical" — it's *mathematically identical*. The 15% of the dataset that consists of stereoisomers is invisible to the fingerprint.

### Prediction consistency: stereoisomers are perfectly (trivially) consistent

| Endpoint | Stereoisomer CV | Scaffold dec. CV | Random CV |
|---|---|---|---|
| LogD | 0.000 | 0.274 | 0.249 |
| KSOL | 0.000 | 0.128 | 0.179 |
| HLM CLint | 0.000 | 0.136 | 0.177 |
| MLM CLint | 0.000 | 0.099 | 0.159 |
| Caco-2 Papp A>B | 0.000 | 0.154 | 0.304 |
| Caco-2 Efflux | 0.000 | 0.165 | 0.289 |
| MPPB | 0.000 | 0.114 | 0.156 |
| MBPB | 0.000 | 0.170 | 0.305 |
| MGMB | 0.000 | 0.142 | 0.236 |

Stereoisomer prediction CV is 0.000 (or negligibly close) for every endpoint — because identical fingerprints produce identical predictions when molecules fall in the same test fold. The small non-zero values for HLM CLint (0.0001) and MLM CLint (0.00002) arise from RDKit 2D descriptors that capture minor stereochemical differences.

Scaffold decorations show prediction CV 2–3× lower than random pairs across most endpoints, meaning the model does learn scaffold-level trends. The exception is LogD, where scaffold decoration CV (0.274) slightly exceeds random CV (0.249) — LogD varies substantially within scaffolds.

### Consistency ratio: model smooths over true biological variation

| Endpoint | Stereoisomer ratio | Scaffold dec. ratio | Random ratio |
|---|---|---|---|
| LogD | 0.000 | 1.059 | 1.195 |
| KSOL | 0.050 | 4.579 | 8.029 |
| HLM CLint | 0.001 | 1.843 | 1.139 |
| MLM CLint | 0.000 | 1.827 | 1.545 |
| Caco-2 Papp A>B | 0.000 | 2.760 | 1.951 |
| Caco-2 Efflux | 0.000 | 2.015 | 2.635 |
| MPPB | 0.000 | 1.191 | 3.221 |
| MBPB | 0.000 | 2.214 | 3.880 |
| MGMB | 0.000 | 2.703 | 17.483 |

Stereoisomer ratio is ~0 everywhere: the model predicts zero spread despite real biological differences between enantiomers (true std ~0.02–0.14). For scaffold decorations, ratios > 1 indicate the model *amplifies* substituent differences — particularly for KSOL (4.6×) and Caco-2 Papp A>B (2.8×). This suggests the model over-reacts to fingerprint changes from scaffold decorations while being completely blind to stereochemistry.

### Interpretation

The results reveal two distinct failure modes:

1. **Stereoisomer blindness**: ECFP4 encodes no stereochemical information by default, so stereoisomers get identical predictions regardless of real activity differences. For datasets where stereochemistry matters (e.g., chiral drug metabolism), this is a silent failure — the model appears perfectly consistent but is missing real biology. The fix is straightforward: use `useChirality=True` in Morgan fingerprint generation, or add 3D descriptors.

2. **Scaffold decoration amplification**: The model shows lower prediction CV than random pairs (good — it has learned scaffold-level chemistry) but consistency ratios > 1 mean it amplifies substituent effects beyond what the true activity warrants. This is the fingerprint artifact Srijit highlighted: minor decorations on a shared scaffold produce disproportionate fingerprint changes, which the model translates into disproportionate prediction changes.

The practical takeaway: always check whether your molecular representation captures the structural variation that matters for your endpoint. ECFP4 gives perfect consistency for stereoisomers not because it has learned that enantiomers should be similar, but because it literally cannot tell them apart.

## Plots

- `data/processed/2.13-seal-molecular-variants/fingerprint_distances.png` — Intra-group Tanimoto distance distributions (stereoisomers vs scaffold decorations vs random)
<!-- Paste: fingerprint_distances.png -->
- `data/processed/2.13-seal-molecular-variants/prediction_consistency.png` — Per-endpoint boxplots of within-group prediction CV
<!-- Paste: prediction_consistency.png -->
- `data/processed/2.13-seal-molecular-variants/consistency_heatmap.png` — Heatmap of mean prediction CV (endpoints × variant types)
<!-- Paste: consistency_heatmap.png -->
- `data/processed/2.13-seal-molecular-variants/spread_scatter.png` — Predicted range vs true activity range per variant type
<!-- Paste: spread_scatter.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.13-seal-molecular-variants.py
```

## Source

`notebooks/2.13-seal-molecular-variants.py`
