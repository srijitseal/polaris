# CheMeleon is not a "smoother" representation than ECFP4 — the amplification of small structural edits is scale-dependent, not architecture-dependent

## Summary

The 2.13 write-up explains scaffold-decoration amplification under XGBoost as a consequence of Morgan discreteness (one decorating atom flips many bits) and asserts that the graph foundation model's representation "is likely smoother." Notebook `2.18` tests that assertion directly against the frozen pretrained CheMeleon embedding.

**The assertion does not hold as stated.** Whether CheMeleon looks smoother depends entirely on which distance is used on its embedding:

- **CheMeleon cosine**: median decoration pair sits at 0.17× the median random-pair distance, vs 0.41× for ECFP4 Tanimoto → looks 2.4× smoother.
- **CheMeleon Euclidean**: 0.41× — indistinguishable from ECFP4.

Same embedding, same pairs, opposite conclusion. Ratio-to-background is not invariant to a monotone rescaling of the distance, so it cannot support a claim about the representation itself.

On a **scale-free** measure — the percentile rank of a pair's partner among all 7,608 molecules, invariant to any monotone transform — the two representations are **effectively identical**, and CheMeleon is marginally *coarser*: median neighbour rank 0.118% (ECFP4) vs 0.145% (CheMeleon, both metrics). A decoration partner is roughly the 9th-nearest molecule under ECFP4 and the 11th-nearest under CheMeleon.

The conclusion the paper wants — that amplification is intrinsic to predicting properties from local-substructure encodings, not an artifact of circular-fingerprint discreteness — survives, but for a different reason than stated. It is not that amplification persists *despite* a smooth representation; it is that **the pretrained graph embedding is not measurably smoother in the first place**, so it was never a test of discreteness.

Where the representations genuinely differ is stereochemistry, not decoration: ECFP4 places 9.0% of stereoisomer pairs at exactly zero distance; CheMeleon places 0.3% there.

## Method

Notebook: `notebooks/2.18-seal-representation-sensitivity.py`

Matched pairs from the 2.13 variant groups (`variant_groups.csv`) provide controlled minimal edits:

- **stereoisomer** — 679 pairs, identical constitution, differing only in stereochemistry
- **scaffold_decoration** — 9,661 pairs, shared Murcko scaffold, differing peripheral substituents

Three representations over all 7,608 dataset molecules:

1. **ECFP4 Tanimoto** — chirality-aware Morgan, radius 2, 2,048 bits (matching every modelling notebook).
2. **CheMeleon cosine** and **3. CheMeleon Euclidean** — the frozen pretrained `BondMessagePassing` block from `chemeleon_mp.pt` in eval mode, molecule vector = mean aggregation of atom states. No fine-tuning, so this measures the foundation model's representation rather than any endpoint-specific head. Cached to `data/interim/chemeleon_pretrained_embeddings.npz`.

Two scales:

- **ratio-to-background** — median pair distance / median random-pair distance (200,000 random pairs). Interpretable, but metric-dependent.
- **neighbour rank** — percentile rank of the partner among all 7,608 molecules by distance from the pair's first member. Invariant to any monotone transform of the distance, so it compares representations fairly.

Also reported: blind-spot rate (pairs at exactly zero distance) and perturbation size stratified by heavy-atom difference within the pair.

## Key Findings

### Sensitivity summary

| Representation | Variant | n pairs | Median distance | Ratio to background | Median neighbour rank | Zero-distance pairs |
|---|---|---|---|---|---|---|
| ECFP4 Tanimoto | stereoisomer | 679 | 0.0667 | 0.078 | 0.0131% | **8.98%** |
| ECFP4 Tanimoto | scaffold decoration | 9,661 | 0.3494 | **0.411** | **0.118%** | 0.18% |
| CheMeleon cosine | stereoisomer | 679 | 0.00002 | 0.0001 | 0.0131% | **0.29%** |
| CheMeleon cosine | scaffold decoration | 9,661 | 0.0482 | **0.172** | **0.145%** | 0.00% |
| CheMeleon Euclidean | stereoisomer | 679 | 0.0155 | 0.008 | 0.0131% | 0.15% |
| CheMeleon Euclidean | scaffold decoration | 9,661 | 0.8350 | **0.409** | **0.145%** | 0.00% |

The two bolded columns tell opposite stories. Ratio-to-background separates cosine (0.172) from ECFP4 (0.411); Euclidean on the *identical* embedding gives 0.409. Neighbour rank — which cannot be gamed by rescaling — puts all three within 0.118–0.145%.

### 1. "Smoother" is a property of the chosen metric, not the representation

Cosine and Euclidean distance on the same 2,048-d embedding are not monotone transforms of one another (vector norms vary across molecules), so they disagree on ratio-to-background. Any claim of the form "representation A perturbs less relative to typical distances than representation B" is therefore unfalsifiable without fixing a metric — and the two natural choices here land on opposite sides of ECFP4.

### 2. On a scale-free measure the representations are near-identical

Median neighbour rank for decoration pairs: 0.118% (ECFP4) vs 0.145% (CheMeleon, both metrics) of a 7,608-molecule library. For stereoisomer pairs all three give a median rank of 0.0131% — a single molecule closer. CheMeleon does not localise minimal edits any more tightly than ECFP4 does; if anything, slightly less.

### 3. ECFP4 discreteness is real and large, but it does not produce excess *relative* displacement

A typical molecule sets 54 of 2,048 ECFP4 bits. A one-heavy-atom decoration difference flips a median of **19 bits — 35% of a molecule's on-bits**, and pairs differing by zero heavy atoms (isomeric rearrangements) still flip a median of 23. The mechanism described in 2.13 is confirmed: a single-atom change is not a single-bit change.

But that discreteness does not translate into an outsized position in the distance distribution. Both representations grow smoothly and similarly with edit size:

| Heavy-atom difference | n pairs | Median bits flipped | ECFP4 relative | CheMeleon cosine relative | CheMeleon Euclidean relative |
|---|---|---|---|---|---|
| 0 | 2,065 | 23 | 0.403 | 0.135 | 0.360 |
| 1 | 3,829 | 19 | 0.362 | 0.131 | 0.350 |
| 2 | 1,829 | 26 | 0.443 | 0.204 | 0.450 |
| 3 | 1,053 | 28 | 0.470 | 0.283 | 0.523 |
| 4 | 484 | 30 | 0.515 | 0.369 | 0.590 |
| 5 | 175 | 34 | 0.549 | 0.448 | 0.646 |
| 6 | 85 | 31 | 0.533 | 0.399 | 0.651 |
| 7 | 54 | 39.5 | 0.646 | 0.560 | 0.685 |
| 8 | 26 | 39.5 | 0.635 | 0.621 | 0.744 |

CheMeleon Euclidean tracks ECFP4 almost exactly across the whole range. Cosine starts lower but converges by 8 heavy atoms. Pairwise decoration distances correlate strongly across representations (Spearman ρ = 0.688, ECFP4 vs CheMeleon cosine).

### 4. The genuine difference is stereochemistry

ECFP4's zero-distance rate for stereoisomer pairs is 9.0% (61/679) — the blind spot characterised in #15, caused by RDKit only folding an atom's chiral tag into its environment hash once the substituent branches become distinguishable within the radius; when they aren't, `useChirality=True` has no effect at that atom. The Chemprop atom featurizer encodes the chiral tag directly (2 of 72 atom features differ between enantiomeric atoms), so CheMeleon's blind-spot rate is 0.29% (cosine) / 0.15% (Euclidean).

The caveat: CheMeleon's stereoisomer distances, while non-zero, are vanishingly small (median cosine 2×10⁻⁵, ratio-to-background 0.0001, vs 0.078 for ECFP4) and correlate barely at all with ECFP4's (ρ = 0.108). The signal exists but is far below the representation's own noise floor — consistent with 2.13's finding that both models under-predict stereoisomer variation.

### 5. Separately: the amplification result itself is an artifact of mean aggregation

Re-aggregating 2.13's per-group consistency ratios by median as well as mean (`consistency_ratio_robustness.csv`) changes the direction of the finding:

| Model | Endpoint | n groups | Ratio mean | Ratio median | % groups > 1 | mean/median |
|---|---|---|---|---|---|---|
| XGBoost | LogD | 978 | 0.917 | 0.783 | 29.0 | 1.17 |
| XGBoost | KSOL | 994 | 3.444 | 0.812 | 44.6 | 4.24 |
| XGBoost | HLM CLint | 633 | 1.228 | 0.459 | 19.7 | 2.68 |
| XGBoost | MLM CLint | 762 | 1.205 | 0.462 | 22.3 | 2.61 |
| XGBoost | Caco-2 Papp A>B | 490 | 2.042 | 0.524 | 26.7 | 3.90 |
| XGBoost | Caco-2 Efflux | 490 | 1.570 | 0.534 | 28.4 | 2.94 |
| XGBoost | MPPB | 214 | 1.135 | 0.670 | 29.0 | 1.69 |
| XGBoost | MBPB | 164 | 2.055 | 0.729 | 31.1 | 2.82 |
| XGBoost | MGMB | 57 | 2.699 | 0.658 | 36.8 | 4.10 |
| CheMeleon | LogD | 978 | 1.015 | 0.905 | 37.7 | 1.12 |
| CheMeleon | KSOL | 994 | 2.925 | 0.835 | 45.1 | 3.50 |
| CheMeleon | HLM CLint | 633 | 1.646 | 0.619 | 30.2 | 2.66 |
| CheMeleon | MLM CLint | 762 | 1.665 | 0.672 | 32.8 | 2.48 |
| CheMeleon | Caco-2 Papp A>B | 490 | 2.154 | 0.613 | 32.2 | 3.52 |
| CheMeleon | Caco-2 Efflux | 490 | 1.877 | 0.701 | 33.5 | 2.68 |
| CheMeleon | MPPB | 214 | 1.327 | 0.862 | 38.8 | 1.54 |
| CheMeleon | MBPB | 164 | 2.230 | 0.972 | 47.6 | 2.29 |
| CheMeleon | MGMB | 57 | 2.359 | 0.877 | 40.4 | 2.69 |

**Mean ratio > 1 in 8/9 (XGBoost) and 9/9 (CheMeleon) endpoints. Median ratio > 1 in 0/9 under both models.** Only 20–48% of decoration groups have a ratio above 1.

The consistency ratio is a quotient of two positive quantities and is heavily right-skewed (mean/median 1.1–4.2); its mean is dominated by the minority of groups whose *measured* within-group variation is near zero, which inflates the quotient without implying the prediction spread is large. In the typical decoration group, both models **under-predict** the observed variation — the same direction as the stereoisomer result, not the opposite.

This does not erase the finding, but it restates it: amplification is concentrated in a minority of decoration groups rather than being the models' typical behaviour. The claim "both representations amplify substituent effects beyond true biological variation" is not supported as a general statement.

### 6. The "2–3-fold lower CV than random pairs" figure

Recomputed per endpoint (`prediction_cv_comparison.csv`), using group medians: **1.3–4.6-fold** lower, not 2–3-fold. LogD is the weakest (1.32× XGBoost, 1.31× CheMeleon) and MGMB the strongest (4.51×, 4.59×). Mean-based fold changes are less stable still — CheMeleon LogD gives 0.34× (i.e. *higher* CV than random) because a handful of near-zero-mean predictions blow up the CV.

## Implications for the manuscript

The sentence "the graph foundation model's representation is likely smoother" should be removed or replaced. Two defensible replacements:

1. **Drop the mechanism claim**: state only that amplification is observed under both a discrete circular fingerprint and a continuous pretrained graph embedding, so it is not specific to fingerprint discreteness.
2. **State the measured result**: the pretrained CheMeleon embedding is *not* measurably smoother than ECFP4 on a scale-free measure (median neighbour rank 0.145% vs 0.118%), so the persistence of amplification across architectures reflects a shared reliance on local-substructure similarity rather than a contrast between discrete and smooth encodings.

Option 2 is stronger and now has data behind it. The word "likely" in the current draft is the tell that this was speculation — it should not survive into the manuscript in either form.

## Outputs

`data/processed/2.18-seal-representation-sensitivity/`

- `sensitivity_summary.csv` — one row per (representation, variant type)
- `pair_distances.csv` — 10,340 pairs with distances, relative distances, neighbour ranks, bit-flip counts, heavy-atom differences
- `decoration_size_effect.csv` — perturbation size vs heavy-atom difference
- `metric_dependence.csv` — ratio-to-background under monotone transforms of the ECFP4 distance
- `consistency_ratio_robustness.csv` — mean vs median consistency ratio and % groups > 1, both models
- `prediction_cv_comparison.csv` — decoration vs random prediction CV, mean and median based
- `representation_sensitivity.png` — relative-distance distributions (top) and scale-free neighbour-rank ECDFs (bottom), per variant type
- `decoration_size_effect.png` — bits flipped and relative distance vs heavy-atom difference
- `consistency_ratio_robustness.png` — mean vs median consistency ratio per endpoint, both models

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.18-seal-representation-sensitivity.py
```

Requires `data/processed/2.13-seal-molecular-variants/variant_groups.csv`. CheMeleon weights auto-download from Zenodo on first run; embeddings are cached thereafter.

## Related

- #15 — molecular variants: stereoisomer blindness and scaffold-decoration amplification (the analysis this one audits)
- #21 — resonance form ambiguity (the other representation-failure mode)
