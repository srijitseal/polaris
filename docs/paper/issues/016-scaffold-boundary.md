> Note: please check [reproduce](#reproduce) for generating the figures referenced in this issue.

# Scaffold boundary analysis: scaffold splits are a blunt instrument you can't tune

## Summary

Scaffold splits give the user no control over the distribution shift being tested — and what they produce by default is structurally leaky and nearly indistinguishable from random. Despite the Murcko scaffold carrying real, endpoint-specific biological signal (same-scaffold pairs have smaller activity differences at any given fingerprint distance), the scaffold split procedure cannot leverage this because of degenerate group sizes (67.5% singletons) and pervasive cross-scaffold structural proximity (56.2% boundary violation rate). Distance-based methods, by contrast, let the user explicitly parametrize the degree of structural novelty.

## Hypothesis

The scaffold boundary is an untunable, arbitrary discretization of continuous chemical space. While the Murcko framework may encode biological information, the scaffold *split procedure* cannot leverage it, and distance-based approaches offer the flexibility to match the evaluation to the deployment scenario.

## Method

1. **Scaffold boundary violations**: For each molecule, identified the nearest neighbor overall and the nearest neighbor with a different Murcko scaffold. Computed the fraction whose overall 1-NN has a different scaffold (= boundary violation rate).
2. **Cross-scaffold proximity**: Compared distributions of overall 1-NN, cross-scaffold 1-NN, and within-scaffold 1-NN distances.
3. **Activity concordance**: For all ~29M molecule pairs per endpoint, binned by Tanimoto distance (width 0.05), computed mean |Δactivity| for same-scaffold vs. different-scaffold pairs. Mann-Whitney U test within each bin.

## Key Findings

### The scaffold boundary is structurally leaky

![Cross-scaffold proximity](../../../data/processed/2.15-araripe-scaffold-boundary/cross_scaffold_proximity.png)

*Figure 1. Left: overall vs. cross-scaffold 1-NN distance distributions nearly overlap (medians 0.188 vs. 0.233). Center: among non-singleton molecules, within-scaffold and cross-scaffold 1-NN distributions are comparable. Right: cumulative fraction of molecules with a cross-scaffold neighbor within a given distance threshold.*

| Metric | Value |
|--------|-------|
| Scaffold boundary violation rate | 56.2% (4,276/7,608 molecules) |
| Overall 1-NN median distance | 0.188 |
| Cross-scaffold 1-NN median distance | 0.233 |
| Within-scaffold 1-NN median (non-singletons) | 0.206 |
| Non-singletons with closer cross-scaffold neighbor | 36.8% |
| Unique Murcko scaffolds | 3,337 |
| Singleton scaffolds | 2,252 (67.5%) |

More than half of all molecules have their nearest neighbor in a *different* scaffold group. The scaffold boundary does not separate structurally similar molecules. The user has no knob to tune — the boundary is fixed by the Murcko decomposition.

### Scaffold membership carries endpoint-specific biological signal — but the split can't use it

![Activity concordance](../../../data/processed/2.15-araripe-scaffold-boundary/activity_concordance.png)

*Figure 2. Mean activity difference (|Δy|) as a function of Tanimoto distance, stratified by scaffold membership (same vs. different Murcko framework), for all 9 endpoints. At a given fingerprint distance, same-scaffold pairs tend to have smaller activity differences — but the magnitude and consistency of this effect is highly endpoint-specific.*

Of 133 Mann-Whitney U tests (9 endpoints x distance bins with sufficient data), 94 were significant at p < 0.05 (expected by chance: 6.7). Among significant tests, 84/94 (89%) showed same-scaffold pairs having *smaller* activity differences at the same fingerprint distance.

**Endpoint-specific patterns**:

| Endpoint | Pattern | Interpretation |
|----------|---------|----------------|
| Caco-2 Efflux | Strongest effect; same-scaffold |Δy| flat (~0.25) while different-scaffold rises to 0.52 | P-gp substrate recognition is highly scaffold-dependent |
| HLM CLint | Strong consistent effect across all distance bins | CYP450 metabolic soft-spot accessibility is scaffold-dependent |
| MLM CLint | Strong consistent effect, even stronger than HLM | Same CYP mechanism, confirmed across species |
| LogD | Moderate, consistent gap; both curves rise together | Scaffold and substituents both contribute to partitioning |
| Caco-2 Papp A>B | Moderate effect, weaker than Efflux | Passive permeability partly scaffold-dependent |
| KSOL | Effect *reverses* at high distances (>0.65): same-scaffold |Δy| *exceeds* different-scaffold | Crystal packing is scaffold-dependent but highly substituent-sensitive |
| MPPB | Moderate effect in mid-range distances | Surface hydrophobicity partly scaffold-driven |
| MBPB | Mostly non-significant | Protein binding may depend more on surface substituents |
| MGMB | Some significant bins but sparse data (431 molecules) | Underpowered |

This endpoint-specificity is itself the problem: a scaffold split applies the *same rigid boundary* regardless of endpoint. It cannot be tuned to match the biological relevance of the scaffold for the property being predicted.

### Scaffold splits give you no control; distance-based splits are parametric

![Boundary violation summary](../../../data/processed/2.15-araripe-scaffold-boundary/boundary_violation_summary.png)

*Figure 3. Left: scatter of within-scaffold vs. cross-scaffold 1-NN distance for non-singleton molecules — 36.8% have a closer neighbor in a different scaffold group (below diagonal). Right: fraction of same-scaffold pairs by distance bin, averaged across endpoints — same-scaffold pairs are a small minority at the distances relevant to model evaluation (>0.2).*

**Scaffold split**: Fixed by the Murcko decomposition. No parameters. Produces median test-to-train 1-NN of 0.246 (NB 2.11), yielding MA-RAE 0.534 — barely worse than random (0.480).

**Distance-based split**: The user controls cluster cutoff, number of clusters, minimum test-to-train distance. Cluster-based splitting produces median 1-NN of 0.424 and MA-RAE 0.724 (NB 2.11). The degree of structural novelty is explicitly parametrized and can be matched to the deployment scenario.

## Interpretation

The Murcko scaffold captures pharmacophore-relevant features — ring system identity, binding mode geometry, metabolic soft-spot accessibility — that fingerprints encode imperfectly. The activity concordance analysis confirms this: same-scaffold pairs have more predictable SAR, especially for endpoints governed by specific binding interactions (efflux, CYP metabolism).

But the scaffold *split procedure* cannot leverage this because:

1. **No tunability**: The boundary is fixed by the decomposition — the user cannot adjust the degree of distribution shift to match their deployment scenario.
2. **Degenerate size distribution**: 67.5% of scaffolds are singletons. Greedy assignment of singletons to folds is indistinguishable from random assignment.
3. **Structural leakage**: 56.2% of molecules have their nearest neighbor in a different scaffold group. The most informative training molecule sits on the wrong side of the boundary.
4. **Endpoint-agnostic**: The scaffold effect varies dramatically across endpoints (strong for Caco-2 Efflux, negligible for MBPB), but the split applies one boundary to all.

Distance-based methods avoid all four problems: the user sets the distance threshold, the grouping is based on the same feature space as the model, there are no singleton artifacts, and the split can be calibrated per endpoint if needed.

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.15-araripe-scaffold-boundary.py
```

## Source

`notebooks/2.15-araripe-scaffold-boundary.py`
