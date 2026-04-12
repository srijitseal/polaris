# ExpansionRX compounds are structurally novel relative to ChEMBL 36 — 98.2% have no close analogue (Tc > 0.7) in public chemical matter

## Summary

Max Tanimoto similarity analysis of all 7,608 ExpansionRX compounds against 2.85M ChEMBL 36 compounds shows that this RNA-targeted dataset occupies chemical space poorly represented in public databases. The median max similarity to any ChEMBL compound is just 0.444, and 98.2% of molecules have no close analogue (Tc > 0.7) in all of ChEMBL. Against the ADME-assayed subset (308K compounds), novelty is even more pronounced: median max similarity drops to 0.388, with 55.7% below 0.4.

## Method

1. Computed 2048-bit Morgan fingerprints (radius 2, useChirality=True) for all 7,608 ExpansionRX compounds
2. Computed Morgan fingerprints for all 2,854,639 ChEMBL 36 compounds and a 307,652-compound ADME-assayed subset (assay_type = 'A')
3. For each ExpansionRX compound, computed max Tanimoto similarity to any compound in each reference set

## Key Findings

### Most ExpansionRX compounds have no close public analogue

| Metric | All ChEMBL (n=2,854,639) | ADME subset (n=307,652) |
|--------|--------------------------|------------------------|
| Median max Tc | 0.444 | 0.388 |
| Mean max Tc | 0.461 | 0.403 |
| % below Tc 0.4 | 25.0% | 55.7% |
| % below Tc 0.7 | 98.2% | 99.8% |
| % below Tc 0.85 | 99.8% | 100.0% |
| Min max Tc | 0.286 | 0.253 |
| Max max Tc | 1.000 | 1.000 |

### Interpretation

1. **Structural novelty is extreme**: Only 1.8% of ExpansionRX compounds have a ChEMBL neighbour within Tc > 0.7. The median max similarity of 0.444 falls in the "weakly similar" regime where ECFP4 Tanimoto correlations with shared activity break down.
2. **ADME data coverage is even sparser**: Against ChEMBL compounds with ADME assay data, the median drops to 0.388 — more than half of ExpansionRX molecules have no ADME-measured analogue above Tc 0.4.
3. **RNA-targeting chemical space is distinct**: These compounds were designed for RNA targets, yielding larger, flatter, more polar molecules than typical protein-targeting drug candidates.
4. **Implications for benchmarking**: Public ADMET benchmarks predominantly draw from ChEMBL. Models benchmarked on these datasets evaluate interpolation within well-covered chemical space. Performance on ExpansionRX represents a genuine extrapolation test.

## Plots

- `data/processed/0.03-araripe-chembl-tanimoto/max_tanimoto_all_chembl.png` — Max Tanimoto distribution vs all ChEMBL
<!-- Paste: max_tanimoto_all_chembl.png -->
- `data/processed/0.03-araripe-chembl-tanimoto/max_tanimoto_comparison.png` — All ChEMBL vs ADME subset comparison
<!-- Paste: max_tanimoto_comparison.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/0.03-araripe-chembl-tanimoto.py
```

## Source

`notebooks/0.03-araripe-chembl-tanimoto.py`
