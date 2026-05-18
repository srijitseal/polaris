# Expansion Tx ADMET dataset has high sparsity and confirms time-split feasibility

## Summary

Initial exploration of the ML-ready Expansion Tx dataset (7,608 molecules, 9 endpoints) reveals highly variable endpoint coverage (6–96%), heavy right-skew in clearance endpoints, and clear temporal separation between train and test splits via ordinal molecule indices. Physicochemical properties are mostly within Lipinski limits — these are not the heavy/flat profile sometimes ascribed to RNA-targeted compounds.

## Key Findings

### Dataset overview
- 7,608 molecules total: 5,326 train + 2,282 test, zero overlap
- 9 ML-ready endpoints (RLM CLint excluded — only in raw)

### Endpoint coverage

| Endpoint | Count | Coverage |
|----------|-------|----------|
| LogD | 7,309 | 96% |
| KSOL | 7,298 | 96% |
| MLM CLint | 5,692 | 75% |
| HLM CLint | 4,541 | 60% |
| Caco-2 Efflux | 3,777 | 50% |
| Caco-2 Papp A>B | 3,773 | 50% |
| MPPB | 1,756 | 23% |
| MBPB | 1,426 | 19% |
| MGMB | 431 | 6% |

- Median molecule has 4 endpoints measured (mean 4.73); distribution is multimodal with peaks at 2 (n=1,251), 4 (n=1,804), and 6 (n=1,364) endpoints
- Protein binding endpoints (MPPB, MBPB, MGMB) are severely sparse

### Endpoint distributions
- CLint endpoints (HLM, MLM) are heavily right-skewed with long tails (MLM max: 10,355 mL/min/kg)
- LogD is approximately normal (median 2.1, range -2.0 to 5.2)
- KSOL is bimodal (peaks near 0 and 200+ uM)
- Caco-2 Efflux has extreme outliers (max 607, median 2.0)

### Ordinal ordering confirms time-split
- Train median molecule index: 15,918
- Test median molecule index: 22,478
- Test molecules are clearly later in the ordinal sequence — validates time-split approach

### Physicochemical properties

| Property | Median | Mean | % Lipinski violation |
|----------|--------|------|----------------------|
| MolWt | 382.9 | 393.2 | 3.5% (>500) |
| MolLogP | 3.33 | 3.34 | 3.9% (>5) |
| NumHBA | 4 | 4.78 | 0.6% (>10) |
| NumHBD | 1 | 1.01 | 0.0% (>5) |
| NumAromaticRings | 3 | 3.33 | — |
| FractionCSP3 | 0.36 | 0.35 | — |
| TPSA | 63.2 | 65.9 | — |
| NumRotatableBonds | 5 | 4.71 | — |

Combining the four core Ro5 rules (MW ≤ 500, LogP ≤ 5, HBA ≤ 10, HBD ≤ 5), **93.5% of the dataset passes all four**. The coarse physicochemical panel here does not by itself show a distinctive RNA-targeting signature — median MW ~383, median 3 aromatic rings, FractionCSP3 ~0.36.

## Plots

<!-- Paste: data/processed/0.01-seal-dataset-exploration/endpoint_missingness.png -->

<!-- Paste: data/processed/0.01-seal-dataset-exploration/endpoint_distributions.png -->

<!-- Paste: data/processed/0.01-seal-dataset-exploration/ordinal_ordering.png -->

<!-- Paste: data/processed/0.01-seal-dataset-exploration/physicochemical_properties.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/0.01-seal-dataset-exploration.py
```

Outputs: `data/processed/0.01-seal-dataset-exploration/`

## Conclusion

The dataset is suitable for the planned generalization analyses, but endpoint sparsity means some endpoints (especially MGMB, MBPB, MPPB) may have insufficient data for robust splitting experiments. LogD and KSOL have near-complete coverage and should be prioritized. The ordinal ordering clearly supports time-split experiments. Physicochemical properties are mostly within Lipinski space (93.5% pass all four core Ro5 rules) — the coarse physchem panel does not by itself show a distinctive RNA-targeting signature.
