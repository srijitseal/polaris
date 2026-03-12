# Expansion Tx ADMET dataset has high sparsity and confirms time-split feasibility

## Summary

Initial exploration of the ML-ready Expansion Tx dataset (7,608 molecules, 9 endpoints) reveals highly variable endpoint coverage (6-96%), heavy right-skew in clearance endpoints, and clear temporal separation between train and test splits via ordinal molecule indices. Physicochemical properties suggest these RNA-targeting compounds frequently exceed Lipinski limits, consistent with the hypothesis that they differ from typical protein modulators.

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

- Most molecules have 3-5 endpoints measured
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

### Physicochemical properties (RNA-targeting hypothesis)
- Median MW ~480, but substantial fraction exceeds Lipinski 500 limit
- LogP centered ~3.5, mostly within Lipinski limit of 5
- High HBA counts (median ~6, many exceed Lipinski limit of 10)
- High aromatic ring count (median ~4-5) and low FractionCSP3 (~0.2) — consistent with RNA-binding requiring planar, aromatic scaffolds
- These properties differ from typical protein-targeting drug-like compounds

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

The dataset is suitable for the planned generalization analyses, but endpoint sparsity means some endpoints (especially MGMB, MBPB, MPPB) may have insufficient data for robust splitting experiments. LogD and KSOL have near-complete coverage and should be prioritized. The ordinal ordering clearly supports time-split experiments. The RNA-targeting physicochemical profile (high MW, high aromaticity, low sp3 fraction) is a distinctive feature worth highlighting in the paper.
