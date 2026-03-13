# OOD performance degrades 1.2–12.9x vs IID when evaluating across chemical series

## Summary

Training XGBoost on the largest chemical series (Butina cluster 0, n=2,578) and evaluating on held-out molecules from the same series (IID) vs a different series (cluster 2, n=1,306, OOD) shows consistent and often dramatic performance degradation. This is the "hero" example (paper Fig 7) — it works because this dataset uniquely provides both temporal ordering (for time-split within a series) and chemical series structure (for OOD evaluation across series). The result is intuitive but quantifying it requires dataset properties that public benchmarks almost never have.

## Method

1. **Select two largest Butina clusters** (cutoff 0.7, from notebook 2.01): cluster 0 (n=2,578) and cluster 2 (n=1,306). These represent two distinct chemical series from the Expansion Tx drug discovery campaigns.
2. **Time-split cluster 0** using ordinal molecule index (E-XXXXXXX naming encodes temporal order): earlier 80% = train (n=2,062), later 20% = IID validation (n=516). This mimics how a medicinal chemist would deploy a model: train on past compounds, predict new ones in the same series.
3. **Cluster 2 = OOD test set** (n=1,306). This mimics predicting an entirely different chemical series — a harder but common real-world scenario (e.g., hit-to-lead on a new scaffold).
4. **Train default XGBoost** (`XGBRegressor(random_state=42)`, no hyperparameter tuning) per endpoint on ECFP4 (2048-bit) + full RDKit 2D descriptors (~200), with dimorphite_dl protonation at assay-relevant pH. No tuning is intentional — this is a case study about distribution shift, not model optimization.
5. **Compare squared error distributions** between IID and OOD predictions. Log-transform applied to all endpoints except LogD, matching the competition evaluation protocol.

## Key Findings

### Split characterization

| Set | n | Cluster | 1-NN median | 1-NN mean |
|-----|---|---------|-------------|-----------|
| Train | 2,062 | 0 (largest) | — | — |
| IID val | 516 | 0 (time-split, later 20%) | 0.282 | 0.295 |
| OOD test | 1,306 | 2 (second largest) | 0.761 | 0.761 |

The distance distributions have near-zero overlap — IID molecules are structurally close to training data (same chemical series, Tanimoto distance ~0.28), while OOD molecules are far away (different series, ~0.76). This clean separation is what makes the comparison compelling: any performance difference can be attributed to distribution shift rather than noise.

### Per-endpoint performance

| Endpoint | IID R² | OOD R² | IID Spearman | OOD Spearman | IID RAE | OOD RAE | IID median SE | OOD median SE | SE fold-change |
|----------|--------|--------|-------------|-------------|---------|---------|--------------|--------------|----------------|
| MLM CLint | -0.06 | -2.15 | 0.512 | 0.025 | 0.983 | 1.863 | 0.085 | 1.094 | **12.9x** |
| KSOL | 0.24 | -0.02 | 0.501 | 0.269 | 0.824 | 0.947 | 0.039 | 0.393 | **10.0x** |
| MPPB | 0.08 | -0.02 | 0.421 | 0.108 | 0.920 | 1.015 | 0.016 | 0.112 | **7.1x** |
| HLM CLint | -0.61 | -0.17 | 0.305 | 0.060 | 1.160 | 1.114 | 0.048 | 0.257 | **5.3x** |
| LogD | 0.53 | 0.09 | 0.750 | 0.315 | 0.637 | 0.960 | 0.138 | 0.596 | **4.3x** |
| Caco-2 Efflux | 0.48 | -0.37 | 0.737 | -0.005 | 0.608 | 1.140 | 0.052 | 0.136 | **2.6x** |
| MBPB | 0.14 | -0.17 | 0.439 | 0.414 | 0.919 | 1.174 | 0.034 | 0.086 | **2.5x** |
| Caco-2 Papp A>B | 0.47 | -0.15 | 0.697 | 0.126 | 0.667 | 1.081 | 0.053 | 0.063 | **1.2x** |

R² goes negative on OOD for 7 of 8 endpoints — the model is worse than predicting the mean when deployed on a new chemical series. Only LogD retains positive (but low) R² on OOD, likely because LogD is the most well-covered endpoint (96% of molecules) and is driven by global physicochemical properties rather than series-specific SAR.

Spearman ρ collapses on OOD: from 0.50–0.75 (IID) to near-zero or negative (OOD) for most endpoints. RAE exceeds 1.0 on OOD for 6 of 8 endpoints, meaning the model is worse than predicting the mean — a baseline that requires no training at all.

The clearance endpoints (MLM CLint, HLM CLint) show the largest degradation, consistent with metabolic stability being highly sensitive to specific structural features within a chemical series. Caco-2 Papp A>B shows the smallest gap (1.2x), suggesting permeability may generalize better across series — possibly because it depends more on bulk molecular properties (size, polarity) than on specific pharmacophoric features.

MGMB was skipped due to insufficient data in both clusters (only 431 molecules total, sparsely distributed).

### Why this matters

Standard IID evaluation (random split or even time-split within a single series) dramatically overestimates real-world performance. When a model is deployed on a genuinely new chemical series — the common scenario in hit identification — performance collapses. This analysis is only possible because the Expansion Tx dataset provides:
- **Chemical series structure**: Butina clustering identifies coherent series with hundreds of members
- **Temporal ordering**: Ordinal molecule naming enables realistic time-split within a series

Public ADMET benchmarks typically lack both properties, making this kind of IID vs OOD decomposition impossible.

## Plots

- `data/processed/2.09-seal-iid-vs-ood-series/distance_characterization.png` — 1-NN distance distributions showing clean IID/OOD separation
<!-- Paste: distance_characterization.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/squared_error_distributions.png` — 3x3 grid of per-endpoint SE histograms (log-scale x-axis)
<!-- Paste: squared_error_distributions.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/median_se_by_endpoint.png` — Per-endpoint median SE comparison
<!-- Paste: median_se_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/mae_by_endpoint.png` — Per-endpoint MAE comparison
<!-- Paste: mae_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/r2_by_endpoint.png` — Per-endpoint R² comparison
<!-- Paste: r2_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/spearman_by_endpoint.png` — Per-endpoint Spearman ρ comparison
<!-- Paste: spearman_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/kendall_by_endpoint.png` — Per-endpoint Kendall τ comparison
<!-- Paste: kendall_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/rae_by_endpoint.png` — Per-endpoint RAE comparison
<!-- Paste: rae_by_endpoint.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.09-seal-iid-vs-ood-series.py
```

## Source

`notebooks/2.09-seal-iid-vs-ood-series.py`
