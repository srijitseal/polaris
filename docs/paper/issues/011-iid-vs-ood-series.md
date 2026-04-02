# OOD performance degrades 1.0–7.4x vs IID when evaluating across chemical series

## Summary

Training Optuna TPE-tuned XGBoost on the largest chemical series (Butina cluster 0, n=2,572) and evaluating on held-out molecules from the same series (IID) vs a different series (cluster 1, n=1,301, OOD) shows consistent and often dramatic performance degradation. This is the "hero" example (paper Fig 7) — it works because this dataset uniquely provides both temporal ordering (for time-split within a series) and chemical series structure (for OOD evaluation across series). The result is intuitive but quantifying it requires dataset properties that public benchmarks almost never have.

## Method

1. **Select two largest Butina clusters** (cutoff 0.7, from notebook 2.01): cluster 0 (n=2,572) and cluster 1 (n=1,301). These represent two distinct chemical series from the Expansion Tx drug discovery campaigns.
2. **Time-split cluster 0** using ordinal molecule index (E-XXXXXXX naming encodes temporal order): earlier 80% = train (n=2,057), later 20% = IID validation (n=515). This mimics how a medicinal chemist would deploy a model: train on past compounds, predict new ones in the same series.
3. **Cluster 1 = OOD test set** (n=1,301). This mimics predicting an entirely different chemical series — a harder but common real-world scenario (e.g., hit-to-lead on a new scaffold).
4. **Train Optuna TPE-tuned XGBoost** per endpoint on ECFP4 (2048-bit) + full RDKit 2D descriptors (~200), with dimorphite_dl protonation at assay-relevant pH. Hyperparameter optimization uses Optuna TPE Bayesian optimization (30 trials, 3-fold CV, MAE scoring) tuning 9 hyperparameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`. See ADR-002 (`docs/decisions/002-optuna-hyperparameter-tuning.md`).
5. **Compare squared error distributions** between IID and OOD predictions. Log-transform applied to all endpoints except LogD, matching the competition evaluation protocol.

## Key Findings

### Split characterization

| Set | n | Cluster | 1-NN median | 1-NN mean |
|-----|---|---------|-------------|-----------|
| Train | 2,057 | 0 (largest) | — | — |
| IID val | 515 | 0 (time-split, later 20%) | 0.288 | 0.296 |
| OOD test | 1,301 | 1 (second largest) | 0.763 | 0.763 |

The distance distributions have near-zero overlap — IID molecules are structurally close to training data (same chemical series, Tanimoto distance ~0.28), while OOD molecules are far away (different series, ~0.76). This clean separation is what makes the comparison compelling: any performance difference can be attributed to distribution shift rather than noise.

### Per-endpoint performance

| Endpoint | IID R² | OOD R² | IID Spearman | OOD Spearman | IID RAE | OOD RAE | IID median SE | OOD median SE | SE fold-change |
|----------|--------|--------|-------------|-------------|---------|---------|--------------|--------------|----------------|
| MLM CLint | 0.08 | -1.07 | 0.458 | 0.121 | 0.931 | 1.451 | 0.078 | 0.577 | **7.4x** |
| KSOL | 0.30 | 0.15 | 0.513 | 0.434 | 0.827 | 0.892 | 0.058 | 0.396 | **6.8x** |
| MPPB | 0.15 | 0.14 | 0.383 | 0.460 | 0.934 | 0.938 | 0.020 | 0.118 | **5.9x** |
| MBPB | 0.26 | -0.48 | 0.458 | 0.478 | 0.842 | 1.335 | 0.023 | 0.131 | **5.6x** |
| LogD | 0.64 | 0.11 | 0.804 | 0.518 | 0.579 | 0.956 | 0.129 | 0.626 | **4.9x** |
| HLM CLint | -0.12 | 0.04 | 0.415 | 0.175 | 0.963 | 0.977 | 0.033 | 0.157 | **4.7x** |
| Caco-2 Efflux | 0.47 | -0.01 | 0.754 | 0.169 | 0.610 | 0.990 | 0.048 | 0.098 | **2.1x** |
| Caco-2 Papp A>B | 0.44 | -0.09 | 0.709 | 0.240 | 0.699 | 1.059 | 0.067 | 0.067 | **1.0x** |

R² goes negative on OOD for 3 of 8 endpoints (MLM CLint, MBPB, Caco-2 Papp A>B), and near-zero for 2 more (Caco-2 Efflux at -0.01, HLM CLint at 0.04). Tuning improves OOD R² relative to default XGBoost for most endpoints — notably KSOL (-0.04 to 0.15), MPPB (-0.19 to 0.14), and LogD (-0.11 to 0.11) — but the IID-to-OOD degradation pattern persists. MLM CLint actually worsens on OOD under tuning (R² -0.38 to -1.07), suggesting tuning can overfit to the training series for some endpoints.

Spearman rho drops on OOD for most endpoints, though the decline is less severe than with default XGBoost: LogD 0.804 to 0.518, Caco-2 Efflux 0.754 to 0.169, Caco-2 Papp 0.709 to 0.240, MLM CLint 0.458 to 0.121. Two protein binding endpoints (MPPB, MBPB) are exceptions where OOD Spearman is comparable to or slightly exceeds IID, though both have small IID sample sizes (n=53, n=51).

RAE exceeds 1.0 on OOD for 3 of 8 endpoints (MLM CLint 1.45, MBPB 1.34, Caco-2 Papp 1.06), an improvement from 7 of 8 with default XGBoost. The remaining endpoints hover just below 1.0 on OOD.

The SE fold-change range is 1.0–7.4x (vs 1.4–12.0x with default XGBoost). MLM CLint shows the largest degradation (7.4x), while Caco-2 Papp A>B shows essentially no degradation in median SE (1.0x), though its OOD R² is still negative (-0.09).

MGMB was skipped due to insufficient data in both clusters (only 431 molecules total, sparsely distributed).

### Why this matters

Standard IID evaluation (random split or even time-split within a single series) dramatically overestimates real-world performance. When a model is deployed on a genuinely new chemical series — the common scenario in hit identification — performance collapses. Hyperparameter tuning narrows the gap somewhat but does not eliminate it: the median SE fold-change across endpoints is still ~5x. This analysis is only possible because the Expansion Tx dataset provides:
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
- `data/processed/2.09-seal-iid-vs-ood-series/spearman_by_endpoint.png` — Per-endpoint Spearman rho comparison
<!-- Paste: spearman_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/kendall_by_endpoint.png` — Per-endpoint Kendall tau comparison
<!-- Paste: kendall_by_endpoint.png -->
- `data/processed/2.09-seal-iid-vs-ood-series/rae_by_endpoint.png` — Per-endpoint RAE comparison
<!-- Paste: rae_by_endpoint.png -->

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.09-seal-iid-vs-ood-series.py
```

## Source

`notebooks/2.09-seal-iid-vs-ood-series.py`
