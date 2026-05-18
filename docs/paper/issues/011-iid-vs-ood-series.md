# OOD performance degrades up to 6.6x vs IID when evaluating across chemical series

## Summary

Training Optuna TPE-tuned XGBoost on the largest chemical series (Butina cluster 0, n=2,572) and evaluating on held-out molecules from the same series (IID) vs a different series (cluster 1, n=1,301, OOD) shows consistent and often dramatic performance degradation. This is the IID-vs-OOD case study mapped to **Fig 5** in the current outline (`docs/paper/outline.md`). It works because this dataset uniquely provides both temporal ordering (for time-split within a series) and chemical series structure (for OOD evaluation across series). The result is intuitive but quantifying it requires dataset properties that public benchmarks almost never have.

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
| KSOL | 0.291 | 0.119 | 0.514 | 0.399 | 0.835 | 0.902 | 0.062 | 0.404 | **6.5x** |
| MLM CLint | 0.107 | -0.822 | 0.458 | 0.228 | 0.915 | 1.340 | 0.073 | 0.466 | **6.4x** |
| MPPB | 0.162 | 0.146 | 0.413 | 0.464 | 0.927 | 0.933 | 0.020 | 0.110 | **5.6x** |
| HLM CLint | -0.029 | -0.012 | 0.438 | 0.086 | 0.923 | 1.003 | 0.031 | 0.162 | **5.2x** |
| LogD | 0.645 | 0.055 | 0.809 | 0.447 | 0.575 | 0.982 | 0.127 | 0.651 | **5.1x** |
| MBPB | 0.237 | -0.488 | 0.467 | 0.539 | 0.843 | 1.340 | 0.025 | 0.128 | **5.1x** |
| Caco-2 Efflux | 0.456 | -0.005 | 0.756 | 0.141 | 0.621 | 1.002 | 0.051 | 0.109 | **2.2x** |
| Caco-2 Papp A>B | 0.448 | -0.041 | 0.714 | 0.318 | 0.691 | 1.034 | 0.060 | 0.059 | **1.0x** |

R² goes negative on OOD for **5 of 8 endpoints** (HLM CLint, MLM CLint, Caco-2 Papp, Caco-2 Efflux, MBPB) and is barely positive for two more (LogD 0.055, KSOL 0.119). Only MPPB sustains comparable R² across IID and OOD (0.162 vs 0.146).

Spearman rho drops on OOD for 6 of 8 endpoints — most sharply for HLM CLint (0.438 → 0.086), Caco-2 Efflux (0.756 → 0.141), MLM CLint (0.458 → 0.228), and Caco-2 Papp (0.714 → 0.318). MPPB and MBPB are exceptions where OOD Spearman is comparable to or slightly exceeds IID, though both have very small IID sample sizes (n=53, n=51).

RAE exceeds 1.0 on OOD for **5 of 8 endpoints** (MLM CLint 1.34, MBPB 1.34, Caco-2 Papp 1.03, HLM CLint 1.00, Caco-2 Efflux 1.00) — i.e. predictions are no better than predicting the mean OOD activity.

Median SE fold-change ranges from 1.0× (Caco-2 Papp) to 6.5× (KSOL). KSOL and MLM CLint show the largest degradation (6.4–6.5×), while Caco-2 Papp A>B shows essentially no change in median SE (1.0×) yet still has negative OOD R² (−0.04) — the residual variance shifts even when the median doesn't.

MGMB was skipped due to insufficient data in both clusters (only 431 molecules total, sparsely distributed).

### Why this matters

Standard IID evaluation (random split or even time-split within a single series) dramatically overestimates real-world performance. When a model is deployed on a genuinely new chemical series — the common scenario in hit identification — performance collapses: median SE fold-change is ~5× across most endpoints, R² goes negative on the majority, and ranking quality (Spearman) collapses on most endpoints. This analysis is only possible because the Expansion Tx dataset provides:
- **Chemical series structure**: Butina clustering identifies coherent series with hundreds of members
- **Temporal ordering**: Ordinal molecule naming enables realistic time-split within a series

Public ADMET benchmarks typically lack both properties, making this kind of IID vs OOD decomposition impossible.

## Plots

- `data/processed/2.09-seal-iid-vs-ood-series/distance_characterization.png` — 1-NN distance distributions showing clean IID/OOD separation
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/squared_error_distributions.png` — per-endpoint SE histograms
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/median_se_by_endpoint.png` — median SE comparison
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/mae_by_endpoint.png` — MAE comparison
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/r2_by_endpoint.png` — R² comparison
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/spearman_by_endpoint.png` — Spearman ρ comparison
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/kendall_by_endpoint.png` — Kendall τ comparison
- `data/processed/2.09-seal-iid-vs-ood-series/xgboost/rae_by_endpoint.png` — RAE comparison
- Combined XGBoost vs CheMeleon panels: `data/processed/2.09-seal-iid-vs-ood-series/combined/`

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.09-seal-iid-vs-ood-series.py
```

## Source

`notebooks/2.09-seal-iid-vs-ood-series.py`

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran with CheMeleon (BondMessagePassing + 2×1024 FFN, fine-tuned per endpoint, single model). Same train/IID/OOD split as the XGBoost run.

**Trends match. OOD degradation persists and in fact worsens for most endpoints — the foundation model does not close the IID–OOD gap; on 6/8 endpoints it widens it.**

### IID vs OOD R²

| Endpoint | R² IID XGB | R² IID CM | R² OOD XGB | R² OOD CM |
|----------|-----------|----------|-----------|----------|
| LogD | 0.645 | 0.640 | +0.055 | **+0.444** |
| KSOL | 0.291 | 0.157 | +0.119 | −0.560 |
| HLM CLint | −0.029 | −0.645 | −0.012 | −0.096 |
| MLM CLint | 0.107 | 0.142 | −0.822 | −1.113 |
| Caco-2 Papp | 0.448 | 0.448 | −0.041 | −0.320 |
| Caco-2 Efflux | 0.456 | 0.402 | −0.005 | −0.062 |
| MPPB | 0.162 | 0.405 | +0.146 | +0.172 |
| MBPB | 0.237 | 0.417 | −0.488 | −1.231 |

R² goes negative OOD for **6/8 endpoints under CheMeleon** (vs 5/8 under XGBoost). Only LogD shows a meaningful OOD improvement (R² 0.05 → 0.44).

### Median squared-error fold change (OOD/IID)

| Endpoint | XGB | CM |
|----------|-----|-----|
| Caco-2 Efflux | 2.15× | 1.97× |
| Caco-2 Papp | 0.98× | 1.34× |
| HLM CLint | 5.16× | 4.62× |
| KSOL | 6.55× | **46.93×** |
| LogD | 5.11× | 4.40× |
| MBPB | 5.09× | **19.00×** |
| MLM CLint | 6.35× | 8.06× |
| MPPB | 5.55× | 2.79× |

KSOL and MBPB extreme fold-change reflects tighter IID fit (median SE 0.006 / 0.012) without proportional OOD improvement — better within-series fit makes the cliff at the series boundary larger.

### Source

- `data/processed/2.09-seal-iid-vs-ood-series/chemeleon/summary_metrics.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/11#issuecomment-4376602653
- Reproduce: `pixi run -e cheminformatics python notebooks/2.09-seal-iid-vs-ood-series.py --model chemeleon`
