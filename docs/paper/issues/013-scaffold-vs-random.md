# Naive scaffold splits are no harder than random splits — cluster-based splitting produces genuine distribution shift

## Summary

Naive Bemis-Murcko scaffold splits produce nearly identical XGBoost performance and test-to-train distance distributions as random splits across all 9 Expansion Tx endpoints. Cluster-based splitting (from 2.03) produces notably worse metrics and larger structural distances, confirming that it creates genuine distribution shift. This supports Greg Landrum's observation that scaffold splits are widely used but without careful considerations they're no better than random.

## Method

1. **Three splitting strategies** applied per endpoint (molecules with non-null values only):
   - **Scaffold split**: Compute generic Murcko scaffolds via RDKit, group molecules by scaffold, sort scaffolds by frequency (largest first), greedily assign each scaffold group to the fold with fewest molecules (5 folds)
   - **Random split**: 5-fold random assignment (seed=42)
   - **Cluster split**: Precomputed cluster-based folds from 2.03 (repeat 0, 5 folds)
2. **Train and evaluate**: Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters per endpoint per fold; see ADR-002) on ECFP4 + RDKit 2D descriptors (protonated at assay pH), log-transformed targets (except LogD)
3. **Competition metrics**: MAE, R², Spearman ρ, Kendall τ, RAE per fold per endpoint
4. **Distance characterization**: Pooled test-to-train 1-NN Tanimoto distances per strategy

## Key Findings

### Performance comparison (MA-RAE)

| Strategy | MA-RAE |
|----------|--------|
| Random | 0.474 |
| Scaffold | 0.508 |
| Cluster | 0.674 |

Scaffold and random splits produce similar metrics across all endpoints. Cluster-based splitting consistently produces worse performance — higher MAE, higher RAE, lower R², lower rank correlations — reflecting genuine distribution shift.

### Per-endpoint RAE (mean across folds)

| Endpoint | Scaffold | Random | Cluster |
|----------|----------|--------|---------|
| LogD | 0.377 | 0.350 | 0.503 |
| KSOL | 0.556 | 0.528 | 0.700 |
| HLM CLint | 0.640 | 0.589 | 0.826 |
| MLM CLint | 0.585 | 0.546 | 0.748 |
| Caco-2 Papp A>B | 0.519 | 0.495 | 0.694 |
| Caco-2 Efflux | 0.454 | 0.434 | 0.620 |
| MPPB | 0.514 | 0.461 | 0.648 |
| MBPB | 0.432 | 0.403 | 0.590 |
| MGMB | 0.492 | 0.462 | 0.734 |

Scaffold–random RAE differences are 0.02–0.07 (small). Cluster–random differences are 0.13–0.27 (large, consistent).

### Per-endpoint R² (mean across folds)

| Endpoint | Scaffold | Random | Cluster |
|----------|----------|--------|---------|
| LogD | 0.839 | 0.859 | 0.720 |
| KSOL | 0.603 | 0.639 | 0.438 |
| HLM CLint | 0.552 | 0.610 | 0.252 |
| MLM CLint | 0.636 | 0.674 | 0.397 |
| Caco-2 Papp A>B | 0.664 | 0.689 | 0.443 |
| Caco-2 Efflux | 0.729 | 0.745 | 0.530 |
| MPPB | 0.688 | 0.744 | 0.546 |
| MBPB | 0.752 | 0.780 | 0.600 |
| MGMB | 0.660 | 0.701 | 0.343 |

### Distance distributions

| Strategy | Median 1-NN | Mean 1-NN | Q75 | Q90 |
|----------|-------------|-----------|-----|-----|
| Random | 0.203 | 0.210 | 0.264 | 0.328 |
| Scaffold | 0.246 | 0.248 | 0.314 | 0.387 |
| Cluster | 0.424 | 0.431 | 0.524 | 0.619 |

Scaffold and random 1-NN distributions overlap substantially (medians 0.246 vs 0.203). The cluster-based split produces a right-shifted distribution with ~2× larger median distance (0.424), confirming genuine structural separation.

### Statistical tests (KS)

| Comparison | KS statistic D | p-value | n_1 | n_2 |
|---|---|---|---|---|
| scaffold vs random | 0.2028 | < 10^-10 | 36,003 | 36,003 |
| cluster vs random | 0.6617 | < 10^-10 | 36,003 | 36,003 |
| cluster vs scaffold | 0.5356 | < 10^-10 | 36,003 | 36,003 |

Two-sample Kolmogorov-Smirnov tests on pooled 1-NN distance arrays confirm that while the scaffold vs random difference is statistically significant (D = 0.203), it is modest compared to the cluster vs random difference (D = 0.662). All three comparisons are highly significant (p ≈ 0), but the effect sizes tell the real story: scaffold splits create only a small shift from random, while cluster-based splits create a qualitatively different distribution.

### Interpretation

The naive scaffold split fails to create distribution shift because:
1. Many scaffolds appear only once — these are randomly scattered across folds regardless of strategy
2. Large scaffold groups keep structurally identical molecules together, but structurally *similar* molecules with different scaffolds still leak across folds
3. The greedy balancing ensures roughly equal fold sizes but not structural separation

The cluster-based approach succeeds because it groups molecules by fingerprint similarity in kernel space, ensuring that test molecules are genuinely distant from training molecules.

**Practical takeaway**: Scaffold splits are a poor proxy for real-world deployment conditions. Researchers should use distance-aware splitting (cluster-based, scaffold with distance constraints, or similar) to honestly evaluate model generalization.

## Plots

- `data/processed/2.11-seal-scaffold-vs-random/xgboost/metric_comparison.png` — combined metric comparison (main Fig S2 candidate)
- `data/processed/2.11-seal-scaffold-vs-random/xgboost/distance_distributions.png` — overlaid 1-NN distance histograms
- `data/processed/2.11-seal-scaffold-vs-random/xgboost/strategy_summary.png` — MA-RAE by strategy
- `data/processed/2.11-seal-scaffold-vs-random/xgboost/metric_mae.png`, `metric_r2.png`, `metric_rae.png`, `metric_spearman_r.png`, `metric_kendall_tau.png` — per-metric panels
- `data/processed/2.11-seal-scaffold-vs-random/scaffold_group_sizes.png` — scaffold group size distribution (representation-level, not model-specific)
- Combined XGBoost vs CheMeleon panels: `data/processed/2.11-seal-scaffold-vs-random/combined_<strategy>_<metric>.png`

## Reproduce

```bash
pixi run -e cheminformatics python notebooks/2.11-seal-scaffold-vs-random.py
```

## Source

`notebooks/2.11-seal-scaffold-vs-random.py`

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran with CheMeleon (5-fold CV per strategy, fine-tuned per fold). **Trend matches XGBoost verbatim: scaffold ≈ random ≪ cluster.**

### MA-RAE by strategy

| Strategy | XGBoost | CheMeleon |
|----------|---------|-----------|
| Random | 0.474 | 0.430 |
| Scaffold | 0.508 | 0.471 |
| Cluster | **0.674** | **0.623** |

### Per-endpoint scaffold–random and cluster–scaffold RAE gaps

| Endpoint | scaf−rand XGB | scaf−rand CM | clus−scaf XGB | clus−scaf CM |
|----------|---------------|--------------|---------------|--------------|
| LogD | +0.027 | +0.029 | +0.126 | +0.100 |
| KSOL | +0.028 | +0.033 | +0.144 | +0.133 |
| HLM CLint | +0.051 | +0.036 | +0.186 | +0.190 |
| MLM CLint | +0.038 | +0.061 | +0.163 | +0.155 |
| Caco-2 Papp | +0.024 | +0.033 | +0.175 | +0.234 |
| Caco-2 Efflux | +0.020 | +0.024 | +0.166 | +0.148 |
| MPPB | +0.052 | +0.028 | +0.134 | +0.164 |
| MBPB | +0.029 | +0.059 | +0.158 | +0.120 |
| MGMB | +0.030 | +0.069 | +0.243 | +0.128 |

Scaffold–random RAE gap remains 0.02–0.07 in both models (negligible). Cluster–scaffold gap remains 0.10–0.24 (substantial). The Murcko-boundary argument is a property of the scaffolds and the fingerprint, not the model — switching to a graph-based foundation model leaves the gap pattern intact.

### Source

- `data/processed/2.11-seal-scaffold-vs-random/chemeleon/aggregated_metrics.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/13#issuecomment-4381106421
- Reproduce: `pixi run -e cheminformatics python notebooks/2.11-seal-scaffold-vs-random.py --model chemeleon`
