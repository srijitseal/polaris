# Resonance form ambiguity causes up to 27% RMSE swing in ADMET predictions — worsening consistently exceeds improvement

## Summary

Resonance form ambiguity is a representation-level vulnerability in fingerprint-based ADMET models. Using a protonate-enumerate-reprotonate-deduplicate pipeline, 19.6–53.4% of molecules have >1 distinct resonance form at pH 7.4 (dropping to 1.8% at pH 6.5 for Caco-2 endpoints). These forms occupy distant regions of fingerprint space (median ECFP4 Tanimoto distance 0.462). With Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) under cluster-split CV, the worst-case resonance form increases RMSE by up to 20.4% (LogD), while the best-case form would only reduce RMSE by 7.0% — a systematic asymmetry where worsening exceeds improvement across all 9 endpoints. Total RMSE swing ranges from 0.4% (Caco-2 Efflux) to 27.4% (LogD). Individual molecules can experience prediction error ranges of several hundred percent.

## Method

1. **Resonance enumeration** (NB 2.14): For each molecule, protonate at endpoint-specific pH (7.4 or 6.5) using dimorphite-DL, enumerate resonance forms via RDKit `ResonanceMolSupplier` (default flags, max 50), re-protonate each form at the same pH, deduplicate by canonical SMILES. Separate CSV files generated per pH value.
2. **Model evaluation** (NB 2.15): Train Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters; ADR-002) on chirality-aware ECFP4 + RDKit 2D descriptors using cluster-split CV (5 folds, repeat 0). For each test molecule, predict all distinct resonance forms. Compute RIGR-aligned metrics: Baseline RMSE (input form), RMS MinRD (best-case), RMS MaxRD (worst-case), improvement %, worsening %, total RMSE swing.
3. **Fingerprint characterization**: Compute ECFP4 Tanimoto distances between all pairwise resonance forms within each group at pH 7.4, compared to 10,000 random molecule pairs.

## Key Findings

### Resonance prevalence is pH-dependent

- pH 7.4 (7 endpoints): 19.6–53.4% of molecules have >1 distinct resonance form (HLM CLint lowest at 19.6%, MGMB highest at 53.4%)
- pH 6.5 (Caco-2 endpoints): only 1.8% — lower pH protonation collapses most resonance variants
- This is a real chemical effect, not a computational artifact

### Resonance forms are far apart in fingerprint space

| Metric | Value |
|--------|-------|
| Resonance pair Tanimoto distance (median) | 0.462 |
| Random pair Tanimoto distance (median) | 0.849 |
| Scaffold decoration distance (median, NB 2.13) | 0.349 |
| Stereoisomer distance (median, NB 2.13) | 0.067 |

Resonance forms — chemically identical molecules — produce larger fingerprint shifts than scaffold decorations. A distance of 0.46 falls in the "structurally novel" regime on performance-over-distance curves.

### RMSE impact: worsening consistently exceeds improvement

| Endpoint | n total | n resonance | % | Baseline RMSE | Improve (%) | Worsen (%) | Total swing (%) |
|----------|---------|-------------|------|---------------|-------------|------------|-----------------|
| LogD | 7,309 | 2,060 | 28.2 | 0.5994 | 7.0 | 20.4 | 27.4 |
| MLM CLint | 5,692 | 1,581 | 27.8 | 0.5367 | 3.5 | 9.1 | 12.6 |
| MBPB | 1,426 | 367 | 25.7 | 0.2638 | 3.7 | 7.6 | 11.3 |
| KSOL | 7,298 | 2,060 | 28.2 | 0.5391 | 4.0 | 6.4 | 10.3 |
| HLM CLint | 4,541 | 888 | 19.6 | 0.5108 | 1.7 | 7.3 | 9.0 |
| MPPB | 1,756 | 456 | 26.0 | 0.2929 | 3.4 | 4.9 | 8.3 |
| MGMB | 431 | 230 | 53.4 | 0.2676 | 1.8 | 3.8 | 5.7 |
| Caco-2 Papp A>B | 3,773 | 68 | 1.8 | 0.3181 | 0.5 | 1.1 | 1.5 |
| Caco-2 Efflux | 3,777 | 68 | 1.8 | 0.3192 | 0.2 | 0.2 | 0.4 |

The asymmetry (worsening > improvement) holds for all endpoints. The canonical input form is already a reasonable representation; deviations from it are more likely to degrade predictions than improve them.

### Endpoint-specific patterns reflect biology

- **LogD** (27.4% swing): Lipophilicity depends on global electronic distribution, which resonance directly alters
- **Microsomal clearance** (9-12.6%): CYP450 metabolism depends on local electronic environments at soft spots
- **Protein binding** (5.7-11.3%): Intermediate — binding depends on both shape and local electrostatics
- **Caco-2** (0.4-1.5%): Least affected — pH 6.5 collapses most variants; passive permeability depends on global properties preserved across forms

### Per-molecule risk can be extreme

Individual molecules can experience prediction error ranges of several hundred percent relative to baseline error. A practitioner submitting a resonance form of a molecule would receive a different prediction with no indication that the result is unstable.

## Plots

- `data/processed/2.15-zalte-resonance-variants/xgboost/resonance_sensitivity_panel.png` — main figure: (A) prevalence, (B) improvement vs worsening, (C) per-molecule error range
- `data/processed/2.15-zalte-resonance-variants/fingerprint_distances.png` — SI: ECFP4 Tanimoto distance distributions for resonance forms vs random pairs (representation-level, not model-specific)
- Combined XGBoost vs CheMeleon panels: `data/processed/2.15-zalte-resonance-variants/combined/`

## Reproduce

```bash
pixi run python notebooks/2.14-zalte-resonance-generation.py
pixi run python notebooks/2.15-zalte-resonance-variants.py
```

Cached Optuna hyperparameters in `data/interim/optuna_cache/`. Cached predictions in `data/processed/2.15-zalte-resonance-variants/raw_predictions_cache.json`.

## Source

`notebooks/2.14-zalte-resonance-generation.py`, `notebooks/2.15-zalte-resonance-variants.py`

## Reference

Zalte, A. et al. RIGR: Representation Instability in Generalization and Robustness. *J. Chem. Inf. Model.* (2025).

---

## Update: CheMeleon foundation model (2026-05-05)

Re-ran with CheMeleon (cluster-split CV, 5 folds, fine-tuned per fold). Resonance form generation pipeline unchanged — same protonate-enumerate-reprotonate-deduplicate at endpoint pH (7.4 / 6.5), same `n_multi` per endpoint.

**Trend matches XGBoost. Resonance ambiguity creates substantial RMSE swings; worsening still ≥ improvement for 9/9 endpoints; LogD remains most affected. Magnitudes are mixed under CheMeleon — smaller for the highest-swing endpoints (LogD, MLM, HLM) but comparable or slightly larger for MBPB, MGMB, and Caco-2 Efflux. Foundation model does not uniformly mitigate the failure mode.**

### RMSE swing comparison (% of baseline RMSE)

| Endpoint | n_multi | coverage | swing XGB | swing CM | improve XGB | improve CM | worsen XGB | worsen CM |
|----------|---------|----------|-----------|----------|-------------|------------|------------|-----------|
| LogD | 2060 | 28.2% | **27.4%** | **20.6%** | 7.0% | 4.3% | 20.4% | 16.3% |
| MLM CLint | 1581 | 27.8% | 12.6% | 7.1% | 3.5% | 3.0% | 9.1% | 4.1% |
| MBPB | 367 | 25.7% | 11.3% | 11.9% | 3.7% | 2.1% | 7.6% | 9.8% |
| KSOL | 2060 | 28.2% | 10.3% | 9.1% | 4.0% | 4.1% | 6.4% | 5.0% |
| HLM CLint | 888 | 19.6% | 9.0% | 4.7% | 1.7% | 1.8% | 7.3% | 2.9% |
| MPPB | 456 | 26.0% | 8.3% | 6.3% | 3.4% | 2.9% | 4.9% | 3.4% |
| MGMB | 230 | 53.4% | 5.7% | 5.7% | 1.8% | 1.3% | 3.8% | 4.3% |
| Caco-2 Papp | 68 | 1.8% | 1.5% | 1.2% | 0.5% | 0.4% | 1.1% | 0.8% |
| Caco-2 Efflux | 68 | 1.8% | 0.4% | 0.5% | 0.2% | 0.3% | 0.2% | 0.3% |

In both models: worsen ≥ improve for 9/9 endpoints; LogD most affected; endpoint ranking by swing is largely preserved (LogD > MLM ≈ MBPB > KSOL ≈ HLM > MPPB > MGMB ≫ Caco-2). CheMeleon swings are smaller for LogD (27.4% → 20.6%), MLM CLint (12.6% → 7.1%), and HLM CLint (9.0% → 4.7%), comparable for KSOL/MGMB/MPPB/Caco-2, and slightly larger for MBPB (11.3% → 11.9%) and Caco-2 Efflux (0.4% → 0.5%). The graph representation is *not* uniformly more robust to bond-order changes — the asymmetry (worsen ≥ improve) and per-molecule risk persist in both models, and at least one endpoint sees a small increase in swing.

### Source

- `data/processed/2.15-zalte-resonance-variants/chemeleon/consistency_summary.csv`
- GitHub comment: https://github.com/srijitseal/polaris/issues/21#issuecomment-4381236703
- Reproduce: `pixi run -e cheminformatics python notebooks/2.15-zalte-resonance-variants.py --model chemeleon`
