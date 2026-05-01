# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You must never coauthor commits with Claude.

## Commands

### Development Setup

```bash
# Install dependencies using pixi
pixi install

# Download dataset from HuggingFace (also runs automatically on cheminformatics env activation)
pixi run download

# Run Python (default env)
pixi run python

# Run an analysis notebook (cheminformatics env required for RDKit, Chemprop, PyArrow)
pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py
pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py --model chemprop
pixi run -e cheminformatics python notebooks/2.08-seal-baseline-performance.py --combined
```

### Adding Dependencies

- Add Python packages to `[project] dependencies` in `pyproject.toml`, not `[tool.pixi.dependencies]` (which is for conda packages)
- RDKit, PyArrow, and Chemprop are conda-only; they go in `[tool.pixi.feature.cheminformatics.dependencies]`

### Code Quality

```bash
# Run linter (ruff)
pixi run -e lint ruff check src/polaris_generalization

# Run formatter
pixi run -e lint ruff format src/polaris_generalization

# Run tests
pixi run -e test pytest
```

## Architecture

This is a generalization evaluation framework for molecular ML, analyzing how model performance degrades when deployed on data distributions different from training data.

### Key Components

1. **Source Package** (`src/polaris_generalization/`): Shared modules:
   - `config.py`: Centralized path definitions using pathlib
   - `visualization.py`: Plotting utilities — `set_style()`, `DEFAULT_DPI`, `MODEL_COLORS`, `MODEL_LABELS`, `plot_model_comparison_bars()`
   - `tuning.py`: Optuna TPE XGBoost hyperparameter tuning with JSON caching (`tune_xgboost`)
   - `chemprop_utils.py`: Chemprop D-MPNN training with prediction caching (`train_chemprop`)

2. **Notebooks** (`notebooks/`): Analysis notebooks following phase-based naming
   - Convention: `<phase>.<sequence>-<initials>-<description>.py`
   - Phases: 0 = exploration, 1 = data loading, 2 = analysis, 3 = figures
   - All modeling notebooks support `--model xgboost|chemprop` and `--combined` flags

3. **Data Organization** (cookiecutter data science):
   - `data/external/`: External datasets
   - `data/raw/`: Original immutable data
   - `data/interim/`: Intermediate data — CV folds, Tanimoto matrix, `optuna_cache/`, `chemprop_pred_cache/`
   - `data/processed/`: Analysis outputs, organized as `{notebook}/{model}/` with a `combined/` subdirectory

4. **Configuration** (`configs/`): Model and experiment configuration files

### Modeling: XGBoost vs Chemprop

Every modeling notebook (2.07–2.15) supports two model backends:

- **XGBoost** (`--model xgboost`, default): ECFP4 (2048-bit) + ~200 RDKit 2D descriptors, Optuna TPE-tuned per endpoint. Hyperparameters cached as JSON in `data/interim/optuna_cache/`.
- **Chemprop** (`--model chemprop`): D-MPNN trained directly on protonated SMILES, default hyperparameters (hidden_dim=300, depth=3, 50 epochs). Predictions cached as `.npy` in `data/interim/chemprop_pred_cache/`.

Both models use dimorphite_dl protonation at assay-relevant pH before training. Log-transform protocol (`log10(clip(x, 1e-10) + 1)`) applies to all endpoints except LogD for both models.

Outputs go to `data/processed/{notebook}/{model}/` with shared figure helpers in `visualization.py`. Running `--combined` generates side-by-side comparison figures in `data/processed/{notebook}/combined/`.

### Dataset

Source: [openadmet/openadmet-expansionrx-challenge-data](https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data) (HuggingFace)

**Files** (in `data/raw/`):
- `expansion_data_raw.csv` (7,618 molecules) — full dataset with out-of-range modifiers (e.g., ">", "<")
- `expansion_data_train.csv` (5,326 molecules) — ML-ready train split (in-range only)
- `expansion_data_test.csv` (2,282 molecules) — ML-ready test split (in-range only)

**Columns**: `Molecule Name`, `SMILES`, plus 10 ADME endpoints:
- `LogD`, `KSOL` (kinetic solubility, uM)
- `HLM CLint`, `RLM CLint`, `MLM CLint` (liver microsomal clearance, mL/min/kg) — RLM only in raw
- `Caco-2 Permeability Papp A>B` (10^-6 cm/s), `Caco-2 Permeability Efflux`
- `MPPB`, `MBPB`, `MGMB` (protein binding, % unbound)

**Key features**: Ordinal molecule naming (enables time-split), chemical series structure (enables IID vs OOD), multi-CRO provenance, RNA-small molecule drug discovery context.

Download: `pixi run download` (also auto-runs on cheminformatics env activation) — data is gitignored.

## Git Commands

- Never use `--no-stat` with `git show` — this flag doesn't exist
- Use `-s` or `--no-patch` to suppress diff output
- Use `--stat` to show diffstat only

## GitHub CLI

- `gh issue list` has no `--sort` flag — use `-S "sort:updated-desc"` in the search query instead
- To edit issue bodies: `gh issue view N --json body -q '.body' > /tmp/issueN.md`, edit the file, then `gh issue edit N --body-file /tmp/issueN.md` — prefer body edits over posting correction comments

## Quick Reference

- `docs/paper/`: Paper outline and analysis checklist (source of truth for analysis status)
- `docs/decisions/`: Decision records for significant technical choices
- `scripts/`: Shell scripts and pipeline orchestration
- `notebooks/`: All Python analysis
  - Phase 0: exploration (`0.01`, `0.02`, `0.03`)
  - Phase 1: data loading (`1.01`)
  - Phase 2: analysis (`2.01`–`2.16`)
  - Phase 3: figures (`3.02`)
- Always use the typer library for argument parsing
- When running python, always use `pixi run -e cheminformatics python` for any notebook
