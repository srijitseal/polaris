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

# Run Python
pixi run python
```

### Adding Dependencies

- Add Python packages to `[project] dependencies` in `pyproject.toml`, not `[tool.pixi.dependencies]` (which is for conda packages)

### Code Quality

```bash
# Run linter (ruff)
pixi run ruff check src/polaris_generalization

# Run formatter
pixi run ruff format src/polaris_generalization

# Run tests
pixi run pytest
```

## Architecture

This is a generalization evaluation framework for molecular ML, analyzing how model performance degrades when deployed on data distributions different from training data.

### Key Components

1. **Source Package** (`src/polaris_generalization/`): Modules for:
   - `config.py`: Centralized path definitions using pathlib
   - `visualization.py`: Shared plotting utilities
   - Future modules: dataset_analysis, splitting, distance_metrics, evaluation

2. **Notebooks** (`notebooks/`): Analysis notebooks following phase-based naming
   - Convention: `<phase>.<sequence>-<initials>-<description>.py`
   - Phases: 0 = exploration, 1 = data loading, 2 = analysis, 3 = figures

3. **Data Organization** (cookiecutter data science):
   - `data/external/`: External datasets (Expansion Tx ADMET data)
   - `data/raw/`: Original immutable data
   - `data/interim/`: Intermediate transformed data
   - `data/processed/`: Final analysis outputs

4. **Configuration** (`configs/`): Model and experiment configuration files

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

- `docs/paper/`: Paper outline, analysis checklist, and planning documents
- `docs/decisions/`: Decision records for significant technical choices
- `scripts/`: Shell scripts and pipeline orchestration
- `notebooks/`: All Python analysis (use phase 0: `0.xx-seal-description.py` for exploration)
- Always use the typer library for argument parsing
- When running python, always use `pixi run python`
