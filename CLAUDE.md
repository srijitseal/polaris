# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You must never coauthor commits with Claude.

## Commands

### Development Setup

```bash
# Install dependencies using pixi
pixi install

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

### Dataset Context

- **Expansion Tx ADMET dataset**: 7,618 molecules, 10 ADME endpoints
- 4 CROs + internal data — multi-CRO provenance enables cross-lab evaluation
- Ordinal ordering enables time-split experiments
- Large chemical series enable IID vs OOD evaluation
- RNA-small molecule drug discovery context

### Data Storage

Data is stored on GitHub (no S3/AWS). Use `data/` directories directly.

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
