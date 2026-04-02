# Decision Records

This directory contains decision records documenting significant technical decisions made in the project.

## Format

Each decision is documented in a separate markdown file with:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What problem we're solving
- **Decision**: What we decided to do
- **Implementation**: How to implement it
- **Consequences**: What are the tradeoffs

## Decisions

| ID | Title | Status |
|----|-------|--------|
| [001](001-overclustering-for-balanced-folds.md) | Over-cluster then greedily assign to folds for balanced CV splits | Accepted |
| [002](002-optuna-hyperparameter-tuning.md) | Switch from HalvingRandomSearchCV to Optuna TPE for hyperparameter tuning | Accepted |
