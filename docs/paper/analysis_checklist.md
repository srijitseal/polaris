# Analysis Checklist

Status tracking for all planned analyses. This is the source of truth for analysis status.

## Phase 0: Exploration

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `0.01-seal-dataset-exploration.py` | Initial Expansion Tx dataset exploration |
| TODO | `0.02-seal-ecfp-distance-exploration.py` | ECFP6 Tanimoto distance distributions |

## Phase 1: Data Loading

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `1.01-seal-dataset-loader.py` | Dataset loading, validation, CRO annotation |

## Phase 2: Analysis

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `2.01-seal-chemical-space-analysis.py` | Chemical space (1-NN distances, Butina clustering) |
| TODO | `2.02-seal-target-distribution.py` | Endpoint distribution analysis (10 ADME endpoints) |
| TODO | `2.03-seal-cluster-split.py` | Cluster-based split (EKM + k-means) |
| TODO | `2.04-seal-time-split.py` | Time-split using ordinal ordering |
| TODO | `2.05-seal-target-split.py` | Target distribution split (rolling window CV) |
| TODO | `2.06-seal-split-quality.py` | Split quality checks |
| TODO | `2.07-seal-performance-distance.py` | Performance-over-distance curves |
| TODO | `2.08-seal-iid-vs-ood-series.py` | IID vs OOD on chemical series |
| TODO | `2.10-seal-activity-cliff-eval.py` | Activity cliff evaluation |
| TODO | `2.11-seal-scaffold-vs-random.py` | Scaffold vs random split comparison |
| TODO | `2.12-seal-split-variance.py` | Split variance study |
| TODO | `2.13-seal-molecular-variants.py` | Molecular variant consistency |

## Phase 3: Figures

| Status | Notebook | Description |
|--------|----------|-------------|
| TODO | `3.01-seal-paper-figures.py` | Publication figures |
