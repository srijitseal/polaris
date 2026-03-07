# Paper Outline: Polaris Model Validation

## Central Contribution

A framework for evaluating molecular ML generalization using realistic ADMET data, demonstrating that standard random-split benchmarks systematically overestimate real-world performance.

## Abstract (5 sentences)

1. **Opportunity**: Machine learning models for molecular property prediction are increasingly used in drug discovery, but their real-world performance depends on generalization to novel chemical matter.
2. **Problem**: Current benchmarks rely on random splits that leak information between train and test sets, systematically overestimating deployment performance.
3. **Approach**: Here, we present a generalization evaluation framework using the Expansion Tx ADMET dataset (7,618 molecules, 10 ADME endpoints, 4 CROs + internal data) with splitting strategies that mimic real-world deployment scenarios.
4. **Results**: [To be filled with key findings and numbers]
5. **Impact**: This framework and dataset enable the community to build and evaluate molecular ML models that generalize to realistic drug discovery scenarios.

## Introduction (4 paragraphs)

### Para 1: Field gap
- ML-driven molecular property prediction is transforming drug discovery, but model evaluation practices have not kept pace.

### Para 2: Subfield gap
- ADMET prediction benchmarks predominantly use random splits, which allow structurally similar molecules to appear in both train and test sets, inflating performance metrics.

### Para 3: Paper gap
- It remains unclear how much performance degrades under realistic deployment conditions, where test molecules differ systematically from training data in scaffold, time of synthesis, or target distribution.

### Para 4: "Here we..."
- Here, we introduce a generalization evaluation framework built on the Expansion Tx ADMET dataset, implementing cluster-based, time-split, and target-distribution splitting strategies with diagnostic tools to quantify the gap between benchmark and deployment performance.

## Results

### 1. The Expansion Tx dataset provides realistic ADMET data from drug discovery
- Dataset overview: 7,618 molecules, 10 ADME endpoints, 4 CROs + internal
- Ordinal ordering, chemical series structure, multi-CRO provenance

### 2. Chemical space and target distribution characterize dataset diversity
- 1-NN distance distributions, Butina clustering
- Endpoint distribution analysis across 10 ADME endpoints

### 3. Splitting strategies that mimic deployment reveal performance gaps
- Cluster-based split (EKM + k-means on ECFP6)
- Time-split using ordinal ordering
- Target distribution split (rolling window CV)

### 4. Split quality diagnostics validate splitting approaches
- Train/test distribution overlap metrics
- Scaffold diversity comparisons

### 5. Performance degrades with distance from training data
- Performance-over-distance curves
- Quantifying the random-split optimism gap

### 6. Case studies illustrate generalization failure modes
- IID vs OOD evaluation on chemical series
- Scaffold vs random split comparison
- Split variance study
- Activity cliff evaluation
- Molecular variant consistency

## Discussion

### Summary
- Restate: random splits overestimate performance; framework quantifies the gap

### Limitations
- Single therapeutic area (RNA-small molecule)
- Dataset size relative to large-scale benchmarks
- Endpoint selection and CRO coverage

### Community Opportunities
- Framework extensibility to other datasets
- Standardized splitting for ADMET benchmarks

## Methods

- Dataset curation and preprocessing
- Fingerprint computation (ECFP6)
- Splitting algorithms (cluster, time, target distribution)
- Evaluation metrics
- Statistical analysis

## Figures

| Figure | Title | Source |
|--------|-------|--------|
| Fig 1 | Dataset overview and chemical space | Notebooks 1.01, 2.01 |
| Fig 2 | Splitting strategy comparison | Notebooks 2.03-2.05 |
| Fig 3 | Performance-over-distance curves | Notebook 2.07 |
| Fig 4 | Case studies (IID vs OOD, scaffolds) | Notebooks 2.08, 2.11 |
| Fig S1 | Split quality diagnostics | Notebook 2.06 |
| Fig S2 | Split variance analysis | Notebook 2.12 |
