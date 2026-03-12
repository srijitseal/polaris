# Three dominant chemical series account for 63% of the dataset

## Summary

Butina clustering (cutoff 0.7) on the Expansion Tx dataset (7,608 molecules) reveals 135 clusters with highly concentrated chemical series structure: three clusters contain 4,769 molecules (63%), while 33 clusters (24%) are singletons. This concentration has major implications for splitting strategies — naive random splits will leak series information between train and test.

## Key Findings

### Cluster statistics

| Metric | Value |
|--------|-------|
| Total clusters | 135 |
| Singletons | 33 (24.4%) |
| Median cluster size | 4 |
| Top 5 sizes | 2,578 / 1,306 / 885 / 258 / 70 |

### Key observations

- **Highly concentrated series**: The largest cluster (2,578 molecules, 34%) dominates the dataset. Together with clusters 2 (1,306) and 3 (885), these three series contain 63% of all molecules.
- **Long tail**: Beyond the top 5 clusters, most clusters have fewer than 10 members. The median cluster size of 4 reflects this skewed distribution.
- **Singletons indicate scaffold diversity**: 33 molecules have no neighbor within Tanimoto distance 0.7, representing truly unique scaffolds.
- **Implications for splitting**: Random splits will place members of the same large cluster in both train and test, inflating performance. Cluster-based splits (notebook 2.03) should use these clusters to create meaningful distribution shift.

### Implications for paper

- The concentrated series structure makes this dataset ideal for the IID vs OOD case study (notebook 2.08) — the two largest clusters can serve as the two chemical series.
- Butina cutoff 0.7 produces interpretable clusters at a useful granularity for splitting. The k-means approach in 2.03 will complement this by producing more equally-sized folds.

## Plots

- `data/processed/2.01-seal-chemical-space-analysis/cluster_size_distribution.png` — bar chart + histogram
- `data/processed/2.01-seal-chemical-space-analysis/top_clusters_grid.png` — molecule grids from top 5 clusters
- `data/processed/2.01-seal-chemical-space-analysis/cluster_*.png` — individual cluster grids

## Data artifacts

- `data/interim/butina_clusters.parquet` — cluster assignments (Molecule Name, cluster_id, cluster_size) for reuse by downstream notebooks
- `data/processed/2.01-seal-chemical-space-analysis/cluster_stats.csv` — all 135 clusters with sizes and example molecules

## Source

`notebooks/2.01-seal-chemical-space-analysis.py`
