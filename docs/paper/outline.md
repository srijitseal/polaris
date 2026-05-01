# Paper Outline: Polaris Model Validation

## Central Contribution

Chemical datasets are full of non-obvious biases (spurious correlations) and the chemical coverage of current benchmarks is extremely limited. Together, these lead to inflated performance measures that do not account for generalization. We provide a framework of industry-informed case studies for critically evaluating and reporting molecular ML model strengths and weaknesses, demonstrated on a realistic ADMET dataset from actual drug discovery campaigns.

## Framing

This paper sits between a review and an opinion paper. Instead of prescriptive "if-this-then-that" guidelines (like the decision tree in the Polaris Method Comparison preprint), we center on a **collection of case studies** in model validation — extending beyond dataset splitting to include model interpretability. The common thread: standard evaluation systematically overestimates real-world performance.

## Abstract (5 sentences)

1. **Opportunity**: Machine learning models for molecular property prediction are increasingly used in drug discovery, but their real-world performance depends on generalization to novel chemical matter — a property that current benchmarks do not adequately measure.
2. **Problem**: Standard benchmarks rely on random splits that leak structural information between train and test sets, and researchers frequently lack the industrial experience to ask the right validation questions, leading to claims that lack translational credibility.
3. **Approach**: Here, we present a generalization evaluation framework using the Expansion Tx ADMET dataset (7,618 molecules, 10 ADME endpoints, 4 CROs + internal, from RNA-small molecule drug discovery) with splitting strategies that mimic real-world deployment and case studies that expose common failure modes.
4. **Results**: We show that random splits overestimate performance by 42% (MA-RAE 0.474 vs 0.672 with cluster-based splitting), that naive scaffold splits offer no improvement over random (MA-RAE 0.510), and that models trained on one chemical series degrade 1.0–7.4× when applied to a different series. We further demonstrate that chirality-aware ECFP4 fingerprints partially resolve stereoisomer blindness but still under-predict true biological variation (15% of the dataset), that activity cliffs cause systematic prediction failures for 7 of 9 endpoints, and that random CV gives precise estimates of the wrong thing (tight variance at an optimistic level).
5. **Impact**: This framework enables the community to critically assess model generalization, moving beyond aggregate metrics to understand where and why models fail.

## Introduction (4 paragraphs)

### Para 1: Field gap
- ML-driven molecular property prediction is transforming drug discovery, but model evaluation practices have not kept pace. Claims of high performance often lack credibility because they are not validated against the distribution shifts encountered in real deployment. Lacking industrial experience and interdisciplinary expertise, researchers frequently fail to ask the right questions to validate performance for real-world applications.

### Para 2: Subfield gap
- ADMET prediction benchmarks predominantly use random splits, which allow structurally similar molecules to appear in both train and test sets, inflating performance metrics. Furthermore, the chemical coverage of current benchmarks is extremely limited. The area of dataset splitting is becoming increasingly crowded, but there remains no consensus framework that connects splitting choices to specific deployment scenarios (hit identification vs. lead optimization) or that systematically exposes the non-obvious biases in chemical datasets.

### Para 3: Paper gap
- It remains unclear how much performance degrades under realistic deployment conditions. Sheridan et al. (Merck) showed correlation between distance to training set and performance, but replicating these findings with a simple, compelling answer has proven elusive. We need a framework that goes beyond dataset splitting to probe model strengths and weaknesses through multiple complementary lenses — including activity cliffs, molecular variants, and chemical series.

### Para 4: "Here we..."
- Here, we introduce a model validation framework demonstrated on the Expansion Tx ADMET dataset — a one-of-a-kind public dataset from actual drug discovery campaigns at Expansion Therapeutics. Through a collection of case studies, we show how cluster-based, time-split, and target-distribution splitting strategies — each corresponding to a distinct deployment scenario in drug discovery — combined with performance-over-distance curves and targeted evaluations, reveal the gap between benchmark and deployment performance. Building on our companion work on method comparison best practices [19], we extend the argument for structured evaluation from method comparison to model evaluation itself.

## Results

### 1. ADMET data sparsity is endpoint-dependent

**Claim**: ADMET datasets are sparse with asymmetric coverage reflecting drug discovery realities; characterizing this is essential before any evaluation.

**Content**:
- 7,608 ML-ready molecules (5,326 train + 2,282 test, zero overlap) from Expansion Therapeutics' RNA-small molecule drug discovery programs
- 9 ML-ready ADME endpoints (RLM CLint only in raw): LogD, KSOL, HLM/MLM CLint, Caco-2 Papp A>B, Caco-2 Efflux, MPPB, MBPB, MGMB
- Data generated by 4 CROs (Aragen, Chempartner, Pharmaron, Wuxi) + internal; vast majority from Pharmaron
- Compounds tested across projects given importance of selectivity in RNA-targeting
- Ordinal molecule naming (E-XXXXXXX) preserves temporal ordering — train median index 15,918 vs test median 22,478 — enables time-split without explicit timestamps
- Several large chemical series (>500 compounds) — enables IID vs OOD evaluation
- Raw dataset includes out-of-range modifiers (">", "<"); ML-ready version has in-range only
- Endpoint coverage is highly asymmetric, reflecting drug discovery testing cascade:
  - Well-covered: LogD (7,309, 96%), KSOL (7,298, 96%), MLM CLint (5,692, 75%)
  - Moderate: HLM CLint (4,541, 60%), Caco-2 Papp/Efflux (~3,775, 50%)
  - Sparse: MPPB (1,756, 23%), MBPB (1,426, 19%), MGMB (431, 6%)
  - All 9 endpoints show significant train/test distribution shift (KS test p < 0.01), especially Caco-2 Efflux (KS=0.37) and Caco-2 Papp (KS=0.28) — consistent with permeability assays added later in campaign
- Physicochemical properties confirm RNA-targeting hypothesis: median MW ~480 (many exceed Lipinski 500), high aromaticity (median ~4-5 rings), low FractionCSP3 (~0.2), high HBA counts — differs from typical protein modulators
- Competition split: test-to-train 1-NN median distance 0.375, nearly double within-train median 0.203 — reflects natural evolution of medicinal chemistry exploring increasingly distant chemical space

**Open questions**:
- Can CRO annotations be obtained per compound? (important for understanding data consistency, cf. Landrum & Riniker)
- Can repeat measurements be used to characterize aleatoric uncertainty per assay?
- Can actual targets be shared? (selectivity context)

**Figures**: Figure 1 (train vs test distance distributions), Table 1 (endpoint coverage)

### 2. Chemical space reveals concentrated series structure

**Claim**: In real drug discovery programs, the majority of medicinal chemistry effort is concentrated on a few promising chemical series, while smaller exploratory series and singleton hits populate the long tail. This series structure creates a natural challenge for generalization.

**Content**:
- Butina clustering (cutoff 0.7) to identify chemical series — show cluster size distribution
  - 135 clusters total; top 5 sizes: 2,572, 1,301, 885, 762, 282; 33 singletons (24.4%); median size 4
  - Three dominant clusters contain ~62.5% of molecules — dataset has concentrated chemical series
- Models trained predominantly on one or two major series must extrapolate when encountering a distinct series with different pharmacophoric features and SAR patterns
- Visualize representative molecules from the 5 largest clusters (6 random examples each)

**Figures**: Figure 2 (cluster size distribution)

### 3. Three splitting strategies probe complementary generalization axes

**Claim**: Splitting strategies should be motivated by the distribution shift encountered in real-world deployment scenarios, and must be compatible with cross-validation. No single split captures all deployment realities.

**Three deployment scenarios** (each strategy corresponds to a distinct drug discovery phase):

#### 3a. Cluster-based splitting (Hit Identification / Virtual Screening)
- Goal: generalize to structurally novel compounds not represented in training data
- EKM + MiniBatchKMeans (k=20) → greedy assignment to 5 folds
- 5 repeats × 5 folds = 25 train/test splits per endpoint
- Produces largest structural distances (LogD median 1-NN: 0.369–0.440 across 5 folds) with near-zero structural overlap

#### 3b. Time-based splitting (Prospective Deployment)
- Exploit ordinal molecule naming (E-XXXXXXX) as temporal proxy
- Expanding window CV: 4 folds, each trains on all earlier data, tests on next temporal chunk
- Intermediate distances (0.301–0.386) with non-monotonic per-fold patterns — chemical series exploration is non-linear in time
- Endpoint value distributions shift across folds (KS test p < 0.05), confirming temporal drift in both chemical and property space

#### 3c. Target-value splitting (Lead Optimization)
- Molecules sorted by endpoint value, expanding window CV (4 folds)
- Lowest structural distances (0.245–0.286) but maximal target distribution shift (KS 0.956–0.988)
- Tests whether models can extrapolate to value ranges not seen during training

All strategies produced balanced fold sizes (max/min ratio < 1.15). Distance distributions illustrated for LogD as representative endpoint.

**Figures**: Figure 3 (test-to-train 1-NN distance distributions by strategy)

### 4. Performance degrades systematically with structural distance

**Claim**: Aggregate performance metrics hide systematic degradation at the frontier of the training distribution. A performance-over-distance curve reveals where models actually fail.

**Model**: Optuna TPE-tuned XGBoost (30 trials, 3-fold CV, 9 hyperparameters per endpoint per fold) — same tuning procedure across all 9 endpoints and 3 split strategies to ensure performance differences reflect the splitting strategy, not arbitrary hyperparameter choices.

**Key findings** (all 9 endpoints, competition metrics):
- **Cluster-split** (MA-RAE 0.673) best overall — each fold sees structurally diverse training data
- **Time-split** (MA-RAE 0.771) worse because early folds have small training sets (expanding window)
- **Target-split** (MA-RAE 6.118) catastrophic — tree-based models fundamentally cannot extrapolate beyond training value range
- **Structural degradation**: cluster-split shows 2–4× RMSE degradation for most endpoints:
  - LogD 3.60× (bulk thermodynamic property, but ECFP4 representation changes dramatically between distant scaffolds)
  - HLM CLint 3.04× (metabolic soft spot accessibility diluted by rest of molecule in fingerprint)
  - KSOL 1.98×, MBPB 1.90×, MGMB 1.87×
  - Caco-2 endpoints show minimal degradation (Papp A>B 0.98×, Efflux 0.95×) — passive permeability governed by global properties (size, lipophilicity, H-bonding) captured by RDKit 2D descriptors

**Figures**: Figure 4 (performance-over-distance curves, 9 endpoints × 3 strategies), Table 2 (per-endpoint RAE by strategy)

### 5. IID versus OOD: degradation across chemical series boundaries

**Claim**: A model validated on historical compounds from a single program may appear excellent yet fail silently when applied to a new chemical series entering the pipeline. Uses two things normally unavailable in public datasets — time split and chemical series.

**Setup** (Optuna TPE-tuned XGBoost):
- Train on largest Butina cluster (n=2,572), temporally split: first 80% (n=2,057) training, last 20% (n=515) IID validation
- OOD test: second-largest cluster (n=1,301) — a structurally distinct chemical series, no molecules in common
- IID 1-NN median distance: 0.288; OOD 1-NN median: 0.763 — nearly 3× further
- Endpoint coverage verified: LogD/KSOL 98–100% in both clusters; clearance/permeability 25–73%; MGMB too sparse (7–17%) → excluded

**Key findings**: Substantial OOD degradation across all endpoints:
- R² goes negative OOD for 4 of 8 endpoints (MLM CLint −1.07, MBPB −0.48, Caco-2 Papp −0.09, Caco-2 Efflux −0.005)
- Median squared error increases 1.0–7.4× (MLM CLint 7.4×, KSOL 6.8×, MPPB 5.9×, MBPB 5.6×, LogD 4.8×)
- HLM CLint shows negative R² on IID (−0.12) — clearance hard even within-series
- Caco-2 Papp most robust (1.0× degradation) — consistent with performance-over-distance finding
- LogD, despite best IID performance (R²=0.64), degrades 4.8× to R²=0.11 on OOD — within-series performance is fundamentally unreliable predictor of cross-series generalization

**Mechanistic interpretation**:
- MLM CLint (7.4×): tuning may overfit to training series SAR patterns; CYP450 metabolic soft-spot accessibility is highly scaffold-dependent
- KSOL (6.8×): solubility depends on crystal packing forces, highly scaffold-dependent
- LogD (4.8×): ECFP4 representation changes between distant scaffolds even when global polarity conserved

**Figures**: Figure 5 (IID vs OOD squared error distributions), Table 3 (IID vs OOD metrics: R², Spearman, SE fold-change)

### 6. Scaffold splits offer no improvement; cluster-based splitting is required

**Claim**: Naive Bemis-Murcko scaffold splits provide no meaningful advantage over random splits. Distance-aware methods are necessary to ensure genuine structural separation.

**Setup** (Optuna TPE-tuned XGBoost, 5-fold CV, all 9 endpoints):
- Naive scaffold splits: greedy assignment of Murcko scaffold groups to 5 folds
- Random splits: seed=42
- Cluster-based splits: EKM + KMeans (as in section 3)

**Key findings**:
- Random MA-RAE: 0.474, Scaffold MA-RAE: 0.510, Cluster MA-RAE: 0.672
- Scaffold-random RAE gap: only 0.02–0.06 per endpoint (negligible)
- Scaffold-cluster RAE gap: 0.13–0.27 per endpoint (substantial)
- 1-NN median distances: scaffold 0.246 vs random 0.203 vs cluster 0.424
- **KS tests**: scaffold vs random D = 0.203 (modest), cluster vs random D = 0.662 (large), cluster vs scaffold D = 0.536, all p < 10^-10
- **Structural explanation**: 3,337 unique Murcko scaffolds; 2,252 (67.5%) contain only a single molecule, median size 1. Greedy assignment of singletons to folds ≈ random assignment. Structurally similar molecules with different Murcko frameworks (ring size change, heteroatom substitution) get assigned to different groups but remain proximate in fingerprint space

**Figures**: Figure S2 (performance comparison), Table S2 (per-endpoint RAE by strategy) — supplementary

### 6b. Scaffold boundary analysis: scaffold splits are a blunt instrument you can't tune

**Claim**: While the Murcko scaffold carries real, endpoint-specific biological signal (same-scaffold pairs have smaller activity differences at a given fingerprint distance), the scaffold *split procedure* cannot leverage this because of degenerate group sizes, pervasive cross-scaffold structural proximity, and lack of tunability. Distance-based methods let the user explicitly parametrize the degree of structural novelty.

**Three analyses** (NB 2.16, no model needed — pure dataset analysis):

**1. Scaffold boundary violations**:
- For each molecule, identified nearest neighbor overall and nearest neighbor with a different Murcko scaffold
- **56.4% boundary violation rate** (4,289/7,608 molecules): more than half of all molecules have their nearest neighbor in a *different* scaffold group
- The scaffold boundary does not separate structurally similar molecules

**2. Cross-scaffold proximity**:
- Overall 1-NN median distance: 0.189; cross-scaffold 1-NN median: 0.233 — distributions nearly overlap
- 37.2% of non-singleton molecules have a closer cross-scaffold neighbor than within-scaffold
- The most informative training molecule often sits on the wrong side of the scaffold boundary

**3. Activity concordance conditioned on scaffold**:
- All ~29M molecule pairs per endpoint, binned by Tanimoto distance (width 0.05)
- Compared mean |Δactivity| for same-scaffold vs different-scaffold pairs per bin
- 95/136 Mann-Whitney U tests significant uncorrected (p < 0.05); 91/136 remain significant after Benjamini-Hochberg FDR correction; expected by chance at α=0.05: 6.8
- Among BH-significant tests, 83/91 (91%) showed same-scaffold pairs having *smaller* activity differences
- **Endpoint-specific patterns** reveal the scaffold carries real biological signal:
  - Caco-2 Efflux: strongest effect — same-scaffold |Δy| flat (~0.25) while different-scaffold rises to 0.52 (P-gp substrate recognition is scaffold-dependent)
  - HLM CLint, MLM CLint: strong consistent effect across all distance bins (CYP450 metabolic soft-spot accessibility)
  - LogD: moderate, consistent gap (scaffold and substituents both contribute to partitioning)
  - KSOL: effect *reverses* at high distances (>0.65) — crystal packing is scaffold-dependent but highly substituent-sensitive
  - MBPB: mostly non-significant (protein binding may depend more on surface substituents)
  - MGMB: sparse data (431 molecules), underpowered

**Why scaffold splits fail despite real scaffold signal** (four reasons):
1. **No tunability**: boundary is fixed by the Murcko decomposition — user cannot adjust degree of distribution shift
2. **Degenerate size distribution**: 67.5% singletons → greedy assignment to folds ≈ random
3. **Structural leakage**: 56.2% boundary violation rate → most informative training molecule on wrong side
4. **Endpoint-agnostic**: scaffold effect varies dramatically across endpoints but split applies one boundary to all

**Contrast with distance-based methods**: user controls cluster cutoff, number of clusters, minimum test-to-train distance. Degree of structural novelty is explicitly parametrized and can be matched to the deployment scenario.

**Figures**: 3 figures in issue #16 (cross-scaffold proximity, activity concordance, boundary violation summary) — candidate for supplementary

### 7. Random cross-validation gives precise estimates of the wrong thing

**Claim**: Random CV's low between-repeat variance is a structural artefact of sampling design — each fold is an unbiased draw from the same joint, so different seeds produce near-identical marginals and near-identical estimates. The tightness is not informative; it is a precision estimate of a quantity (performance on near-duplicates leaking across folds) that does not correspond to deployment. The between-strategy MA-RAE gap dwarfs within-strategy spread by an order of magnitude, so splitting strategy dominates repeat count regardless of which within-strategy reproducibility is tighter on any given endpoint.

**Setup** (Optuna TPE-tuned XGBoost):
- 5 independent random 5-fold CV repeats (seeds 0–4)
- 5 cluster-based split repeats (stochastic EKM + MiniBatchKMeans)
- ~450 models total

**Key findings**:
- Random splits: RAE std 0.002–0.012, ranges 0.006–0.033 — remarkably tight, as expected from unbiased draws over the same joint
- Cluster splits: RAE std 0.007–0.064, ranges 0.018–0.182 — cluster/random std ratio spans 0.56× (KSOL, where cluster is tighter) to 14× (Caco-2 Efflux), median ~5×
- MGMB most extreme: cluster R² spans 0.299–0.577 across 5 repeats — "poor" or "moderate" depending on seed
- **Mann-Whitney U tests**: U = 25.0, p = 0.0079 for all 9 endpoints — cluster and random RAE distributions entirely non-overlapping (U=25 is maximum possible for n1=n2=5; p=0.0079 is minimum achievable)
- MA-RAE gap between strategies (~0.20) dwarfs within-strategy variance by ~10×, even for KSOL where cluster is the more reproducible strategy
- Random splits leak structurally similar molecules across fold boundaries (median test-to-train 1-NN of 0.203)

**Practical recommendation**: Use distance-aware splits with multiple repeats and report confidence intervals. Five cluster repeats with wide CIs give an honest picture; narrow random-CV CIs are a property of sampling, not of generalization.

**Figures**: Figure S3 (RAE distributions), Table S3 (R² variance) — supplementary

### 8. Activity cliffs expose interpolation failures

**Claim**: Activity cliffs — structurally similar molecules with unexpectedly large activity differences — violate the smoothness assumption underlying fingerprint-based models. This is a qualitatively distinct failure mode from extrapolation failure.

**Definition**: Cliff pairs = Tanimoto similarity > 0.85 (ECFP4) + absolute activity difference ≥ 1.0 log units (≈ 10-fold change in raw measurement). Activity differences computed in log₁₀(x+1) space (matching target transform); for LogD, computed in raw units. Any molecule in ≥1 cliff pair labeled a cliff molecule.

**Key findings** (Optuna TPE-tuned XGBoost, cluster-split CV, 5 folds):
- Cliff molecules comprise 0–4.6% of molecules per endpoint (absolute threshold ≥ 1.0 log units)
- Cliff RAE exceeds non-cliff RAE for all 7 endpoints with sufficient cliff populations:
  - Worst: Caco-2 Efflux cliff RAE = 0.95 vs non-cliff 0.58; Caco-2 Papp cliff RAE = 0.91 vs non-cliff 0.64
  - HLM CLint cliff RAE = 0.87 vs non-cliff 0.81
- MBPB (0 cliffs) and MPPB (4 cliffs) had insufficient cliff molecules for evaluation
- Endpoints may be particularly cliff-prone when governed by specific structural motifs (CYP450 recognition, P-gp substrate features)

**Figures**: Figure S4 (squared error distributions), Table S4 (cliff prevalence and performance) — supplementary

### 9. Molecular representation limits prediction consistency

**Claim**: The fingerprint's implicit similarity function may not align with the biological similarity function for the endpoint of interest. Two distinct failure modes: under-sensitivity to stereochemistry and amplification of scaffold decorations.

**Stereoisomer analysis** (Optuna TPE-tuned XGBoost, cluster-split CV, 5 folds):
- 548 stereoisomer groups (1,152 molecules, 15.1% of dataset) — pairs/sets of enantiomers and diastereomers
- Chirality-aware ECFP4 (useChirality=True) produces non-zero but small distances (median 0.067, up from 0.000)
- Prediction CV = 0.000–0.015 across all 9 endpoints — 10–50× lower than scaffold decorations (0.074–0.442) and random pairs (0.128–0.404)
- Only 3 of ~200 RDKit 2D descriptors differ between stereoisomers (`NumAtomStereoCenters`, `NumUnspecifiedAtomStereoCenters`, `Ipc`), contributing negligibly
- Consistency ratios < 1 for most endpoints — model *under-predicts* true biological variation between enantiomers
- Biological significance: (R)/(S)-enantiomers can differ by orders of magnitude in clearance (stereoselective CYP450), binding (chiral pockets), efflux (stereoselective P-gp)

**Scaffold decoration analysis**:
- 1,050 scaffold decoration groups (3,835 molecules, 50.4%) — analogs sharing same Bemis-Murcko core, size 2–20
- Median intra-group Tanimoto distance 0.349; prediction CV 2–3× lower than random pairs (model learns scaffold-level trends)
- Consistency ratios > 1 across most endpoints (up to 3.4× for KSOL, 2.0× for Caco-2 Papp A>B) — model *amplifies* substituent effects beyond true biological variation
- Amplification mechanism: single atom change in decoration modifies all circular substructures including that atom, flipping dozens of ECFP4 bits

**Resonance form analysis** (Optuna TPE-tuned XGBoost, cluster-split CV, 5 folds; RIGR framework, Zalte et al. 2025 JCIM):
- Protonate-enumerate-reprotonate-deduplicate pipeline at endpoint-specific pH (7.4 or 6.5) ensures fair evaluation matching training preprocessing
- **pH-dependent prevalence**: 20-53% of molecules have >1 distinct form at pH 7.4; only 1.8% at pH 6.5 (Caco-2) — protonation at lower pH collapses resonance variants
- Resonance forms have median ECFP4 Tanimoto distance of 0.462 (cf. scaffold decorations 0.349, stereoisomers 0.067, random pairs 0.849) — larger shifts than different substituents despite chemical identity
- **RMSE swing up to 27%** (LogD): worst-case form increases RMSE by 20%, best-case reduces by 7.0%. Worsening meets or exceeds improvement across all 9 endpoints
- Endpoint-specific: LogD most affected (electronic distribution), clearance moderate (CYP450 soft spots), Caco-2 least affected (global properties + pH 6.5 collapse)
- **Per-molecule risk**: individual molecules can experience prediction error ranges of several hundred percent — practitioner receives no signal of instability
- Bond order changes, aromaticity flag shifts, and formal charge differences alter the molecular graph that Morgan fingerprints encode — a representation-level vulnerability distinct from extrapolation or interpolation failures

**Practical caveat**: Many public/pharma datasets contain unreliable stereochemistry annotations (racemic synthesis, incomplete chiral separation). Chirality-aware modeling can be counterproductive by fitting noise.

**Figures**: Figure 6 (fingerprint distances), Table 4 (prediction consistency), Figures S5–S6 (prediction CV, spread scatter), Figure 7 (resonance sensitivity panel — main text), Figure S8 (resonance fingerprint distances — supplementary)

## Discussions

### Summary of four failure modes
- **Extrapolation failure** (IID vs OOD): 1.0–7.4× worse performance across chemical series boundaries, R² negative for 4 of 8 endpoints on OOD — models trained on one series fail silently on a new series
- **Interpolation failure** (activity cliffs): 0–4.6% of molecules (absolute ≥ 1.0 log units) show systematically higher error for all 7 endpoints with sufficient cliff populations
- **Representation failure** (molecular variants): chirality-aware ECFP4 partially resolves stereoisomer blindness but still under-predicts true biological variation (15% of dataset); scaffold decorations amplified beyond true biological variation (up to 3.4× for KSOL); resonance form ambiguity causes up to 27% RMSE swing (LogD) with worsening meeting or exceeding improvement — a representation-level vulnerability where chemically identical molecules occupy distant fingerprint regions (median Tanimoto 0.46)
- **Evaluation failure** (scaffold ≈ random; split variance): random CV gives precise estimates of the wrong thing; scaffold splits are indistinguishable from random (MA-RAE 0.510 vs 0.474); single cluster-split R² spans 0.30–0.58 for MGMB
- The framework connects evaluation choices to deployment scenarios (hit identification vs. lead optimization)
- The Expansion Tx dataset uniquely enables these analyses due to its real-world provenance (ordinal ordering, chemical series, multi-endpoint coverage)

### General principles
- The four failure modes are not dataset-specific but structural features of molecular ML: any dataset with concentrated chemical series → extrapolation failure; any fingerprint-based model → activity cliff failure; any fixed-radius fingerprint → representation blind spots; any evaluation not matching deployment distribution → misleading metrics
- Encourage researchers to apply the framework to their own datasets rather than treating Expansion Tx results as universal benchmarks

### Practical recommendations (six)
1. Start from the deployment scenario — choose splitting strategy that creates the relevant distribution shift by construction
2. Characterize dataset before splitting — chemical series structure, endpoint coverage, distance distributions determine which analyses are feasible
3. Use distance-aware splits with multiple repeats and report confidence intervals; avoid random splits and single-number metrics
4. Probe specific failure modes — performance-over-distance curves, activity cliff analysis, molecular variant consistency each reveal blind spots that aggregate metrics hide
5. Audit molecular representation for blind spots using variant group analysis; the fingerprint's implicit similarity function may not align with the biological similarity function for your endpoint
6. Accompany every prediction with a distance-based confidence estimate — performance-over-distance curves map nearest-neighbor distance to expected error range; non-technical end users (medicinal chemists, biologists, project leaders) need this context

### Conclusion (within Discussion)
- Cultural shift needed: journal/conference reviewers should require distribution-relevant splits, not just random splits
- Authors should characterize the distribution shift their model will face in deployment
- Companion work on method comparison best practices [19] argued for structured reporting; here we extend to model evaluation
- The community has the tools — what is needed now is the expectation that rigorous evaluation is the standard, not the exception
- Framework + Expansion Tx dataset openly available to encourage rigorous evaluation

### Limitations
- Single therapeutic area (RNA-small molecule) — compounds may have different properties from protein modulators; generalizability to other target classes unknown
- Dataset size (7,608) is modest relative to large-scale benchmarks — some analyses may be power-limited
- Endpoint sparsity — not all molecules have all 9 endpoints measured (MGMB only 6%)
- Distance metric choice (ECFP4 + Tanimoto/Jaccard) is pragmatic but endpoint-specific metrics could perform better
- Data curation assumed — detailed curation guidance out of scope (planned for future work)
- CRO consistency not fully characterized (pending annotation availability)
- Chirality-aware ECFP4 partially resolves stereoisomer blindness but 3D descriptors could further improve stereochemical sensitivity

## Methods

### Dataset
- Source: Expansion Therapeutics, via OpenADMET
- 7,618 raw molecules (10 endpoints incl. RLM CLint); 7,608 ML-ready (9 endpoints, in-range only)
- Raw vs ML-ready versions (out-of-range modifier handling)
- Train/test split from competition (5,326 / 2,282, zero overlap)

### Fingerprints and Distance
- ECFP4, 2048-bit, Tanimoto similarity/distance
- Justification: prioritize simplicity over perfection; no single metric captures full molecular richness

### Clustering
- Butina clustering (cutoff 0.7) for chemical series identification — 135 clusters, top 3 contain 62.5% of molecules
- k-means with Empirical Kernel Map for splitting (compatible with Tanimoto distance)
- Mini Batch K-Means for scalability and stochasticity
- Over-cluster (k=20) then greedily assign to 5 folds for balance (ADR-001)

### Splitting Algorithms
- Cluster-based: EKM + k-means, per-endpoint fold assignments (only molecules with data for each endpoint)
- Time-split: ordinal molecule naming as temporal proxy
- Target distribution: Rolling Window CV for regression

### Features
- 2048-bit ECFP4 fingerprints + full RDKit 2D descriptor suite (~200 descriptors, zero-variance removed, StandardScaler-normalized)
- Molecules protonated at assay-relevant pH using dimorphite_dl (pH 7.4 for most endpoints, pH 6.5 for Caco-2) before feature computation
- Features computed per unique pH, then selected per endpoint

### Evaluation Metrics
- Competition metrics: MAE, R², Spearman ρ, Kendall's τ, RAE, MA-RAE (macro-averaged RAE across 9 endpoints)
- Targets log-transformed via `log10(clip(x, 1e-10) + 1)` for all endpoints except LogD, matching competition protocol
- Performance-over-distance curves with sliding window bins (RMSE per bin)
- Cross-validation with confidence intervals (5x5 CV where applicable)

### Models
- **Optuna TPE-tuned XGBoost** (all notebooks): 30 trials of Bayesian optimization via Tree-structured Parzen Estimator (TPE), 3-fold inner CV, MAE scoring. 9 hyperparameters tuned: `n_estimators` (100–1000), `max_depth` (3–12), `learning_rate` (0.01–0.3, log), `subsample` (0.5–1.0), `colsample_bytree` (0.3–1.0), `min_child_weight` (1–10), `gamma` (0.0–5.0), `reg_alpha` (0.0–1.0), `reg_lambda` (0.5–3.0). Same tuning procedure applied uniformly across all 8 analysis notebooks (NB 2.07–2.13, 2.15) — ensures observed differences reflect splitting strategy or molecular subpopulation, not arbitrary hyperparameter choices.
- Best parameters cached per (endpoint, split_strategy, fold) in JSON files (`data/interim/optuna_cache/`); tuning cost paid once, subsequent runs load cached params. Shared utility: `src/polaris_generalization/tuning.py` (ADR-002).
- Focus is on evaluation framework, not model architecture

## Figures (current manuscript mapping)

| Figure | Title | Source | File |
|--------|-------|--------|------|
| Fig 1 | Train vs test distance distributions | NB 0.02 | `train_vs_test_distances.png` |
| Fig 2 | Butina cluster size distribution | NB 2.01 | `cluster_size_distribution.png` |
| Fig 3 | Test-to-train 1-NN distance by splitting strategy | NB 2.06 | `distance_distributions_by_strategy.png` |
| Fig 4 | Performance-over-distance curves (9 endpoints × 3 strategies) | NB 2.07 | `performance_over_distance.png` |
| Fig 5 | IID vs OOD squared error distributions | NB 2.09 | `squared_error_distributions.png` |
| Fig 6 | Intra-group fingerprint distances (stereo/scaffold/random) | NB 2.13 | `fingerprint_distances.png` |
| Fig S1 | Baseline scatter (per-endpoint tuned, competition split) | NB 2.08 | `scatter_predictions.png` |
| Fig S2 | Scaffold vs random vs cluster performance comparison | NB 2.11 | `metric_comparison.png` |
| Fig S3 | Split variance RAE distributions (5 random + 5 cluster) | NB 2.12 | `rae_distributions.png` |
| Fig S4 | Activity cliff squared error distributions | NB 2.10 | `squared_error_distributions.png` |
| Fig S5 | Prediction CV by variant type | NB 2.13 | `prediction_consistency.png` |
| Fig S6 | Predicted vs true range scatter by variant type | NB 2.13 | `spread_scatter.png` |
| Fig 7 | Resonance sensitivity panel (prevalence, improve/worsen, per-molecule error) | NB 2.15 | `resonance_sensitivity_panel.png` |
| Fig S7 | Activity concordance by scaffold membership | NB 2.16 | `activity_concordance.png` |
| Fig S8 | Resonance form fingerprint distance distributions | NB 2.15 | `fingerprint_distances.png` |

## Key References

1. Vamathevan, J. et al. Applications of machine learning in drug discovery and development. *Nat. Rev. Drug Discov.* **18**, 463–477 (2019).
2. Walters, W. P. & Barzilay, R. Critical assessment of AI in drug discovery. *Expert Opin. Drug Discov.* **16**, 937–947 (2021).
3. Sheridan, R. P. Time-split cross-validation as a method for estimating the goodness of prospective prediction. *J. Chem. Inf. Model.* **53**, 783–790 (2013).
4. Landrum, G. The problem(s) with scaffold splits / Using intake for chemistry. *RDKit Blog* (2023).
5. Walters, W. P. Some Thoughts on Evaluating Predictive Models. *Practical Cheminformatics Blog* (2019).
6. Tossou, P., Wognum, C., et al. Real-World Molecular Out-Of-Distribution: Specification and Investigation. *J. Chem. Inf. Model.* **64**, 697–711 (2024).
7. Hauptmann, T. et al. SPECTRA: a tool for enhanced performance assessment. *J. Cheminform.* (2024).
8. Tilborg, D. van et al. Exposing the limitations of molecular machine learning with activity cliffs. *J. Chem. Inf. Model.* **62**, 5938–5951 (2022).
9. Sheridan, R. P. et al. Experimental error, kurtosis, activity cliffs, and methodology: What limits the predictivity of quantitative structure–activity relationship models? *J. Chem. Inf. Model.* **60**, 1969–1982 (2020).
10. Tilborg, D. van et al. MoleculeACE: evaluating predictive performance on activity cliff compounds. *J. Cheminform.* **14**, 45 (2022).
11. Seal, S. LinkedIn post on molecular variant consistency in ADMET predictions (2025).
12. OpenADMET Expansion Rx Challenge (2025). Dataset: https://huggingface.co/datasets/openadmet/openadmet-expansionrx-challenge-data
13. Landrum, G. RDKit: Open-source cheminformatics software. https://www.rdkit.org
14. Ropp, P. J. et al. Dimorphite-DL: an open-source program for enumerating the ionization states of drug-like small molecules. *J. Cheminform.* **11**, 14 (2019).
15. Butina, D. Unsupervised data base clustering based on daylight's fingerprint and Tanimoto similarity. *J. Chem. Inf. Comput. Sci.* **39**, 747–750 (1999).
16. Ralaivola, L. et al. Graph kernels for chemical informatics. *Neural Netw.* **18**, 1093–1110 (2005).
17. Chen, T. & Guestrin, C. XGBoost: A scalable tree boosting system. In *Proc. KDD* 785–794 (2016).
18. Bemis, G. W. & Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* **39**, 2887–2893 (1996).
19. Ash, J. R., Wognum, C., et al. Practically Significant Method Comparison Protocols for Machine Learning in Small Molecule Drug Discovery. *J. Chem. Inf. Model.* **65**, (18) (2025). DOI: 10.1021/acs.jcim.5c01609.
20. Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **50**, 742–751 (2010).
21. Zalte, A. et al. RIGR: Representation Instability in Generalization and Robustness. *J. Chem. Inf. Model.* (2025).
