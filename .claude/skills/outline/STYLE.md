# Carpenter-Singh Lab Writing Style Guide

Extracted from lab papers and Anne's writing advice for consistent outlining.

---

## Abstract Structure (Anne's Recipe)

Based on Eric Lander's grant structure, adapted for papers:

| Element         | Purpose                   | Example pattern                                                       |
| --------------- | ------------------------- | --------------------------------------------------------------------- |
| **Opportunity** | Why this matters broadly  | "X offers valuable insights..." / "A major challenge is..."           |
| **Problem**     | The specific gap          | "...but publicly available Y only includes..." / "However, it is..."  |
| **Approach**    | What we did               | "Here, we..." / "Here we present..."                                  |
| **Results**     | Key findings with numbers | "Analysis revealed..." / "We found X% (N)..."                         |
| **Impact**      | Broader significance      | "This resource enables..." / "Our approach could replace..."          |

---

## Introduction Paragraphs

Progressive gap narrowing - each paragraph ends on the "unknown":

| Paragraph | Scope        | Pattern                                           |
| --------- | ------------ | ------------------------------------------------- |
| 1         | Field gap    | Broad importance -> what's missing at field level |
| 2         | Subfield gap | Specific area -> what's unknown in this subfield  |
| 3         | Paper gap    | Clues/hypothesis -> specific untested question    |
| 4         | "Here we..." | Summary of approach (no detailed results)         |

### Characteristic Phrases

- **Para 1:** "A major challenge is...", "Historically, X are determined painstakingly..."
- **Para 2:** "However, the field lacks...", "Thus far, the largest... is...", "It is unknown whether..."
- **Para 3:** "Evidence in favor... include...", "It is also unknown whether..."
- **Para 4:** "Here, we present...", "Here we..." (appears only in final paragraph)

---

## Results Paragraphs

Each paragraph follows: **Question -> Data -> Answer**

### First Sentence Patterns (Setup)

- "To assess X, we investigated whether..."
- "We next benchmarked X against..."
- "To study X, a scientist must..."

### Middle (Data with Specifics)

- Include actual numbers: "68% (10,352) yielded..."
- Include statistics: "p-value = 0.026 using 844 random permutations"
- Include comparisons: "four times more often than..."

### Last Sentence Patterns (Answer)

- "These results suggest..."
- "Genes do not cluster by X, possibly because..."
- "This demonstrates that..."

### Subsection Titles

**DO:** Declarative statements (conclusions)

- "Performance degrades with distance from training data"
- "Cluster-based splits reveal optimistic bias in random splits"

**DON'T:** Methods or vague descriptions

- "Analysis of splitting strategies" (too vague)
- "Distance experiments" (methods, not findings)

### Results Section Organization

Results subsections follow a **logical arc**: Setup -> Validation -> Discovery -> Applications

| Position | Purpose | Example Title |
| -------- | ------- | ------------- |
| First | Establish the data/assay works | "The Expansion Tx dataset provides realistic ADMET data" |
| Early | Characterize the landscape | "Chemical space analysis reveals diverse scaffolds" |
| Middle | Present core findings | "Performance degrades with distance from training data" |
| Late | Show practical implications | "Case studies illustrate generalization failure modes" |

**Key principle:** Each subsection builds on the previous. Early sections establish credibility (validation), enabling readers to trust later sections (discovery).

---

## Quantitative Claim Patterns

Always include the actual numbers:

| Pattern               | Example                                                    |
| --------------------- | ---------------------------------------------------------- |
| Percentage with count | "68% (10,352) yielded a detectable phenotype"              |
| Fraction of total     | "7,031 genes (56% of tested genes)"                        |
| Statistical test      | "p-value = 0.026 using 844 random permutations"            |
| Comparison            | "four times more often than for the remainder"             |
| Range                 | "percent replicating scores (57% to 83%, low to high)"     |
| Precision/recall      | "100% recall and 100% precision in correctly calling..."   |

---

## Discussion Structure

| Paragraph | Content      | Pattern                                       |
| --------- | ------------ | --------------------------------------------- |
| 1         | Summary      | Restate findings + meaning (gist of Results)  |
| 2-N       | Limitations  | Weakness -> Context -> Future direction       |
| Final     | Significance | Connect to Introduction gap, broader impact   |

### Limitation Paragraph Structure

1. **Acknowledge weakness directly:** "This dataset has some limitations."
2. **Provide specific details:** "Most notably, only 7,618 molecules were tested..."
3. **Add context/mitigation:** "The experiment focused on a single therapeutic area..."
4. **Point to future:** "For this reason, it is crucial to evaluate across diverse programs..."

---

## Figure References

**DO:** Indirect reference (conclusion first)

- "Performance drops sharply beyond 0.4 Tanimoto distance (Figure X)."

**DON'T:** Direct reference (figure as subject)

- "Figure X shows that performance drops."

---

## Figure Legend Titles

**First sentence:** Bold, states the main conclusion

Example:

> **Performance degrades with distance from training data.** (a) Overview of the evaluation framework...

---

## Voice and Style

| Aspect   | Guideline                                                |
| -------- | -------------------------------------------------------- |
| Voice    | Active: "We tested..." not "X was tested..."             |
| Pronouns | Use "we" for group work                                  |
| Tense    | Past for methods/results, present for established facts  |
| Length   | Shorter is better - cut every unnecessary word           |
| Jargon   | Define on first use, avoid when possible                 |

---

## First-Sentence Test

After writing an outline, read only the first sentence of each paragraph. If it tells the complete story, the structure is sound.

The first sentences should form a readable summary that captures:

- What gap exists
- What was done
- What was found
- What it means
