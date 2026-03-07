---
name: outline
description: Create structured paper outlines using Ten Rules framework. Use when starting a new section, planning results, or restructuring content.
allowed-tools: Read, Glob, Grep, Edit, Write, AskUserQuestion
---

# Paper Outlining Skill

Generate structured one-sentence-per-paragraph outlines following the "Ten Simple Rules for Structuring Papers" (Mensh & Kording) and writing conventions.

## Usage

```text
/outline [section]

Examples:
  /outline abstract
  /outline introduction
  /outline results
  /outline discussion
  /outline          # (asks which section)
```

## Instructions

### Phase 1: Survey Available Findings

**Before asking any questions, read all available analysis results and present a summary.**

1. Read `docs/paper/analysis_checklist.md` for analysis status
2. Check `notebooks/` for completed analyses
3. Group findings into themes
4. Present a table summarizing available findings

### Phase 2: Confirm Selection

**Do not proceed until the user confirms.**

1. Based on user input, list the selected findings explicitly
2. Ask: "Does this selection look right? Any changes before I generate the outline?"
3. **Wait for explicit confirmation** before proceeding.

### Phase 3: Gather Section-Specific Context

Only after selection is confirmed, ask section-specific questions:

For **Abstract**: "What is the central claim in one sentence?"
For **Introduction**: "What's the field-level gap?"
For **Results**: "What logical order should these findings appear?"
For **Discussion**: "Which limitations are most important to acknowledge?"

### Phase 4: Generate Outline

Apply structure from `TENRULES.md` and style from `STYLE.md`.

**Output format:** Numbered sentences with annotations:

```markdown
## [Section]: [Central claim]

### Subsection Title (if Results)

1. [Context] First sentence establishing context...
2. [Data] Second sentence with key finding/data...
3. [Conclusion] Third sentence interpreting the result...
```

### Phase 5: Iterate

After presenting the outline:
- Ask if any sections need expansion or reordering
- Check against the relevant checklist in `CHECKLISTS.md`
- Offer to revise based on feedback

## Files Referenced

| File                               | Purpose                                |
| ---------------------------------- | -------------------------------------- |
| `TENRULES.md`                      | Condensed actionable rules             |
| `STYLE.md`                         | Writing patterns and examples          |
| `CHECKLISTS.md`                    | Section-specific verification          |
| `docs/paper/outline.md`           | Current paper outline                  |
| `docs/paper/analysis_checklist.md` | Analysis status tracking               |

## Critical Rules

1. **Never generate an outline without showing available findings first**
2. **Never proceed without explicit user confirmation on selection**
3. **Always show what you're including AND excluding**
4. **Outlines are one sentence per paragraph - no prose yet**
