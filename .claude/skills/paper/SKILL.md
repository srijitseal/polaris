---
name: paper
description: Load context from the paper draft and analysis checklist before working on an analysis. Use when starting work on a paper-related analysis to get relevant context.
allowed-tools: Read, Grep, Glob
---

# Paper Analysis Context

Load relevant context from the paper and checklist before starting analysis work.

## Usage

```
/paper [analysis description or notebook name]

Examples:
  /paper 2.01 chemical space
  /paper cluster split
  /paper performance distance
```

## Instructions

1. **Read the context files:**
   - `docs/paper/analysis_checklist.md` - analysis tracking (source of truth for status)
   - `docs/paper/outline.md` - the paper outline

2. **Find relevant sections** based on the user's query:
   - Search checklist for analysis status and notebook details
   - Search outline for related sections

3. **Present a context summary:**
   ```
   ## Analysis: [name]

   ### From Checklist (source of truth)
   - Status: [TODO/IN_PROGRESS/DONE]
   - Expected outputs: [what it should produce]
   - Dependencies: [data or prior analyses needed]

   ### From Outline
   - Related sections: [where this fits in the paper]
   - Key context: [relevant excerpts or summary]

   ### Ready to Start
   - Inputs available: [yes/no, with paths]
   - Suggested notebook: [path if exists]
   ```

4. **Then stop and let the user drive.** Don't start the analysis automatically.

5. **After work is done**, remind the user:
   - Update checklist status if needed
   - Use `/writeup` to document findings as a GitHub Issue

## Notes

- **Workflow**: `/paper` (context) -> analysis -> `/writeup` (GitHub Issue) -> update checklist
- **Source of truth for status**: `analysis_checklist.md` only
- All files evolve - always read fresh, don't assume content
- Keep summaries concise - extract only what's relevant
