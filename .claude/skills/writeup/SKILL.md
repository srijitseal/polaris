---
name: writeup
description: Write up analysis findings as a GitHub Issue following lab notebook workflow. Use when documenting experiments, analysis results, or conclusions from the conversation.
allowed-tools: Bash(gh issue create:*), Bash(gh issue edit:*), Bash(gh issue view:*), Bash(gh issue list:*), Read
---

# Write Up Analysis Findings

Document analysis findings from the current conversation as a GitHub Issue.

## Match Existing Style

Before writing, check existing examples to match conventions:
- **Issues**: `gh issue list` then `gh issue view NUMBER` to see format/structure of recent issues

## Instructions

1. **Gather from conversation**:
   - What question was investigated?
   - What were the key results (with specific numbers)?
   - What code/queries were used? (include executable snippets)
   - What's the takeaway?

2. **Title**: Use a conclusion statement, not a question
   - Good: "Random splits overestimate ADMET model performance by 15-30%"
   - Avoid: "How do random vs cluster splits compare?"

3. **Format the issue** (keep it concise):

```markdown
## Summary
[1-2 sentences: what was found]

## Key Findings
[Table or bullet points with specific numbers]

## Plots
[Placeholders with file paths - user will paste actual images]
<!-- Paste: data/processed/example_plot.png -->

## Reproduce
[Executable code block or notebook path]

## Conclusion
[1-2 sentences: the takeaway, what to do differently]
```

4. **Present to user for review** before creating - they may want to simplify or reframe

5. **Create or edit issue**:
```bash
gh issue create --title "Title" --body "Body"
```

## Working with Plots

- You cannot upload images directly. Use HTML comment placeholders with file paths
- Tell the user which images to paste and where
- When editing an existing issue, **preserve uploaded image URLs**
