# Workflow

Use this loop for every development pass.

1. Restate the current objective in one sentence.
2. Inspect the existing artifacts before changing anything.
3. Choose the smallest testable experiment.
4. Implement the smallest change that advances the experiment.
5. Verify the change with a deterministic check or a concrete artifact.
6. Record the result, the risk, and the next step.

## Decision rule

- Prefer local repository changes before introducing new infrastructure.
- Prefer the smallest useful abstraction.
- If a choice affects safety, privacy, or app-store viability, document it before coding.

## Output format for each iteration

- What changed.
- What was verified.
- What remains uncertain.
- What to do next.

## When blocked

- Identify the exact missing fact.
- Check whether it can be discovered from the repo, MCP tools, or official docs.
- Ask the smallest possible question only if the fact cannot be inferred safely.

