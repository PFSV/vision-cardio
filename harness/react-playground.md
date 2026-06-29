# ReAct Playground

Use this as the working notebook for autonomous iterations.

## Loop

Observe

- Read the current docs and code.
- Identify the current state in one sentence.

Plan

- Pick one concrete action.
- State the expected artifact or verification signal.

Act

- Make the smallest reasonable change.
- Keep the change scoped to the chosen action.

Verify

- Run the lightest meaningful check.
- Capture the result, not just the intent.

Decide

- Continue, pivot, or stop.
- If continuing, name the next single step.

## Template

```text
Objective:
State:
Action:
Verification:
Result:
Next:
```

## Auto-evolve rule

- Prefer short cycles over large refactors.
- After each cycle, tighten the harness if the last pass exposed ambiguity.
- If a recurring failure appears twice, add a doc or script that prevents it on the third pass.

