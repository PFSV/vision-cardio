# Exercise Policy

## Goal

Use the passive heart-rate signal plus user behavior to learn better exercise direction over time, while keeping the system bounded, on-device, and safe.

## Product shape

- Input: passive HR/RHR estimates, workout completion, user feedback, and simple recovery signals.
- Output: a recommended workout direction such as easy, maintain, or push.
- Learning: user-specific adaptation on the phone only.

## What may adapt

- Workout intensity recommendation.
- Weekly load progression.
- Timing of the next suggestion.
- Per-user calibration from observed completion and recovery.

## What must stay fixed

- Safety rules.
- No diagnosis claims.
- No automatic escalation beyond bounded levels.
- No hidden background capture.

## Learning signals

- Whether the user completed the suggestion.
- Post-workout recovery quality.
- Resting-heart-rate trend stability.
- Session consistency over time.
- Optional explicit thumbs-up/down feedback.

## Safety gates

- Never increase intensity by more than one step per update.
- Hold or reduce intensity if recovery worsens.
- Hold or reduce intensity if adherence collapses.
- Freeze updates when the data quality is poor.
- Surface the recommendation as wellness guidance, not medical advice.

## State

- Fixed base policy.
- Local user state.
- Version number.
- Warmup flag.
- Freeze flag.

## First implementation

1. Build a deterministic rule-based policy engine.
2. Persist per-user state locally.
3. Update the policy from recent user history.
4. Add a simulator for offline testing.
5. Add a future hook for lightweight on-device learning if the rule-based policy is stable.

