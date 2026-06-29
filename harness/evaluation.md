# Evaluation

Do not treat the work as ready for users until these gates are met.

## Technical gates

- The capture path is explicit and bounded.
- The inference path is deterministic enough to reproduce on the same input class.
- Confidence gating is in place for noisy frames.
- The app behaves reasonably under low light, motion, partial face occlusion, and camera permission denial.

## Product gates

- The UI explains what is being measured.
- The user can opt out and delete data.
- The app does not overstate diagnostic certainty.
- The app does not require a wearable for the core value proposition.

## Safety and compliance gates

- No medical diagnosis language without a regulatory plan.
- No hidden background camera use.
- Privacy policy and data-handling story are written before release.
- Any clinical-claims path is separated from consumer wellness claims.

## Research-to-product gate

If the implementation diverges from the Nature paper pattern, document why:

- different camera timing,
- different model architecture,
- different gating strategy,
- different target population,
- different metrics.

