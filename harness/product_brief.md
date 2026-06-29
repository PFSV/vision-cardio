# Product Brief

## Source inspiration

The paper *Passive heart-rate monitoring during smartphone use in everyday life* (Nature, 2026) shows that front-camera facial video after phone unlock can support passive heart-rate and resting heart-rate estimation.

Key product implications:

- Use short, opportunistic camera captures after unlock or explicit consent.
- Prefer on-device inference and confidence gating.
- Aggregate daily measurements into a trend, not just a single reading.
- Treat the result as wellness monitoring unless and until a clinical pathway exists.
- Keep self-evolution bounded to per-user personalization, not free retraining.
- Use the signal to adapt exercise direction over time, within safe bounds.

## Product goal

Build a consumer mobile app that passively estimates heart rate during normal smartphone use and surfaces a daily resting-heart-rate trend that is understandable, private, and low-friction.

## Default scope

- iOS-first prototype.
- Front-facing camera only.
- On-device processing for the critical path.
- Minimal data retention.
- Explicit opt-in and clear revocation.
- Fixed base model with a small on-device personalization layer.
- Bounded workout direction output such as easy, maintain, or push.

Recommended first-pass stack:

- `SwiftUI`
- `AVFoundation`
- `Vision`
- `Core ML`

## Non-goals for the first pass

- No diagnosis claims.
- No emergency triage.
- No always-on camera.
- No dependence on a wearable for the core feature.

## Reference metrics from the paper

- HR estimation target: MAPE under 10 percent.
- Daily RHR target: MAE under 5 bpm.
- Pay attention to parity across skin tones, lighting, motion, and face coverings.

## Self-evolving constraint

- Allow only bounded personalization on-device.
- Do not let the model rewrite its own safety policy.
- Freeze or roll back on any sign of drift.
- Keep exercise recommendations in a small, auditable action space.
