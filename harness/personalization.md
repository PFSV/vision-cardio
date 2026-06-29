# Personalization

## Goal

Allow the app to improve for an individual user using only on-device data, without letting the core system drift unpredictably.

## Model shape

- Keep one fixed base model for HR estimation.
- Add a small user-specific adaptation layer or calibration state.
- Version the base model and the personalization state separately.

## What may adapt

- Confidence thresholding.
- User-specific signal-quality filters.
- Baseline heart-rate trend smoothing.
- Per-user calibration offsets or low-rank adapters, if they can be safely bounded.

## What must stay fixed

- Safety policy.
- Medical interpretation language.
- Privacy policy.
- Rollback logic.
- The core output schema.

## Update rules

- Update only after enough validated user data accumulates.
- Never update from low-confidence or heavily corrupted clips.
- Gate updates on simple quality checks and a held-out recent window.
- Keep a rollback path to the previous stable state.

## Data policy

- Process user data on device by default.
- Do not retain raw video unless the user explicitly opts in.
- Prefer derived features, embeddings, and summary statistics over raw media.
- Make the learning toggle visible and reversible.

## Failure handling

- If personalization degrades performance, freeze updates and revert.
- If the device lacks enough data, keep the base model behavior.
- If the user changes phone behavior materially, re-enter a warmup period.

## First implementation

1. Freeze the base HR model.
2. Add a tiny calibration state for quality and baseline smoothing.
3. Persist the state locally with versioning.
4. Add a rollback command in the app.
5. Evaluate whether per-user adaptation improves daily RHR stability.

