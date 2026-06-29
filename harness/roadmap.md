# Roadmap

## Phase 1: Proof of direction

- Verify the paper-derived product assumptions.
- Lock the initial platform and stack: iOS-native with `SwiftUI`, `AVFoundation`, `Vision`, and `Core ML`.
- Define the first measurable offline evaluation target.

## Phase 2: Capture and inference prototype

- Build the front-camera capture flow.
- Add confidence gating.
- Produce one heart-rate estimate from one short clip.

## Phase 3: Daily aggregation

- Aggregate multiple estimates into a daily resting-heart-rate trend.
- Track uncertainty and missing-data behavior.
- Keep the user-facing interpretation conservative.

## Phase 4: Bounded personalization

- Add a local personalization layer or calibration state.
- Version the base model and the personalization state separately.
- Add warmup, freeze, and rollback behavior.

## Phase 5: Adaptive exercise policy

- Translate user patterns into easy, maintain, or push recommendations.
- Learn from completion, recovery, and resting-heart-rate trend stability.
- Keep recommendations in a small safe action space.

## Phase 6: App hardening

- Add permission handling, privacy text, and data deletion.
- Measure latency, battery, and failure modes.
- Prepare store-facing copy that avoids diagnostic claims.

## Phase 7: Validation

- Check performance across lighting, motion, and skin-tone groups.
- Compare against a ground-truth reference where available.
- Decide whether the product remains wellness-only or needs a clinical path.
