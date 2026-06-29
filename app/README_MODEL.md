# Wiring the trained model into the iOS app

The Swift app is built so the trained model **drops straight in** — no code
changes needed once you have the `.mlpackage`.

## 1. Export the trained checkpoint to Core ML (on any machine with the repo)

```bash
python -m ml.export_coreml \
  --ckpt artifacts/scamps_hr_model_real.pt \
  --out  artifacts/VisionCardioHR.mlpackage
```

This reads the architecture (`video`/width), clip dims, and `min_bpm` from the
checkpoint and writes them into the model metadata, so the app stays in sync
automatically.

## 2. Add it to the Xcode target

Drag `VisionCardioHR.mlpackage` into the app target (check "Copy items if
needed" and the app target membership). Xcode compiles it to `.mlmodelc` in the
bundle. `HeartRateModel` looks it up by the name `VisionCardioHR`.

## 3. Build & run (needs Xcode on macOS)

The pipeline is already wired:

```
CameraManager (front cam frames)
   -> FrameClipBuffer   resize to HxW, RGB, 0..255, build [1,3,T,H,W]
   -> HeartRateModel    Core ML predict -> argmax bin + min_bpm = bpm
   -> CoachViewModel    -> CoachSignal -> CoachEngine.decide
   -> ContentView       live HR / confidence / recommendation
```

If the model isn't bundled yet, the app still builds and runs — it shows
"model not bundled" and skips inference.

## Contract (must match `ml/export_coreml.py`)

| | |
|---|---|
| input `clip` | MLMultiArray `[1, 3, T, H, W]`, float **0–255, RGB, not normalized** |
| output `logits` | MLMultiArray `[1, numBins]` |
| decode | `bpm = argmax(logits) + min_bpm` ; confidence = softmax of argmax bin |
| metadata | `clip_frames`, `clip_size`, `min_bpm`, `num_bins` read at load time |

## Honest caveats

- **Domain gap**: the model is trained on **synthetic SCAMPS** faces. Real-camera
  accuracy is unverified until tested on real subjects — treat on-device numbers
  as a plumbing check, not a validated vital sign.
- **Preprocessing parity is load-bearing**: training feeds raw 0–255 RGB with no
  normalization. `FrameClipBuffer` matches that. If you change training
  normalization, change the buffer too.
- Frame rate / window: the buffer collects `clip_frames` frames before the first
  prediction (~4 s at 30 fps), then predicts once per second.
