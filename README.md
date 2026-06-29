# Vision Cardio

On-device, offline **heart-rate estimation from the front camera** (rPPG) with an
exercise-coaching layer — a SwiftUI iOS app plus the training/eval pipeline behind it.

> Wellness/research only. Not a medical device; no diagnostic claims.

## What it does

- **Camera → heart rate**: a PhysNet-style 3D-CNN reads the faint pulse signal (rPPG)
  from a short front-camera clip and outputs a waveform; HR is the FFT peak (0.7–3.0 Hz).
- **Exercise coaching** (after a baseline-HR calibration):
  - **Running** — live HR zones / phases (1–5).
  - **Weights** — set/rest detection from HR trend, with recovery-vs-baseline timing
    ("rest / keep going / next set").
- Runs fully **offline, on device** (Core ML, GPU/CPU). Korean UI.

## Results (PhysNet, real-face UBFC-rPPG)

Trained on synthetic **SCAMPS**, then fine-tuned on **UBFC-rPPG** with a strict
by-participant split (no subject leakage):

| stage | HR MAE vs contact-PPG |
|---|---|
| zero-shot (SCAMPS only) on UBFC | 5.63 bpm |
| **after UBFC fine-tune** (val) | **2.80 bpm** |

Trained weights + Core ML model: **[hyunseop/vision-cardio-rppg](https://huggingface.co/hyunseop/vision-cardio-rppg)** (Hugging Face).

## Layout

```
app/        SwiftUI iOS app (camera → clip → Core ML → HR → coaching UI)
ml/         PhysNet model, UBFC/SCAMPS loaders, train / eval / fine-tune, Core ML export
scripts/    SLURM launchers + demos (set DATA paths via env / --flags)
harness/    product & design notes (roadmap, policy, evaluation)
paper.md    reference write-up
```

## Pipeline (high level)

```bash
# 1. train base rPPG model (point --pool-dir/--ppg-dir at your own extracted data)
torchrun --standalone --nproc_per_node=8 -m ml.train_rppg --pool-dir <DATA>/scamps_pool --ppg-dir <DATA>/scamps_pool_ppg

# 2. fine-tune on UBFC-rPPG (by-participant split)
python -m ml.finetune_rppg_ubfc --init-ckpt artifacts/rppg_physnet.pt --ubfc-root <DATA>/ubfc-rppg/rppg-data

# 3. honest cross-dataset eval
python -m ml.eval_rppg --ckpt artifacts/rppg_physnet_ubfc.pt --ubfc-root <DATA>/ubfc-rppg/rppg-data

# 4. export to Core ML for the app
python -m ml.export_coreml --ckpt artifacts/rppg_physnet_ubfc.pt --out app/VisionCardioHR.mlpackage
```

iOS app: generate the Xcode project with [XcodeGen](https://github.com/yonaskolb/XcodeGen)
(`xcodegen generate`), then build `VisionCardio` (iOS 16+). Drop the Core ML model from
Hugging Face into `app/VisionCardioHR.mlpackage`.

## Contract (app ↔ model)

```
input  "clip"     : (1, 3, 128, 112, 112) float, RGB, [0,1]   (~20 s window resampled to 128 frames)
output "waveform" : (1, 128)  rPPG pulse  ->  HR = FFT peak in 0.7-3.0 Hz, fs = 6.4 Hz
```

## Caveats

- rPPG degrades under motion / low light — best with the face well-lit, framed, and relatively still.
- Datasets (SCAMPS, UBFC-rPPG) are access-gated by their owners; bring your own and point the
  `--pool-dir` / `--ubfc-root` flags at them. No data or model weights are committed here.

## Author

Created and developed by **hyeonseop yoon** (PFSV) — model training, Core ML pipeline, and SwiftUI app.

## License

MIT © 2026 hyeonseop yoon.
