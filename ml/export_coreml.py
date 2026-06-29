"""Convert the trained PhysNet rPPG model to a Core ML .mlpackage for iOS.

Input: face clip (1, 3, T, H, W) in [0,1] (a `clip_seconds` window resampled to T
frames). Output: rPPG **waveform** (1, T). On device: HR = FFT peak of the
waveform in 0.7-3.0 Hz, with fs = T / clip_seconds. clip_frames / clip_size /
clip_seconds / fs are written into the model metadata so the Swift app stays in
sync automatically.

Usage:
    python -m ml.export_coreml --ckpt artifacts/rppg_physnet.pt \
        --out app/VisionCardioHR.mlpackage
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .physnet import PhysNet


def main() -> None:
    p = argparse.ArgumentParser(description="Export PhysNet rPPG model to Core ML.")
    p.add_argument("--ckpt", default="artifacts/rppg_physnet.pt")
    p.add_argument("--out", default="app/VisionCardioHR.mlpackage")
    p.add_argument("--clip-size", type=int, default=112)
    args = p.parse_args()

    import coremltools as ct

    ck = torch.load(args.ckpt, map_location="cpu")
    width = int(ck.get("width", 32))
    clip_frames = int(ck.get("clip_frames", 128))
    clip_size = args.clip_size
    clip_seconds = float(ck.get("clip_seconds", 20.0))

    model = PhysNet(width=width)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    example = torch.zeros(1, 3, clip_frames, clip_size, clip_size)
    traced = torch.jit.trace(model, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="clip", shape=example.shape)],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    md = mlmodel.user_defined_metadata
    md["clip_frames"] = str(clip_frames)
    md["clip_size"] = str(clip_size)
    md["clip_seconds"] = str(clip_seconds)
    md["fs"] = str(clip_frames / clip_seconds)
    md["hr_band_hz"] = "0.7,3.0"
    mlmodel.short_description = (
        "Vision Cardio rPPG. Input clip (1,3,%d,%d,%d) in [0,1]; output PPG waveform "
        "(1,%d). HR = FFT peak (0.7-3.0 Hz), fs = %.3f." % (
            clip_frames, clip_size, clip_size, clip_frames, clip_frames / clip_seconds)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"saved {out}  clip={clip_frames}x{clip_size}x{clip_size} fs={clip_frames/clip_seconds:.2f}")


if __name__ == "__main__":
    main()
