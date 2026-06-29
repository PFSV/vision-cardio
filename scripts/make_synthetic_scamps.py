"""Generate synthetic SCAMPS-format .mat clips for pipeline validation.

The real SCAMPS bundle and the Google rPPG assets are gated behind access
approval (see README "Reality check"), and this container ships no GPU and no
data. This generator produces small `.mat` files that match the schema
`ml/scamps.py` expects (`Xsub` frames, `d_ppg` waveform, `hr` label) so the
training and inference pipeline can be exercised end-to-end without internet.

Each clip embeds a real, recoverable heart-rate signal: the green channel of a
center ROI oscillates at the clip's target BPM, and `d_ppg` carries the same
waveform. A model that learns to read the RGB trace therefore has a genuine
signal to fit, not random noise. This is for pipeline validation only, not
product-grade training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import savemat


def make_clip(
    hr_bpm: float,
    fps: float = 30.0,
    duration_s: float = 5.0,
    size: int = 36,
    rng: np.random.Generator | None = None,
) -> dict:
    """Build one synthetic SCAMPS sample with a recoverable HR signal."""
    rng = rng if rng is not None else np.random.default_rng(0)
    n_frames = int(round(fps * duration_s))
    t = np.arange(n_frames) / fps
    hr_hz = hr_bpm / 60.0

    # Pulse waveform shared by the green channel and the ground-truth PPG.
    pulse = np.sin(2.0 * np.pi * hr_hz * t).astype(np.float32)

    frames = np.empty((n_frames, size, size, 3), dtype=np.uint8)
    # Per-frame channel means carry the signal; spatial texture is light noise
    # so the center-ROI mean trace (what the model reads) stays informative.
    green = 150.0 + 45.0 * pulse
    red = 130.0 + 8.0 * pulse
    blue = 110.0 + 4.0 * pulse
    for i in range(n_frames):
        spatial = rng.normal(0.0, 3.0, size=(size, size, 3)).astype(np.float32)
        spatial[..., 0] += red[i]
        spatial[..., 1] += green[i]
        spatial[..., 2] += blue[i]
        frames[i] = np.clip(spatial, 0, 255).astype(np.uint8)

    # Ground-truth PPG with mild measurement noise, same band as the pulse.
    d_ppg = pulse + rng.normal(0.0, 0.05, size=n_frames).astype(np.float32)

    return {
        "Xsub": frames,
        "d_ppg": d_ppg.astype(np.float32),
        "hr": np.float32(hr_bpm),
        "fps": np.float32(fps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic SCAMPS .mat clips.")
    parser.add_argument("--out", default="synthetic_data/scamps", help="Output directory")
    parser.add_argument("--count", type=int, default=24, help="Number of clips")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--size", type=int, default=36)
    parser.add_argument("--min-bpm", type=float, default=50.0)
    parser.add_argument("--max-bpm", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Spread target BPMs across the range so the bin classifier sees many labels.
    targets = np.linspace(args.min_bpm, args.max_bpm, args.count)
    for idx, hr in enumerate(targets):
        clip = make_clip(
            hr_bpm=float(hr),
            fps=args.fps,
            duration_s=args.duration_s,
            size=args.size,
            rng=rng,
        )
        path = out_dir / f"synth_{idx:03d}_hr{int(round(hr))}.mat"
        savemat(path, clip, do_compression=True)

    print(f"wrote {args.count} clips to {out_dir}")


if __name__ == "__main__":
    main()
