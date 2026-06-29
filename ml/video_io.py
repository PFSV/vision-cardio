from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VideoFrames:
    frames: np.ndarray  # (T, H, W, 3), uint8 RGB
    fps: float


def load_video_frames(
    video_path: str | Path,
    fps: int = 15,
    duration_s: int = 8,
    size: int = 64,
) -> VideoFrames:
    """Load a short RGB clip using ffmpeg without extra Python dependencies."""
    video_path = str(video_path)
    frame_count = fps * duration_s
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vf",
        f"fps={fps},scale={size}:{size}",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    expected = frame_count * size * size * 3
    if len(raw) < expected:
        raise ValueError(
            f"ffmpeg returned too few bytes: got {len(raw)}, expected {expected}"
        )
    raw = raw[:expected]
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, size, size, 3)
    return VideoFrames(frames=frames, fps=float(fps))


def center_crop(frames: np.ndarray, crop_fraction: float = 0.7) -> np.ndarray:
    """Keep the center region as a naive face ROI."""
    if not 0 < crop_fraction <= 1:
        raise ValueError("crop_fraction must be in (0, 1]")
    _, height, width, _ = frames.shape
    crop_h = max(1, int(height * crop_fraction))
    crop_w = max(1, int(width * crop_fraction))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return frames[:, y0 : y0 + crop_h, x0 : x0 + crop_w, :]


def rgb_trace(frames: np.ndarray, crop_fraction: float = 0.7) -> np.ndarray:
    """Return mean RGB trace over a central ROI."""
    roi = center_crop(frames, crop_fraction=crop_fraction).astype(np.float32)
    trace = roi.mean(axis=(1, 2))
    return trace

