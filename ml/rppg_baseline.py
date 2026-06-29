from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.signal import butter, detrend, filtfilt, welch

from .video_io import load_video_frames, rgb_trace


@dataclass(frozen=True)
class HRPrediction:
    bpm: float
    confidence: float
    method: str
    peak_strength: float
    band_energy: float


def _bandpass(signal: np.ndarray, fps: float, low_hz: float = 0.7, high_hz: float = 4.0) -> np.ndarray:
    nyquist = 0.5 * fps
    b, a = butter(3, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return filtfilt(b, a, signal)


def _spectral_hr(signal: np.ndarray, fps: float) -> tuple[float, float, float]:
    freqs, power = welch(signal, fs=fps, nperseg=min(len(signal), 256))
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        raise ValueError("No valid physiological band in spectrum")
    freqs = freqs[band]
    power = power[band]
    peak_idx = int(np.argmax(power))
    peak_hz = float(freqs[peak_idx])
    peak_power = float(power[peak_idx])
    band_energy = float(power.sum())
    median_power = float(np.median(power) + 1e-8)
    confidence = float(np.clip((peak_power / median_power - 1.0) / 10.0, 0.0, 1.0))
    return peak_hz * 60.0, confidence, peak_power / (band_energy + 1e-8)


def chrom(trace: np.ndarray, fps: float) -> tuple[np.ndarray, float]:
    """CHROM-style pulse trace from mean RGB over time."""
    rgb = trace.astype(np.float32)
    rgb = rgb / (rgb.mean(axis=0, keepdims=True) + 1e-8)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b
    alpha = np.std(x) / (np.std(y) + 1e-8)
    s = x - alpha * y
    s = detrend(s)
    s = _bandpass(s, fps)
    return s, alpha


def pos(trace: np.ndarray, fps: float) -> tuple[np.ndarray, float]:
    """POS-style pulse trace from mean RGB over time."""
    rgb = trace.astype(np.float32)
    rgb = rgb / (rgb.mean(axis=0, keepdims=True) + 1e-8)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    x = g - b
    y = -2.0 * r + g + b
    alpha = np.std(x) / (np.std(y) + 1e-8)
    s = x + alpha * y
    s = detrend(s)
    s = _bandpass(s, fps)
    return s, alpha


def estimate_hr_from_frames(
    frames: np.ndarray,
    fps: float,
    method: Literal["auto", "chrom", "pos"] = "auto",
) -> HRPrediction:
    """Estimate HR from an RGB face clip using a naive offline rPPG baseline."""
    trace = rgb_trace(frames)
    candidates: list[tuple[str, np.ndarray]] = []
    if method in ("auto", "chrom"):
        candidates.append(("chrom", chrom(trace, fps)[0]))
    if method in ("auto", "pos"):
        candidates.append(("pos", pos(trace, fps)[0]))
    if not candidates:
        raise ValueError(f"Unsupported method: {method}")

    best: HRPrediction | None = None
    for name, pulse in candidates:
        bpm, confidence, peak_strength = _spectral_hr(pulse, fps)
        # Keep HR within plausible resting/active range.
        if 40.0 <= bpm <= 180.0:
            band_energy = float(np.mean(np.square(pulse)))
            pred = HRPrediction(
                bpm=float(bpm),
                confidence=float(confidence),
                method=name,
                peak_strength=float(peak_strength),
                band_energy=band_energy,
            )
            if best is None or pred.confidence > best.confidence:
                best = pred
    if best is None:
        raise ValueError("No plausible HR estimate produced by baseline")
    return best


def estimate_hr_from_video(video_path: str, method: Literal["auto", "chrom", "pos"] = "auto") -> HRPrediction:
    video = load_video_frames(video_path)
    return estimate_hr_from_frames(video.frames, video.fps, method=method)

