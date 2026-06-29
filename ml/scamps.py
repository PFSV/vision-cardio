from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

try:
    import h5py
except ImportError:  # optional dependency; used for MATLAB v7.3 SCAMPS files
    h5py = None


@dataclass(frozen=True)
class ScampsSample:
    path: Path
    frames: np.ndarray  # (T, H, W, C) or (C, H, W, T)
    ppg: np.ndarray
    heart_rate_bpm: float


def _to_t_h_w_c(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D frame array, got shape {arr.shape}")
    # The repo documents either [t, w, h, c] or [c, w, h, t]. Normalize to [t, h, w, c].
    if arr.shape[-1] in (3, 4):
        # Common case already close to time-first.
        if arr.shape[1] >= 32 and arr.shape[2] >= 32:
            return arr
        if arr.shape[0] >= 32 and arr.shape[1] >= 32:
            return np.transpose(arr, (3, 1, 2, 0))
    if arr.shape[0] in (3, 4):
        return np.transpose(arr, (3, 2, 1, 0))
    if arr.shape[0] > arr.shape[-1]:
        return np.transpose(arr, (3, 2, 1, 0))
    return arr


def _hr_from_ppg(ppg: np.ndarray, fps: float = 30.0) -> float:
    signal = np.asarray(ppg).astype(np.float32).squeeze()
    if signal.ndim != 1:
        signal = signal.reshape(-1)
    signal = signal - np.mean(signal)
    freqs, power = welch(signal, fs=fps, nperseg=min(len(signal), 1024))
    band = (freqs >= 0.7) & (freqs <= 4.0)
    if not np.any(band):
        raise ValueError("No physiological band found in PPG signal")
    freqs = freqs[band]
    power = power[band]
    peak_hz = float(freqs[int(np.argmax(power))])
    return peak_hz * 60.0


def _load_h5py_mat(mat_path: Path) -> dict[str, np.ndarray]:
    if h5py is None:
        raise NotImplementedError("h5py is required to read MATLAB v7.3 files")

    payload: dict[str, np.ndarray] = {}
    with h5py.File(mat_path, "r") as f:
        for key in f.keys():
            payload[key] = np.array(f[key])
    return payload


def load_scamps_mat(mat_path: str | Path) -> ScampsSample:
    mat_path = Path(mat_path)
    try:
        payload = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        payload = _load_h5py_mat(mat_path)

    frame_key = "Xsub" if "Xsub" in payload else "RawFrames"
    if frame_key not in payload:
        raise KeyError(f"Could not find RawFrames or Xsub in {mat_path}")
    frames = _to_t_h_w_c(payload[frame_key])
    ppg_key = next((key for key in ("d_ppg", "ppg", "GT_ppg") if key in payload), None)
    if ppg_key is None:
        raise KeyError(f"Could not find d_ppg/ppg/GT_ppg in {mat_path}")
    ppg = np.asarray(payload[ppg_key])

    hr = None
    for key in ("hr", "heart_rate", "heart_rate_bpm", "HR", "gt_hr"):
        if key in payload:
            value = np.asarray(payload[key]).squeeze()
            if np.ndim(value) == 0:
                hr = float(value)
                break
    if hr is None:
        hr = _hr_from_ppg(ppg)

    return ScampsSample(path=mat_path, frames=frames, ppg=ppg, heart_rate_bpm=hr)


def iter_scamps_files(root: str | Path):
    root = Path(root)
    for path in sorted(root.rglob("*.mat")):
        yield path


def sample_summary(sample: ScampsSample) -> dict:
    return {
        "path": str(sample.path),
        "frames_shape": list(sample.frames.shape),
        "ppg_shape": list(np.asarray(sample.ppg).shape),
        "heart_rate_bpm": float(sample.heart_rate_bpm),
    }
