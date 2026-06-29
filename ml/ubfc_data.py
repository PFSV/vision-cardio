"""UBFC-rPPG loader -> windows matching the SCAMPS training representation.

Real-face benchmark for honest cross-dataset eval of the PhysNet rPPG model.

Layout (after cleanup, under data/ubfc-rppg/rppg-data/):
- DATASET_1/<k>-gt/vid.avi + gtdump.xmp
    gtdump.xmp = CSV cols: 0 time(ms), 1 HR(bpm), 2 SpO2, 3 PPG
- DATASET_2/subject<k>/vid.avi + ground_truth.txt
    rows: 0 PPG, 1 HR(bpm), 2 time(s)

CRITICAL: the model was trained on SCAMPS clips = 20 s of video resampled to 128
frames (effective fs = 6.4 Hz). So UBFC is windowed by `clip_seconds` (=20 s) and
each window's frames+PPG are resampled to `clip_frames` (=128) — NOT 128
consecutive frames, which would be the wrong time scale. Video decoded with
ffmpeg (cv2 absent), scaled to clip_size.
"""
from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np


def parse_gt(gt_path: str | Path):
    """-> (ppg, hr, t_seconds), one row each, native sampling rate."""
    p = Path(gt_path)
    if p.name == "ground_truth.txt":                       # DATASET_2
        r = p.read_text().strip().split("\n")
        ppg = np.array(r[0].split(), dtype=np.float32)
        hr = np.array(r[1].split(), dtype=np.float32)
        t = np.array(r[2].split(), dtype=np.float32)
    else:                                                  # DATASET_1 gtdump.xmp
        a = np.array([ln.split(",") for ln in p.read_text().strip().splitlines()],
                     dtype=np.float32)
        t = a[:, 0] / 1000.0
        hr = a[:, 1]
        ppg = a[:, 3]
    return ppg, hr, t


def video_fps(path: str | Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", str(path)],
        capture_output=True, text=True).stdout.strip()
    if "/" in out:
        a, b = out.split("/"); return float(a) / float(b)
    return float(out or 30.0)


def decode_video(path: str | Path, size: int, crop: str | None = None):
    """-> (frames (T,size,size,3) uint8, fps). crop = ffmpeg 'w:h:x:y' face ROI
    applied before scale (drops background so the faint pulse isn't diluted)."""
    fps = video_fps(path)
    vf = (f"crop={crop}," if crop else "") + f"scale={size}:{size}"
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-vf", vf, "-"]
    buf = np.frombuffer(subprocess.run(cmd, capture_output=True).stdout, dtype=np.uint8)
    fr = size * size * 3
    n = buf.size // fr
    return buf[: n * fr].reshape(n, size, size, 3), fps


def _resample_z(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 2:
        return np.zeros(n, dtype=np.float32)
    w = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x).astype(np.float32)
    return (w - w.mean()) / (w.std() + 1e-8)


def subject_windows(avi, gt, clip_seconds=20.0, clip_frames=128, clip_size=112, crop=None):
    """Yield (x (3,clip_frames,clip_size,clip_size) float32 [0,1],
              ppg (clip_frames,) z-normed,
              sensor_hr float) per non-overlapping clip_seconds window."""
    ppg, hr, t = parse_gt(gt)
    frames, fps = decode_video(avi, clip_size, crop=crop)
    T = len(frames)
    vid_t = np.arange(T) / fps
    n_win = int((T / fps) // clip_seconds)
    for w in range(n_win):
        t0, t1 = w * clip_seconds, (w + 1) * clip_seconds
        fmask = (vid_t >= t0) & (vid_t < t1)
        fr = frames[fmask].astype(np.float32) / 255.0          # (m,H,W,3)
        if len(fr) < clip_frames // 2:
            continue
        idx = np.linspace(0, len(fr) - 1, clip_frames).astype(int)
        x = np.transpose(fr[idx], (3, 0, 1, 2)).astype(np.float32)   # (3,cf,H,W)
        pmask = (t >= t0) & (t < t1)
        y = _resample_z(ppg[pmask], clip_frames)
        shr = float(np.median(hr[pmask])) if pmask.any() else float("nan")
        yield x, y, shr


def discover(root: str | Path):
    """List (avi, gt, name) across DATASET_1 (gtdump.xmp) + DATASET_2 (ground_truth.txt)."""
    root = Path(root)
    pairs = []
    for gt in list(root.rglob("ground_truth.txt")) + list(root.rglob("gtdump.xmp")):
        avi = gt.with_name("vid.avi")
        if avi.exists():
            pairs.append((str(avi), str(gt), gt.parent.name))
    return sorted(pairs, key=lambda x: x[2])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    for avi, gt, name in discover(args.root):
        print(f"{name:18s} fps={video_fps(avi):.2f}  {Path(gt).name}")
    print("total:", len(discover(args.root)))
