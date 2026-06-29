"""Companion pass: extract the per-frame PPG waveform from each SCAMPS .mat,
resample to clip_frames, z-normalize, save ppg_<idx>.npy (clip_frames,) float32.

Aligned by the SAME 0-based .mat member index as clip_<idx>_bin*.npy in the frame
pool, so (frames clip_<idx>, waveform ppg_<idx>) form a training pair. This is the
dense supervision target for the rPPG waveform model (negative-Pearson loss),
replacing the single HR bin that the failed classifier used.

ISA-L inflate over the streaming tar (one serial gzip stream = the only cost;
reading just d_ppg per .mat is instant). See notes/POSTMORTEM_rppg_3dconv.md.
"""
from __future__ import annotations
import argparse, io, json, time
from pathlib import Path

import h5py
import numpy as np
from isal import igzip
import tarfile


def resample_z(ppg: np.ndarray, n: int) -> np.ndarray:
    p = np.asarray(ppg, dtype=np.float32).squeeze()
    if p.ndim != 1:
        p = p.reshape(-1)
    x_old = np.linspace(0.0, 1.0, len(p))
    x_new = np.linspace(0.0, 1.0, n)
    w = np.interp(x_new, x_old, p).astype(np.float32)
    return (w - w.mean()) / (w.std() + 1e-8)      # zero-mean unit-std for neg-Pearson


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarball", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clip-frames", type=int, default=128)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = igzip.open(args.tarball, "rb")
    tf = tarfile.open(fileobj=raw, mode="r|")
    seen = saved = 0
    t0 = time.time()
    print(json.dumps({"event": "start", "out": str(out), "clip_frames": args.clip_frames}), flush=True)
    try:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".mat")):
                continue
            idx = seen; seen += 1
            if args.max_clips is not None and idx >= args.max_clips:
                break
            outp = out / f"ppg_{idx}.npy"
            if outp.exists():
                continue
            data = tf.extractfile(m).read()
            with h5py.File(io.BytesIO(data), "r") as f:
                key = next((k for k in ("d_ppg", "ppg", "GT_ppg") if k in f), None)
                if key is None:
                    continue
                ppg = np.array(f[key])
            wav = resample_z(ppg, args.clip_frames)
            tmp = out / f"_tmp_{idx}.npy"
            np.save(tmp, wav); tmp.rename(outp)
            saved += 1
            if saved % args.log_every == 0:
                rate = saved / max(1e-6, time.time() - t0) * 3600
                print(json.dumps({"event": "progress", "saved": saved, "last_idx": idx,
                                  "clips_per_hr": round(rate, 1)}), flush=True)
    except (tarfile.ReadError, EOFError, OSError) as e:
        print(json.dumps({"event": "stream_end", "saved": saved, "error": repr(e)[:200]}), flush=True)
    print(json.dumps({"event": "done", "saved": saved, "seen": seen}), flush=True)


if __name__ == "__main__":
    main()
