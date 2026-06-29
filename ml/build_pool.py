"""Build a PERSISTENT, preprocessed SCAMPS clip pool on shared storage.

This decouples the (slow, network-bound) download from the (fast, compute-bound)
GPU training. It streams the single SCAMPS tarball sequentially, preprocesses
each clip ONCE into the exact model-input tensor (3, T, H, W) saved as a small
float16 ``.npy``, and writes it ADD-ONLY into a persistent directory (intended to
live on shared cluster storage, e.g. cepheid -- NOT node-local scratch that a job
wipes on exit). Training jobs then read this growing pool read-only.

Each file is ``clip_<index>_bin<b>.npy`` where ``b = bpm_to_bin(hr)``, so the
label travels with the file and no ``.mat`` is kept.

Why run this OFF the GPU nodes: the Azure blob throttles to ~6 MiB/s per IP, so
the bottleneck is the network, not compute. Burning an 8xH100 node to run what is
effectively ``curl`` wastes the GPUs. Run this on a login/dev box; point training
at the same ``--out-dir``.

Resumability: the tarball is a single gzip stream and gzip is not seekable, so a
restart re-opens the stream from byte 0 and *skips* members whose target ``.npy``
already exists (idempotent by index). That re-reads -- but does not re-save -- the
already-cached prefix; it is the only correct way to resume a non-seekable stream.
Network drops are caught and the stream is reopened with the same skip logic.

Usage:
    python -m ml.build_pool --out-dir data/scamps_pool \
        --max-clips 2000 --clip-frames 128 --clip-size 112
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .hr_bins import bpm_to_bin
from .scamps import load_scamps_mat
from .stream_train_scamps import DEFAULT_URL, _drain_members, _open_stream
from .pool_train_scamps import prepare_clip_tensor


def _count_existing(out_dir: Path) -> int:
    """Highest contiguous index already cached. We name files by stream index,
    so the number of existing clip_*.npy is exactly how many members to skip."""
    return len(list(out_dir.glob("clip_*.npy")))


def _cache_member(tf, member, out_dir: Path, index: int,
                  clip_frames: int, clip_size: int) -> bool:
    """Extract one .mat from the stream, preprocess, save .npy atomically.
    Returns True on success. The .mat bytes are written to a temp file (scipy
    needs a real path), loaded, turned into the (3,T,H,W) fp16 tensor, then the
    temp .mat is removed -- only the small .npy persists."""
    tmp_mat = out_dir / f"_tmp_{index}.mat"
    fobj = tf.extractfile(member)
    if fobj is None:
        return False
    try:
        with open(tmp_mat, "wb") as h:
            while True:
                chunk = fobj.read(1 << 20)
                if not chunk:
                    break
                h.write(chunk)
        sample = load_scamps_mat(tmp_mat)
        tensor = prepare_clip_tensor(sample.frames, clip_frames, clip_size)
        b = bpm_to_bin(sample.heart_rate_bpm)
    finally:
        tmp_mat.unlink(missing_ok=True)
    tmp_npy = out_dir / f"_tmp_{index}.npy"
    np.save(tmp_npy, tensor.numpy())
    tmp_npy.rename(out_dir / f"clip_{index}_bin{b}.npy")
    return True


def build(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = _count_existing(out_dir)
    target = args.max_clips
    print(json.dumps({"event": "start", "out_dir": str(out_dir),
                      "already_cached": saved, "target": target,
                      "clip": [args.clip_frames, args.clip_size]}), flush=True)
    if target is not None and saved >= target:
        print(json.dumps({"event": "done", "reason": "target_already_met",
                          "cached": saved}), flush=True)
        return

    attempt = 0
    while True:
        attempt += 1
        t0 = time.time()
        seen = 0          # members walked this pass (== stream index)
        new_this_pass = 0
        try:
            tf = _open_stream(args.url)
            for member in _drain_members(tf):
                idx = seen
                seen += 1
                if idx < saved:
                    continue  # already cached in a prior pass; skip-advance
                if target is not None and idx >= target:
                    break
                ok = _cache_member(tf, member, out_dir, idx,
                                   args.clip_frames, args.clip_size)
                if ok:
                    saved = max(saved, idx + 1)
                    new_this_pass += 1
                    if saved % args.log_every == 0:
                        rate = new_this_pass / max(1e-6, time.time() - t0)
                        print(json.dumps({"event": "progress", "cached": saved,
                                          "new_this_pass": new_this_pass,
                                          "clips_per_s": round(rate, 3)}), flush=True)
            try:
                tf.close()
            except Exception:
                pass
            # Reached end of stream (or target) cleanly.
            print(json.dumps({"event": "done", "cached": saved,
                              "members_seen": seen}), flush=True)
            return
        except Exception as exc:  # network drop, truncated gzip, etc.
            wait = min(args.max_backoff, args.backoff * attempt)
            print(json.dumps({"event": "reconnect", "attempt": attempt,
                              "cached": saved, "error": repr(exc)[:200],
                              "sleep_s": wait}), flush=True)
            if target is not None and saved >= target:
                print(json.dumps({"event": "done", "reason": "target_met_after_error",
                                  "cached": saved}), flush=True)
                return
            time.sleep(wait)


def main() -> None:
    p = argparse.ArgumentParser(description="Build a persistent preprocessed SCAMPS clip pool.")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--out-dir", required=True,
                   help="persistent dir (put on shared storage, NOT node-local scratch)")
    p.add_argument("--max-clips", type=int, default=None,
                   help="stop after caching this many clips (None = whole dataset ~2800)")
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-size", type=int, default=112)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--backoff", type=float, default=10.0, help="reconnect backoff base seconds")
    p.add_argument("--max-backoff", type=float, default=120.0)
    args = p.parse_args()
    try:
        build(args)
    except KeyboardInterrupt:
        print(json.dumps({"event": "interrupted"}), flush=True)
        sys.exit(130)


if __name__ == "__main__":
    main()
