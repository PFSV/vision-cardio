"""Streaming, parallel SCAMPS clip-pool extractor (~4x faster than serial).

This is a faster sibling of ``ml/extract_growing.py`` for the case where the
tarball is ALREADY fully downloaded on disk. It produces byte-identical clips to
that path -- same ``Xsub -> _to_t_h_w_c -> prepare_clip_tensor`` frame math, same
``clip_<idx>_bin<b>.npy`` naming where ``idx`` is the 0-based order of ``.mat``
members in the tar -- but wins on wall-clock two ways:

1. Decompressor. cepheid reads at ~61 MB/s but stock zlib only inflates at
   ~32 MB/s (CPU-bound), so the decode, not the disk, is the bottleneck. ISA-L's
   ``igzip`` inflates at ~59 MB/s, roughly matching the disk. We therefore wrap
   the file in ``isal.igzip.open(...)`` and hand tarfile an ALREADY-decompressed
   byte stream via ``mode='r|'`` (uncompressed streaming tar), NOT ``'r|gz'``.

2. Per-.mat load. ``ml.scamps._load_h5py_mat`` materializes ALL 21 HDF5 datasets
   (RawFrames 1.1GB + Xsub 829MB + masks 550MB + signals + AUs), but only Xsub +
   d_ppg are ever used -- ~14s/clip of wasted reads. Workers here read ONLY the
   frame dataset and the ppg, then run the exact repo transforms. With Xsub
   present, RawFrames/masks/AUs are never touched.

The main process owns the single sequential inflate+tar walk (cheap, IO-bound)
and the only thing each worker gets is the raw member bytes; the expensive HDF5
parse + trilinear resize fan out across ``--workers`` processes. A bounded
in-flight count keeps RAM in check (each decompressed Xsub is ~829MB).

Usage (tarball already downloaded):
    python -m ml.build_pool_fast \
        --tarball /mnt/.../scamps/scamps_videos.tar.gz \
        --out-dir /mnt/.../scamps_pool \
        --workers 16
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import tarfile
import time
from pathlib import Path

import numpy as np
import h5py

from isal import igzip

from .hr_bins import bpm_to_bin
from .pool_train_scamps import prepare_clip_tensor
from .scamps import _hr_from_ppg, _to_t_h_w_c


# ---------------------------------------------------------------------------
# Worker: parse one .mat from memory and write its clip .npy.
# ---------------------------------------------------------------------------

def _worker(idx: int, data: bytes, out_dir: str, clip_frames: int, clip_size: int):
    """Parse a single .mat (held in ``data``) and atomically save its clip.

    Reads ONLY the frame dataset (Xsub preferred, RawFrames fallback) and the
    ppg dataset -- never RawFrames/masks/AUs when Xsub exists -- then runs the
    SAME math as ``load_scamps_mat`` + ``extract_growing._save_clip`` so the
    output tensor is byte-identical to the serial path. Returns (idx, bin, err).
    """
    out = Path(out_dir)
    try:
        with h5py.File(io.BytesIO(data), "r") as f:
            frame_key = "Xsub" if "Xsub" in f else "RawFrames"
            if frame_key not in f:
                raise KeyError("Could not find Xsub or RawFrames")
            # Read only the two datasets we actually use.
            raw_frames = np.array(f[frame_key])
            ppg_key = next((k for k in ("d_ppg", "ppg", "GT_ppg") if k in f), None)
            if ppg_key is None:
                raise KeyError("Could not find d_ppg/ppg/GT_ppg")
            ppg = np.array(f[ppg_key])

        # Identical normalization to load_scamps_mat: SCAMPS v7.3 files carry no
        # scalar hr key, so that path also falls back to _hr_from_ppg(ppg).
        frames = _to_t_h_w_c(raw_frames)
        hr = _hr_from_ppg(ppg)
        tensor = prepare_clip_tensor(frames, clip_frames, clip_size)
        b = bpm_to_bin(hr)

        tmp_npy = out / f"_tmp_{idx}.npy"
        np.save(tmp_npy, tensor.numpy())
        tmp_npy.rename(out / f"clip_{idx}_bin{b}.npy")
        return idx, int(b), None
    except Exception as exc:  # pragma: no cover - reported, not fatal
        return idx, None, repr(exc)[:300]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _existing_idxs(out_dir: Path) -> set[int]:
    """Set of clip indices already present in the pool (any bin)."""
    done: set[int] = set()
    for p in out_dir.glob("clip_*_bin*.npy"):
        try:
            done.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return done


def _open_tar(tarball: Path):
    """ISA-L decompress -> uncompressed streaming tar (the decode speed win)."""
    raw = igzip.open(str(tarball), "rb")
    tf = tarfile.open(fileobj=raw, mode="r|")  # 'r|' over already-inflated bytes
    return raw, tf


# ---------------------------------------------------------------------------
# Main: sequential inflate + tar walk in the parent, parsing fanned out.
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Fast parallel SCAMPS clip-pool extractor (ISA-L + multiprocessing).")
    p.add_argument("--tarball", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-clips", type=int, default=None,
                   help="stop after this many .mat members are seen (cheap testing)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-size", type=int, default=112)
    p.add_argument("--log-every", type=int, default=5)
    args = p.parse_args()

    tarball = Path(args.tarball)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_idxs(out_dir)
    print(json.dumps({"event": "start", "tarball": str(tarball), "out_dir": str(out_dir),
                      "already_cached": len(existing), "workers": args.workers,
                      "max_clips": args.max_clips}), flush=True)

    # Bounded in-flight: block the inflate loop when ~2*workers tasks pend, so RAM
    # (each decompressed Xsub ~829MB) stays bounded regardless of worker speed.
    in_flight = mp.Semaphore(2 * args.workers)
    saved = 0          # clips written this run
    submitted = 0      # tasks submitted this run
    t0 = time.time()
    errors = 0

    def _on_done(result):
        nonlocal saved, errors
        in_flight.release()
        idx, b, err = result
        if err is not None:
            errors += 1
            print(json.dumps({"event": "clip_error", "idx": idx, "error": err}), flush=True)
            return
        saved += 1
        if saved % args.log_every == 0:
            elapsed = max(1e-6, time.time() - t0)
            rate = saved / elapsed
            print(json.dumps({"event": "progress", "saved_this_run": saved,
                              "last_idx": idx, "clips_per_s": round(rate, 4),
                              "clips_per_hr": round(rate * 3600, 1)}), flush=True)

    raw, tf = _open_tar(tarball)
    pool = mp.Pool(processes=args.workers)
    seen = 0  # 0-based .mat member counter == clip idx (matches extract_growing)
    try:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".mat")):
                continue
            idx = seen
            seen += 1
            if args.max_clips is not None and idx >= args.max_clips:
                break
            if idx in existing:
                # Already cached: advance the stream cheaply, don't parse.
                fobj = tf.extractfile(member)
                if fobj is not None:
                    fobj.read()  # consume so the tar stream stays aligned
                continue
            # Read member bytes in the MAIN process, then fan out the parse.
            fobj = tf.extractfile(member)
            if fobj is None:
                continue
            data = fobj.read()
            in_flight.acquire()  # block here when too many parses are pending
            submitted += 1
            pool.apply_async(
                _worker,
                (idx, data, str(out_dir), args.clip_frames, args.clip_size),
                callback=_on_done,
                error_callback=lambda e: (_on_done((-1, None, repr(e)[:300]))),
            )
            del data  # drop the parent's reference promptly
    except (tarfile.ReadError, EOFError, OSError) as exc:
        print(json.dumps({"event": "stream_end", "members_seen": seen,
                          "error": repr(exc)[:200]}), flush=True)
    finally:
        # Drain: let every submitted parse finish before tearing down.
        pool.close()
        pool.join()
        try:
            tf.close()
        except Exception:
            pass
        try:
            raw.close()
        except Exception:
            pass

    elapsed = max(1e-6, time.time() - t0)
    print(json.dumps({"event": "done", "members_seen": seen, "submitted": submitted,
                      "saved_this_run": saved, "errors": errors,
                      "total_cached": len(_existing_idxs(out_dir)),
                      "wall_s": round(elapsed, 1),
                      "clips_per_hr": round(saved / elapsed * 3600, 1)}), flush=True)


if __name__ == "__main__":
    main()
