"""Extract SCAMPS clips from a tarball that is STILL downloading.

Pairs with an ``aria2c --stream-piece-selector=inorder --file-allocation=none``
download: aria2 fills the ``.tar.gz`` contiguously front-to-back at the full
per-IP rate (~4-6 MiB/s with 16 connections -- ~8x a single urllib stream), and
this process tails that growing file, decompressing the tar stream and writing
each clip's preprocessed ``(3,T,H,W)`` fp16 ``.npy`` into the persistent pool as
soon as its bytes land. So GPU training can start on real data while the rest of
the 200 GB+ keeps downloading.

``GrowingFile`` is a tail -f for a regular file: ``read(n)`` blocks until ``n``
bytes are available or the download is finished (aria2's ``<name>.aria2`` control
file disappears on completion). A single long-lived gzip/tar decode follows the
download frontier -- no re-decompression, no polling re-globs.

Usage (alongside the aria2 download of the same file):
    python -m ml.extract_growing \
        --tarball /path/to/vision_cardio_data/scamps/scamps_videos.tar.gz \
        --out-dir /path/to/vision_cardio_data/scamps_pool \
        --max-clips 1100
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path

import numpy as np

from .hr_bins import bpm_to_bin
from .scamps import load_scamps_mat
from .pool_train_scamps import prepare_clip_tensor


class GrowingFile:
    """File-like wrapper that follows a file still being appended to.

    aria2 ``inorder`` prefers the front-most piece, but with 16 connections in
    flight a stalled early connection can leave a HOLE far behind the byte-size
    frontier (``st_size`` is the highest written offset, not the contiguous
    prefix). Reading into a hole yields sparse zeros (-> "invalid compressed
    data"). So the frontier is the true contiguous prefix from offset 0, found
    with ``lseek(SEEK_HOLE)`` on the sparse file (``--file-allocation=none`` ->
    not-yet-written ranges are real holes), minus a small tail margin for the
    boundary piece still being written. ``lag`` is only a fallback for
    filesystems without ``SEEK_HOLE``."""

    def __init__(self, path: Path, done_fn, lag_bytes: int = 384 << 20,
                 poll: float = 2.0, startup_grace: float = 600.0,
                 tail_margin: int = 4 << 20) -> None:
        self.path = Path(path)
        self.done_fn = done_fn
        self.lag = lag_bytes
        self.tail_margin = tail_margin
        self.poll = poll
        self.startup_grace = startup_grace
        self._f = None
        self._probe_fd = None
        self._waited_for_start = 0.0

    def _ensure_open(self):
        while self._f is None:
            if self.path.exists():
                self._f = open(self.path, "rb")
                self._probe_fd = os.open(self.path, os.O_RDONLY)
                return
            if self._waited_for_start >= self.startup_grace:
                raise FileNotFoundError(f"{self.path} never appeared")
            time.sleep(self.poll)
            self._waited_for_start += self.poll

    def _frontier(self) -> int:
        """Highest byte offset safe to read up to right now."""
        try:
            sz = self.path.stat().st_size
        except OSError:
            return 0
        if self.done_fn():
            return sz                       # download complete: whole file is valid
        # True contiguous prefix from offset 0: lseek to the first hole. A
        # separate fd is used so the tar decode's read position is untouched.
        try:
            prefix = os.lseek(self._probe_fd, 0, os.SEEK_HOLE)
        except OSError:
            prefix = max(0, sz - self.lag)  # no SEEK_HOLE support: lag fallback
        prefix = min(prefix, sz)
        return max(0, prefix - self.tail_margin)  # stay behind the boundary piece

    def read(self, size: int = -1) -> bytes:
        self._ensure_open()
        unbounded = size is None or size < 0
        buf = bytearray()
        while unbounded or len(buf) < size:
            pos = self._f.tell()
            front = self._frontier()
            can = front - pos
            if can <= 0:
                if self.done_fn():
                    chunk = self._f.read(1 << 20 if unbounded else size - len(buf))
                    if not chunk:
                        break               # true EOF on a finished download
                    buf.extend(chunk)
                    if unbounded:
                        continue
                    continue
                time.sleep(self.poll)        # frontier hasn't advanced enough yet
                continue
            want = (1 << 20) if unbounded else (size - len(buf))
            chunk = self._f.read(min(want, can))
            if chunk:
                buf.extend(chunk)
            elif self.done_fn():
                break
            else:
                time.sleep(self.poll)
        return bytes(buf)

    def close(self):
        if self._f is not None:
            self._f.close()
        if self._probe_fd is not None:
            os.close(self._probe_fd)
            self._probe_fd = None


def _count_existing(out_dir: Path) -> int:
    return len(list(out_dir.glob("clip_*.npy")))


def _save_clip(tf, member, out_dir: Path, index: int,
               clip_frames: int, clip_size: int) -> bool:
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


def main() -> None:
    p = argparse.ArgumentParser(description="Extract clips from a still-downloading SCAMPS tarball.")
    p.add_argument("--tarball", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-size", type=int, default=112)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--lag-mb", type=int, default=384,
                   help="stay this many MiB behind the download frontier (avoids in-flight holes)")
    args = p.parse_args()

    tarball = Path(args.tarball)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl = Path(str(tarball) + ".aria2")

    def download_done() -> bool:
        # aria2 removes the .aria2 control file when the file is fully downloaded.
        return tarball.exists() and not ctrl.exists()

    already = _count_existing(out_dir)
    saved = already
    print(json.dumps({"event": "start", "tarball": str(tarball), "out_dir": str(out_dir),
                      "already_cached": already, "target": args.max_clips}), flush=True)

    gf = GrowingFile(tarball, download_done, lag_bytes=args.lag_mb << 20, poll=args.poll)
    tf = tarfile.open(fileobj=gf, mode="r|gz")
    seen = 0
    t0 = time.time()
    try:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".mat")):
                continue
            idx = seen
            seen += 1
            if idx < already:
                continue  # already cached by a previous pass / the seed
            if args.max_clips is not None and idx >= args.max_clips:
                break
            if _save_clip(tf, member, out_dir, idx, args.clip_frames, args.clip_size):
                saved = max(saved, idx + 1)
                if saved % args.log_every == 0:
                    rate = (saved - already) / max(1e-6, time.time() - t0)
                    print(json.dumps({"event": "progress", "cached": saved,
                                      "clips_per_s": round(rate, 4),
                                      "clips_per_hr": round(rate * 3600, 1)}), flush=True)
    except (tarfile.ReadError, EOFError, OSError) as exc:
        # truncated stream at frontier after download_done flipped, or transient
        print(json.dumps({"event": "stream_end", "cached": saved, "error": repr(exc)[:200]}), flush=True)
    finally:
        try:
            tf.close()
        except Exception:
            pass
        gf.close()
    print(json.dumps({"event": "done", "cached": saved, "members_seen": seen}), flush=True)


if __name__ == "__main__":
    main()
