"""Stitch strided range-download chunks back into the streaming tarball, IN ORDER.

Pairs with `ml.pdl_worker` (many nodes writing chunk_<idx>.done out of order) and
keeps the existing `ml.extract_growing` streaming pipeline running unchanged: it
appends chunk 0, then 1, then 2 ... onto the tarball's contiguous tail and
deletes each consumed chunk. The extractor tails that growing tarball exactly as
it did under aria2, so download -> extract -> train never stops.

It keeps a sentinel control file (`<tarball>.aria2`) present until every chunk is
stitched, so the extractor's ``download_done()`` stays False and it keeps reading
a safe margin behind the live append tail (never off the end mid-append).

Usage:
    python -m ml.pdl_stitch --tarball <tar.gz> --chunks-dir <chunks> \
        --base-offset <bytes already on disk> --total-bytes <N> --chunk-mb 256
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Ordered chunk stitcher for streaming download.")
    p.add_argument("--tarball", required=True)
    p.add_argument("--chunks-dir", required=True)
    p.add_argument("--base-offset", type=int, required=True,
                   help="bytes already in the tarball (chunk 0 starts here)")
    p.add_argument("--total-bytes", type=int, required=True)
    p.add_argument("--chunk-mb", type=int, default=256)
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    tarball = Path(args.tarball)
    chunks = Path(args.chunks_dir)
    chunk = args.chunk_mb << 20
    n_chunks = (args.total_bytes - args.base_offset + chunk - 1) // chunk
    ctrl = Path(str(tarball) + ".aria2")
    ctrl.write_text("stitching\n")  # keep extractor in streaming mode until done

    # Guard: tarball must already be exactly base-offset bytes (contiguous prefix).
    cur = tarball.stat().st_size if tarball.exists() else 0
    if cur != args.base_offset:
        print(json.dumps({"event": "warn", "msg": "tarball size != base-offset",
                          "tarball_bytes": cur, "base_offset": args.base_offset}), flush=True)

    print(json.dumps({"event": "stitch_start", "n_chunks": n_chunks,
                      "base_gb": round(args.base_offset / 1e9, 2),
                      "total_gb": round(args.total_bytes / 1e9, 2)}), flush=True)

    nxt = 0
    t0 = time.time()
    appended = 0
    with open(tarball, "ab") as out:
        while nxt < n_chunks:
            cpath = chunks / f"chunk_{nxt:07d}.done"
            if not cpath.exists():
                time.sleep(args.poll)         # wait for this chunk (downloaded out of order)
                continue
            with open(cpath, "rb") as f:
                while True:
                    b = f.read(1 << 20)
                    if not b:
                        break
                    out.write(b)
            out.flush()
            os.fsync(out.fileno())
            cpath.unlink(missing_ok=True)      # reclaim space; tarball is the source of truth
            nxt += 1
            appended += 1
            if appended % args.log_every == 0:
                size = tarball.stat().st_size
                rate = (size - args.base_offset) / max(1e-6, time.time() - t0) / 1e6
                print(json.dumps({"event": "progress", "stitched": nxt, "of": n_chunks,
                                  "tarball_gb": round(size / 1e9, 2),
                                  "MBps": round(rate, 2)}), flush=True)

    ctrl.unlink(missing_ok=True)               # download complete -> extractor drains to EOF
    print(json.dumps({"event": "stitch_done", "chunks": nxt,
                      "tarball_gb": round(tarball.stat().st_size / 1e9, 2),
                      "secs": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
