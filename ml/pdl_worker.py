"""Parallel multi-node range-download worker.

The SCAMPS blob throttles to ~3.5 MiB/s PER IP. One node can't go faster no
matter how many connections. This worker runs on ONE node (= one IP) and pulls a
STRIDED subset of fixed-size byte chunks; run K copies on K different nodes and
aggregate throughput is ~K x 3.5 MiB/s. A separate stitcher (`ml.pdl_stitch`)
appends the chunks back in order into the tarball the extractor tails.

Assignment is work-stealing, NOT static striping: every worker walks chunk
indices from 0 upward and claims each via an atomic ``O_EXCL`` create of
chunk_<idx>.part; whoever wins downloads it, everyone else skips. So the LOW
indices (the front the stitcher needs first, in order) always get filled densely
no matter how many workers are actually running -- a pending or dead node just
means the survivors cover its chunks, instead of leaving permanent gaps that
stall the in-order stitch. Each chunk -> chunk_<idx>.part -> atomic rename to
chunk_<idx>.done once its byte length is verified.

Within the node a thread pool opens ~16 concurrent range requests (one node's
share of the per-connection rate). --num-workers/--worker-id are accepted for
logging only; correctness no longer depends on them.

Usage (one per node):
    python -m ml.pdl_worker --url <U> --out-dir <chunks> \
        --base-offset <bytes already on disk> --total-bytes <N>
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def head_total(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch_chunk(url: str, start: int, end: int, out_path: Path, retries: int = 100000) -> int:
    """Claim + download inclusive byte range [start, end] -> out_path (atomic).

    The ``.part`` file IS the cross-process claim: created with ``O_EXCL`` so only
    one worker (across all nodes) downloads any given chunk. Returns bytes written
    if we did it, or 0 if it's already done / claimed by someone else (caller just
    moves to the next index). Retries transient network errors on the claim we own;
    releases the claim if it ultimately gives up so another worker can retry."""
    if out_path.exists():
        return 0  # already done by some worker
    expected = end - start + 1
    tmp = out_path.with_suffix(".part")
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return 0  # another worker owns this chunk
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                if r.status not in (206, 200):
                    raise IOError(f"status {r.status}")
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            got = tmp.stat().st_size
            if got != expected:
                raise IOError(f"short {got}/{expected}")
            os.rename(tmp, out_path)
            return got
        except Exception:  # noqa: BLE001 - transient network, keep retrying
            time.sleep(min(30.0, 1.5 ** min(attempt, 8)))
    tmp.unlink(missing_ok=True)  # release claim so another worker can retry
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Strided multi-node range-download worker.")
    p.add_argument("--url", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base-offset", type=int, default=0,
                   help="byte offset already present on disk; download [base, total)")
    p.add_argument("--total-bytes", type=int, default=0, help="0 -> HEAD the server")
    p.add_argument("--chunk-mb", type=int, default=256)
    p.add_argument("--num-workers", type=int, required=True)
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--threads", type=int, default=16, help="concurrent conns on this node")
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = args.total_bytes or head_total(args.url)
    chunk = args.chunk_mb << 20
    n_chunks = (total - args.base_offset + chunk - 1) // chunk

    print(json.dumps({"event": "worker_start", "worker": args.worker_id,
                      "num_workers": args.num_workers, "total_gb": round(total / 1e9, 1),
                      "base_gb": round(args.base_offset / 1e9, 1),
                      "n_chunks": n_chunks, "chunk_mb": args.chunk_mb,
                      "threads": args.threads}), flush=True)

    def job(idx: int) -> int:
        start = args.base_offset + idx * chunk
        end = min(total - 1, start + chunk - 1)
        return fetch_chunk(args.url, start, end, out_dir / f"chunk_{idx:07d}.done")

    # Submit 0..n_chunks-1 IN ORDER; the pool's FIFO queue + cross-process
    # O_EXCL claim => every running worker races on the lowest unclaimed chunk,
    # so the contiguous front fills first regardless of fleet size.
    downloaded = 0
    bytes_got = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futs = [ex.submit(job, idx) for idx in range(n_chunks)]
        for fut in as_completed(futs):
            got = fut.result()
            if got > 0:
                downloaded += 1
                bytes_got += got
                if downloaded % args.log_every == 0:
                    rate = bytes_got / max(1e-6, time.time() - t0) / 1e6
                    print(json.dumps({"event": "progress", "worker": args.worker_id,
                                      "downloaded": downloaded, "MBps": round(rate, 2)}),
                          flush=True)
    print(json.dumps({"event": "worker_done", "worker": args.worker_id,
                      "downloaded": downloaded, "MB": round(bytes_got / 1e6, 1),
                      "secs": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
