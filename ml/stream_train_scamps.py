"""Stream-train on SCAMPS without storing the full dataset.

Downloads the SCAMPS tarball as a single sequential HTTP stream, extracts a
small rolling buffer of clips at a time, trains the HR model on each shard, then
deletes the shard before pulling the next one. Peak disk stays at ~(val + a
couple of shards) of .mat files instead of the full 593GB.

A background producer thread does all the network/tar reads and fills a bounded
queue; the main thread trains on the GPU(s). The bounded queue gives natural
backpressure -- when the GPU is busy the producer blocks, which pauses the TCP
stream, so disk never grows past `queue_max` shards.

Note: throughput is network-bound (the Azure blob throttles to a few MiB/s), so
extra GPUs do not speed up a pass -- the win here is low disk + whole-dataset
coverage, not wall-clock. Prefetch overlaps download with compute so wall time
~= download time rather than download + train.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import tarfile
import threading
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .distributed import select_device, unwrap_model, wrap_multi_gpu
from .hr_bins import bin_to_bpm
from .train_scamps import ScampsDataset, TinyHRNet, TinyVideoHRNet, evaluate

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

DEFAULT_URL = (
    "https://facesyntheticspubwedata.z6.web.core.windows.net/"
    "neurips-2022/scamps_videos.tar.gz"
)


def _open_stream(url: str) -> tarfile.TarFile:
    req = urllib.request.Request(url, headers={"User-Agent": "vision-cardio-stream"})
    resp = urllib.request.urlopen(req, timeout=60)  # noqa: S310 - trusted MS blob
    return tarfile.open(fileobj=resp, mode="r|gz")


def _drain_members(tf: tarfile.TarFile):
    """Yield .mat TarInfo members in stream order."""
    for member in tf:
        if member.isfile() and member.name.endswith(".mat"):
            yield member


def _extract_clip(tf: tarfile.TarFile, member: tarfile.TarInfo, dest_dir: Path) -> Path:
    fobj = tf.extractfile(member)
    if fobj is None:
        raise IOError(f"could not extract {member.name}")
    out = dest_dir / Path(member.name).name
    with open(out, "wb") as handle:
        shutil.copyfileobj(fobj, handle, length=1 << 20)
    return out


class ShardProducer(threading.Thread):
    """Reads the tar stream, batches clips into shard dirs, queues them."""

    def __init__(self, tf, member_iter, buffer_root: Path, shard_clips: int,
                 max_clips: int | None, out_queue: "queue.Queue") -> None:
        super().__init__(daemon=True)
        self.tf = tf
        self.member_iter = member_iter
        self.buffer_root = buffer_root
        self.shard_clips = shard_clips
        self.max_clips = max_clips
        self.out_queue = out_queue
        self.error: Exception | None = None

    def run(self) -> None:
        try:
            shard_idx = 0
            seen = 0
            shard_dir = self.buffer_root / f"shard_{shard_idx:05d}"
            shard_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            for member in self.member_iter:
                if self.max_clips is not None and seen >= self.max_clips:
                    break
                _extract_clip(self.tf, member, shard_dir)
                count += 1
                seen += 1
                if count >= self.shard_clips:
                    self.out_queue.put((shard_idx, shard_dir))  # blocks if full -> backpressure
                    shard_idx += 1
                    shard_dir = self.buffer_root / f"shard_{shard_idx:05d}"
                    shard_dir.mkdir(parents=True, exist_ok=True)
                    count = 0
            if count > 0:
                self.out_queue.put((shard_idx, shard_dir))
            else:
                shard_dir.rmdir()
        except Exception as exc:  # surface to main thread
            self.error = exc
        finally:
            self.out_queue.put(None)  # sentinel


def _build_loader(root: Path, args, shuffle: bool) -> DataLoader:
    ds = ScampsDataset(
        root,
        representation=args.representation,
        clip_frames=args.clip_frames,
        clip_size=args.clip_size,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream-train HR model on SCAMPS (download->train->delete).")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--buffer-dir", default=os.environ.get("SLURM_TMPDIR", "/tmp") + "/scamps_stream")
    parser.add_argument("--out", default="artifacts/scamps_hr_model_stream.pt")
    parser.add_argument("--shard-clips", type=int, default=96)
    parser.add_argument("--val-clips", type=int, default=64)
    parser.add_argument("--max-clips", type=int, default=None, help="cap total clips (None = whole dataset)")
    parser.add_argument("--queue-max", type=int, default=2, help="shards buffered on disk (backpressure)")
    parser.add_argument("--local-epochs", type=int, default=2, help="passes over each shard before deleting it")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--representation", choices=["trace", "raw"], default="raw")
    parser.add_argument("--clip-frames", type=int, default=128)
    parser.add_argument("--clip-size", type=int, default=96)
    parser.add_argument("--ckpt-every", type=int, default=5, help="checkpoint every N shards")
    parser.add_argument("--wandb-project", default="vision_cardio")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    buffer_root = Path(args.buffer_dir)
    if buffer_root.exists():
        shutil.rmtree(buffer_root)
    buffer_root.mkdir(parents=True, exist_ok=True)

    device = select_device()
    base_model = TinyVideoHRNet() if args.representation == "raw" else TinyHRNet()
    model = wrap_multi_gpu(base_model.to(device), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(json.dumps({"device": str(device), "gpu_count": gpu_count,
                      "representation": args.representation}, sort_keys=True))

    wandb_run = None
    if wandb is not None and (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        try:
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                   config={**vars(args), "device": str(device), "gpu_count": gpu_count})
        except Exception as exc:  # pragma: no cover
            print(f"wandb disabled: {exc}")

    # Open the single sequential stream; hold out the first val-clips for a fixed val set.
    tf = _open_stream(args.url)
    members = _drain_members(tf)
    val_dir = buffer_root / "val"
    val_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(args.val_clips):
        try:
            _extract_clip(tf, next(members), val_dir)
        except StopIteration:
            break
    val_loader = _build_loader(val_dir, args, shuffle=False)
    print(json.dumps({"val_clips": len(list(val_dir.glob('*.mat')))}))

    # Producer streams the rest into shard dirs; main thread trains + deletes.
    shard_queue: "queue.Queue" = queue.Queue(maxsize=args.queue_max)
    producer = ShardProducer(tf, members, buffer_root, args.shard_clips,
                             args.max_clips, shard_queue)
    producer.start()

    history = []
    total_clips = 0
    shards_done = 0
    while True:
        item = shard_queue.get()
        if item is None:
            break
        shard_idx, shard_dir = item
        n_clips = len(list(shard_dir.glob("*.mat")))
        if n_clips == 0:
            shutil.rmtree(shard_dir, ignore_errors=True)
            continue
        loader = _build_loader(shard_dir, args, shuffle=True)
        model.train()
        losses = []
        for _ in range(args.local_epochs):
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                loss = criterion(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        del loader
        shutil.rmtree(shard_dir, ignore_errors=True)  # <-- delete after training

        total_clips += n_clips
        shards_done += 1
        stats = evaluate(model, val_loader, device)
        row = {"shard": shard_idx, "clips_seen": total_clips,
               "train_loss": float(np.mean(losses)) if losses else None, **stats}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if wandb_run is not None:
            wandb_run.log(row, step=total_clips)

        if shards_done % args.ckpt_every == 0:
            _save(model, args, history, total_clips)

    producer.join()
    if producer.error is not None:
        print(f"producer error: {producer.error}")

    _save(model, args, history, total_clips)
    if wandb_run is not None:
        wandb_run.summary["clips_seen"] = total_clips
        wandb_run.summary["artifact_path"] = str(args.out)
        wandb_run.finish()
    shutil.rmtree(buffer_root, ignore_errors=True)
    print(json.dumps({"final_clips_seen": total_clips, "out": str(args.out)}))


def _save(model, args, history, clips_seen) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": unwrap_model(model).state_dict(),
        "min_bpm": 40, "max_bpm": 180,
        "representation": args.representation,
        "clip_frames": args.clip_frames, "clip_size": args.clip_size,
        "clips_seen": clips_seen, "history": history,
    }, out_path)


if __name__ == "__main__":
    main()
