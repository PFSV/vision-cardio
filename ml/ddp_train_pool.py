"""Real multi-GPU (DDP) training over a PERSISTENT, growing SCAMPS clip pool.

This replaces the old ``nn.DataParallel`` path (single-process, GIL-bound, poor
8-GPU scaling) with ``torch.nn.parallel.DistributedDataParallel`` launched by
``torchrun`` -- one process per GPU, NCCL all-reduce, near-linear scaling.

It reads a directory of pre-cached ``clip_<i>_bin<b>.npy`` tensors (built by
``ml.build_pool`` onto shared storage). Crucially it RE-SCANS the pool at the
start of every epoch, so while the downloader keeps adding clips the training set
keeps growing -- the GPUs stay busy and absorb new data as it lands instead of
waiting for the whole 552 GiB download to finish.

Fixes vs the old trainer (which plateaued at MAE ~42):
  * DDP instead of DataParallel (real 8-GPU utilization).
  * Leak-free, deterministic val split (``index % val_mod == 0``) that is stable
    as the pool grows -- a val clip is never trained on, ever.
  * Soft-ordinal target: a Gaussian over HR bins centred on the true bpm (KLDiv),
    so being 2 bpm off costs far less than 80 bpm off. Hard 141-way CE gave the
    label no ordinal structure.
  * LR 5e-4 with linear warmup + cosine decay (old 2e-3 diverged on small pools).
  * bf16 autocast.

Launch (Slurm, one 8-GPU node):
    torchrun --standalone --nproc_per_node=8 -m ml.ddp_train_pool \
        --pool-dir /path/to/vision_cardio_data/scamps_pool \
        --epochs 200 --batch-size 64 --arch video --width 64

Single-process CPU/GPU smoke test (no torchrun):
    python -m ml.ddp_train_pool --pool-dir <pool> --epochs 1 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .hr_bins import bin_to_bpm
from .train_scamps import build_model

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

NUM_BINS = 141  # 40..180 bpm inclusive


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def ddp_setup() -> tuple[bool, int, int, int, torch.device]:
    """Initialise the process group from torchrun env vars. Returns
    (is_dist, rank, world_size, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            device = torch.device("cuda", local)
        else:
            device = torch.device("cpu")
        return True, rank, world, local, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, 0, device


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def _clip_index(path: Path) -> int:
    # clip_<i>_bin<b>.npy
    return int(path.stem.split("_")[1])


def scan_pool(pool_dir: Path, val_mod: int) -> tuple[list[Path], list[Path]]:
    """Return (train_files, val_files). Val = clips whose stream index % val_mod
    == 0. Deterministic and stable as the pool grows, so no val clip is ever
    trained on."""
    files = sorted(pool_dir.glob("clip_*.npy"), key=_clip_index)
    train, val = [], []
    for f in files:
        (val if _clip_index(f) % val_mod == 0 else train).append(f)
    return train, val


class NpyClipDataset(Dataset):
    """Loads pre-cached (3,T,H,W) float16 tensors; label parsed from filename."""

    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path)  # (3,T,H,W) float16
        x = torch.from_numpy(arr).float()
        b = int(Path(path).stem.split("_bin")[-1])
        return x, torch.tensor(b, dtype=torch.long)


def broadcast_file_list(files: list[Path], is_dist: bool, device: torch.device) -> list[Path]:
    """All ranks must train on the SAME epoch file list. The downloader adds
    files between globs, so rank 0's list is the source of truth and is
    broadcast to every rank to keep DDP in lockstep."""
    if not is_dist:
        return files
    payload = [[str(p) for p in files]]
    dist.broadcast_object_list(payload, src=0)
    return [Path(s) for s in payload[0]]


# --------------------------------------------------------------------------- #
# soft-ordinal target
# --------------------------------------------------------------------------- #
def build_soft_labels(num_bins: int, sigma_bpm: float) -> torch.Tensor:
    """SOFT[t] = normalised Gaussian over bins centred at true bin t (1 bin == 1
    bpm). Row-normalised so each is a proper distribution for KLDiv."""
    idx = torch.arange(num_bins, dtype=torch.float32)
    diff = idx[None, :] - idx[:, None]          # (t, j) = j - t
    w = torch.exp(-0.5 * (diff / sigma_bpm) ** 2)
    return w / w.sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool) -> dict:
    model.eval()
    errs = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=use_amp and device.type == "cuda"):
            logits = model(x)
        pred = logits.argmax(dim=-1).cpu().tolist()
        errs.extend(abs(bin_to_bpm(int(p)) - bin_to_bpm(int(t)))
                    for p, t in zip(pred, y.tolist()))
    return {"mae": float(np.mean(errs)) if errs else None, "n": len(errs)}


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #
def save_ckpt(model, args, history) -> None:
    core = model.module if isinstance(model, DDP) else model
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": core.state_dict(),
        "arch": args.arch, "width": args.width,
        "representation": "raw", "min_bpm": 40, "max_bpm": 180,
        "clip_frames": args.clip_frames, "clip_size": args.clip_size,
        "history": history,
    }, out)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="DDP training over a growing SCAMPS npy pool.")
    p.add_argument("--pool-dir", required=True)
    p.add_argument("--out", default="artifacts/scamps_hr_model_real.pt")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64, help="per-GPU batch size")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--arch", default="video", choices=["video", "tiny_video"])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-size", type=int, default=112)
    p.add_argument("--sigma-bpm", type=float, default=3.0, help="Gaussian soft-label width")
    p.add_argument("--val-mod", type=int, default=7, help="val = clips with index %% val_mod == 0")
    p.add_argument("--min-train", type=int, default=8, help="wait until pool has this many train clips")
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--resume", type=int, default=0,
                   help="warm-start from --out if it exists (for successive jobs over a grown pool)")
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--wandb-project", default="vision_cardio")
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    is_dist, rank, world, local, device = ddp_setup()
    use_amp = bool(args.amp) and device.type == "cuda"
    pool_dir = Path(args.pool_dir)

    model = build_model(args.arch, num_bins=NUM_BINS, width=args.width).to(device)
    if args.resume and Path(args.out).exists():
        ck = torch.load(args.out, map_location=device)
        try:
            model.load_state_dict(ck["state_dict"])
            if is_main(rank):
                print(json.dumps({"event": "resumed", "from": str(args.out),
                                  "prev_epochs": len(ck.get("history", []))}), flush=True)
        except Exception as exc:
            if is_main(rank):
                print(json.dumps({"event": "resume_failed", "error": repr(exc)[:200]}), flush=True)
    if is_dist and device.type == "cuda":
        model = DDP(model, device_ids=[local], output_device=local)
    elif is_dist:
        model = DDP(model)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    soft = build_soft_labels(NUM_BINS, args.sigma_bpm).to(device)
    kld = nn.KLDivLoss(reduction="batchmean")

    def lr_at(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return args.lr * (epoch + 1) / max(1, args.warmup_epochs)
        prog = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if is_main(rank):
        print(json.dumps({"event": "init", "device": str(device), "world": world,
                          "gpu_count": gpu_count, "arch": args.arch, "width": args.width,
                          "amp": use_amp, "clip": [args.clip_frames, args.clip_size]},
                         sort_keys=True), flush=True)

    wandb_run = None
    if is_main(rank) and wandb is not None and (
            os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        try:
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                   config={**vars(args), "world": world})
        except Exception as exc:  # pragma: no cover
            print(f"wandb disabled: {exc}", flush=True)

    # Fixed val set: snapshot once from whatever is present at start (grows only
    # via the deterministic index rule, never overlaps train).
    history = []
    val_loader = None
    last_val_n = -1

    for epoch in range(args.epochs):
        # rank 0 decides this epoch's file lists; broadcast to keep ranks aligned.
        if is_main(rank):
            train_files, val_files = scan_pool(pool_dir, args.val_mod)
        else:
            train_files, val_files = [], []
        train_files = broadcast_file_list(train_files, is_dist, device)
        val_files = broadcast_file_list(val_files, is_dist, device)

        # wait for enough data (downloader still filling the pool)
        if len(train_files) < args.min_train:
            if is_main(rank):
                print(json.dumps({"event": "waiting", "epoch": epoch + 1,
                                  "train_clips": len(train_files),
                                  "need": args.min_train}), flush=True)
            time.sleep(15)
            if is_dist:
                dist.barrier()
            continue

        # (re)build val loader only when val set changed (rank 0 evals)
        if is_main(rank) and len(val_files) != last_val_n:
            val_loader = DataLoader(NpyClipDataset(val_files), batch_size=args.batch_size,
                                    shuffle=False, num_workers=args.num_workers,
                                    pin_memory=device.type == "cuda")
            last_val_n = len(val_files)

        ds = NpyClipDataset(train_files)
        if is_dist:
            sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                         shuffle=True, drop_last=False)
            sampler.set_epoch(epoch)
            loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")
        else:
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=device.type == "cuda")

        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)

        model.train()
        losses = []
        t0 = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            target = soft[y]  # (B, num_bins) soft distribution
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                logp = F.log_softmax(logits.float(), dim=-1)
                loss = kld(logp, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        if is_main(rank):
            # Eval on the UNWRAPPED module: calling the DDP wrapper on rank 0
            # only would fire a buffer-broadcast collective the other ranks
            # (parked at the barrier below) never join -> deadlock / 100%-util hang.
            eval_model = model.module if isinstance(model, DDP) else model
            stats = evaluate(eval_model, val_loader, device, use_amp) if val_loader else {"mae": None, "n": 0}
            row = {"epoch": epoch + 1, "train_clips": len(train_files),
                   "val_clips": len(val_files), "lr": round(lr_at(epoch), 6),
                   "train_loss": float(np.mean(losses)) if losses else None,
                   "epoch_s": round(time.time() - t0, 1), **stats}
            history.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            if wandb_run is not None:
                wandb_run.log(row, step=epoch + 1)
            if (epoch + 1) % args.ckpt_every == 0:
                save_ckpt(model, args, history)
        if is_dist:
            dist.barrier()

    if is_main(rank):
        save_ckpt(model, args, history)
        if wandb_run is not None:
            wandb_run.summary["final_mae"] = history[-1]["mae"] if history else None
            wandb_run.finish()
        print(json.dumps({"event": "done", "out": str(args.out),
                          "epochs": len(history)}), flush=True)
    if is_dist:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
