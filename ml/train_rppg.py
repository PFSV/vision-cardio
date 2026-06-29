"""DDP trainer for the PhysNet rPPG waveform model.

Pairs each frame clip  scamps_pool/clip_<idx>_bin*.npy  (3,T,H,W) fp16
with its PPG label    scamps_pool_ppg/ppg_<idx>.npy     (T,)    fp32
(same .mat member index). Loss = negative-Pearson(pred_wave, gt_wave). Eval HR
= FFT peak of the predicted waveform vs the GT waveform -> MAE (bpm).

Launch: torchrun --standalone --nproc_per_node=8 -m ml.train_rppg --pool-dir ... --ppg-dir ...
See scripts/train_rppg_h100.slurm.
"""
from __future__ import annotations
import argparse, json, math, os, re
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .physnet import PhysNet, neg_pearson_loss, hr_from_wave

IDX_RE = re.compile(r"clip_(\d+)_bin")


def is_main(rank: int) -> bool:
    return rank == 0


class RppgDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        clip_path, ppg_path = self.pairs[i]
        x = torch.from_numpy(np.load(clip_path)).float()        # (3,T,H,W)
        y = torch.from_numpy(np.load(ppg_path)).float()         # (T,)
        return x, y


def build_pairs(pool_dir: Path, ppg_dir: Path):
    pairs = []
    for clip in sorted(pool_dir.glob("clip_*_bin*.npy")):
        m = IDX_RE.search(clip.name)
        if not m:
            continue
        idx = int(m.group(1))
        ppg = ppg_dir / f"ppg_{idx}.npy"
        if ppg.exists():
            pairs.append((str(clip), str(ppg), idx))
    return pairs


@torch.no_grad()
def evaluate(model, val_pairs, device, fs, clip_frames):
    model.eval()
    errs = []
    for clip_path, ppg_path, _ in val_pairs:
        x = torch.from_numpy(np.load(clip_path)).float().unsqueeze(0).to(device)
        gt = np.load(ppg_path)
        pred = model(x)[0].float().cpu().numpy()
        hr_p = hr_from_wave(pred, fs)
        hr_g = hr_from_wave(gt, fs)
        if not (math.isnan(hr_p) or math.isnan(hr_g)):
            errs.append(abs(hr_p - hr_g))
    return {"mae": float(np.mean(errs)) if errs else None, "n": len(errs)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool-dir", default="/path/to/vision_cardio_data/scamps_pool")
    p.add_argument("--ppg-dir", default="/path/to/vision_cardio_data/scamps_pool_ppg")
    p.add_argument("--out", default="artifacts/rppg_physnet.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-seconds", type=float, default=20.0, help="SCAMPS = 600f @ 30fps")
    p.add_argument("--val-mod", type=int, default=7)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--wandb-project", default="vision_cardio")
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    fs = args.clip_frames / args.clip_seconds
    pairs = build_pairs(Path(args.pool_dir), Path(args.ppg_dir))
    train = [pr for pr in pairs if pr[2] % args.val_mod != 0]
    val = [pr for pr in pairs if pr[2] % args.val_mod == 0]

    if is_main(rank):
        print(json.dumps({"event": "init", "pairs": len(pairs), "train": len(train),
                          "val": len(val), "fs": round(fs, 3), "width": args.width,
                          "world": world}), flush=True)

    use_wandb = is_main(rank)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       config=vars(args))
        except Exception as e:  # noqa: BLE001
            print("wandb off:", e, flush=True); use_wandb = False

    model = PhysNet(width=args.width).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and torch.cuda.is_available())

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return args.lr * (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    ds = RppgDataset([(c, g) for (c, g, _i) in train])
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if distributed else None
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None),
                        sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        if sampler:
            sampler.set_epoch(epoch)
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        losses = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(args.amp) and torch.cuda.is_available()):
                pred = model(x)
                loss = neg_pearson_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        if is_main(rank):
            eval_model = model.module if isinstance(model, DDP) else model
            stats = evaluate(eval_model, val, device, fs, args.clip_frames)
            row = {"epoch": epoch + 1, "lr": round(lr_at(epoch), 6),
                   "train_loss": float(np.mean(losses)) if losses else None,
                   "mae": stats["mae"], "n": stats["n"],
                   "train_pairs": len(train), "val_pairs": len(val)}
            print(json.dumps(row), flush=True)
            if use_wandb:
                import wandb
                wandb.log({k: v for k, v in row.items() if v is not None})
            if stats["mae"] is not None and stats["mae"] < best:
                best = stats["mae"]
                core = model.module if isinstance(model, DDP) else model
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": core.state_dict(), "width": args.width,
                            "clip_frames": args.clip_frames, "clip_seconds": args.clip_seconds,
                            "best_mae": best, "epoch": epoch + 1}, args.out)
        if distributed:
            dist.barrier()

    if is_main(rank):
        print(json.dumps({"event": "done", "best_mae": best}), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
