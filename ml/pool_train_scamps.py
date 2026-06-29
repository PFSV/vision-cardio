"""Throughput training: keep the 8xH100 node compute-bound, not IO-bound.

A background producer streams clips from the SCAMPS tarball, and -- crucially --
preprocesses each clip ONCE at download time into the exact model-input tensor
(3, T, H, W) saved as a small float16 .npy on node-local NVMe. The training loop
then reads those tiny tensors directly (no scipy.loadmat, no trilinear resize in
the hot path), so the GPUs are fed fast enough to stay busy across many epochs
over the resident pool. Pool is on ephemeral node-local scratch (not cepheid).

Each cached file is named `clip_<n>_bin<b>.npy`, where b = bpm_to_bin(hr), so the
label travels with the file and no .mat is kept around.

Usage:
    python -m ml.pool_train_scamps --buffer-dir /raid/$USER/scamps_pool \
        --pool-clips 1200 --max-epochs 300 --batch-size 128 --num-workers 16
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .distributed import select_device, unwrap_model, wrap_multi_gpu
from .hr_bins import bin_to_bpm, bpm_to_bin
from .scamps import load_scamps_mat
from .stream_train_scamps import _drain_members, _open_stream
from .train_scamps import build_model

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None


def prepare_clip_tensor(frames: np.ndarray, clip_frames: int, clip_size: int) -> torch.Tensor:
    """Same transform as ScampsDataset(representation='raw'), returned as float16."""
    clip = np.asarray(frames, dtype=np.float32)
    if clip.ndim != 4:
        raise ValueError(f"expected 4D frames, got {clip.shape}")
    total = clip.shape[0]
    if total > clip_frames:
        idx = np.linspace(0, total - 1, clip_frames).astype(np.int64)
        clip = clip[idx]
    t = torch.from_numpy(clip).permute(3, 0, 1, 2)  # C, T, H, W
    t = F.interpolate(t.unsqueeze(0), size=(clip_frames, clip_size, clip_size),
                      mode="trilinear", align_corners=False).squeeze(0)
    return t.half()


class NpyClipDataset(Dataset):
    """Loads pre-cached (3,T,H,W) float16 tensors; label parsed from filename."""

    def __init__(self, files) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path)  # (3,T,H,W) float16
        x = torch.from_numpy(arr).float()
        b = int(Path(path).stem.split("_bin")[-1])
        return x, torch.tensor(b, dtype=torch.long)


class PoolProducer(threading.Thread):
    """Stream -> preprocess once -> cache .npy. First `val_clips` to val_dir,
    then up to `pool_clips` to pool_dir. Add-only, atomic rename."""

    def __init__(self, url, val_dir, pool_dir, val_clips, pool_clips,
                 clip_frames, clip_size, val_ready) -> None:
        super().__init__(daemon=True)
        self.url = url
        self.val_dir, self.pool_dir = val_dir, pool_dir
        self.val_clips, self.pool_clips = val_clips, pool_clips
        self.clip_frames, self.clip_size = clip_frames, clip_size
        self.val_ready = val_ready
        self.error: Exception | None = None
        self.n_val = 0
        self.n_pool = 0

    def _cache(self, tf, member, dest_dir: Path, n: int) -> bool:
        # extract .mat bytes to a temp file, load, preprocess, save .npy, drop .mat
        tmp_mat = dest_dir / ("_tmp_%d.mat" % n)
        fobj = tf.extractfile(member)
        if fobj is None:
            return False
        with open(tmp_mat, "wb") as h:
            while True:
                chunk = fobj.read(1 << 20)
                if not chunk:
                    break
                h.write(chunk)
        try:
            sample = load_scamps_mat(tmp_mat)
            tensor = prepare_clip_tensor(sample.frames, self.clip_frames, self.clip_size)
            b = bpm_to_bin(sample.heart_rate_bpm)
        finally:
            tmp_mat.unlink(missing_ok=True)
        tmp_npy = dest_dir / ("_tmp_%d.npy" % n)
        np.save(tmp_npy, tensor.numpy())
        tmp_npy.rename(dest_dir / ("clip_%d_bin%d.npy" % (n, b)))
        return True

    def run(self) -> None:
        try:
            tf = _open_stream(self.url)
            members = _drain_members(tf)
            for member in members:
                if self.n_val >= self.val_clips:
                    break
                if self._cache(tf, member, self.val_dir, self.n_val):
                    self.n_val += 1
            self.val_ready.set()
            for member in members:
                if self.n_pool >= self.pool_clips:
                    break
                if self._cache(tf, member, self.pool_dir, self.n_pool):
                    self.n_pool += 1
        except Exception as exc:
            self.error = exc
            self.val_ready.set()


def _loader(files, args, shuffle: bool) -> DataLoader:
    return DataLoader(NpyClipDataset(files), batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=False, drop_last=False)


def evaluate(model, loader, device) -> dict:
    model.eval()
    errs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(dim=-1).cpu().tolist()
            errs.extend(abs(bin_to_bpm(int(p)) - bin_to_bpm(int(t)))
                        for p, t in zip(pred, y.tolist()))
    return {"mae": float(np.mean(errs)) if errs else None, "n": len(errs)}


def main() -> None:
    p = argparse.ArgumentParser(description="Compute-bound SCAMPS training on node-local NVMe (npy cache).")
    p.add_argument("--url", default=None)
    p.add_argument("--buffer-dir", default=(os.environ.get("SLURM_TMPDIR", "/tmp") + "/scamps_pool"))
    p.add_argument("--out", default="artifacts/scamps_hr_model_real.pt")
    p.add_argument("--pool-clips", type=int, default=1200)
    p.add_argument("--val-clips", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--arch", default="video", choices=["video", "tiny_video"])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-size", type=int, default=112)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--min-pool", type=int, default=16, help="start training once pool has this many clips")
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--wandb-project", default="vision_cardio")
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    from .stream_train_scamps import DEFAULT_URL
    url = args.url or DEFAULT_URL

    buf = Path(args.buffer_dir)
    val_dir, pool_dir = buf / "val", buf / "pool"
    for d in (val_dir, pool_dir):
        d.mkdir(parents=True, exist_ok=True)

    device = select_device()
    use_amp = bool(args.amp) and device.type == "cuda"
    model = wrap_multi_gpu(build_model(args.arch, width=args.width).to(device), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(json.dumps({"device": str(device), "gpu_count": gpu_count, "arch": args.arch,
                      "width": args.width, "amp": use_amp,
                      "clip": [args.clip_frames, args.clip_size]}, sort_keys=True))

    wandb_run = None
    if wandb is not None and (os.environ.get("WANDB_API_KEY") or Path.home().joinpath(".netrc").exists()):
        try:
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                                   config={**vars(args), "gpu_count": gpu_count})
        except Exception as exc:  # pragma: no cover
            print(f"wandb disabled: {exc}")

    val_ready = threading.Event()
    producer = PoolProducer(url, val_dir, pool_dir, args.val_clips, args.pool_clips,
                            args.clip_frames, args.clip_size, val_ready)
    producer.start()
    val_ready.wait()
    if producer.error is not None:
        raise RuntimeError(f"producer failed during val fetch: {producer.error}")
    val_loader = _loader(sorted(val_dir.glob("*.npy")), args, shuffle=False)

    history = []
    for epoch in range(args.max_epochs):
        pool_files = sorted(pool_dir.glob("*.npy"))
        if len(pool_files) < args.min_pool:
            if producer.error is not None:
                raise RuntimeError(f"producer failed: {producer.error}")
            time.sleep(5)
            continue
        loader = _loader(pool_files, args, shuffle=True)
        model.train()
        losses = []
        t0 = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        stats = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "pool_clips": len(pool_files),
               "train_loss": float(np.mean(losses)) if losses else None,
               "epoch_s": round(time.time() - t0, 1), **stats}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if wandb_run is not None:
            wandb_run.log(row, step=epoch + 1)
        if (epoch + 1) % args.ckpt_every == 0:
            _save(model, args, history)

    _save(model, args, history)
    if wandb_run is not None:
        wandb_run.summary["pool_clips"] = producer.n_pool
        wandb_run.finish()
    print(json.dumps({"done": True, "pool_clips": producer.n_pool, "out": str(args.out)}))


def _save(model, args, history) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": unwrap_model(model).state_dict(),
        "arch": args.arch, "width": args.width,
        "representation": "raw", "min_bpm": 40, "max_bpm": 180,
        "clip_frames": args.clip_frames, "clip_size": args.clip_size,
        "history": history,
    }, out)


if __name__ == "__main__":
    main()
