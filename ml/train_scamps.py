from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .distributed import select_device, unwrap_model, wrap_multi_gpu
from .hr_bins import bin_to_bpm, bpm_to_bin
from .scamps import iter_scamps_files, load_scamps_mat
from .video_io import rgb_trace

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


class ScampsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        limit: int | None = None,
        representation: str = "trace",
        clip_frames: int = 128,
        clip_size: int = 96,
    ) -> None:
        self.paths = list(iter_scamps_files(root))
        if limit is not None:
            self.paths = self.paths[:limit]
        self.representation = representation
        self.clip_frames = clip_frames
        self.clip_size = clip_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        sample = load_scamps_mat(self.paths[idx])
        if self.representation == "trace":
            trace = rgb_trace(sample.frames)
            x = torch.from_numpy(trace.astype(np.float32).T)
        elif self.representation == "raw":
            frames = self._prepare_raw_clip(sample.frames)
            x = frames
        else:
            raise ValueError(f"Unsupported representation: {self.representation}")
        y = torch.tensor(bpm_to_bin(sample.heart_rate_bpm), dtype=torch.long)
        return x, y

    def _prepare_raw_clip(self, frames: np.ndarray) -> torch.Tensor:
        clip = np.asarray(frames, dtype=np.float32)
        if clip.ndim != 4:
            raise ValueError(f"Expected 4D frames, got {clip.shape}")
        total = clip.shape[0]
        if total > self.clip_frames:
            indices = np.linspace(0, total - 1, self.clip_frames).astype(np.int64)
            clip = clip[indices]
        clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # C, T, H, W
        clip = F.interpolate(
            clip.unsqueeze(0),
            size=(self.clip_frames, self.clip_size, self.clip_size),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
        return clip


class TinyHRNet(nn.Module):
    def __init__(self, num_bins: int = 141) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(64, num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global average over time == AdaptiveAvgPool1d(1); written as mean so
        # coremltools can convert it (adaptive_avg_pool1d is unsupported).
        return self.head(self.net(x).mean(dim=-1))


class TinyVideoHRNet(nn.Module):
    def __init__(self, num_bins: int = 141) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 96, kernel_size=3, padding=1, stride=(1, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(96, 192, kernel_size=3, padding=1, stride=(2, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(192, 384, kernel_size=3, padding=1, stride=(2, 2, 2)),
            nn.ReLU(),
            nn.Conv3d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(384, num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # global average over (T,H,W) == AdaptiveAvgPool3d(1); written as mean
        # because coremltools can't convert adaptive_avg_pool3d.
        x = self.net(x).mean(dim=(2, 3, 4))
        return self.head(x)


class VideoHRNet(nn.Module):
    """Scalable 3D-conv HR net. `width` sets the channel base; depth is 4 stages.

    HR is a TEMPORAL FREQUENCY (how fast skin pixels oscillate). The earlier
    version pooled the whole (T,H,W) volume to a scalar before the head
    (``mean(dim=(2,3,4))``), which collapses the pulse waveform to a DC value and
    discards the very signal that distinguishes 60 from 140 bpm -> the model
    could only predict the mean (val MAE frozen ~30-37 bpm regardless of data or
    width). Classical CHROM on the SAME clips recovers HR at ~3 bpm median, so
    the signal is present; the architecture was throwing it away.

    Fix: the conv stages downsample SPATIAL only (stride (1,2,2)), keeping all 128
    time steps (eff 6.4 Hz -> Nyquist 3.2 Hz covers 40-180 bpm). A Conv1d temporal
    head then learns oscillation/frequency features over the full trace before the
    time axis is pooled. Export-safe (Conv3d/Conv1d, BN, ReLU, mean, Linear -- no
    adaptive pooling, no FFT)."""

    def __init__(self, num_bins: int = 141, width: int = 64) -> None:
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8

        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
                nn.Conv3d(cout, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(cout),
                nn.ReLU(inplace=True),
            )

        # stride (1,2,2) every stage: shrink H,W, PRESERVE the time axis so the
        # cardiac frequency survives to the head.
        self.net = nn.Sequential(
            block(3, c1, stride=(1, 2, 2)),
            block(c1, c2, stride=(1, 2, 2)),
            block(c2, c3, stride=(1, 2, 2)),
            block(c3, c4, stride=(1, 2, 2)),
        )
        # temporal head: spatial-pooled per-frame features -> Conv1d over the full
        # T trace (learns periodicity) -> pool time AFTER frequency is extracted.
        self.temporal = nn.Sequential(
            nn.Conv1d(c4, c4, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
            nn.Conv1d(c4, c4, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(c4, num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Remove per-pixel temporal DC (skin tone, lighting). The rPPG pulse is a
        # ~1% AC fluctuation riding on a large static color; without this the conv
        # stack latches onto the static color (subject identity) and ignores the
        # faint signal -> predict-the-mean collapse. Subtracting each pixel's
        # time-average makes the AC pulse the dominant variance (DeepPhys-style
        # normalization). Export-safe (mean + subtract).
        x = x - x.mean(dim=2, keepdim=True)
        x = self.net(x).mean(dim=(3, 4))   # (B, c4, T) -- spatial pool only, keep time
        x = self.temporal(x)               # (B, c4, T) -- learn oscillation/frequency
        x = x.mean(dim=-1)                 # (B, c4)    -- pool time AFTER freq features
        return self.head(x)


def build_model(arch: str, num_bins: int = 141, width: int = 64) -> nn.Module:
    if arch == "video":
        return VideoHRNet(num_bins=num_bins, width=width)
    if arch == "tiny_video":
        return TinyVideoHRNet(num_bins=num_bins)
    if arch == "trace":
        return TinyHRNet(num_bins=num_bins)
    raise ValueError(f"unknown arch: {arch}")


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    errs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.tensor([bin_to_bpm(int(idx)) for idx in logits.argmax(dim=-1).cpu().tolist()], device=device)
            true = torch.tensor([bin_to_bpm(int(idx)) for idx in y.cpu().tolist()], device=device)
            errs.extend((pred - true).abs().cpu().numpy().tolist())
    return {"mae": float(np.mean(errs)) if errs else None, "n": len(errs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny HR model on SCAMPS.")
    parser.add_argument("--root", required=True, help="Path to scamps videos directory or extracted example archive")
    parser.add_argument("--out", default="artifacts/scamps_hr_model.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--representation", choices=["trace", "raw"], default="trace")
    parser.add_argument("--clip-frames", type=int, default=128)
    parser.add_argument("--clip-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers; raise on GPU nodes to keep the device fed")
    parser.add_argument("--wandb-project", default="vision_cardio")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    ds = ScampsDataset(
        args.root,
        limit=args.limit,
        representation=args.representation,
        clip_frames=args.clip_frames,
        clip_size=args.clip_size,
    )
    if len(ds) < 2:
        raise ValueError("Need at least two SCAMPS samples")
    val_count = max(1, len(ds) // 5)
    train_count = len(ds) - val_count
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_count, val_count], generator=torch.Generator().manual_seed(7))
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    device = select_device()
    if args.representation == "raw":
        base_model = TinyVideoHRNet()
    else:
        base_model = TinyHRNet()
    model = wrap_multi_gpu(base_model.to(device), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    wandb_run = None
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    if wandb is not None and wandb_api_key:
        try:
            wandb.login(key=wandb_api_key)
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config={
                    "root": str(args.root),
                    "out": str(args.out),
                    "limit": args.limit,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "representation": args.representation,
                    "clip_frames": args.clip_frames,
                    "clip_size": args.clip_size,
                    "device": str(device),
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                },
            )
        except Exception as exc:  # pragma: no cover - best effort telemetry
            print(f"wandb disabled: {exc}")
            wandb_run = None

    print(
        json.dumps(
            {
                "device": str(device),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "representation": args.representation,
            },
            sort_keys=True,
        )
    )

    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        stats = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **stats}
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if wandb_run is not None:
            wandb_run.log(row, step=epoch + 1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": unwrap_model(model).state_dict(),
            "min_bpm": 40,
            "max_bpm": 180,
            "representation": args.representation,
            "clip_frames": args.clip_frames,
            "clip_size": args.clip_size,
            "history": history,
        },
        out_path,
    )
    if wandb_run is not None:
        wandb_run.summary["artifact_path"] = str(out_path)
        wandb_run.finish()


if __name__ == "__main__":
    main()
