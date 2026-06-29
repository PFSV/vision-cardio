from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .distributed import select_device, unwrap_model, wrap_multi_gpu
from .hr_bins import bin_to_bpm, bpm_to_bin
from .video_io import load_video_frames, rgb_trace


@dataclass(frozen=True)
class Sample:
    video_path: str
    hr_bpm: float
    participant_id: str
    split: str


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, split: str = "train") -> None:
        self.samples: list[Sample] = []
        with open(manifest_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("split", "train") != split:
                    continue
                self.samples.append(
                    Sample(
                        video_path=row["video_path"],
                        hr_bpm=float(row["hr_bpm"]),
                        participant_id=row.get("participant_id", ""),
                        split=row.get("split", "train"),
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video = load_video_frames(sample.video_path)
        trace = rgb_trace(video.frames)
        # Trace shape: (T, 3). Convert to (3, T).
        x = torch.from_numpy(trace.astype(np.float32).T)
        y = torch.tensor(bpm_to_bin(sample.hr_bpm), dtype=torch.long)
        return x, y


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
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(64, num_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x).squeeze(-1)
        return self.head(x)


def bin_to_bpm(bin_idx: int, min_bpm: int = 40) -> int:
    return min_bpm + bin_idx


def bpm_to_bin(bpm: float, min_bpm: int = 40, max_bpm: int = 180) -> int:
    bpm = int(round(bpm))
    return int(np.clip(bpm, min_bpm, max_bpm) - min_bpm)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    errs = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred_bins = logits.argmax(dim=-1)
            pred_bpm = torch.tensor([bin_to_bpm(int(idx)) for idx in pred_bins.cpu().tolist()], device=device)
            true_bpm = torch.tensor([bin_to_bpm(int(idx)) for idx in y.cpu().tolist()], device=device)
            errs.extend((pred_bpm - true_bpm).abs().cpu().numpy().tolist())
    return {
        "mae": float(np.mean(errs)) if errs else None,
        "n": len(errs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny offline HR model from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="artifacts/hr_model.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_ds = ManifestDataset(args.manifest, split="train")
    val_ds = ManifestDataset(args.manifest, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device()
    model = wrap_multi_gpu(TinyHRNet().to(device), device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

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
        val_stats = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **val_stats})
        print(json.dumps(history[-1], sort_keys=True))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": unwrap_model(model).state_dict(),
            "min_bpm": 40,
            "max_bpm": 180,
            "history": history,
        },
        out_path,
    )


if __name__ == "__main__":
    main()
