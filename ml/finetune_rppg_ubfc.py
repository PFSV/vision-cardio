"""Fine-tune the SCAMPS-trained PhysNet rPPG model on real UBFC-rPPG faces.

Self-contained: does NOT touch ml/train_rppg.py (the SCAMPS pool DDP trainer).
Builds (clip, ppg) windows straight from UBFC videos via ml.ubfc_data, inits from
a pretrained PhysNet checkpoint, splits BY PARTICIPANT (whole subjects held out
for val -> no leakage), fine-tunes with negative-Pearson loss, and saves the
best-by-val checkpoint ONLY if it beats the zero-shot baseline. Single GPU is
plenty (UBFC ~150 windows).

  python -m ml.finetune_rppg_ubfc \
    --init-ckpt artifacts/rppg_physnet.pt \
    --ubfc-root data/ubfc-rppg/rppg-data \
    --out artifacts/rppg_physnet_ubfc.pt --epochs 40 --lr 3e-4 --holdout-every 5
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .physnet import PhysNet, neg_pearson_loss, hr_from_wave
from .ubfc_data import discover, subject_windows


def build_windows(ubfc_root, clip_seconds, clip_frames, clip_size, crop, max_subjects=0):
    """Decode every subject once -> list of {x fp16 (3,T,H,W), y fp32 (T,), subj}."""
    items = []
    subs = sorted(discover(ubfc_root), key=lambda r: str(r[2]))
    if max_subjects:
        subs = subs[:max_subjects]
    for avi, gt, name in subs:
        nw = 0
        for x, y, _shr in subject_windows(avi, gt, clip_seconds, clip_frames, clip_size, crop=crop):
            items.append({"x": x.astype(np.float16), "y": y.astype(np.float32), "subj": str(name)})
            nw += 1
        print(json.dumps({"event": "decoded", "subj": str(name), "windows": nw}), flush=True)
    return items


class WinDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return torch.from_numpy(it["x"].astype(np.float32)), torch.from_numpy(it["y"])


@torch.no_grad()
def eval_mae(model, items, device, fs):
    model.eval()
    errs = []
    for it in items:
        x = torch.from_numpy(it["x"].astype(np.float32)).unsqueeze(0).to(device)
        pred = model(x)[0].float().cpu().numpy()
        hp = hr_from_wave(pred, fs)
        hg = hr_from_wave(it["y"], fs)
        if not (math.isnan(hp) or math.isnan(hg)):
            errs.append(abs(hp - hg))
    return (float(np.mean(errs)) if errs else float("nan")), len(errs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init-ckpt", default="artifacts/rppg_physnet.pt")
    p.add_argument("--ubfc-root", required=True)
    p.add_argument("--out", default="artifacts/rppg_physnet_ubfc.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip-frames", type=int, default=128)
    p.add_argument("--clip-seconds", type=float, default=20.0)
    p.add_argument("--clip-size", type=int, default=112)
    p.add_argument("--crop", default=None, help="ffmpeg face ROI 'w:h:x:y'")
    p.add_argument("--holdout-every", type=int, default=5, help="every Nth subject -> val")
    p.add_argument("--max-subjects", type=int, default=0, help="smoke-test cap (0=all)")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fs = args.clip_frames / args.clip_seconds

    ck = torch.load(args.init_ckpt, map_location="cpu")
    width = int(ck.get("width", 32))
    model = PhysNet(width=width).to(device)
    model.load_state_dict(ck["state_dict"])
    print(json.dumps({"event": "init", "init_ckpt": args.init_ckpt,
                      "pretrain_best_mae": ck.get("best_mae"), "width": width,
                      "fs": round(fs, 3), "device": str(device)}), flush=True)

    items = build_windows(args.ubfc_root, args.clip_seconds, args.clip_frames,
                          args.clip_size, args.crop, args.max_subjects)
    subj_order = []
    for it in items:
        if it["subj"] not in subj_order:
            subj_order.append(it["subj"])
    val_subj = set(subj_order[::args.holdout_every]) if args.holdout_every > 0 else set()
    train_items = [it for it in items if it["subj"] not in val_subj]
    val_items = [it for it in items if it["subj"] in val_subj]
    print(json.dumps({"event": "split", "subjects": len(subj_order),
                      "val_subjects": len(val_subj), "train_win": len(train_items),
                      "val_win": len(val_items)}), flush=True)

    base_mae, base_n = eval_mae(model, val_items, device, fs)
    print(json.dumps({"event": "baseline", "val_mae": base_mae, "n": base_n}), flush=True)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:  # noqa: BLE001
            print("wandb off:", e, flush=True)
            use_wandb = False

    loader = DataLoader(WinDataset(train_items), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(ep):
        if ep < args.warmup_epochs:
            return args.lr * (ep + 1) / max(1, args.warmup_epochs)
        prog = (ep - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * args.lr * (1 + math.cos(math.pi * min(1.0, prog)))

    best = base_mae if not math.isnan(base_mae) else float("inf")
    best_saved = False
    for epoch in range(args.epochs):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(epoch)
        losses = []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = neg_pearson_loss(pred, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        vmae, vn = eval_mae(model, val_items, device, fs)
        row = {"epoch": epoch + 1, "lr": round(lr_at(epoch), 6),
               "train_loss": float(np.mean(losses)) if losses else None,
               "val_mae": vmae, "n": vn}
        print(json.dumps(row), flush=True)
        if use_wandb:
            import wandb
            wandb.log({k: v for k, v in row.items() if v is not None})
        if not math.isnan(vmae) and vmae < best:
            best = vmae
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "width": width,
                        "clip_frames": args.clip_frames, "clip_seconds": args.clip_seconds,
                        "best_mae": best, "epoch": epoch + 1, "finetune": "ubfc"}, args.out)
            best_saved = True

    improved = (base_mae - best) if not math.isnan(base_mae) else None
    print(json.dumps({"event": "done", "baseline_val_mae": base_mae, "best_val_mae": best,
                      "improved_bpm": improved, "saved": best_saved, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
