"""Cross-dataset eval: run a trained PhysNet on UBFC-rPPG real faces.

HR(pred FFT of predicted waveform) vs HR(FFT of contact PPG) per 20 s window,
per subject -> MAE (bpm). This is the HONEST number (synthetic SCAMPS MAE is
within-distribution only). Zero-shot unless --ckpt is a UBFC-fine-tuned model.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch

from .physnet import PhysNet, hr_from_wave
from .ubfc_data import discover, subject_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/rppg_physnet.pt")
    ap.add_argument("--ubfc-root", required=True)
    ap.add_argument("--clip-frames", type=int, default=128)
    ap.add_argument("--clip-seconds", type=float, default=20.0)
    ap.add_argument("--clip-size", type=int, default=112)
    ap.add_argument("--crop", default=None, help="ffmpeg face ROI 'w:h:x:y' (e.g. 360:360:140:60)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    m = PhysNet(width=ck.get("width", 32))
    m.load_state_dict(ck["state_dict"])
    m.eval()
    fs = args.clip_frames / args.clip_seconds
    print(f"ckpt={args.ckpt} synthetic_best_mae={ck.get('best_mae')} fs={fs:.2f}Hz")

    all_ppg, all_sensor = [], []
    for avi, gt, name in discover(args.ubfc_root):
        ep, es = [], []
        for x, y, shr in subject_windows(avi, gt, args.clip_seconds, args.clip_frames,
                                         args.clip_size, crop=args.crop):
            with torch.no_grad():
                pred = m(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            hp = hr_from_wave(pred, fs)
            hg = hr_from_wave(y, fs)
            if not (np.isnan(hp) or np.isnan(hg)):
                ep.append(abs(hp - hg))
            if not (np.isnan(hp) or np.isnan(shr)):
                es.append(abs(hp - shr))
        all_ppg += ep
        all_sensor += es
        msg = (f"MAE vs PPG={np.mean(ep):5.2f}  vs sensorHR={np.mean(es):5.2f}"
               if ep else "no windows")
        print(f"  {name:18s} n={len(ep):2d}  {msg}")

    if all_ppg:
        print(f"\nOVERALL  vs contact-PPG = {np.mean(all_ppg):.2f} bpm (n={len(all_ppg)})")
        print(f"         vs sensor-HR   = {np.mean(all_sensor):.2f} bpm")


if __name__ == "__main__":
    main()
