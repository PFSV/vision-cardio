"""Live wandb progress bar for the SCAMPS pool extraction.

A side-car: it does NOT touch the running extractor (slurm job 132429). It just
counts clip_*.npy in the pool every INTERVAL and logs clips / percent / rate /
ETA to wandb so you get a live chart at wandb.ai/pfsv/vision_cardio. Stops when
the pool hits TARGET or the slurm job leaves the queue and the pool stops growing.

Run:  JID=132429 TARGET=2800 python3 scripts/extract_wandb_monitor.py
"""
from __future__ import annotations
import glob, os, subprocess, time
import wandb

POOL = os.environ.get("POOL", "/path/to/vision_cardio_data/scamps_pool")
TARGET = int(os.environ.get("TARGET", "2800"))
JID = os.environ.get("JID", "132429")
INTERVAL = float(os.environ.get("INTERVAL", "30"))


def count() -> int:
    return len(glob.glob(os.path.join(POOL, "clip_*.npy")))


def job_alive() -> bool:
    r = subprocess.run(["squeue", "-j", JID, "-h"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def main() -> None:
    run = wandb.init(project="vision_cardio", name=f"scamps-extract-{JID}",
                     job_type="extract", config={"target": TARGET, "slurm_job": JID})
    n0 = count(); t0 = time.time()
    prev, t_prev = n0, t0
    stale = 0
    while True:
        time.sleep(INTERVAL)
        n = count(); t = time.time()
        win_rate = (n - prev) / max(1e-6, t - t_prev) * 3600.0      # clips/hr, window
        avg_rate = (n - n0) / max(1e-6, t - t0) * 3600.0            # clips/hr, since start
        eta_h = (TARGET - n) / win_rate if win_rate > 1 else float("nan")
        pct = 100.0 * n / TARGET
        bar = "#" * int(pct / 2.5) + "-" * (40 - int(pct / 2.5))
        wandb.log({"clips": n, "percent": pct, "remaining": TARGET - n,
                   "clips_per_hr": round(win_rate, 1),
                   "clips_per_hr_avg": round(avg_rate, 1),
                   "eta_hours": round(eta_h, 2) if eta_h == eta_h else 0})
        print(f"[{bar}] {n}/{TARGET} ({pct:4.1f}%)  {win_rate:5.0f}/hr  ETA {eta_h:4.2f}h", flush=True)
        prev, t_prev = n, t
        if n >= TARGET:
            print("TARGET reached", flush=True); break
        if not job_alive():
            stale = stale + 1 if n == prev else 0
            if stale >= 2:
                print("job gone + pool stable -> stop", flush=True); break
    wandb.summary["final_clips"] = count()
    run.finish()


if __name__ == "__main__":
    main()
