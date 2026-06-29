"""Optuna hyperparameter search over ml.ddp_train_pool.

Runs inside ONE 8-GPU slurm job (holds the node for the whole study). Each trial
subprocess-launches a SHORT DDP training with sampled hyperparams; the objective
is the best val MAE that trial reaches. TPE (Bayesian) sampler picks the next
config. After the study the best params are written to artifacts/hpo_best.json.

NOTE: clip-size/clip-frames are NOT searched here — the pool .npy are baked at
112x112x128. If every trial floors near the predict-the-mean baseline (~31 bpm),
the bottleneck is the data representation / architecture, not these knobs.

Run via scripts/hpo_search.slurm.
"""
from __future__ import annotations
import json, os, re, subprocess

import optuna

POOL = os.environ.get("POOL_DIR", "/path/to/vision_cardio_data/scamps_pool")
GPUS = int(os.environ.get("GPUS", "8"))
TRIAL_EPOCHS = int(os.environ.get("TRIAL_EPOCHS", "40"))
N_TRIALS = int(os.environ.get("N_TRIALS", "20"))


def run_trial(num: int, p: dict) -> float:
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={GPUS}", "-m", "ml.ddp_train_pool",
           "--pool-dir", POOL, "--epochs", str(TRIAL_EPOCHS), "--arch", "video", "--amp", "1",
           "--ckpt-every", "100000",                       # don't litter checkpoints
           "--warmup-epochs", str(p["warmup"]),
           "--lr", str(p["lr"]), "--width", str(p["width"]),
           "--weight-decay", str(p["wd"]), "--sigma-bpm", str(p["sigma"]),
           "--batch-size", str(p["bs"]), "--num-workers", "8",
           "--wandb-project", "vision_cardio", "--wandb-run-name", f"hpo-t{num}",
           "--out", f"artifacts/hpo_t{num}.pt"]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    maes = [float(m) for m in re.findall(r'"mae": ([0-9.]+)', r.stdout)]
    best = min(maes) if maes else 999.0
    print(f"[trial {num}] params={p} best_mae={best:.3f} epochs_seen={len(maes)}", flush=True)
    if not maes:
        print("[trial stderr tail]", r.stderr[-1000:], flush=True)
    return best


def objective(t: optuna.Trial) -> float:
    p = {
        "lr": t.suggest_float("lr", 1e-4, 2e-3, log=True),
        "width": t.suggest_categorical("width", [32, 64, 96]),
        "wd": t.suggest_float("wd", 1e-5, 1e-1, log=True),
        "sigma": t.suggest_float("sigma", 1.0, 8.0),
        "bs": t.suggest_categorical("bs", [16, 32, 64]),
        "warmup": t.suggest_int("warmup", 0, 10),
    }
    return run_trial(t.number, p)


def main() -> None:
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    print(f"[hpo] start: {N_TRIALS} trials x {TRIAL_EPOCHS} epochs on {GPUS} GPUs", flush=True)
    study.optimize(objective, n_trials=N_TRIALS)
    best = {"params": study.best_params, "mae": study.best_value,
            "trial_epochs": TRIAL_EPOCHS, "n_trials": N_TRIALS}
    with open("artifacts/hpo_best.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"[hpo] BEST_MAE={study.best_value:.3f} BEST_PARAMS={json.dumps(study.best_params)}", flush=True)
    # leaderboard
    rows = sorted(((t.value, t.number, t.params) for t in study.trials if t.value is not None))
    for v, n, pr in rows[:5]:
        print(f"[hpo] #{n} mae={v:.3f} {json.dumps(pr)}", flush=True)


if __name__ == "__main__":
    main()
