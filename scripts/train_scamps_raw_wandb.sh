#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running this script}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONPATH="/tmp/h5py_only${PYTHONPATH:+:$PYTHONPATH}" \
WANDB_DIR="${WANDB_DIR:-/tmp/wandb}" \
python3 -m ml.train_scamps \
  --root /NAS2/hsy/vision_cardio/data/scamps_example/scamps_videos_example \
  --limit 10 \
  --epochs 1 \
  --batch-size 8 \
  --representation raw \
  --clip-frames 320 \
  --clip-size 160 \
  --wandb-project vision_cardio \
  --wandb-run-name scamps-raw-gpu-b8-c320 \
  --out artifacts/scamps_hr_model_gpu_raw_wandb.pt
