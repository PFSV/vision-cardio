#!/usr/bin/env bash
# Train the SCAMPS HR model on a single H100 node (raw 3D-conv path).
# Submit from the repo root:  sbatch scripts/train_scamps_h100.slurm
#SBATCH --job-name=vc-scamps
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

BASE_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
cd "$BASE_DIR"
mkdir -p logs artifacts

export PYTHONUNBUFFERED=1
# wandb auth comes from ~/.netrc (shared home); set WANDB_API_KEY here to override.

DATA_ROOT="${DATA_ROOT:-data/scamps/full}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-48}"

echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) data=$DATA_ROOT"
nvidia-smi || true
python3 -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

python3 -m ml.train_scamps \
  --root "$DATA_ROOT" \
  --representation raw \
  --clip-frames 128 --clip-size 96 \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --wandb-project vision_cardio \
  --wandb-run-name "scamps-raw-h100-${SLURM_JOB_ID:-manual}" \
  --out artifacts/scamps_hr_model_real.pt

echo "[done] -> artifacts/scamps_hr_model_real.pt"
