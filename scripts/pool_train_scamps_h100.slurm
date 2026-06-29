#!/usr/bin/env bash
# Throughput training on a full 8xH100 node. Streams a pool of clips onto
# node-local NVMe, then trains many AMP epochs over the resident pool so the
# GPUs stay compute-bound instead of idling on the ~14 MiB/s download.
# Submit from repo root:  sbatch scripts/pool_train_scamps_h100.slurm
#SBATCH --job-name=vc-scamps-pool
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail

BASE_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"
cd "$BASE_DIR"
mkdir -p logs artifacts
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

# Pick the fastest node-local scratch for the pool (ephemeral; not cepheid quota).
if mkdir -p "/raid/$USER/scamps_pool_${SLURM_JOB_ID:-manual}" 2>/dev/null; then
  BUF="/raid/$USER/scamps_pool_${SLURM_JOB_ID:-manual}"
elif [ -n "${SLURM_TMPDIR:-}" ]; then
  BUF="$SLURM_TMPDIR/scamps_pool"
else
  BUF="/tmp/scamps_pool_${SLURM_JOB_ID:-manual}"
fi
trap 'rm -rf "$BUF"' EXIT   # always free node-local scratch when the job ends

POOL_CLIPS="${POOL_CLIPS:-1200}"
MAX_EPOCHS="${MAX_EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WIDTH="${WIDTH:-64}"
VAL_CLIPS="${VAL_CLIPS:-8}"      # tiny warmup so GPUs engage fast
MIN_POOL="${MIN_POOL:-8}"        # start training after just 8 pool clips
CLIP_FRAMES="${CLIP_FRAMES:-128}"
CLIP_SIZE="${CLIP_SIZE:-112}"

echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) buffer=$BUF"
df -h "$(dirname "$BUF")" | tail -1
nvidia-smi -L || true
python3 -c "import torch;print('cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())"

python3 -m ml.pool_train_scamps \
  --buffer-dir "$BUF" \
  --pool-clips "$POOL_CLIPS" \
  --val-clips "$VAL_CLIPS" \
  --min-pool "$MIN_POOL" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --arch video --width "$WIDTH" \
  --clip-frames "$CLIP_FRAMES" --clip-size "$CLIP_SIZE" \
  --amp 1 \
  --lr 2e-3 \
  --ckpt-every 5 \
  --wandb-project vision_cardio \
  --wandb-run-name "scamps-pool-h100-${SLURM_JOB_ID:-manual}" \
  --out artifacts/scamps_hr_model_real.pt

echo "[done] -> artifacts/scamps_hr_model_real.pt"
