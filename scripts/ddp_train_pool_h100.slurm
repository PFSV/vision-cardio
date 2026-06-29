#!/usr/bin/env bash
# Real 8xH100 DDP training over the PERSISTENT cepheid clip pool that
# ml.build_pool keeps filling. One process per GPU via torchrun (NCCL),
# re-scanning the pool each epoch so the run absorbs newly-downloaded clips.
# Submit from repo root:  sbatch scripts/ddp_train_pool_h100.slurm
#SBATCH --job-name=vc-ddp
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
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8
# Large model near the 80 GB ceiling fragments the caching allocator -> a big
# contiguous alloc fails even with GBs free (observed OOM at 44/80 GB used).
# expandable_segments lets the allocator grow segments instead of fragmenting.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Persistent, shared pool (NOT node-local scratch) so download progress and
# checkpoints survive across jobs.
POOL_DIR="${POOL_DIR:-/path/to/vision_cardio_data/scamps_pool}"
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"      # per GPU
NUM_WORKERS="${NUM_WORKERS:-8}"
WIDTH="${WIDTH:-64}"
CLIP_FRAMES="${CLIP_FRAMES:-128}"
CLIP_SIZE="${CLIP_SIZE:-112}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"   # raise to fight overfit (e.g. 0.05)
OUT="${OUT:-artifacts/scamps_hr_model_real.pt}"

echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) pool=$POOL_DIR gpus=$GPUS"
nvidia-smi -L || true
python3 -c "import torch;print('cuda',torch.cuda.is_available(),'gpus',torch.cuda.device_count())"
echo "pool clips at launch: $(ls "$POOL_DIR"/clip_*.npy 2>/dev/null | wc -l)"

torchrun --standalone --nproc_per_node="$GPUS" -m ml.ddp_train_pool \
  --pool-dir "$POOL_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --arch video --width "$WIDTH" \
  --clip-frames "$CLIP_FRAMES" --clip-size "$CLIP_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --amp 1 \
  --ckpt-every 5 \
  --wandb-project vision_cardio \
  --wandb-run-name "scamps-ddp-h100-${SLURM_JOB_ID:-manual}" \
  --out "$OUT"

echo "[done] -> artifacts/scamps_hr_model_real.pt"
