#!/usr/bin/env bash
# Stream-train on the WHOLE SCAMPS dataset with low disk: download a shard ->
# train on it -> delete it -> next shard, all in one H100 job. Peak disk is a
# couple of shards of .mat on node-local storage; nothing persisted but ckpts.
# Submit from repo root:  sbatch scripts/stream_train_scamps_h100.slurm
#SBATCH --job-name=vc-scamps-stream
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
# Node-local scratch for the rolling shard buffer (deleted as we go).
BUFFER_DIR="${SLURM_TMPDIR:-/tmp}/scamps_stream_${SLURM_JOB_ID:-manual}"

MAX_CLIPS="${MAX_CLIPS:-}"            # empty = whole dataset (~2800 clips)
SHARD_CLIPS="${SHARD_CLIPS:-96}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-32}"

echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) buffer=$BUFFER_DIR"
nvidia-smi -L || true
python3 -c "import torch; print('cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

EXTRA=""
[ -n "$MAX_CLIPS" ] && EXTRA="--max-clips $MAX_CLIPS"

python3 -m ml.stream_train_scamps \
  --buffer-dir "$BUFFER_DIR" \
  --representation raw \
  --clip-frames 128 --clip-size 96 \
  --shard-clips "$SHARD_CLIPS" \
  --val-clips 64 \
  --local-epochs "$LOCAL_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --lr 1e-3 \
  --ckpt-every 5 \
  --wandb-project vision_cardio \
  --wandb-run-name "scamps-stream-h100-${SLURM_JOB_ID:-manual}" \
  --out artifacts/scamps_hr_model_stream.pt \
  $EXTRA

echo "[done] -> artifacts/scamps_hr_model_stream.pt"
