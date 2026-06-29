#!/usr/bin/env bash
# PhysNet rPPG waveform trainer (neg-Pearson) on 8xH100 DDP. Pairs the frame pool
# with the PPG-waveform labels (ml/extract_ppg_labels.py). Replaces the failed
# HR-bin classifier (notes/POSTMORTEM_rppg_3dconv.md).
#SBATCH --job-name=vc-rppg
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

POOL_DIR="${POOL_DIR:-/path/to/vision_cardio_data/scamps_pool}"
PPG_DIR="${PPG_DIR:-/path/to/vision_cardio_data/scamps_pool_ppg}"
GPUS="${GPUS:-8}"
EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WIDTH="${WIDTH:-32}"
LR="${LR:-1e-3}"
OUT="${OUT:-artifacts/rppg_physnet.pt}"

echo "job=${SLURM_JOB_ID:-manual} host=$(hostname) gpus=$GPUS"
echo "pairs ~= $(comm -12 <(ls "$POOL_DIR" | grep -oE 'clip_[0-9]+' | sort -u) <(ls "$PPG_DIR" 2>/dev/null | grep -oE 'ppg_[0-9]+' | sed 's/ppg_/clip_/' | sort -u) | wc -l)"

torchrun --standalone --nproc_per_node="$GPUS" -m ml.train_rppg \
  --pool-dir "$POOL_DIR" --ppg-dir "$PPG_DIR" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --width "$WIDTH" --lr "$LR" \
  --num-workers 8 --amp 1 --ckpt-every 5 \
  --wandb-project vision_cardio --wandb-run-name "rppg-physnet-${SLURM_JOB_ID:-manual}" \
  --out "$OUT"
echo "[done] -> $OUT"
