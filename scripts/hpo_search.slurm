#!/usr/bin/env bash
# Optuna HPO over ml.ddp_train_pool, held in ONE 8-GPU job. Sequential short
# trials; best params -> artifacts/hpo_best.json. ~N_TRIALS x TRIAL_EPOCHS.
#SBATCH --job-name=vc-hpo
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
export POOL_DIR="${POOL_DIR:-/path/to/vision_cardio_data/scamps_pool}"
export GPUS=8 TRIAL_EPOCHS="${TRIAL_EPOCHS:-40}" N_TRIALS="${N_TRIALS:-20}"

python3 -c "import optuna" 2>/dev/null || pip install --user --quiet optuna
echo "hpo on $(hostname): $N_TRIALS trials x $TRIAL_EPOCHS epochs, $GPUS GPUs, pool=$(ls "$POOL_DIR"/clip_*.npy 2>/dev/null | wc -l) clips"
python3 -u -m ml.hpo_search
echo "[done] best -> artifacts/hpo_best.json"
