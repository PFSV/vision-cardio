#!/usr/bin/env bash
# Detached auto-fire: wait until the SCAMPS pool is full (or the extract job
# ends), then launch the real 8xH100 DDP retrain. Runs as nohup so it survives
# the Claude session ending. Fresh init (NOT --resume: old ckpt overfit 254
# clips), finite EPOCHS for a clean wandb finish, writes a NEW checkpoint so the
# old one is preserved.
set -u
POOL=/path/to/vision_cardio_data/scamps_pool
JID=132429
REPO=/path/to/ByeongYeok_RnD_NLP/TUTORIAL_DRILL/neuro-nlp/vision_cardio

while true; do
  n=$(ls "$POOL"/clip_*.npy 2>/dev/null | wc -l)
  st=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
  echo "$(date +%H:%M:%S) pool=$n extract=${st:-DONE}"
  if [ "$n" -ge 2790 ] || [ -z "$st" ]; then break; fi
  sleep 30
done

cd "$REPO" || exit 2
export EPOCHS=300 WIDTH=128 OUT=artifacts/scamps_hr_model_full.pt
echo "FIRING retrain at pool=$(ls "$POOL"/clip_*.npy 2>/dev/null | wc -l) $(date)"
sbatch --reservation=reflex --account=idle_gpu scripts/ddp_train_pool_h100.slurm \
  || sbatch --reservation=reflex scripts/ddp_train_pool_h100.slurm \
  || sbatch scripts/ddp_train_pool_h100.slurm
echo "sbatch chain exit=$? $(date)"
