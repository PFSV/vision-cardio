#!/usr/bin/env bash
set -u
PPGDIR=/path/to/vision_cardio_data/scamps_pool_ppg
REPO=/path/to/ByeongYeok_RnD_NLP/TUTORIAL_DRILL/neuro-nlp/vision_cardio
while true; do
  n=$(ls "$PPGDIR"/ppg_*.npy 2>/dev/null | wc -l)
  st=$(squeue -j 132482 -h -o '%T' 2>/dev/null)
  echo "$(date +%H:%M:%S) ppg=$n extract=${st:-DONE}"
  if [ "$n" -ge 2700 ] || [ -z "$st" ]; then break; fi
  sleep 60
done
cd "$REPO" || exit 2
echo "FIRING rppg train at ppg=$(ls "$PPGDIR"/ppg_*.npy 2>/dev/null|wc -l) $(date)"
# small model -> 4 GPUs is plenty and fits the free slots; fall back to 2.
GPUS=4 EPOCHS=120 WIDTH=32 BATCH_SIZE=16 OUT=artifacts/rppg_physnet.pt \
  sbatch --gres=gpu:4 scripts/train_rppg_h100.slurm \
  || GPUS=2 EPOCHS=120 WIDTH=32 BATCH_SIZE=16 OUT=artifacts/rppg_physnet.pt \
     sbatch --gres=gpu:2 scripts/train_rppg_h100.slurm
echo "sbatch exit=$? $(date)"
