#!/usr/bin/env bash
# Quick status of the Vision Cardio download + training pipeline.
# Usage: bash scripts/vc_status.sh
set -u
DATADIR=/path/to/vision_cardio_data/scamps
POOL=/path/to/vision_cardio_data/scamps_pool
TAR="$DATADIR/scamps_videos.tar.gz"

echo "==== DOWNLOAD ===="
if pgrep -f 'aria2c.*scamps_videos' >/dev/null; then
  echo "aria2: RUNNING (pid $(pgrep -f 'aria2c.*scamps_videos' | tr '\n' ' '))"
else
  echo "aria2: NOT running"
fi
if [ -f "$TAR" ]; then
  sz=$(stat -c %s "$TAR")
  awk -v s="$sz" 'BEGIN{printf "tarball: %.2f GB / 593 GB (%.1f%% of 200GB target)\n", s/1e9, 100*s/200e9}'
fi
echo "last aria2 rate: $(grep -oE 'DL:[0-9.]+[KMG]iB' logs/aria2.log 2>/dev/null | tail -1)"

echo "==== EXTRACTOR ===="
pgrep -f 'ml.extract_growing|ml.build_pool' >/dev/null && echo "extractor: RUNNING" || echo "extractor: NOT running"
tail -1 logs/extract.log 2>/dev/null || tail -1 logs/build_pool.log 2>/dev/null

echo "==== POOL ===="
echo "clips: $(ls "$POOL"/clip_*.npy 2>/dev/null | wc -l)   (200 GB target ~= 1015 clips)"

echo "==== H100 JOBS ===="
squeue -u "$USER" 2>/dev/null || echo "(slurm unavailable)"

echo "==== LATEST CHECKPOINT ===="
ls -lt artifacts/*.pt 2>/dev/null | head -2 || echo "(none yet)"
