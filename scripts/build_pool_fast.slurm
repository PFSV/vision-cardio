#!/usr/bin/env bash
# Fast SCAMPS clip-pool builder as a CPU slurm job (NOT the login node).
# ISA-L igzip inflate + N parallel .mat->.npy workers; ~8.5x the serial
# ml/extract_growing.py. CPU-only (--gres=gpu:0) so it packs onto idle cores of
# GPU-busy nodes (224 cores / ~1.9 TB each). Single sequential gzip stream means
# the inflate is the ceiling, so ~16 workers saturate it; more cores don't help.
#SBATCH --job-name=vc-extract
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1

TARBALL="${TARBALL:-/path/to/vision_cardio_data/scamps/scamps_videos.tar.gz}"
OUTDIR="${OUTDIR:-/path/to/vision_cardio_data/scamps_pool}"
WORKERS="${WORKERS:-16}"
MAXCLIPS="${MAXCLIPS:-}"

# ISA-L lives in ~/.local (pip --user); ensure it's importable on this node.
python3 -c "import isal" 2>/dev/null || pip install --user --quiet isal || true

echo "extract on $(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} workers=$WORKERS out=$OUTDIR"
python3 -u -m ml.build_pool_fast --tarball "$TARBALL" --out-dir "$OUTDIR" \
  --workers "$WORKERS" ${MAXCLIPS:+--max-clips "$MAXCLIPS"}
echo "[done] pool: $(ls "$OUTDIR"/clip_*.npy 2>/dev/null | wc -l) clips"
