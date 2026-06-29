#!/usr/bin/env bash
# One parallel-download worker = one node = one IP. Submit N copies pinned to N
# distinct nodes (different --nodelist), each with a unique WORKER_ID in
# [0, NUM_WORKERS). CPU-only (--gres=gpu:0) so it packs onto GPU-busy reserved
# nodes. Chunks land on SHARED cepheid storage where the stitcher reads them.
#SBATCH --job-name=vc-dl
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONUNBUFFERED=1

URL="${URL:?set URL}"
OUTDIR="${OUTDIR:?set OUTDIR}"
BASE_OFFSET="${BASE_OFFSET:?set BASE_OFFSET}"
TOTAL_BYTES="${TOTAL_BYTES:?set TOTAL_BYTES}"
NUM_WORKERS="${NUM_WORKERS:?set NUM_WORKERS}"
WORKER_ID="${WORKER_ID:?set WORKER_ID}"
CHUNK_MB="${CHUNK_MB:-256}"
THREADS="${THREADS:-16}"

echo "worker $WORKER_ID/$NUM_WORKERS on $(hostname) ip=$(hostname -I | awk '{print $1}')"
python3 -u -m ml.pdl_worker --url "$URL" --out-dir "$OUTDIR" \
  --base-offset "$BASE_OFFSET" --total-bytes "$TOTAL_BYTES" \
  --chunk-mb "$CHUNK_MB" --num-workers "$NUM_WORKERS" --worker-id "$WORKER_ID" \
  --threads "$THREADS"
