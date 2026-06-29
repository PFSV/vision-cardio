#!/usr/bin/env bash
set -euo pipefail

python3 "$(dirname "$0")/exercise_coach_policy.py" \
  --current-hr-bpm 132 \
  --resting-hr-bpm 68 \
  --hr-confidence 0.91 \
  --recovery-score 0.78 \
  --rhr-trend-bpm -1.5 \
  --completed-last-session
