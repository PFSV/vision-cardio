#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CODEX_HOME="${CODEX_HOME:-$repo_root/.codex-local}"
mkdir -p "$CODEX_HOME"

exec codex -p vision_cardio -a never -C "$repo_root" "$@"
