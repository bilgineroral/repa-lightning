#!/usr/bin/env bash
# Check the machine before a benchmark run: GPUs, versions, SDPA backends, FP8 support, scheduler, disk, dataset.
#   DATA_ROOT=$HOME/imagenet bash bench/preflight.sh
# Exit code 1 if something the run needs is missing (see bench/preflight.py).
set -uo pipefail
cd "$(dirname "$0")/.."
source bench/env.sh || exit 1

echo "== host: $(hostname), $(date)"
echo "== git: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD)$( [ -n "$(git status --porcelain --untracked-files=no)" ] && echo ' (uncommitted changes)')"
echo "== python: $PY"
echo "== scheduler: sbatch $(command -v sbatch || echo 'not found'), tmux $(command -v tmux || echo 'not found')"
echo "== CPUs: $(nproc)"
free -g 2>/dev/null | head -2
echo "== disk (repo and results)"
df -h . | tail -1
if [ -n "${DATA_ROOT:-}" ]; then echo "== disk (DATA_ROOT)"; df -h "$DATA_ROOT" 2>/dev/null | tail -1; fi
echo "== nvidia-smi"
nvidia-smi 2>&1 | head -25 || echo "nvidia-smi not available"

"$PY" bench/preflight.py ${DATA_ROOT:+--data-root "$DATA_ROOT"} ${PREFLIGHT_JSON:+--json "$PREFLIGHT_JSON"}
