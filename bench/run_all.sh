#!/usr/bin/env bash
# Tier 3 benchmark sequence on one GPU (usage: bench/README.md): vanilla baseline (no Tier 1 flags),
# preflight, profiles, compile modes, FP8 recipes, best-settings profile + re-rank rule, SDPA backends, real vs synthetic data.
#   bash bench/run_all.sh --gpu 0 [--data-root $HOME/imagenet]       # full run (Arcturus, RTX 6000 Ada)
#   bash bench/run_all.sh --gpu 0 --dry-run --data-root ~/repa_fake  # local check (RTX 3060)
# Without --gpu, a single-GPU CUDA_VISIBLE_DEVICES (as set by bench/launch.sh or Slurm) is used.
# Resumable: results go to bench/results/<run_id>/<step>.json; rerunning with the same --run-id (the default id is
# fixed per host, GPU and dry/full) skips finished steps. Other options: python bench/kern_runner.py --help.
# The Tier 1/2 run that used to live here is bench/run_tier12.sh.
#
# 1. bench/kern_preflight.py (standard library only): GPU model, compute capability (8.9 expected, else a warning),
#    driver version -> conda env (>= 580: repa-kern, 525-579: repa-kern-cu126), refuses a busy GPU unless --force.
# 2. bench/kern_runner.py with that env's python: the steps, a per-step timeout, a summary table at the end.
# Exit codes: 0 all ran steps ok, 1 some step failed, 2 GPU busy, 3 preflight failed (GPU / driver / env).
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="" FORCE=() ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --gpu) GPU=${2:-}; shift 2 ;;
    --gpu=*) GPU=${1#--gpu=}; shift ;;
    --force) FORCE=(--force); shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if [ -z "$GPU" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "$CUDA_VISIBLE_DEVICES" != *,* ]]; then
    GPU=$CUDA_VISIBLE_DEVICES
    echo "run_all: no --gpu, using CUDA_VISIBLE_DEVICES=$GPU"
  else
    echo "run_all: pass --gpu <index> (the physical GPU, as nvidia-smi numbers it)"; exit 3
  fi
fi

SYS_PY=$(command -v python3 || command -v python || true)
[ -n "$SYS_PY" ] || { echo "run_all: no python3 on PATH for the preflight"; exit 3; }
GATE=$(mktemp -t kern_preflight.XXXXXX.json)
trap 'rm -f "$GATE"' EXIT
"$SYS_PY" bench/kern_preflight.py --gpu "$GPU" --json "$GATE" "${FORCE[@]}"
rc=$?
[ $rc -eq 0 ] || exit $rc
PY=$("$SYS_PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['python'])" "$GATE")

# GPU-pinned: the runner gives every step CUDA_VISIBLE_DEVICES=$GPU
CUDA_VISIBLE_DEVICES=$GPU "$PY" bench/kern_runner.py --gpu "$GPU" --python "$PY" --preflight-json "$GATE" "${ARGS[@]}"
