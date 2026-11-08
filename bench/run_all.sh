#!/usr/bin/env bash
# The whole benchmark run on one GPU: preflight, tests, Tier 1 matrix (S and B), profiles. Usually started by
# bench/launch.sh. Results and the log go to bench/results/<timestamp>/ (bench/results/latest points to it).
#   DATA_ROOT=$HOME/imagenet bash bench/run_all.sh
# Optional env vars: MODELS ("S B"), WARMUP (20), STEPS (100), NUM_WORKERS (8), ENV_NAME / PY (see env.sh).
# Keeps going after a failed step and lists the failures at the end (exit code 1 if any).
set -uo pipefail
cd "$(dirname "$0")/.."
source bench/env.sh || exit 1

TS=$(date +%Y%m%d-%H%M%S)
OUT="bench/results/$TS"
mkdir -p "$OUT"
ln -sfn "$TS" bench/results/latest
LOG="$OUT/run_all.log"
exec > >(tee -a "$LOG") 2>&1

MODELS=${MODELS:-"S B"}
WARMUP=${WARMUP:-20}
STEPS=${STEPS:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
FAILURES=()

step() { # step <name> <command...>: run, log, record failure, never abort
  local name=$1; shift
  echo; echo "########## $name ($(date +%H:%M:%S))"
  if "$@"; then echo "########## $name: ok"; return 0; fi
  echo "########## $name: FAILED"; FAILURES+=("$name"); return 1
}

echo "run_all: results in $OUT, python $PY, DATA_ROOT=${DATA_ROOT:-unset}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# 1. preflight; a bad DATA_ROOT turns off the real-data runs instead of stopping everything
REAL_DATA=${DATA_ROOT:+1}
if ! PREFLIGHT_JSON="$OUT/preflight.json" step preflight bash bench/preflight.sh; then
  if [ -n "${DATA_ROOT:-}" ] && "$PY" -c "
import json, sys
p = json.load(open('$OUT/preflight.json'))['problems']
sys.exit(0 if any(x.startswith('DATA_ROOT') for x in p) else 1)" 2>/dev/null; then
    echo "DATA_ROOT check failed: skipping real-data runs"; REAL_DATA=""
  fi
fi
DATAS="synthetic${REAL_DATA:+ real}"

# 2. tests
step pytest "$PY" -m pytest tests/ -q -p no:cacheprovider

# 3. batch size per model: the config's, or the largest that fits (halving), checked for baseline and all flags.
#    Gradient accumulation keeps the config's global batch on one GPU.
pick_batch() { # pick_batch <S|B> -> prints "batch accum"
  local model=$1 cfg_batch global b
  read -r cfg_batch global < <("$PY" -c "
import sys; sys.path.insert(0, '.')
from config import PretrainConfig, PretrainConfigSmall
C = {'S': PretrainConfigSmall, 'B': PretrainConfig}['$model']
print(C.batch_size, C.global_batch_size)")
  b=$cfg_batch
  while [ "$b" -ge 8 ]; do
    if "$PY" bench/step_time.py --model "$model" --flags none --batch "$b" --optimizer --warmup 2 --steps 2 \
         --out "$OUT/probe" >"$OUT/probe_${model}_b${b}_none.log" 2>&1 \
       && "$PY" bench/step_time.py --model "$model" --flags all --batch "$b" --optimizer --warmup 2 --steps 2 \
         --out "$OUT/probe" >"$OUT/probe_${model}_b${b}_all.log" 2>&1 \
       && "$PY" -c "
import glob, json, sys
# some drivers (e.g. WSL) spill to host memory instead of failing: then it 'fits' but runs many times slower
runs = [json.load(open(f)) for f in glob.glob('$OUT/probe/${model}_*_b${b}_*.json')]
over = [r for r in runs if r['peak_mem_gib'] > 0.95 * r['gpu_mem_total_gib']]
for r in over:
    print(f\"peak {r['peak_mem_gib']:.1f} GiB > 95% of {r['gpu_mem_total_gib']:.1f} GiB GPU memory\", file=sys.stderr)
sys.exit(1 if over or not runs else 0)"; then
      echo "$b $(( global / b ))"; return 0
    fi
    echo "model $model: batch $b does not fit or failed (see $OUT/probe_${model}_b${b}_*.log)" >&2
    b=$(( b / 2 ))
  done
  return 1
}

declare -A BATCH ACCUM
for m in $MODELS; do
  if res=$(pick_batch "$m"); then
    read -r BATCH[$m] ACCUM[$m] <<<"$res"
    echo "model $m: batch ${BATCH[$m]}, gradient accumulation ${ACCUM[$m]}" | tee -a "$OUT/batch_sizes.txt"
  else
    echo "model $m: no batch size fits"; FAILURES+=("batch probe $m")
  fi
done

# 4. Tier 1 matrix
for m in $MODELS; do
  [ -n "${BATCH[$m]:-}" ] || continue
  step "tier1 $m" env MODELS="$m" DATAS="$DATAS" DATA_ROOT="${DATA_ROOT:-}" BATCH_S="${BATCH[$m]}" BATCH_B="${BATCH[$m]}" \
    ACCUM_S="${ACCUM[$m]}" ACCUM_B="${ACCUM[$m]}" WARMUP="$WARMUP" STEPS="$STEPS" NUM_WORKERS="$NUM_WORKERS" \
    OUT="$OUT/tier1" PY="$PY" bash bench/run_tier1.sh
done

# 5. profiles: baseline vs all flags
for m in $MODELS; do
  [ -n "${BATCH[$m]:-}" ] || continue
  for f in none all; do
    step "profile $m $f" "$PY" bench/profile_step.py --model "$m" --flags "$f" --batch "${BATCH[$m]}" --warmup "$WARMUP" \
      --steps 10 --out "$OUT/profiles"
  done
done

# 6. FP8 (Tier 2): only if preflight found a GPU with compute capability >= 8.9
if "$PY" -c "import json, sys; sys.exit(0 if json.load(open('$OUT/preflight.json')).get('fp8_possible') else 1)" 2>/dev/null; then
  for m in $MODELS; do
    [ -n "${BATCH[$m]:-}" ] || continue
    for data in $DATAS; do
      if [ "$data" = "real" ]; then extra=(--data-root "$DATA_ROOT" --num-workers "$NUM_WORKERS" --optimizer --accum "${ACCUM[$m]}");
      else extra=(--no-optimizer); fi
      for f in fp8 all,fp8; do
        step "fp8 $m $data [$f]" "$PY" bench/step_time.py --model "$m" --flags "$f" --batch "${BATCH[$m]}" --data "$data" \
          --warmup "$WARMUP" --steps "$STEPS" --out "$OUT/tier1" "${extra[@]}"
      done
    done
  done
else
  echo; echo "########## fp8: skipped (preflight: GPU capability < 8.9, or preflight.json missing; see $OUT/preflight.json)"
fi

# summary
step summarize "$PY" bench/summarize.py --results "$OUT/tier1" --out "$OUT/summary.md"

echo; echo "########## done ($(date +%H:%M:%S)): results in $OUT"
if [ ${#FAILURES[@]} -gt 0 ]; then
  echo "failed steps (${#FAILURES[@]}):"; printf '  - %s\n' "${FAILURES[@]}"; exit 1
fi
echo "all steps ok"
