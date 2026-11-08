#!/usr/bin/env bash
# Tier 1 run matrix: baseline, each perf flag alone, and all flags; per model and data source. One GPU.
# Synthetic data: forward + backward only. Real data: optimizer step included, real dataloader.
#
#   DATA_ROOT=/path/to/imagenet bash bench/run_tier1.sh            # S and B, synthetic and real
#   MODELS=S DATAS=synthetic bash bench/run_tier1.sh               # subset
#
# Env vars (defaults): MODELS ("S B"), DATAS ("synthetic real"), DATA_ROOT (none: real runs are skipped),
# BATCH_S (256), BATCH_B (128), WARMUP (20), STEPS (100), NUM_WORKERS (8), ACCUM (1; or ACCUM_S / ACCUM_B per model),
# OUT (bench/results), PY (python)
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
MODELS=${MODELS:-"S B"}
DATAS=${DATAS:-"synthetic real"}
BATCH_S=${BATCH_S:-256}
BATCH_B=${BATCH_B:-128}
WARMUP=${WARMUP:-20}
STEPS=${STEPS:-100}
NUM_WORKERS=${NUM_WORKERS:-8}
ACCUM=${ACCUM:-1}
ACCUM_S=${ACCUM_S:-$ACCUM}
ACCUM_B=${ACCUM_B:-$ACCUM}
OUT=${OUT:-bench/results}
FLAG_SETS="none skip_last_content causal_no_mask query_proj_once no_cpu_sync uint8_images fused_adamw ddp_static_graph all"

failed=0
for model in $MODELS; do
  batch=$([ "$model" = "S" ] && echo "$BATCH_S" || echo "$BATCH_B")
  accum=$([ "$model" = "S" ] && echo "$ACCUM_S" || echo "$ACCUM_B")
  for data in $DATAS; do
    if [ "$data" = "real" ] && [ -z "${DATA_ROOT:-}" ]; then
      echo "skipping real-data runs for $model: set DATA_ROOT"; continue
    fi
    if [ "$data" = "real" ]; then
      extra=(--data-root "$DATA_ROOT" --num-workers "$NUM_WORKERS" --optimizer --accum "$accum")
    else
      extra=(--no-optimizer)
    fi
    for flags in $FLAG_SETS; do
      echo "=== $model $data [$flags] batch $batch"
      if ! "$PY" bench/step_time.py --model "$model" --flags "$flags" --batch "$batch" --data "$data" \
          --warmup "$WARMUP" --steps "$STEPS" --out "$OUT" "${extra[@]}"; then
        echo "FAILED: $model $data [$flags]"; failed=$((failed + 1))
      fi
    done
  done
done

"$PY" bench/summarize.py --results "$OUT"
echo "failed runs: $failed"
exit $(( failed > 0 ))
