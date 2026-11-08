#!/usr/bin/env bash
# Architecture-flag speed runs (these CHANGE the model; speed only, no accuracy): each arch flag alone and with all
# Tier 1 perf flags, plus the two baselines, per model. Synthetic data, optimizer on. One GPU.
#
#   MODELS=S BATCH_S=128 ACCUM_S=32 bash bench/run_arch.sh
#
# Env vars (defaults): MODELS ("S B"), BATCH_S (256), BATCH_B (128), ACCUM_S / ACCUM_B (1), WARMUP (20), STEPS (100),
# OUT (bench/results), PY (python), ARCHS (the list below)
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
MODELS=${MODELS:-"S B"}
BATCH_S=${BATCH_S:-256}
BATCH_B=${BATCH_B:-128}
ACCUM_S=${ACCUM_S:-1}
ACCUM_B=${ACCUM_B:-1}
WARMUP=${WARMUP:-20}
STEPS=${STEPS:-100}
OUT=${OUT:-bench/results}
ARCHS=${ARCHS:-"none single_stream_ref query_top_k=6 query_top_k=4 single_stream_target_pos query_mlp_width=2d query_mlp_width=d train_res=160 predict_frac=0.5 predict_frac=0.25"}

failed=0
for model in $MODELS; do
  batch=$([ "$model" = "S" ] && echo "$BATCH_S" || echo "$BATCH_B")
  accum=$([ "$model" = "S" ] && echo "$ACCUM_S" || echo "$ACCUM_B")
  for arch in $ARCHS; do
    for perf in none all; do
      echo "=== $model arch [$arch] perf [$perf] batch $batch"
      if ! "$PY" bench/step_time.py --model "$model" --arch "$arch" --flags "$perf" --batch "$batch" --data synthetic \
          --optimizer --accum "$accum" --warmup "$WARMUP" --steps "$STEPS" --out "$OUT"; then
        echo "FAILED: $model arch [$arch] perf [$perf]"; failed=$((failed + 1))
      fi
    done
  done
done

"$PY" bench/summarize.py --results "$OUT"
echo "failed runs: $failed"
exit $(( failed > 0 ))
