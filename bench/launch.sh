#!/usr/bin/env bash
# Start bench/run_all.sh in the background: a Slurm job if sbatch exists, else a tmux session, else nohup.
#   DATA_ROOT=$HOME/imagenet bash bench/launch.sh
# ---- edit these if needed ----
GPUS=1                       # the benchmarks use one GPU
TIME_LIMIT="24:00:00"        # Slurm time limit
PARTITION=""                 # Slurm partition, empty = cluster default
GPU=${GPU:-0}                # without Slurm: which GPU to use (check nvidia-smi first)
JOB_NAME="repa-bench"
LAUNCH_LOG_DIR="bench/results/launch"
# ------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p "$LAUNCH_LOG_DIR"
[ -n "${DATA_ROOT:-}" ] || echo "WARNING: DATA_ROOT is not set: only synthetic-data runs will happen"

if command -v sbatch >/dev/null 2>&1; then
  out=$(sbatch --job-name="$JOB_NAME" --gres="gpu:$GPUS" --time="$TIME_LIMIT" ${PARTITION:+--partition="$PARTITION"} \
         --output="$LAUNCH_LOG_DIR/%x-%j.log" --export=ALL,DATA_ROOT="${DATA_ROOT:-}" --wrap="bash bench/run_all.sh")
  echo "$out"
  job=${out##* }
  echo "Check progress:"
  echo "  squeue -u $USER"
  echo "  tail -f $LAUNCH_LOG_DIR/$JOB_NAME-$job.log        # or bench/results/latest/run_all.log once it starts"
  echo "Cancel: scancel $job"
  exit 0
fi

# no Slurm: run on GPU $GPU, but not if something else is already using it
busy=$(nvidia-smi --id="$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
if [ -n "$busy" ] && [ -z "${FORCE:-}" ]; then
  echo "GPU $GPU is in use by:"; echo "$busy"
  echo "Pick a free GPU with GPU=<id>, or set FORCE=1 to start anyway (timings will be affected)."
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
  exit 1
fi
log="$LAUNCH_LOG_DIR/$JOB_NAME-$(date +%Y%m%d-%H%M%S).log"
passthru=""; for v in MODELS WARMUP STEPS NUM_WORKERS ENV_NAME PY; do # tmux sessions don't inherit the caller's env
  [ -n "${!v:-}" ] && passthru+="$v='${!v}' "
done
cmd="CUDA_VISIBLE_DEVICES=$GPU DATA_ROOT='${DATA_ROOT:-}' $passthru bash bench/run_all.sh > '$log' 2>&1"
if command -v tmux >/dev/null 2>&1; then
  session="$JOB_NAME-$(date +%H%M%S)"
  tmux new-session -d -s "$session" "cd '$PWD' && $cmd"
  echo "Started in tmux session '$session' on GPU $GPU."
  echo "Check progress:"
  echo "  tail -f $log          # or bench/results/latest/run_all.log"
  echo "  tmux attach -t $session   (detach again with Ctrl-b d)"
else
  nohup bash -c "$cmd" >/dev/null 2>&1 &
  echo "Started with nohup (pid $!) on GPU $GPU."
  echo "Check progress:"
  echo "  tail -f $log          # or bench/results/latest/run_all.log"
  echo "  ps -p $!"
fi
