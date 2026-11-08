#!/usr/bin/env bash
# Create the conda env for the benchmarks, or report on it if it already exists. Safe to run again.
#   bash bench/setup_env.sh                  # env "repa" from environment.yml, plus pytest and torchao
#   ENV_NAME=repa-bench bash bench/setup_env.sh   # use another name (e.g. if "repa" belongs to someone else)
# An existing env is never modified: missing packages are reported with the command to add them.
set -uo pipefail
cd "$(dirname "$0")/.."
ENV_NAME=${ENV_NAME:-repa}

if ! command -v conda >/dev/null 2>&1; then
  for f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "$HOME/miniforge3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
    if [ -f "$f" ]; then source "$f"; break; fi
  done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found: install Miniconda or load the conda module first"; exit 1; }

report() {
  "$1" - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0], "at", sys.executable)
missing = []
for m in ["torch", "torchvision", "pytorch_lightning", "transformers", "timm", "pytest", "torchao"]:
    try:
        mod = importlib.import_module(m)
        extra = f" (CUDA {mod.version.cuda}, GPU available: {mod.cuda.is_available()})" if m == "torch" else ""
        print(f"  {m:18s} {getattr(mod, '__version__', '?')}{extra}")
    except Exception as e:
        print(f"  {m:18s} MISSING ({type(e).__name__})")
        missing.append(m)
sys.exit(3 if missing else 0)
EOF
}

prefix=$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')
if [ -n "$prefix" ]; then
  echo "conda env '$ENV_NAME' already exists at $prefix; not changing it."
  report "$prefix/bin/python"
  rc=$?
  if [ $rc -eq 3 ]; then
    echo "Some packages are missing. If this env is yours, add them with:"
    echo "  $prefix/bin/python -m pip install pytest torchao"
    echo "If it isn't yours, make a separate one: ENV_NAME=repa-bench bash bench/setup_env.sh (then pass ENV_NAME=repa-bench to the other scripts)"
  fi
  exit $rc
fi

echo "Creating conda env '$ENV_NAME' from environment.yml"
if ! conda env create -f environment.yml -n "$ENV_NAME"; then
  # locally the first attempt once failed with "python=3.14.8 not available" and the retry succeeded
  echo "env create failed; retrying once"
  conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1
  conda env create -f environment.yml -n "$ENV_NAME" || { echo "conda env create failed twice; see the errors above"; exit 1; }
fi
prefix=$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')
"$prefix/bin/python" -m pip install pytest torchao || { echo "pip install pytest torchao failed"; exit 1; }
report "$prefix/bin/python"
