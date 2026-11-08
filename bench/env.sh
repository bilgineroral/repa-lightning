# Sourced by the bench scripts: sets PY to the python of conda env $ENV_NAME (default repa), unless PY is set.
ENV_NAME=${ENV_NAME:-repa}
if [ -z "${PY:-}" ]; then
  if ! command -v conda >/dev/null 2>&1; then
    for f in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
             "$HOME/miniforge3/etc/profile.d/conda.sh" /opt/conda/etc/profile.d/conda.sh; do
      if [ -f "$f" ]; then source "$f"; break; fi
    done
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "env.sh: conda not found (looked on PATH and in ~/miniconda3, ~/anaconda3, ~/miniforge3, /opt/conda). Set PY=/path/to/python." >&2
    return 1 2>/dev/null || exit 1
  fi
  _prefix=$(conda env list | awk -v n="$ENV_NAME" '$1 == n {print $NF}')
  if [ -z "$_prefix" ]; then
    echo "env.sh: conda env '$ENV_NAME' not found. Run: bash bench/setup_env.sh" >&2
    return 1 2>/dev/null || exit 1
  fi
  PY="$_prefix/bin/python"
fi
export PY ENV_NAME
