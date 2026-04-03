#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# User settings
# -----------------------------
CONDA_ENV="poetry-env"
SYSTEM_NAME="${SYSTEM:-local}"   # override by exporting SYSTEM before running if desired
SLEEP_BETWEEN_RUNS_SEC=10        # be nice to shared filesystems

# Experiments in recommended order:
# 1) GBM baselines first (fast, validates pipeline)
# 2) TABM next (can fail due to deps/CUDA)
# 3) Optional: stack-without-bag last (least interpretable)
EXPERIMENTS=(
  "exp_GBM_no_stack_no_bag"
  "exp_GBM_no_stack_bag"
  "exp_GBM_stack_bag"

  "exp_TABM_no_stack_no_bag"
  "exp_TABM_no_stack_bag"
  "exp_TABM_stack_bag"

  # Optional / only if you really want these:
  # "exp_GBM_stack_no_bag"
  # "exp_TABM_stack_no_bag"
)

# -----------------------------
# Helpers
# -----------------------------
log() { echo -e "\n[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# -----------------------------
# Activate conda env
# -----------------------------
log "Activating conda env: ${CONDA_ENV}"

# Ensure conda is available in non-interactive shells
if ! command -v conda >/dev/null 2>&1; then
  log "ERROR: conda not found in PATH. Run: source ~/anaconda3/etc/profile.d/conda.sh (or your conda.sh path)"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# -----------------------------
# Set environment variables
# -----------------------------
export PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export SYSTEM="${SYSTEM_NAME}"

log "PROJECT_ROOT=${PROJECT_ROOT}"
log "SYSTEM=${SYSTEM}"
log "PYTHONPATH=${PYTHONPATH}"

# -----------------------------
# Run experiments
# -----------------------------
for exp in "${EXPERIMENTS[@]}"; do
  log "Starting experiment: ${exp}"
  python3 src/models/train_models_experiment.py --experiment-name "${exp}"
  log "Finished experiment: ${exp}"

  if [[ "${SLEEP_BETWEEN_RUNS_SEC}" -gt 0 ]]; then
    log "Sleeping ${SLEEP_BETWEEN_RUNS_SEC}s..."
    sleep "${SLEEP_BETWEEN_RUNS_SEC}"
  fi
done

log "All experiments completed successfully."