#!/usr/bin/env bash
set -uo pipefail

# Determine project root (parent directory of this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT" || exit 1

# old conda
#source /home/cl403/miniforge3/bin/activate
#conda activate pylos-env

# new uv environment
source .venv/bin/activate

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export SYSTEM=local

mkdir -p logs

experiments=(
  "exp_GBM_no_stack_no_bag"
  "exp_GBM_no_stack_bag"
  "exp_GBM_stack_bag"
  "exp_TABM_no_stack_no_bag"
  "exp_TABM_no_stack_bag"
  "exp_TABM_stack_bag"
)

for exp in "${experiments[@]}"; do
  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_file="logs/${exp}_${timestamp}.log"

  echo "==================================================" | tee -a "$log_file"
  echo "Starting experiment: $exp" | tee -a "$log_file"
  echo "Start time: $(date)" | tee -a "$log_file"
  echo "Host: $(hostname)" | tee -a "$log_file"
  echo "Project root: $PROJECT_ROOT" | tee -a "$log_file"
  echo "==================================================" | tee -a "$log_file"

  if python src/models/train_models_experiment.py --experiment-name "$exp" 2>&1 | tee -a "$log_file"; then
    echo "Finished experiment: $exp at $(date)" | tee -a "$log_file"
  else
    echo "FAILED experiment: $exp at $(date)" | tee -a "$log_file"
  fi

  echo "" | tee -a "$log_file"
done