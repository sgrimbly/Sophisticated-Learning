#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SEED_START="${SEED_START:-1}"
export SEED_END="${SEED_END:-5}"
export MAX_SLOTS="${MAX_SLOTS:-40}"
export POLL_SECONDS="${POLL_SECONDS:-30}"
export RESULTS_ROOT_OVERRIDE="${RESULTS_ROOT_OVERRIDE:-/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/revision_novelty_diagnosis_default_env_smoke}"
export RUN_LABEL_OVERRIDE="${RUN_LABEL_OVERRIDE:-revision_novelty_diagnosis_smoke_defaultenv_h9_t120_s1-5}"

"${SCRIPT_DIR}/revision_novelty_diagnosis_runner.sh"
