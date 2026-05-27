#!/bin/bash
#SBATCH --job-name=sl_perf_verify
#SBATCH --output=run_perf_verify_%j.out
#SBATCH --error=run_perf_verify_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Re-run the canonical SI+SL configs and assert the per-step CSV plus the
# survived array match the golden trace. Mode is taken from $PERF_MODE
# (default 'exact'). Also reports ms/step.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
MODE="${PERF_MODE:-exact}"
echo "[run_perf_verify] cwd=$(pwd) matlab_dir=$MATLAB_DIR mode=$MODE"

cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    perf_check_golden_trace('${MODE}', {'SI','SL'}, 3, 1, 9); \
    fprintf('\n--- timing ---\n'); \
    perf_time_step({'SI','SL'}, 2, 1, 9);"
