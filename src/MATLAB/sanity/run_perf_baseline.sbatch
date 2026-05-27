#!/bin/bash
#SBATCH --job-name=sl_perf_baseline
#SBATCH --output=run_perf_baseline_%j.out
#SBATCH --error=run_perf_baseline_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Capture the per-step golden trace and a baseline ms/step reading for SI and SL
# under the canonical local-dashboard config. Run this ONCE on the baseline
# commit, commit the resulting sanity/golden/* files, then optimise.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
echo "[run_perf_baseline] cwd=$(pwd)  matlab_dir=$MATLAB_DIR"
ls "$MATLAB_DIR/sanity/" | head

cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    fprintf('which perf_capture_golden_trace: %s\n', which('perf_capture_golden_trace')); \
    perf_capture_golden_trace({'SI','SL'}, 3, 1, 9); \
    fprintf('\n--- baseline timing ---\n'); \
    perf_time_step({'SI','SL'}, 2, 1, 9);"
