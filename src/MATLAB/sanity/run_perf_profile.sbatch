#!/bin/bash
#SBATCH --job-name=sl_perf_profile
#SBATCH --output=run_perf_profile_%j.out
#SBATCH --error=run_perf_profile_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Profile SI and SL with MATLAB's profiler. Prints top functions by total
# and self time. Replaces the audit-by-inspection that overestimated some
# function-overhead optimisations in the 2026-05-01 rollout.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
echo "[run_perf_profile] cwd=$(pwd) matlab_dir=$MATLAB_DIR"

cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    perf_profile('SI', 5, 1, 9); \
    fprintf('\n\n'); \
    perf_profile('SL', 5, 1, 9);"
