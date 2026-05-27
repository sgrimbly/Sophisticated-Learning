#!/bin/bash
#SBATCH --job-name=sl_perf_time
#SBATCH --output=run_perf_time_%j.out
#SBATCH --error=run_perf_time_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Production-style ms/step measurement (no profiler overhead).
# Single-thread BLAS; 5 trials post-warmup so total wall time is more stable
# than a 2-trial run.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
echo "[run_perf_time] cwd=$(pwd) matlab_dir=$MATLAB_DIR"
cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    perf_time_step({'SI','SL'}, 5, 1, 9);"
