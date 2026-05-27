#!/bin/bash
#SBATCH --job-name=sl_perf_stat_verify
#SBATCH --output=run_perf_stat_verify_%j.out
#SBATCH --error=run_perf_stat_verify_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Statistical verification for changes that are NOT bit-exact (Phase 2+).
# Re-runs the same 10-seed x 5-trial panel and asserts mean survival is within
# 1 SE of the saved baseline. Use after editing files where FP order changes.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
echo "[run_perf_stat_verify] cwd=$(pwd) matlab_dir=$MATLAB_DIR"

cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    perf_stat_check('SI', 1:10, 5, 9, fullfile('$MATLAB_DIR','sanity','golden','stat_baseline_SI.mat')); \
    fprintf('\n'); \
    perf_stat_check('SL', 1:10, 5, 9, fullfile('$MATLAB_DIR','sanity','golden','stat_baseline_SL.mat'));"
