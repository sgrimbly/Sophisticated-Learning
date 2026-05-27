#!/bin/bash
#SBATCH --job-name=sl_perf_stat_baseline
#SBATCH --output=run_perf_stat_baseline_%j.out
#SBATCH --error=run_perf_stat_baseline_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Capture a multi-seed statistical baseline: survival mean +/- SE across
# 10 seeds x 5 trials @ horizon=9. Saved under sanity/golden/stat_baseline_*.mat.
# Re-run with the same script to verify a candidate change is within 1 SE of
# this baseline (perf_stat_check infers capture-vs-check by file presence).

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
echo "[run_perf_stat_baseline] cwd=$(pwd) matlab_dir=$MATLAB_DIR"

cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    perf_stat_check('SI', 1:10, 5, 9, ''); \
    fprintf('\n'); \
    perf_stat_check('SL', 1:10, 5, 9, '');"
