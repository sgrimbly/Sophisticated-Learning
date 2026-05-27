#!/bin/bash
#SBATCH --job-name=sl_perf_full_panel
#SBATCH --output=run_perf_full_panel_%j.out
#SBATCH --error=run_perf_full_panel_%j.err
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Final-stage sanity check from the rollout plan: confirm SI / SL / BA / BAUCB
# all still behave on the full panel after the tree-search edits. Runs each
# algorithm over 10 seeds x 10 trials and prints survival summary stats so the
# user can eyeball any regression vs prior runs.
#
# This DOES NOT compare to a stored baseline — the modular runner's RNG and
# config differ from perf_run_canonical's so the numbers aren't directly
# diff-able with stat_baseline_*.mat. The point is to surface gross behavioural
# regressions in BA / BAUCB that the SI/SL-only verifies would miss.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
echo "[run_perf_full_panel] cwd=$(pwd) matlab_dir=$MATLAB_DIR"

cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    fprintf('--- SI panel (10 seeds x 10 trials) ---\n'); \
    perf_stat_check('SI', 1:10, 10, 9, ''); \
    fprintf('\n--- SL panel ---\n'); \
    perf_stat_check('SL', 1:10, 10, 9, ''); \
    fprintf('\n--- BA panel ---\n'); \
    perf_stat_check('BA', 1:10, 10, 9, ''); \
    fprintf('\n--- BAUCB panel ---\n'); \
    perf_stat_check('BAUCB', 1:10, 10, 9, '');"
