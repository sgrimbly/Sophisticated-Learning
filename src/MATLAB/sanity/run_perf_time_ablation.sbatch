#!/bin/bash
#SBATCH --job-name=sl_perf_ablation
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/logs/perf_time_ablation_%j.out
#SBATCH --partition=ada
#SBATCH --account=maths
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#
# Per-variant MATLAB ms/step for the canonical-8 ablation set, single-thread
# BLAS, 5 timed trials after a warm-up (matches the existing perf_time_step
# methodology). Prints `ms_per_step` per variant to the log; parsed by
# scripts/analysis/replication_table.py. Timing is RNG-agnostic, so the
# default twister is fine even though the survival reference used threefry.

set -euo pipefail

module load software/matlab-R2024b

MATLAB_DIR=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB
cd "$MATLAB_DIR"

matlab -batch "addpath(genpath('$MATLAB_DIR')); \
    perf_time_step({'SI','SI_noNovelty','SI_novelty_smooth','SI_smooth_noNovelty', \
                    'SL_adaptivePlan','SL_noNovelty_adaptivePlan', \
                    'SL_noSmooth_adaptivePlan','SL_noNovelty_noSmooth_adaptivePlan'}, 5, 1, 9);"
