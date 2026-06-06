#!/bin/bash
#SBATCH --job-name=plan_divergence
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-3
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/logs/plan_divergence_%A_%a.out
#
# Action-divergence diagnostic: one SL_noNovelty_adaptivePlan rollout per seed,
# comparing the chosen action with the roll-forward ON vs OFF at each real step
# (planner runs twice/step, so ~2x a normal seed -> 20h walltime). Writes a
# per-seed JSON.
set -euo pipefail
SEED=${SLURM_ARRAY_TASK_ID}
VENV=/scratch/grmstj001/virtual-envs/act-inf-numba
REPO=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning
OUT=/scratch/grmstj001/plan_divergence_20260606
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$NUMBA_CACHE_DIR" "$OUT"
# Script is run by absolute path, so its own dir (scripts/analysis) lands on
# sys.path, not src/Python -- put the package dir on PYTHONPATH so `import sl` works.
export PYTHONPATH="$REPO/src/Python:${PYTHONPATH:-}"
cd "$REPO/src/Python"
echo "[plan_divergence] seed=$SEED node=$(hostname)"
"$VENV/bin/python" "$REPO/scripts/analysis/plan_divergence.py" "$SEED" 120 \
    "$OUT/plan_divergence_seed${SEED}.json"
