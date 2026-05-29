#!/bin/bash
#SBATCH --job-name=sl_replication
#SBATCH --account=maths
#SBATCH --partition=ada
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=1-50
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/logs/replication_%x_%A_%a.out
#
# Canonical-8 ablation replication: Python (Numba JIT) vs the MATLAB diagnosis
# batch, horizon=9, 120 trials. ONE (variant, seed) per array task -- mirrors
# the MATLAB diagnosis batch's 1600-job structure. Full 120-trial runs are
# heavy (late trials have deep search): SI ~1.4 h/seed, adaptive variants
# more, so each task gets its own multi-hour walltime. Single worker => clean
# per-variant elapsed_seconds (= runtime parity), no oversubscription.
#
# Submit one array per variant; tune --time/--array per cost tier:
#   for A in SI SI_noNovelty SI_novelty_smooth SI_smooth_noNovelty \
#            SL_noSmooth_adaptivePlan SL_noNovelty_noSmooth_adaptivePlan; do
#     sbatch --job-name=rep_$A --time=04:00:00 --array=1-50 \
#       --export=ALL,SL_ALGO=$A,REPLICATION_OUT=/scratch/grmstj001/replication_20260527/python_sweep \
#       run_replication_sweep.sbatch
#   done
#   # heavy:
#   for A in SL_adaptivePlan SL_noNovelty_adaptivePlan; do
#     sbatch --job-name=rep_$A --time=12:00:00 --array=1-50 --export=ALL,SL_ALGO=$A,... ; done

set -euo pipefail

ALGO=${SL_ALGO:?set SL_ALGO (e.g. --export=ALL,SL_ALGO=SI)}
SEED=${SLURM_ARRAY_TASK_ID}
VENV=/scratch/grmstj001/virtual-envs/act-inf-numba
REPO=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning
OUT=${REPLICATION_OUT:-/scratch/grmstj001/replication_$(date +%Y%m%d)/python_sweep}

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
# Per-task, node-local JIT cache: a SHARED NFS cache dir races across the
# array (OSError: Stale file handle on the cache index). One dir per task
# costs only a ~2-4s recompile, which is negligible vs the ~1h+ runtime.
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$NUMBA_CACHE_DIR" "$OUT"

cd "$REPO/src/Python"
echo "[replication] algo=$ALGO seed=$SEED out=$OUT node=$(hostname)"

"$VENV/bin/python" -m sl.sweep \
    --algorithms "$ALGO" \
    --seeds "${SEED}-${SEED}" \
    --num-trials 120 --max-horizon 9 \
    --use-jit-planner --workers 1 \
    --grid-id default10 \
    --output-dir "$OUT" \
    --skip-existing
