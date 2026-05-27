#!/bin/bash
#SBATCH --job-name=sl_sweep
#SBATCH --partition=ada
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#
# Slurm submission template for a paper-scale Python-vs-MATLAB comparison.
#
# Usage:
#   mkdir -p logs
#   sbatch slurm_sweep.sbatch [smoke|small|medium|paper] [/scratch/...]
#
# Time estimates (full JIT, on this 16-core allocation):
#   smoke    ~3 min          paper    ~2h45m
#   small    ~25 min         medium   ~1h25m
#
# Tune --cpus-per-task and --time accordingly. Bumping to 32 CPUs roughly
# halves wall-time. Memory is ~250 MB/worker so 24 G is plenty for 32.

set -euo pipefail

SCOPE="${1:-small}"
OUTPUT_DIR="${2:-/scratch/${USER}/sl_${SCOPE}_${SLURM_JOB_ID}}"

REPO_ROOT="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning"
SCRIPT="${REPO_ROOT}/src/Python/sl/scripts/run_paper_comparison.sh"

# Activate venv if needed.
if [ -f "${HOME}/venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/venv/bin/activate"
fi

bash "$SCRIPT" "$SCOPE" "$OUTPUT_DIR"
