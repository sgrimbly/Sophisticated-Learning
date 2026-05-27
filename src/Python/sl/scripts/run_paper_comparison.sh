#!/usr/bin/env bash
# run_paper_comparison.sh
#
# Run a Python-vs-MATLAB comparison sweep across SI / SL / BA / BAUCB,
# matching the canonical paper config (default 10×10 grid, horizon=9,
# 120 trials per seed). Tunable scope:
#
#   smoke    :  5 seeds × 30 trials  ×  4 algos =  600 trials
#   small    : 30 seeds × 120 trials × 4 algos = 14,400 trials
#   medium   : 100 seeds × 120 trials × 4 algos = 48,000 trials
#   paper    : 200 seeds × 120 trials × 4 algos = 96,000 trials
#
# Time estimates (full JIT, multiprocessing on a 16-core Slurm node, JIT
# cache warm; SL is the long pole). Wall-time is dominated by the slowest
# algorithm × seed combination.
#
#                        smoke   small    medium    paper
#   wall-time (16 cores)  ~3 m   ~25 m   ~1h 25m   ~2h 45m
#   wall-time (32 cores)  ~2 m   ~15 m     ~50 m   ~1h 40m
#
# Usage:
#   bash run_paper_comparison.sh smoke   /scratch/<user>/sl_smoke
#   bash run_paper_comparison.sh small   /scratch/<user>/sl_small
#   bash run_paper_comparison.sh medium  /scratch/<user>/sl_medium
#   bash run_paper_comparison.sh paper   /scratch/<user>/sl_paper
#
# After the sweep, plot results with:
#   python -m sl.scripts.plot_results --input-dir <output_dir> \
#       --include-matlab --algorithms SI SL BA BAUCB

set -euo pipefail

# Pin BLAS thread counts to 1 — multiprocessing.Pool spawns N workers and
# we don't want each to also spawn N OpenMP threads (16-way oversubscribe
# on a 4-core slot). The matmuls inside the JIT planner are small enough
# that single-threaded BLAS is faster than oversubscribed multi-threaded.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SCOPE="${1:-smoke}"
OUTPUT_DIR="${2:-./sl_sweep_$(date +%Y%m%d_%H%M%S)}"

case "$SCOPE" in
    smoke)
        SEEDS="1-5"
        TRIALS=30
        ;;
    small)
        SEEDS="1-30"
        TRIALS=120
        ;;
    medium)
        SEEDS="1-100"
        TRIALS=120
        ;;
    paper)
        SEEDS="1-200"
        TRIALS=120
        ;;
    *)
        echo "Unknown scope '$SCOPE'. Choose: smoke / small / medium / paper" >&2
        exit 2
        ;;
esac

WORKERS="${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-$(nproc)}}"

REPO_ROOT="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning"
PYTHON_DIR="${REPO_ROOT}/src/Python"

cd "${PYTHON_DIR}"

echo "========================================"
echo "Paper comparison sweep"
echo "  scope     : $SCOPE  ($SEEDS, $TRIALS trials)"
echo "  output    : $OUTPUT_DIR"
echo "  workers   : $WORKERS"
echo "  algorithms: SI SL BA BAUCB"
echo "========================================"
mkdir -p "$OUTPUT_DIR"

python -m sl.sweep \
    --algorithms SI SL BA BAUCB \
    --seeds "$SEEDS" \
    --num-trials "$TRIALS" \
    --max-horizon 9 \
    --max-steps 100 \
    --output-dir "$OUTPUT_DIR" \
    --workers "$WORKERS" \
    --use-jit-planner \
    --skip-existing \
    --grid-id "default10"

echo ""
echo "Sweep complete. Plotting..."
python -m sl.scripts.plot_results \
    --input-dir "$OUTPUT_DIR" \
    --algorithms SI SL BA BAUCB \
    --include-matlab \
    --n-trials "$TRIALS" \
    --output-dir "$OUTPUT_DIR/plots"

echo ""
echo "Done. Plots in $OUTPUT_DIR/plots/"
