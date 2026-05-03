#!/bin/bash
#SBATCH --job-name=sl_jax_gpu_largebatch
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/largebatch_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/largebatch_%j.err

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

for BATCH in 16 32 64 128 200; do
  echo
  echo "=== sl_jax batched $BATCH seeds × 30 trials, horizon=4, GPU ==="
  $PY -u <<PYEOF
import sys, time
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_batched import run_batched
grid = GridConfig.from_matlab_indices()
opts = JaxOptions(algorithm='SI', seed=0, num_trials=30, horizon=4)
seeds = list(range(1, $BATCH + 1))
t0 = time.time()
result = run_batched(grid, opts, seeds=seeds)
elapsed = time.time() - t0
trial_seeds = 30 * len(seeds)
print(f'Wall: {elapsed:.1f}s   ({elapsed/trial_seeds*1000:.1f} ms/trial-seed)')
all_means = [np.mean(s) for s in result.survivals]
print(f'Pooled cum_mean(30) over {len(seeds)} seeds: {np.mean(all_means):.2f}')
print(f'Per-seed: {[round(m, 1) for m in all_means]}')
PYEOF
done
