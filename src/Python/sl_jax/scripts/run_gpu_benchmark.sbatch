#!/bin/bash
#SBATCH --job-name=sl_jax_gpu_bench
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/gpu_bench_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/gpu_bench_%j.err

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
# Don't set JAX_PLATFORMS=cpu — let JAX find GPU
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Use the act-inf-gpu venv which should have jax-cuda configured.
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

echo
echo "=== sl_jax single-seed SI horizon=4, 30 trials, GPU ==="
$PY -u <<'PYEOF'
import sys, time
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent import run
grid = GridConfig.from_matlab_indices()
opts = JaxOptions(algorithm='SI', seed=1, num_trials=30, horizon=4)
t0 = time.time()
result = run(grid, opts)
elapsed = time.time() - t0
survs = [tr.t_terminal for tr in result.trials]
print(f'Total: {elapsed:.1f}s = {elapsed/30:.2f}s/trial')
print(f'Survivals: {survs}')
print(f'cum_mean(30): {np.mean(survs):.2f}')
print('CPU baseline (this venv): ~9.2s/trial, cum_mean(30)=24.67')
PYEOF

echo
echo "=== sl_jax batched 8 seeds × 30 trials, horizon=4, GPU ==="
$PY -u <<'PYEOF'
import sys, time
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_batched import run_batched
grid = GridConfig.from_matlab_indices()
opts = JaxOptions(algorithm='SI', seed=0, num_trials=30, horizon=4)
seeds = list(range(1, 9))
t0 = time.time()
result = run_batched(grid, opts, seeds=seeds)
elapsed = time.time() - t0
print(f'Wall: {elapsed:.1f}s ({elapsed/(30*len(seeds)):.3f}s/trial-seed)')
for s, survs in zip(result.seeds, result.survivals):
    print(f'  Seed {s}: cum_mean(30)={np.mean(survs):.2f} max={max(survs)}')
all_means = [np.mean(s) for s in result.survivals]
print(f'Pooled cum_mean(30) over {len(seeds)} seeds: {np.mean(all_means):.2f}')
PYEOF
