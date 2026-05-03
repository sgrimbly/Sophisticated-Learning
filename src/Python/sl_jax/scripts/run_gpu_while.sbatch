#!/bin/bash
#SBATCH --job-name=sl_jax_gpu_while
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/while_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/while_%j.err

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

echo
echo "=== sl_jax_while (lax.while_loop, batch-level early termination) on GPU ==="
$PY -u <<'PYEOF'
import sys, time
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_jax_while import run_while_batched

grid = GridConfig.from_matlab_indices()

# Warmup
print('Warmup compile (batch=4, 1 trial)...', flush=True)
opts = JaxOptions(algorithm='SI', seed=0, num_trials=1, horizon=4)
t0 = time.time()
run_while_batched(grid, opts, seeds=[1, 2, 3, 4])
print(f'  Warmup: {time.time()-t0:.1f}s', flush=True)

# Benchmark sweep
for batch in (8, 32, 64, 128, 200):
    opts = JaxOptions(algorithm='SI', seed=0, num_trials=30, horizon=4)
    seeds = list(range(1, batch+1))
    t0 = time.time()
    result = run_while_batched(grid, opts, seeds=seeds)
    elapsed = time.time() - t0
    trial_seeds = 30 * batch
    means = [np.mean(s) for s in result.survivals]
    print(f'batch={batch}: {elapsed:.1f}s ({elapsed/trial_seeds*1000:.1f} ms/trial-seed)  pooled={np.mean(means):.2f}', flush=True)
PYEOF
