#!/bin/bash
#SBATCH --job-name=sl_jax_120trial
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/120t_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/120t_%j.err

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

echo
echo "=== sl_jax SI 5 seeds × 120 trials, h=4, GPU ==="
$PY -u <<'PYEOF'
import sys, time, os
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_batched import run_batched

grid = GridConfig.from_matlab_indices()
opts = JaxOptions(algorithm='SI', seed=0, num_trials=120, horizon=4)

print('Warming up JIT (small batch)...', flush=True)
t0 = time.time()
run_batched(grid, opts, seeds=[1, 2])
print(f'  Warmup: {time.time()-t0:.1f}s', flush=True)

# Main run
seeds = [1, 2, 3, 4, 5]
print(f'\nMain run: 5 seeds × 120 trials, h=4', flush=True)
t0 = time.time()
result = run_batched(grid, opts, seeds=seeds)
elapsed = time.time() - t0
print(f'Wall: {elapsed:.1f}s', flush=True)

# Per-seed cum_mean at multiple checkpoints
for i, s in enumerate(seeds):
    survs = result.survivals[i]
    cm30 = np.mean(survs[:30])
    cm60 = np.mean(survs[:60])
    cm120 = np.mean(survs)
    print(f'  Seed {s}: cum_mean(30)={cm30:.2f}  cum_mean(60)={cm60:.2f}  cum_mean(120)={cm120:.2f}  max={max(survs)}', flush=True)

# Pooled
all_survs = [result.survivals[i] for i in range(5)]
print(f'\nPooled cum_mean(30)  over 5 seeds: {np.mean([np.mean(s[:30]) for s in all_survs]):.2f}', flush=True)
print(f'Pooled cum_mean(60)  over 5 seeds: {np.mean([np.mean(s[:60]) for s in all_survs]):.2f}', flush=True)
print(f'Pooled cum_mean(120) over 5 seeds: {np.mean([np.mean(s) for s in all_survs]):.2f}', flush=True)

# MATLAB ref
mat_dir = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/revision_novelty_diagnosis_default_env/SI'
mat = []
for s in (1, 2, 3, 4, 5):
    f = os.path.join(mat_dir, f'SI_Seed{s}.txt')
    if os.path.exists(f):
        surv = [float(l)-1 for l in open(f).read().splitlines() if l.strip()]
        mat.append(surv)
print(f'\nMATLAB SI seeds 1-5 ref:')
print(f'  Pooled cum_mean(30)  = {np.mean([np.mean(s[:30]) for s in mat]):.2f}')
print(f'  Pooled cum_mean(60)  = {np.mean([np.mean(s[:60]) for s in mat]):.2f}')
print(f'  Pooled cum_mean(120) = {np.mean([np.mean(s[:120]) for s in mat]):.2f}')
PYEOF
