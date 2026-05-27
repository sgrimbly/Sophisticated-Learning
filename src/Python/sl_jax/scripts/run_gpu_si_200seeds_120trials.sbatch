#!/bin/bash
#SBATCH --job-name=sl_jax_si_panel
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/si_panel_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/si_panel_%j.err

# sl_jax SI cross-check at the same scale as the MATLAB panel:
# 200 seeds x 120 trials, horizon=5 (canonical sl_jax horizon as of 2026-05-04).
# Saves per-seed survivals to a .npz file matching the MATLAB layout for
# downstream side-by-side comparison.
#
# Time budget: at h=5 batch=5 we observed 1445 ms/trial-seed; at batch=200
# we expect ~700-1000 ms/trial-seed (parallel-vmap amortisation similar to
# h=4 where batch=5 -> batch=200 gave ~2x throughput gain). 24,000 trial-seeds
# at 1000 ms/ts = 24,000s = 6.7h main run, plus ~30 min warmup. 10h wall is
# margin for slower-than-expected scaling and JIT.

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

OUT_DIR=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/results/si_200seeds_120trials_h5
mkdir -p "$OUT_DIR"

echo
echo "=== sl_jax SI 200 seeds x 120 trials (horizon=5, canonical) ==="
$PY -u <<PYEOF
import sys, time, os
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_batched import run_batched

grid = GridConfig.from_matlab_indices()

# JIT warmup at the SAME batch size as the main run (avoids re-compile).
# Use a tiny num_trials to keep warmup cheap in wall but trigger full compile.
warmup_seeds = list(range(1, 201))
warmup_opts = JaxOptions(algorithm='SI', seed=0, num_trials=2, horizon=5)
print(f'JIT warmup at batch=200, 2 trials, h=5...', flush=True)
t0 = time.time()
run_batched(grid, warmup_opts, seeds=warmup_seeds)
print(f'  warmup: {time.time()-t0:.1f}s', flush=True)

opts = JaxOptions(algorithm='SI', seed=0, num_trials=120, horizon=5)
seeds = list(range(1, 201))
print(f'\nMain run: {len(seeds)} seeds x {opts.num_trials} trials, horizon={opts.horizon}', flush=True)
t0 = time.time()
result = run_batched(grid, opts, seeds=seeds)
elapsed = time.time() - t0
trial_seeds = opts.num_trials * len(seeds)
print(f'Wall: {elapsed:.1f}s  ({elapsed/trial_seeds*1000:.2f} ms/trial-seed)', flush=True)

survivals = np.array(result.survivals)  # shape (200, 120)
out_npz = '$OUT_DIR/survivals.npz'
np.savez(out_npz, survivals=survivals, seeds=np.array(seeds),
         num_trials=opts.num_trials, horizon=opts.horizon)
print(f'Saved survivals -> {out_npz}', flush=True)

# Summary stats
print(f'\n--- summary ---')
print(f'Pooled cum_mean(30)  = {survivals[:, :30].mean():.2f}')
print(f'Pooled cum_mean(60)  = {survivals[:, :60].mean():.2f}')
print(f'Pooled cum_mean(120) = {survivals.mean():.2f}')
print(f'Pooled max-trial = {survivals.max()}, min-trial = {survivals.min()}')
print(f'Per-seed cum_mean(120) range: [{survivals.mean(axis=1).min():.2f}, {survivals.mean(axis=1).max():.2f}]')

# Compare to MATLAB SI baseline (seeds 1..200)
print(f'\n--- vs MATLAB SI (revision_novelty_diagnosis_default_env, seeds 1..200) ---')
mat_dir = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/revision_novelty_diagnosis_default_env/SI'
mat_rows = []
missing = 0
for s in range(1, 201):
    f = os.path.join(mat_dir, f'SI_Seed{s}.txt')
    if os.path.exists(f):
        surv = [float(l) - 1 for l in open(f).read().splitlines() if l.strip()]
        if len(surv) >= 120:
            mat_rows.append(surv[:120])
        else:
            missing += 1
    else:
        missing += 1
print(f'MATLAB seeds loaded: {len(mat_rows)} (missing {missing})')
if mat_rows:
    mat = np.array(mat_rows)
    print(f'MATLAB pooled cum_mean(30)  = {mat[:, :30].mean():.2f}')
    print(f'MATLAB pooled cum_mean(60)  = {mat[:, :60].mean():.2f}')
    print(f'MATLAB pooled cum_mean(120) = {mat.mean():.2f}')
    print(f'\nDelta (sl_jax h=5 - MATLAB):')
    print(f'  cum_mean(30):  {survivals[:, :30].mean() - mat[:, :30].mean():+.3f}')
    print(f'  cum_mean(60):  {survivals[:, :60].mean() - mat[:, :60].mean():+.3f}')
    print(f'  cum_mean(120): {survivals.mean() - mat.mean():+.3f}')
PYEOF
