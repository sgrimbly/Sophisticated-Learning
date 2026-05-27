#!/bin/bash
#SBATCH --job-name=sl_jax_hsweep
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/hsweep_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/hsweep_%j.err

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

echo
echo "=== sl_jax SI horizon sweep: h=4,5,6 — 5 seeds x 120 trials each ==="
$PY -u <<'PYEOF'
import sys, time, os
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_batched import run_batched

grid = GridConfig.from_matlab_indices()
SEEDS = [1, 2, 3, 4, 5]

results = {}

for h in (4, 5, 6):
    print(f'\n========== HORIZON h={h} (5^{h}={5**h} policies) ==========', flush=True)

    opts = JaxOptions(algorithm='SI', seed=0, num_trials=120, horizon=h)

    print(f'  Warmup JIT (batch=2, full 120 trials)...', flush=True)
    t0 = time.time()
    _ = run_batched(grid, opts, seeds=[1, 2])
    print(f'    Warmup: {time.time()-t0:.1f}s', flush=True)

    print(f'  Main run: 5 seeds x 120 trials, h={h}', flush=True)
    t0 = time.time()
    result = run_batched(grid, opts, seeds=SEEDS)
    elapsed = time.time() - t0
    print(f'    Wall: {elapsed:.1f}s ({elapsed/(5*120)*1000:.1f} ms/trial-seed)', flush=True)

    cm30s, cm60s, cm120s, maxs = [], [], [], []
    for i, s in enumerate(SEEDS):
        survs = result.survivals[i]
        cm30 = np.mean(survs[:30])
        cm60 = np.mean(survs[:60])
        cm120 = np.mean(survs)
        cm30s.append(cm30); cm60s.append(cm60); cm120s.append(cm120); maxs.append(max(survs))
        print(f'    Seed {s}: cm30={cm30:.2f}  cm60={cm60:.2f}  cm120={cm120:.2f}  max={max(survs)}', flush=True)

    results[h] = (np.mean(cm30s), np.mean(cm60s), np.mean(cm120s))
    print(f'  Pooled: cm30={results[h][0]:.2f}  cm60={results[h][1]:.2f}  cm120={results[h][2]:.2f}', flush=True)

# MATLAB ref
mat_dir = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/revision_novelty_diagnosis_default_env/SI'
mat = []
for s in SEEDS:
    f = os.path.join(mat_dir, f'SI_Seed{s}.txt')
    if os.path.exists(f):
        surv = [float(l)-1 for l in open(f).read().splitlines() if l.strip()]
        mat.append(surv)
mat_cm30 = np.mean([np.mean(s[:30]) for s in mat])
mat_cm60 = np.mean([np.mean(s[:60]) for s in mat])
mat_cm120 = np.mean([np.mean(s[:120]) for s in mat])

print(f'\n========== SUMMARY ==========')
print(f'{"h":>3}  {"cm30":>7}  {"cm60":>7}  {"cm120":>7}  {"d30":>6}  {"d60":>6}  {"d120":>6}')
for h in (4, 5, 6):
    cm30, cm60, cm120 = results[h]
    print(f'{h:>3}  {cm30:>7.2f}  {cm60:>7.2f}  {cm120:>7.2f}  {cm30-mat_cm30:>+6.2f}  {cm60-mat_cm60:>+6.2f}  {cm120-mat_cm120:>+6.2f}')
print(f'MAT  {mat_cm30:>7.2f}  {mat_cm60:>7.2f}  {mat_cm120:>7.2f}  (h adaptive <=9)')
PYEOF
