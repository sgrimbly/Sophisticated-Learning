#!/bin/bash
#SBATCH --job-name=sl_jax_max_batch
#SBATCH --partition=l40s
#SBATCH --account=l40sfree
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/maxbatch_%j.out
#SBATCH --error=/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs/maxbatch_%j.err

set -e
mkdir -p /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/sl_jax/scripts/logs

export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/grmstj001/virtual-envs/act-inf-gpu/bin/python

cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python

echo "=== JAX device info ==="
$PY -c "import jax; print('default_backend:', jax.default_backend()); print('devices:', jax.devices())"

# Use agent_batched (the proven best). Push batch sizes to find the
# memory ceiling on a single L40S (48GB VRAM).
echo
echo "=== Push batch up: 200, 400, 800, 1600 ==="
$PY -u <<'PYEOF'
import sys, time, traceback
sys.path.insert(0, '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python')
import numpy as np
from sl_jax.config import GridConfig, JaxOptions
from sl_jax.agent_batched import run_batched

grid = GridConfig.from_matlab_indices()

# We just need 5 trials to test — the question is "does it fit and run?"
# not "what's the cum_mean".
NUM_TRIALS = 5

for batch in (200, 400, 800, 1600):
    opts = JaxOptions(algorithm='SI', seed=0, num_trials=NUM_TRIALS, horizon=4)
    seeds = list(range(1, batch + 1))
    print(f'--- batch={batch} ({batch * NUM_TRIALS} trial-seeds) ---', flush=True)
    try:
        t0 = time.time()
        result = run_batched(grid, opts, seeds=seeds)
        elapsed = time.time() - t0
        ts = batch * NUM_TRIALS
        means = [np.mean(s) for s in result.survivals]
        print(f'  OK: {elapsed:.1f}s  ({elapsed/ts*1000:.1f} ms/trial-seed)  pooled cum_mean={np.mean(means):.2f}', flush=True)
    except Exception as e:
        print(f'  FAIL @ batch={batch}: {type(e).__name__}: {str(e)[:200]}', flush=True)
        # Stop here — bigger won't fit.
        break
    # Optional: report GPU mem
    try:
        import jax
        for d in jax.devices():
            mem = d.memory_stats() if hasattr(d, 'memory_stats') else {}
            if mem:
                print(f'  GPU memory: peak_used={mem.get("peak_bytes_in_use", 0)/1e9:.2f} GB '
                      f'limit={mem.get("bytes_limit", 0)/1e9:.2f} GB', flush=True)
    except Exception:
        pass
PYEOF
