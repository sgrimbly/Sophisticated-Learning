#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the MATLAB module (if needed)
module load software/matlab-R2024b

# Parameters
declare -a ALGORITHMS=("model_free_RL")
declare -a SEEDS=({11..30})  # Seeds 1 to 100
export ROOT_FOLDER="/home/grmstj001"

# SLURM Configuration
export TIME_LIMIT="72:00:00"  # Adjust time limit based on expected runtime
export MEMORY_ALLOCATION="8G"  # Increase if needed
declare SCRIPT_PATH="${HOME}/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB"
declare MAIN_PATH="${HOME}/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
export SL_RESULTS_ROOT="${ROOT_FOLDER}/MATLAB-experiments/Sophisticated-Learning/results"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    export ALGORITHM
    for SEED in "${SEEDS[@]}"
    do
        export SEED
        # Create a unique job name for easy tracking
        DATE=$(date +'%Y-%m-%d_%H-%M-%S')
        JOB_NAME="${ALGORITHM}_Seed${SEED}_$DATE"

        # Copy and modify the SLURM template
        SLURM_SCRIPT="submit_${JOB_NAME}.sh"
        cp "${SCRIPT_DIR}/SLURM_Template.sh" "$SLURM_SCRIPT"

        # Replace placeholders in the SLURM script
        sed -i "s|\$JOB_NAME|$JOB_NAME|g; \
                s|\$TIME_LIMIT|$TIME_LIMIT|g; \
                s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; \
                s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"

        # Add the MATLAB command to run main.m with your parameters
        # echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${MAIN_PATH}')); main('${ALGORITHM}', '${SEED}', '${HORIZON}', '${K_FACTOR}', '${ROOT_FOLDER}', '${MCT}', '${NUM_MCT}'); exit;\"" >> $SLURM_SCRIPT
        # echo "Running MATLAB command:"
        # echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${MAIN_PATH}')); main('${ALGORITHM}', '${SEED}', '1000', '1.5', '${ROOT_FOLDER}', '500', '10', 'false'); exit;\""
        echo "matlab -batch \"addpath(genpath('${MAIN_PATH}')); main('${ALGORITHM}', ${SEED}, 1000, 1.5, '${ROOT_FOLDER}', 500, 10, false); exit;\"" >> "$SLURM_SCRIPT"

        # Create an output directory for experiment logs
        output_dir=~/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/job_data/$ALGORITHM/$JOB_NAME
        mkdir -p "$output_dir"

        # Submit the job
        sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT"
    done
done
