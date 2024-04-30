#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2022b

# Define the parameter ranges
declare -a ALGORITHMS=("SL" "SI" "BA" "BAUCB")
declare -i NUM_SEEDS=30  # Number of seeds to run for each algorithm

# Thsee params are not used (yet) for unknown model algos as of 30/4/2024
export HORIZON="2"
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/media/labs"  # Adjust as necessary

# Base configuration
export JOB_NAME="main_run"
export TIME_LIMIT="72:00:00"
export MEMORY_ALLOCATION="9G"
export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# Loop over each combination of parameters
for ALGORITHM in "${ALGORITHMS[@]}"
do
    for ((SEED=1; SEED<=$NUM_SEEDS; SEED++))
    do
        
        # Configure the job name to include parameter info for easier tracking
        export JOB_NAME="${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}"

        # Prepare the SLURM script
        SLURM_SCRIPT="submit_${JOB_NAME}.sh"
        cp SLURM_Template.sh $SLURM_SCRIPT

        # Replace placeholders with actual configured values
        sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

        # Add the MATLAB run line
        echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', '${SEED}', '${HORIZON}', '${K_FACTOR}', '${ROOT_FOLDER}', '${MCT}', '${NUM_MCT}'); exit;\"" >> $SLURM_SCRIPT

        # Create experiment log directory
        output_dir=~/MATLAB-experiments/experiments/$(date +'%Y-%m-%d_%H-%M-%S')
        mkdir -p "$output_dir"

        # Submit the job
        sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" $SLURM_SCRIPT

    done
done
