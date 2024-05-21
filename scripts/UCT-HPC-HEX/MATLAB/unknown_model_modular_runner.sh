#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2024a

# Define the parameter ranges
declare -a ALGORITHMS=("SI" "SL" "BA" "BAUCB")
declare -a SEEDS=(1 2 3 ... 30)  # Example range of seeds, adjust as needed

# Parameters (Adjust as necessary)
export HORIZON="2"
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"
export TIME_LIMIT="72:00:00"
export MEMORY_ALLOCATION="3G"
export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# Read grid configurations from a file and store them in an array
mapfile -t GRID_CONFIGS < "${SCRIPT_PATH}/grid_configs.txt"

# Prepare and run experiments for each configuration and algorithm
for config in "${GRID_CONFIGS[@]}"; do
    IFS=',' read -ra PARAMS <<< "$config"
    GRID_SIZE=${PARAMS[0]}
    HORIZON=${PARAMS[1]}
    HILL_POS=${PARAMS[2]}
    START_POSITION=${PARAMS[3]}
    FOOD_SOURCES=${PARAMS[4]}
    WATER_SOURCES=${PARAMS[5]}
    SLEEP_SOURCES=${PARAMS[6]}

    for ALGORITHM in "${ALGORITHMS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            DATE=$(date +'%Y-%m-%d_%H-%M-%S')
            JOB_NAME="${ALGORITHM}_Seed${SEED}_Grid${GRID_SIZE}_Hor${HORIZON}_${DATE}"
            export JOB_NAME  # Ensure JOB_NAME is exported

            # Prepare the SLURM script
            SLURM_SCRIPT="submit_${JOB_NAME}.sh"
            cp SLURM_Template.sh $SLURM_SCRIPT

            # Replace placeholders with actual configured values
            sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

            # Add the MATLAB run line, modify to include all parameters and file paths
            echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}, false, '', ${GRID_SIZE}, ${START_POSITION}, ${HILL_POS}, '${FOOD_SOURCES}', '${WATER_SOURCES}', '${SLEEP_SOURCES}'); exit;\"" >> $SLURM_SCRIPT

            # Create experiment log directory with descriptive name
            output_dir="$ROOT_FOLDER/results/$ALGORITHM/$JOB_NAME"

            mkdir -p "$output_dir"

            # Submit the job
            sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" $SLURM_SCRIPT
        done
    done
done
