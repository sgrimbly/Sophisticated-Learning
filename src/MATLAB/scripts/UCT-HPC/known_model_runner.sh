#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2022b

# Define the parameter ranges
declare -a ALGORITHMS=("known_large_MCT")
declare -i NUM_SEEDS=30  # Number of seeds to run for each algorithm
declare -a HORIZONS=("2")
declare -a K_FACTORS=("0.7")
declare -a MCTS=("3")
declare -a NUM_MCTS=("100")
declare -a ROOT_FOLDERS=("/media/labs" "/media/labs2")

# Base configuration
export TIME_LIMIT="72:00:00"
export MEMORY_ALLOCATION="12G"
export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# Loop over each combination of parameters
for ALGORITHM in "${ALGORITHMS[@]}"
do
    export ALGORITHM
    for ((SEED=1; SEED<=$NUM_SEEDS; SEED++))
    do
        export SEED
        for HORIZON in "${HORIZONS[@]}"
        do
            export HORIZON
            for K_FACTOR in "${K_FACTORS[@]}"
            do
                export K_FACTOR
                for MCT in "${MCTS[@]}"
                do
                    export MCT
                    for NUM_MCT in "${NUM_MCTS[@]}"
                    do
                        export NUM_MCT
                        for ROOT_FOLDER in "${ROOT_FOLDERS[@]}"
                        do
                            export ROOT_FOLDER
                            # Configure the job name to include parameter info for easier tracking
                            JOB_NAME="${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}_$(date +'%Y-%m-%d_%H-%M-%S')"
                            export JOB_NAME

                            # Prepare the SLURM script
                            SLURM_SCRIPT="submit_${JOB_NAME}.sh"
                            cp SLURM_Template.sh $SLURM_SCRIPT

                            # Replace placeholders with actual configured values
                            sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

                            # Add the MATLAB run line
                            echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', '${SEED}', '${HORIZON}', '${K_FACTOR}', '${ROOT_FOLDER}', '${MCT}', '${NUM_MCT}'); exit;\"" >> $SLURM_SCRIPT

                            # Submit the job
                            sbatch $SLURM_SCRIPT
                        done
                    done
                done
            done
        done
    done
done
