#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2024a

# Define the parameter ranges
declare -a ALGORITHMS=("SI" "SL" "BA" "BAUCB")
declare -a MISSING_SI_SEEDS=() # All necessary SI seeds from 1 to 130 are accounted for
declare -a MISSING_SL_SEEDS=() # All necessary SL seeds from 1 to 130 are accounted for
declare -a MISSING_BA_SEEDS=() # Adjusted to include only remaining BA seeds
declare -a MISSING_BAUCB_SEEDS=(111 128 129) # Remaining seeds for BAUCB to run

# These params are not used (yet) for unknown model algos as of 30/4/2024
export HORIZON="2"
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"  # Adjust as necessary

# Base configuration
export TIME_LIMIT="72:00:00"
export MEMORY_ALLOCATION="4G"
export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    case "$ALGORITHM" in
        "SI")
            SEEDS_TO_RUN="${MISSING_SI_SEEDS[@]}"
            ;;
        "SL")
            SEEDS_TO_RUN="${MISSING_SL_SEEDS[@]}"
            ;;
        "BA")
            SEEDS_TO_RUN="${MISSING_BA_SEEDS[@]}"
            ;;
        "BAUCB")
            SEEDS_TO_RUN="${MISSING_BAUCB_SEEDS[@]}"
            ;;
    esac

    for SEED in $SEEDS_TO_RUN
    do
        # Configure the job name to include parameter info for easier tracking
        export ALGORITHM  # Export the ALGORITHM for each job
        export SEED       # Export the SEED for each job
        DATE=$(date +'%Y-%m-%d_%H-%M-%S')
        JOB_NAME="${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}_$DATE"
        export JOB_NAME   # Ensure JOB_NAME is also exported

        # Prepare the SLURM script
        SLURM_SCRIPT="submit_${JOB_NAME}.sh"
        cp SLURM_Template.sh $SLURM_SCRIPT

        # Replace placeholders with actual configured values
        sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

        # Add the MATLAB run line
        echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', '${SEED}', '${HORIZON}', '${K_FACTOR}', '${ROOT_FOLDER}', '${MCT}', '${NUM_MCT}'); exit;\"" >> $SLURM_SCRIPT

        # Create experiment log directory with descriptive name
        output_dir=~/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/job_data/$ALGORITHM/$JOB_NAME

        mkdir -p "$output_dir"

        # Submit the job
        sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" $SLURM_SCRIPT
    done
done
