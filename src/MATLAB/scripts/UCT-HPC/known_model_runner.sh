#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2022b

# Define the parameter ranges
declare -i NUM_SEEDS=100  # Number of seeds to run for each experiment
declare -a MEMORY_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
declare -a NO_MEMORY_HORIZONS=("1" "2" "3" "4" "5")
declare -a HYBRID_HORIZONS=("1" "2" "3" "4" "5" "6")  # Total horizon shown on x-axis
declare -a PURE_MC_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")  # MCT settings for pure Monte Carlo
declare K_FACTOR="0.7"
declare ROOT_FOLDER="/home/grmstj001"
declare MCT_HYBRID="3"  # Fixed mct for hybrid if not included in x-axis values

# Base configuration
export TIME_LIMIT="7-00:00:00"
export MEMORY_ALLOCATION="4G"
export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# Submit job function
submit_job() {
    local job_name="$1"
    local horizon="$2"
    local mct="$3"
    local auto_rest="$4"

    SLURM_SCRIPT="submit_${job_name}.sh"
    cp SLURM_Template.sh $SLURM_SCRIPT

    # Replace placeholders with actual configured values
    sed -i "s|\$JOB_NAME|$job_name|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

    # Add the MATLAB run line
    echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('known_large_MCT', '${SEED}', '${horizon}', '${K_FACTOR}', '${ROOT_FOLDER}', '${mct}', '100', '${auto_rest}'); exit;\"" >> $SLURM_SCRIPT

    # Submit the job
    sbatch $SLURM_SCRIPT
}

# Loop over each seed
for ((SEED=1; SEED<=$NUM_SEEDS; SEED++))
do
    export SEED

    # Memory Experiments
    for HORIZON in "${MEMORY_HORIZONS[@]}"
    do
        submit_job "Memory_Seed${SEED}_Hor${HORIZON}" "$HORIZON" "0" "0"
    done

    # No Memory Experiments
    for HORIZON in "${NO_MEMORY_HORIZONS[@]}"
    do
        submit_job "NoMemory_Seed${SEED}_Hor${HORIZON}" "$HORIZON" "0" "1"
    done

    # Hybrid Experiments
    for HORIZON in "${HYBRID_HORIZONS[@]}"
    do
        submit_job "Hybrid_Seed${SEED}_Hor${HORIZON}" "$HORIZON" "$MCT_HYBRID" "0"
    done

    # Pure Monte Carlo Experiments
    for MCT in "${PURE_MC_HORIZONS[@]}"
    do
        submit_job "PureMC_Seed${SEED}_MCT${MCT}" "0" "$MCT" "0"
    done
done
