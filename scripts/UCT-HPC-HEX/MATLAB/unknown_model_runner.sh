#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2022b

# Define the parameter ranges
declare -a ALGORITHMS=("SI" "SL")  # You can add more algorithms here
declare -a SEEDS=({121..300})  # Adjust the range of seeds as needed

# Default configuration parameters
export HORIZON="2"
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"
export TIME_LIMIT="72:00:00"
export SCRIPT_PATH="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
export JOB_TRACKING_FILE="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB/job_submissions.txt"

# Function to check available slots and return the count
check_available_slots() {
    local num_jobs=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)
    local available_slots=$((240 - num_jobs))
    echo "$available_slots"
}

# Function to wait and check slots after some time
wait_and_check_slots() {
    echo "Waiting for 5 minutes before checking available slots again..."
    sleep 300  # 5 minutes
    check_available_slots
}

# Function to submit jobs
submit_jobs() {
    local available_slots=$1
    local submitted_jobs=0

    for ALGORITHM in "${ALGORITHMS[@]}"; do
        export ALGORITHM
        for SEED in "${SEEDS[@]}"; do
            export SEED
            DATE=$(date +'%Y-%m-%d_%H-%M-%S')
            JOB_NAME="${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}_${DATE}"
            SLURM_SCRIPT="submit_${JOB_NAME}.sh"
            cp SLURM_Template.sh "$SLURM_SCRIPT"
            sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"

            echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}); exit;\"" >> "$SLURM_SCRIPT"
            output_dir="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/job_data/$ALGORITHM/$JOB_NAME"
            mkdir -p "$output_dir"

            job_submission=$(sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT")
            job_sub_id=$(echo $job_submission | awk '{print $4}')
            echo "$job_sub_id: $JOB_NAME" >> "$JOB_TRACKING_FILE"
            echo "Submitted: $JOB_NAME with Job ID: $job_sub_id"
            ((submitted_jobs++))
        done
    done
}

# Main loop to check slots and submit jobs
while true; do
    AVAILABLE_SLOTS=$(check_available_slots)
    if [ "$AVAILABLE_SLOTS" -gt 0 ]; then
        submit_jobs "$AVAILABLE_SLOTS"
    fi

    # Wait if there were no slots or after a batch submission
    wait_and_check_slots
done
