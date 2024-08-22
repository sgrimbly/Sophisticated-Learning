#!/bin/bash

# Load the MATLAB module:
# module load software/matlab-R2022b
module load software/matlab-R2024b

# Define the parameter ranges
declare -a ALGORITHMS=("BA" "BAUCB") # "SI" "SL")
declare -a SEEDS=({0..5})
# declare -a SEEDS=(1021 1022 1023) # Test seeds for MATLAB 2024b

# Default configuration parameters
export HORIZON="6"
export K_FACTOR="1.5"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"
export TIME_LIMIT="72:00:00"
export SCRIPT_PATH="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
export JOB_TRACKING_FILE="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB/job_submissions.txt"
export MAX_SLOTS=240

# Function to check available slots and return the count
check_available_slots() {
    local num_jobs=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)
    local available_slots=$((MAX_SLOTS - num_jobs))
    echo "$available_slots"
}

# Function to wait and check slots after some time
wait_and_check_slots() {
    echo "Waiting for 5 minutes before checking available slots again..."
    sleep 300  # 5 minutes
    check_available_slots
}

# Read already submitted jobs into an associative array
declare -A submitted_jobs
if [[ -f "$JOB_TRACKING_FILE" ]]; then
    while IFS= read -r line; do
        job_key=$(echo "$line" | cut -d':' -f2 | xargs | awk -F'_' -v OFS='_' '{print $1, $2, $3, $4, $5, $6}')
        submitted_jobs["$job_key"]=1
    done < "$JOB_TRACKING_FILE"
fi

# Function to submit jobs
submit_jobs() {
    local available_slots=$1
    local submitted_jobs_count=0

    for ALGORITHM in "${ALGORITHMS[@]}"; do
        export ALGORITHM
        for SEED in "${SEEDS[@]}"; do
            export SEED
            if [[ "$submitted_jobs_count" -ge "$available_slots" ]]; then
                echo "Reached slot limit, no more submissions allowed."
                return  # Stop submitting if we reach the available slots limit
            fi
            JOB_KEY="${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}"  # Key for checking duplicates

            if [[ -n "${submitted_jobs[$JOB_KEY]}" ]]; then
                echo "Skipping already submitted job: $JOB_KEY"
                continue  # Skip if job already submitted
            fi

            DATE=$(date +'%Y-%m-%d_%H-%M-%S')
            JOB_NAME="${JOB_KEY}_${DATE}"
            echo "Preparing: $JOB_NAME"

            SLURM_SCRIPT="submit_${JOB_NAME}.sh"
            cp SLURM_Template.sh "$SLURM_SCRIPT"
            sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"
            # echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}); exit;\"" >> "$SLURM_SCRIPT"
            # I was asked to remove -nodisplay -nosplash -nodesktop -r and add -batch for the 2024b version of MATLAB.
            echo "matlab -batch \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}); exit;\"" >> "$SLURM_SCRIPT"
            output_dir="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/job_data/$ALGORITHM/$JOB_NAME"
            mkdir -p "$output_dir"

            job_submission=$(sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT")
            job_sub_id=$(echo $job_submission | awk '{print $4}')
            echo "$job_sub_id: $JOB_NAME" >> "$JOB_TRACKING_FILE"
            echo "Successfully submitted: $JOB_NAME with Job ID: $job_sub_id"
            submitted_jobs["$JOB_KEY"]=1  # Add job key to the array
            ((submitted_jobs_count++))
        done
    done
}

# Main loop to check slots and submit jobs
while true; do
    AVAILABLE_SLOTS=$(check_available_slots)
    if [ "$AVAILABLE_SLOTS" -gt 0 ]; then
        submit_jobs "$AVAILABLE_SLOTS"
    else
        echo "No available slots right now."
    fi

    # Wait if there were no slots or after a batch submission
    wait_and_check_slots
done