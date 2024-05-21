#!/bin/bash

# Load the MATLAB module:
module load software/matlab-R2022b

# Define the parameter ranges
declare -a ALGORITHMS=("SI" "SL")  # Adjust algorithms as needed based on MATLAB function requirements
declare -a SEEDS=({1..30})  # Range of seeds

# Parameters
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"
export TIME_LIMIT="72:00:00"
export SCRIPT_PATH="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
export JOB_TRACKING_FILE="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB/grid_configs_job_submissions.txt"

# Read grid configurations from a file
mapfile -t GRID_CONFIGS < "${SCRIPT_PATH}/grid_configs.txt"

# Check the current number of running/pending jobs and calculate available slots
NUM_JOBS=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)
AVAILABLE_SLOTS=$((240 - NUM_JOBS))

if [ "$AVAILABLE_SLOTS" -le 0 ]; then
    echo "No slots available to submit jobs now. Exiting..."
    exit 0
fi

# Initialize counter for submitted jobs
submitted_jobs=0

# Loop through each config line
for config in "${GRID_CONFIGS[@]}"; do
    # Parsing the config line
    grid_size=$(echo "$config" | grep -oP 'Grid Size: \K\d+')
    HORIZON=$(echo "$config" | grep -oP 'Horizon: \K\d+')
    hill=$(echo "$config" | grep -oP 'Hill: \K\d+')
    start_pos=$(echo "$config" | grep -oP 'Start Position: \K\d+')
    food=$(echo "$config" | grep -oP 'Food\(\K[^\)]+' | tr ',' ' ')
    water=$(echo "$config" | grep -oP 'Water\(\K[^\)]+' | tr ',' ' ')
    sleep=$(echo "$config" | grep -oP 'Sleep\(\K[^\)]+' | tr ',' ' ')

    # Process each algorithm and seed
    for ALGORITHM in "${ALGORITHMS[@]}"; do
        export ALGORITHM
        for SEED in "${SEEDS[@]}"; do
            export SEED
            JOB_ID="${ALGORITHM}_Grid${grid_size}_Hor${HORIZON}_Seed${SEED}_Hill${hill}_Start${start_pos}_Food${food// /_}_Water${water// /_}_Sleep${sleep// /_}"

            # Check if this job has already been submitted by looking for its ID in the tracking file
            if grep -q "$JOB_ID" "$JOB_TRACKING_FILE"; then
                echo "Skipping already submitted job: $JOB_ID"
                continue
            fi

            # Check if there are still slots available
            if [ "$submitted_jobs" -ge "$AVAILABLE_SLOTS" ]; then
                echo "Reached maximum number of job submissions for available slots."
                exit 0
            fi

            DATE=$(date +'%Y-%m-%d_%H-%M-%S')
            JOB_NAME="${JOB_ID}_${DATE}"
            echo "Preparing: $JOB_NAME"

            SLURM_SCRIPT="submit_${JOB_NAME}.sh"
            cp SLURM_Template.sh "$SLURM_SCRIPT"
            sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"
            echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}, false, '', ${grid_size}, ${start_pos}, ${hill}, [${food}], [${water}], [${sleep}]); exit;\"" >> "$SLURM_SCRIPT"

            output_dir="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments/$ALGORITHM/$JOB_NAME"
            mkdir -p "$output_dir"

            # Submit the job
            JOB_SUBMISSION=$(sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT")
            JOB_SUB_ID=$(echo $JOB_SUBMISSION | awk '{print $4}')
            echo "$JOB_SUB_ID: $JOB_NAME" >> "$JOB_TRACKING_FILE"
            echo "Submitted: $JOB_NAME with Job ID: $JOB_SUB_ID"
            ((submitted_jobs++))
        done
    done
done
