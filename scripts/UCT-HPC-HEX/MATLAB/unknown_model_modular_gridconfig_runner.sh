#!/bin/bash

# Load the MATLAB module:
# module load software/matlab-R2022b
module load software/matlab-R2024b

# Define the parameter ranges
declare -a ALGORITHMS=("BA" "BAUCB") # "SI" "SL")
declare -a SEEDS=({0..1000})

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

# Function to check available slots and return the count
check_available_slots() {
    NUM_JOBS=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)
    AVAILABLE_SLOTS=$((140 - NUM_JOBS))
    echo "$AVAILABLE_SLOTS"
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

    for config in "${GRID_CONFIGS[@]}"; do
        export GRID_ID=$(echo "$config" | grep -oP 'Grid ID: \K[^,]+')
        export GRID_SIZE=$(echo "$config" | grep -oP 'Grid Size: \K\d+')
        export HORIZON=$(echo "$config" | grep -oP 'Horizon: \K\d+')
        export HILL=$(echo "$config" | grep -oP 'Hill: \K\d+')
        export START_POS=$(echo "$config" | grep -oP 'Start Position: \K\d+')
        export FOOD=$(echo "$config" | grep -oP 'Food\(\K[^\)]+' | tr ',' ' ')
        export WATER=$(echo "$config" | grep -oP 'Water\(\K[^\)]+' | tr ',' ' ')
        export SLEEP=$(echo "$config" | grep -oP 'Sleep\(\K[^\)]+' | tr ',' ' ')

        for ALGORITHM in "${ALGORITHMS[@]}"; do
            export ALGORITHM
            for SEED in "${SEEDS[@]}"; do
                export SEED
                if [ "$submitted_jobs" -ge "$available_slots" ]; then
                    return 0  # Return if the number of submitted jobs reaches available slots
                fi

                GRID_JOB_NAME="${ALGORITHM}_Seed_${SEED}_${GRID_ID}_Grid${GRID_SIZE}_Hor${HORIZON}_Hill${HILL}_Start${START_POS}_Food${FOOD// /_}_Water${WATER// /_}_Sleep${SLEEP// /_}"
                JOB_ID="${ALGORITHM}_Seed_${SEED}_${GRID_ID}_Grid${GRID_SIZE}_Hor${HORIZON}_Hill${HILL}_Start${START_POS}_Food${FOOD// /_}_Water${WATER// /_}_Sleep${SLEEP// /_}"
                if grep -q "$JOB_ID" "$JOB_TRACKING_FILE"; then
                    echo "Skipping already submitted job: $JOB_ID"
                    continue
                fi

                DATE=$(date +'%Y-%m-%d_%H-%M-%S')
                JOB_NAME="${JOB_ID}_${DATE}"
                echo "Preparing: $JOB_NAME"

                SLURM_SCRIPT="submit_${JOB_NAME}.sh"
                cp SLURM_Template.sh "$SLURM_SCRIPT"
                sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"

                # echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}, false, '', ${GRID_SIZE}, ${START_POS}, ${HILL}, [${FOOD}], [${WATER}], [${SLEEP}], struct('novelty', 10, 'learning', 40, 'epistemic', 1, 'preference', 10), 100, 200, '${GRID_ID}'); exit;\"" >> "$SLURM_SCRIPT"
                echo "matlab -batch \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}, false, '', ${GRID_SIZE}, ${START_POS}, ${HILL}, [${FOOD}], [${WATER}], [${SLEEP}], struct('novelty', 10, 'learning', 40, 'epistemic', 1, 'preference', 10), 100, 200, '${GRID_ID}'); exit;\"" >> "$SLURM_SCRIPT"

                output_dir="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments/$ALGORITHM/${GRID_ID}/$JOB_NAME"
                mkdir -p "$output_dir"

                JOB_SUBMISSION=$(sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT")
                JOB_SUB_ID=$(echo $JOB_SUBMISSION | awk '{print $4}')
                echo "$JOB_SUB_ID: $JOB_NAME" >> "$JOB_TRACKING_FILE"
                echo "Submitted: $JOB_NAME with Job ID: $JOB_SUB_ID"
                ((submitted_jobs++))
            done
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
