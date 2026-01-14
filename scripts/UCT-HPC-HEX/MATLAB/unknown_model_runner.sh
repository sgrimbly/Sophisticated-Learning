#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the MATLAB module:
# module load software/matlab-R2022b
module load software/matlab-R2024b

# Define the parameter ranges
declare -a ALGORITHMS=(
    "SI" "SI_novelty" "SI_smooth" "SI_novelty_smooth"
    "SL" "SL_noNovelty" "SL_noSmooth" "SL_noNovelty_noSmooth"
    "BA" "BAUCB"
)
declare -a SEEDS=({0..1000})
# declare -a SEEDS=(1021 1022 1023) # Test seeds for MATLAB 2024b

# Default configuration parameters
export HORIZON="6"
export K_FACTOR="1.5"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"
export TIME_LIMIT="72:00:00"
export SCRIPT_PATH="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
export MAX_SLOTS=240

# Results root for portable paths used by src/MATLAB/main.m
export SL_RESULTS_ROOT="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/default_env"

# Experiment defaults (kept explicit for reproducibility)
export NUM_TRIALS="120"
export W_NOVELTY="10"
export W_LEARNING="40"
export W_EPISTEMIC="1"
export W_PREFERENCE="10"
export UCB_SCALE="5"
export STATE_SELECTION="sample"          # 'sample' or 'map'
export PREFERENCE_PARAM="weight"         # 'weight' or 'inverse_precision'
export BAUCB_VARIANT="legacy"            # 'legacy' or 'fixed_joint_counts'
export REAL_SMOOTHING="1"                # 1=backward smoothing on real trajectory, 0=filtered-only updates
export ADAPTIVE_LIKELIHOOD_IN_PLAN="0"   # 1=update y from simulated a inside SL planning (TODO ablation)

RUN_LABEL="UCTHEX_defaultenv_${STATE_SELECTION}_${PREFERENCE_PARAM}_pref${W_PREFERENCE}_nov${W_NOVELTY}_learn${W_LEARNING}_epi${W_EPISTEMIC}_ucb${UCB_SCALE}_baucb${BAUCB_VARIANT}_RS${REAL_SMOOTHING}_AP${ADAPTIVE_LIKELIHOOD_IN_PLAN}_T${NUM_TRIALS}"
RUN_LABEL_SAFE=$(echo "$RUN_LABEL" | sed 's/[^a-zA-Z0-9_-]/_/g')
export JOB_TRACKING_FILE="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB/job_submissions_${RUN_LABEL_SAFE}.txt"
touch "$JOB_TRACKING_FILE"

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

# Function to submit jobs
submit_jobs() {
    local available_slots=$1
    local submitted_jobs_count=0

    # Round-robin by seed across algorithms (seed-major ordering).
    for SEED in "${SEEDS[@]}"; do
        export SEED
        for ALGORITHM in "${ALGORITHMS[@]}"; do
            export ALGORITHM

            if [[ "$submitted_jobs_count" -ge "$available_slots" ]]; then
                echo "Reached slot limit, no more submissions allowed."
                return
            fi

            JOB_ID="${RUN_LABEL_SAFE}_${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}"
            if grep -q "$JOB_ID" "$JOB_TRACKING_FILE"; then
                echo "Skipping already submitted job: $JOB_ID"
                continue
            fi

            DATE=$(date +'%Y-%m-%d_%H-%M-%S')
            JOB_NAME="${JOB_ID}_${DATE}"
            echo "Preparing: $JOB_NAME"

            SLURM_SCRIPT="submit_${JOB_NAME}.sh"
            cp "${SCRIPT_DIR}/SLURM_Template.sh" "$SLURM_SCRIPT"
            sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"

	            MATLAB_WEIGHTS_STRUCT="struct('novelty', ${W_NOVELTY}, 'learning', ${W_LEARNING}, 'epistemic', ${W_EPISTEMIC}, 'preference', ${W_PREFERENCE}, 'ucb_scale', ${UCB_SCALE}, 'state_selection', '${STATE_SELECTION}', 'preference_param', '${PREFERENCE_PARAM}', 'baucb_variant', '${BAUCB_VARIANT}', 'real_smoothing', ${REAL_SMOOTHING}, 'adaptive_likelihood_in_plan', ${ADAPTIVE_LIKELIHOOD_IN_PLAN})"

            echo "matlab -batch \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}, false, '', 10, 51, 55, [71 43 57 78], [73 33 48 67], [64 44 49 59], ${MATLAB_WEIGHTS_STRUCT}, 100, ${NUM_TRIALS}, 'default_env'); exit;\"" >> "$SLURM_SCRIPT"

            output_dir="${SL_RESULTS_ROOT}/slurm_logs/${RUN_LABEL_SAFE}/${ALGORITHM}/${JOB_NAME}"
            mkdir -p "$output_dir"

            job_submission=$(sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT")
            job_sub_id=$(echo $job_submission | awk '{print $4}')
            echo "$job_sub_id: $JOB_NAME" >> "$JOB_TRACKING_FILE"
            echo "Successfully submitted: $JOB_NAME with Job ID: $job_sub_id"
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
