#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load software/matlab-R2024b

declare -a ALGORITHMS=(
    "SI"
    "SI_novelty"
    "SI_novelty_smooth"
    "SL_noSmooth"
    "SL"
)
declare -a SEEDS=({1..50})

export ROOT_FOLDER="/home/grmstj001"
export SCRIPT_PATH="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
export RESULTS_ROOT="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/revision_novelty_default_env"
export TIME_LIMIT="72:00:00"
export MAX_SLOTS=120

export GRID_ID="default_env_revision_novelty_hor9"
export GRID_SIZE=10
export START_POS=51
export HILL=55
export FOOD="71 43 57 78"
export WATER="73 33 48 67"
export SLEEP="64 44 49 59"
export NUM_STATES=100
export HORIZON=9
export NUM_TRIALS=120

export W_NOVELTY=10
export W_LEARNING=40
export W_EPISTEMIC=1
export W_PREFERENCE=10
export UCB_SCALE=5
export STATE_SELECTION="sample"
export PREFERENCE_PARAM="weight"
export BAUCB_VARIANT="legacy"
export REAL_SMOOTHING=1
export ADAPTIVE_LIKELIHOOD_IN_PLAN=0
export LEARNING_PRUNE_THRESHOLD=0.2
export SL_LOG_METRICS=1

RUN_LABEL="revision_novelty_defaultenv_h${HORIZON}_t${NUM_TRIALS}_s1-50"
RUN_LABEL_SAFE=$(echo "$RUN_LABEL" | sed 's/[^a-zA-Z0-9_-]/_/g')
export JOB_TRACKING_FILE="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB/job_submissions_${RUN_LABEL_SAFE}.txt"
touch "$JOB_TRACKING_FILE"

check_available_slots() {
    local num_jobs
    num_jobs=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)
    echo $((MAX_SLOTS - num_jobs))
}

wait_and_check_slots() {
    echo "Waiting for 5 minutes before checking available slots again..."
    sleep 300
    check_available_slots
}

submit_jobs() {
    local available_slots=$1
    local submitted_jobs=0

    for SEED in "${SEEDS[@]}"; do
        export SEED
        for ALGORITHM in "${ALGORITHMS[@]}"; do
            export ALGORITHM
            if [ "$submitted_jobs" -ge "$available_slots" ]; then
                return 0
            fi

            JOB_ID="${RUN_LABEL_SAFE}_${ALGORITHM}_Seed${SEED}"
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

            RUN_RESULTS_DIR="${RESULTS_ROOT}/${ALGORITHM}"
            mkdir -p "$RUN_RESULTS_DIR"

            {
                echo "export SL_LOG_METRICS=1"
                echo "rm -f \"${RUN_RESULTS_DIR}/${ALGORITHM}_Seed${SEED}.txt\""
                echo "rm -f \"${RUN_RESULTS_DIR}/${ALGORITHM}_Seed${SEED}_metrics.csv\""
                echo "rm -f \"${RUN_RESULTS_DIR}/${ALGORITHM}_Seed${SEED}_step_metrics.csv\""
                echo "rm -f \"${RUN_RESULTS_DIR}/${ALGORITHM}_Seed_${SEED}_GridID_${GRID_ID}_Cfg_\"*.mat"
                echo "matlab -batch \"setenv('SL_LOG_METRICS','1'); addpath(genpath('${SCRIPT_PATH}')); weights = struct('novelty', ${W_NOVELTY}, 'learning', ${W_LEARNING}, 'epistemic', ${W_EPISTEMIC}, 'preference', ${W_PREFERENCE}, 'ucb_scale', ${UCB_SCALE}, 'state_selection', '${STATE_SELECTION}', 'preference_param', '${PREFERENCE_PARAM}', 'baucb_variant', '${BAUCB_VARIANT}', 'real_smoothing', logical(${REAL_SMOOTHING}), 'adaptive_likelihood_in_plan', logical(${ADAPTIVE_LIKELIHOOD_IN_PLAN}), 'learning_prune_threshold', ${LEARNING_PRUNE_THRESHOLD}); cfg = struct('seed', ${SEED}, 'grid_size', ${GRID_SIZE}, 'start_position', ${START_POS}, 'hill_pos', ${HILL}, 'food_sources', [${FOOD}], 'water_sources', [${WATER}], 'sleep_sources', [${SLEEP}], 'num_states', ${NUM_STATES}, 'num_trials', ${NUM_TRIALS}, 'max_horizon', ${HORIZON}, 'weights', weights); dashboard_run_one('${ALGORITHM}', cfg, [], '${RUN_RESULTS_DIR}', '${GRID_ID}');\""
            } >> "$SLURM_SCRIPT"

            output_dir="${RESULTS_ROOT}/slurm_logs/${RUN_LABEL_SAFE}/${ALGORITHM}/${JOB_NAME}"
            mkdir -p "$output_dir"

            job_submission=$(sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" "$SLURM_SCRIPT")
            job_sub_id=$(echo "$job_submission" | awk '{print $4}')
            echo "$job_sub_id: $JOB_NAME" >> "$JOB_TRACKING_FILE"
            echo "Submitted: $JOB_NAME with Job ID: $job_sub_id"
            ((submitted_jobs++))
        done
    done
}

while true; do
    AVAILABLE_SLOTS=$(check_available_slots)
    if [ "$AVAILABLE_SLOTS" -gt 0 ]; then
        submit_jobs "$AVAILABLE_SLOTS"
    else
        echo "No available slots right now."
    fi

    wait_and_check_slots
done
