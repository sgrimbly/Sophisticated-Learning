#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load the MATLAB module:
module load software/matlab-R2024b

# Define the parameter ranges
declare -a ALGORITHMS=(
    "SI" "SI_novelty" "SI_smooth" "SI_novelty_smooth"
    "SL" "SL_noNovelty" "SL_noSmooth" "SL_noNovelty_noSmooth"
    "BA" "BAUCB"
)
declare -a SEEDS=({0..1000})

# Parameters
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/home/grmstj001"
export TIME_LIMIT="72:00:00"
export SCRIPT_PATH="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# Results root for portable paths used by src/MATLAB/main.m
export SL_RESULTS_ROOT="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments"

# Experiment defaults (kept explicit for reproducibility)
export NUM_TRIALS="200"
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
export LEARNING_PRUNE_THRESHOLD="0.2"    # 0 disables pruning; legacy default is 0.2

export SKIP_REDUNDANT="${SKIP_REDUNDANT:-1}"  # 1 skips provably redundant algorithm variants

if [ "$SKIP_REDUNDANT" -ne 0 ]; then
    is_zero() { awk -v x="$1" 'BEGIN{exit !(x+0==0)}'; }
    filtered=()
    for ALG in "${ALGORITHMS[@]}"; do
        if [ "$ALG" = "SI_smooth" ]; then
            echo "Skipping redundant algorithm: SI_smooth"
            continue
        fi
        if is_zero "$W_NOVELTY"; then
            case "$ALG" in
                SI_novelty*|*noNovelty*)
                    echo "Skipping redundant algorithm (W_NOVELTY=0): $ALG"
                    continue
                    ;;
            esac
        fi
        filtered+=("$ALG")
    done
    ALGORITHMS=("${filtered[@]}")
fi

RUN_LABEL="UCTHEX_modular_${STATE_SELECTION}_${PREFERENCE_PARAM}_pref${W_PREFERENCE}_nov${W_NOVELTY}_learn${W_LEARNING}_epi${W_EPISTEMIC}_lp${LEARNING_PRUNE_THRESHOLD}_ucb${UCB_SCALE}_baucb${BAUCB_VARIANT}_RS${REAL_SMOOTHING}_AP${ADAPTIVE_LIKELIHOOD_IN_PLAN}_T${NUM_TRIALS}"
RUN_LABEL_SAFE=$(echo "$RUN_LABEL" | sed 's/[^a-zA-Z0-9_-]/_/g')
export JOB_TRACKING_FILE="$ROOT_FOLDER/MATLAB-experiments/Sophisticated-Learning/scripts/UCT-HPC-HEX/MATLAB/modular_job_submissions_${RUN_LABEL_SAFE}.txt"
touch "$JOB_TRACKING_FILE"

# Read grid configurations from a file
mapfile -t GRID_CONFIGS < "${SCRIPT_PATH}/grid_configs.txt"

# Function to check available slots and return the count
check_available_slots() {
    NUM_JOBS=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)
    AVAILABLE_SLOTS=$((240 - NUM_JOBS))
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

    # Round-robin by seed across algorithms (and across grid configs): seed-major ordering.
    for SEED in "${SEEDS[@]}"; do
        export SEED
        for config in "${GRID_CONFIGS[@]}"; do
            export GRID_ID=$(echo "$config" | grep -oP 'Grid ID: \K[^,]+')
            export GRID_SIZE=$(echo "$config" | grep -oP 'Grid Size: \K\d+')
            export HORIZON=$(echo "$config" | grep -oP 'Horizon: \K\d+')
            export HILL=$(echo "$config" | grep -oP 'Hill: \K\d+')
            export START_POS=$(echo "$config" | grep -oP 'Start Position: \K\d+')
            export FOOD=$(echo "$config" | grep -oP 'Food\(\K[^\)]+' | tr ',' ' ')
            export WATER=$(echo "$config" | grep -oP 'Water\(\K[^\)]+' | tr ',' ' ')
            export SLEEP=$(echo "$config" | grep -oP 'Sleep\(\K[^\)]+' | tr ',' ' ')

            NUM_STATES=$((GRID_SIZE * GRID_SIZE))

            for ALGORITHM in "${ALGORITHMS[@]}"; do
                export ALGORITHM
                if [ "$submitted_jobs" -ge "$available_slots" ]; then
                    return 0  # Return if the number of submitted jobs reaches available slots
                fi

                JOB_ID="${RUN_LABEL_SAFE}_${ALGORITHM}_Seed_${SEED}_${GRID_ID}_Grid${GRID_SIZE}_Hor${HORIZON}_Hill${HILL}_Start${START_POS}_Food${FOOD// /_}_Water${WATER// /_}_Sleep${SLEEP// /_}"
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

                MATLAB_WEIGHTS_STRUCT="struct('novelty', ${W_NOVELTY}, 'learning', ${W_LEARNING}, 'epistemic', ${W_EPISTEMIC}, 'preference', ${W_PREFERENCE}, 'ucb_scale', ${UCB_SCALE}, 'state_selection', '${STATE_SELECTION}', 'preference_param', '${PREFERENCE_PARAM}', 'baucb_variant', '${BAUCB_VARIANT}', 'real_smoothing', ${REAL_SMOOTHING}, 'adaptive_likelihood_in_plan', ${ADAPTIVE_LIKELIHOOD_IN_PLAN}, 'learning_prune_threshold', ${LEARNING_PRUNE_THRESHOLD})"
                echo "matlab -batch \"addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', ${SEED}, ${HORIZON}, ${K_FACTOR}, '${ROOT_FOLDER}', ${MCT}, ${NUM_MCT}, false, '', ${GRID_SIZE}, ${START_POS}, ${HILL}, [${FOOD}], [${WATER}], [${SLEEP}], ${MATLAB_WEIGHTS_STRUCT}, ${NUM_STATES}, ${NUM_TRIALS}, '${GRID_ID}'); exit;\"" >> "$SLURM_SCRIPT"

                output_dir="${SL_RESULTS_ROOT}/slurm_logs/${RUN_LABEL_SAFE}/${ALGORITHM}/${GRID_ID}/${JOB_NAME}"
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
