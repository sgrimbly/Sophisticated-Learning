#!/bin/bash
# This script automates the submission of MATLAB jobs to a SLURM cluster, managing submissions
# according to real-time cluster queue capacities. It can operate in either continuous or batch mode,
# and uses a job tracking log to prevent duplicate submissions, ensuring efficient cluster usage.

# Script performs the following key functions:
# 1. Loads the required MATLAB module and defines parameters for various types of experiments.
# 2. Exports these parameters to make them accessible for subprocesses and functions within the script.
# 3. Utilizes a job submission function to configure and submit jobs, ensuring that each job's
#    uniqueness is logged to prevent duplication.
# 4. Depending on the operational mode, manages job submissions to either continuously fill the queue 
#    up to a set limit or submit a fixed number of jobs.
# 5. In continuous mode, it dynamically adjusts submissions based on the SLURM queue's state,
#    rechecking and submitting new jobs as slots become available.
# 6. In one-time batch mode, submits a predetermined number of jobs without continuous checking,
#    suitable for less frequent or smaller-scale job submissions.


# Load the MATLAB module:
module load software/matlab-R2024a

# Define the parameter ranges
declare -i NUM_SEEDS=1
declare -a MEMORY_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
declare -a NO_MEMORY_HORIZONS=("1" "2" "3" "4" "5")
declare -a HYBRID_HORIZONS=("1" "2" "3" "4" "5" "6") # Last one will be pure SI horizon=6
declare -a PURE_MC_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
declare K_FACTOR="0.7"
declare ROOT_FOLDER="/home/grmstj001"
declare SCRIPT_PATH="${HOME}/MATLAB-experiments/Sophisticated-Learning/src/MATLAB/scripts/UCT-HPC"
declare MAIN_PATH="${HOME}/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
declare TIME_LIMIT="3-00:00:00"
declare MEMORY_ALLOCATION="4G"

export NUM_SEEDS MEMORY_HORIZONS NO_MEMORY_HORIZONS HYBRID_HORIZONS PURE_MC_HORIZONS K_FACTOR ROOT_FOLDER SCRIPT_PATH TIME_LIMIT MEMORY_ALLOCATION

# Initialize job queue management variables
declare -i MAX_RUNNING=120
declare -i MAX_QUEUED=240
declare -i TOTAL_SUBMITTED=0
declare -i ACTIVE_SUBMISSIONS=0
declare JOB_TRACKING_FILE="known_model_submitted_jobs.txt"

# Read existing job submissions to avoid duplication
declare -A submitted_jobs
if [ -f "$JOB_TRACKING_FILE" ]; then
    while IFS= read -r line; do
        job_id=$(echo "$line" | cut -d' ' -f1)
        submitted_jobs["$job_id"]=1
    done < "$JOB_TRACKING_FILE"
fi

# Define the job submission function
submit_job() {
    local job_type="$7"
    export JOB_NAME="${job_type}_Seed${1}_Hor${2}_MCT_${5}"
    export ALGORITHM="known_large_MCT"
    export SEED="$1"
    export HORIZON="$2"
    export K_FACTOR="$3"
    export ROOT_FOLDER="$4"
    export MCT="$5"
    export NUM_MCT="100"
    export AUTO_REST="$6"

    # Prevent duplicate submissions
    if [[ -n "${submitted_jobs[$JOB_NAME]}" ]]; then
        echo "$JOB_NAME has already been submitted. Skipping..."
        return
    fi

    local SLURM_SCRIPT="${SCRIPT_PATH}/submit_${JOB_NAME}.sh"
    cp "${SCRIPT_PATH}/SLURM_Template.sh" "$SLURM_SCRIPT"
    sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" "$SLURM_SCRIPT"
    echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('$MAIN_PATH')); main('$ALGORITHM', '$SEED', '$HORIZON', '$K_FACTOR', '$ROOT_FOLDER', '$MCT', '$NUM_MCT', '$AUTO_REST'); exit;\"" >> "$SLURM_SCRIPT"

    # Log job submission with correct type
    echo "$JOB_NAME type=$job_type, horizon=$HORIZON, mct=$MCT, seed=$SEED" >> "$JOB_TRACKING_FILE"
    sbatch "$SLURM_SCRIPT"
    ((TOTAL_SUBMITTED++))
    ((ACTIVE_SUBMISSIONS++))
}

# Function to handle multiple experiment types based on seed
function submit_experiments() {
    local seed=$1
    local type=""
    # Memory, No Memory, Hybrid, and Pure MC experiment submissions
    for HORIZON in "${MEMORY_HORIZONS[@]}"; do
        type="Memory"
        submit_job "$seed" "$HORIZON" "$K_FACTOR" "$ROOT_FOLDER" "0" "0" "$type"
    done
    for HORIZON in "${NO_MEMORY_HORIZONS[@]}"; do
        type="NoMemory"
        submit_job "$seed" "$HORIZON" "$K_FACTOR" "$ROOT_FOLDER" "0" "1" "$type"
    done
    for HORIZON in "${HYBRID_HORIZONS[@]}"; do
        local MCT=$((6 - HORIZON))
        type="Hybrid"
        if ((MCT >= 0 && MCT <= 11)); then
            submit_job "$seed" "$HORIZON" "$K_FACTOR" "$ROOT_FOLDER" "$MCT" "0" "$type"
        fi
    done
    for MCT in "${PURE_MC_HORIZONS[@]}"; do
        type="PureMC"
        submit_job "$seed" "0" "$K_FACTOR" "$ROOT_FOLDER" "$MCT" "0" "$type"
    done
}

# Main submission logic for continuous or one-time batch mode
declare -i CONTINUOUS_SUBMISSION=0  # Set to 1 for continuous, 0 for one-time submission
declare -i JOBS_TO_SUBMIT_ONCE=33  # Number of jobs to submit if not running continuously
if (( CONTINUOUS_SUBMISSION == 1 )); then
    while (( TOTAL_SUBMITTED < 3300 )); do
        CURRENT_JOBS=$(squeue -u grmstj001 | grep -E "R|PD" | wc -l)
        let "JOBS_TO_SUBMIT = MAX_QUEUED - CURRENT_JOBS - ACTIVE_SUBMISSIONS"
        if (( JOBS_TO_SUBMIT > 0 )); then
            for ((SEED=1; SEED <= NUM_SEEDS && TOTAL_SUBMITTED < 3300; SEED++)); do
                submit_experiments $SEED
                if (( ACTIVE_SUBMISSIONS >= JOBS_TO_SUBMIT )); then
                    break 2
                fi
            done
        fi
        sleep 30s  # Moderates scheduler query frequency
        ACTIVE_SUBMISSIONS=$(squeue -u grmstj001 | grep -c 'PD')
    done
else
    for ((SEED=1; TOTAL_SUBMITTED < JOBS_TO_SUBMIT_ONCE; SEED++)); do
        submit_experiments $SEED
        if (( TOTAL_SUBMITTED >= JOBS_TO_SUBMIT_ONCE )); then break; fi
    done
fi

echo "Reached the total job submission limit of $TOTAL_SUBMITTED."
