#!/bin/bash

# This script manages the submission of a large number of MATLAB jobs to a SLURM cluster, ensuring adherence
# to the queue limits of the cluster. The script dynamically submits jobs based on the availability of
# slots within the cluster's job queue, respecting the cluster's maximum allowed running and queued jobs.

# The script performs the following tasks:
# 1. Defines the necessary experiment parameters such as the number of seeds, various horizon arrays for
#    different experiment types, and other MATLAB function parameters.
# 2. Exports these parameters to ensure they are accessible in the environment for all subprocesses.
# 3. Contains a function, submit_job, which prepares and submits a job to the SLURM scheduler. This function
#    takes multiple parameters that configure each job uniquely, including the type of experiment and its
#    specific settings.
# 4. Implements a main loop that continuously checks the current count of running and pending jobs for
#    the user. It calculates the number of new jobs that can be submitted without exceeding the limits
#    imposed by the SLURM scheduler.
# 5. Submits new jobs only if there is capacity within the defined limits, ensuring that the cluster is
#    not overloaded and that jobs are submitted as slots become available.
# 6. Waits for a specified interval (30 seconds) before re-evaluating the job queue status and potentially
#    submitting more jobs.

# Key variables:
# - NUM_SEEDS: Specifies the number of seeds for which jobs need to be run.
# - MEMORY_HORIZONS, NO_MEMORY_HORIZONS, HYBRID_HORIZONS, PURE_MC_HORIZONS: Arrays defining the range
#   of experiment settings for different types of runs.
# - K_FACTOR, ROOT_FOLDER, SCRIPT_PATH: Configuration settings for MATLAB scripts and where they are stored.
# - TIME_LIMIT, MEMORY_ALLOCATION: SLURM job submission parameters.

# Usage:
# - Ensure that the SLURM_Template.sh contains the correct SLURM directives and that the placeholders match
#   those expected by the sed commands in the submit_job function.
# - Run this script on a node or a login server where you have permission to submit jobs to the SLURM queue.
# - The script will automatically manage the submission process, keeping the job queue full but within limits.

# Load the MATLAB module:
module load software/matlab-R2024a

# Define the parameter ranges
declare -i NUM_SEEDS=1
declare -a MEMORY_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
declare -a NO_MEMORY_HORIZONS=("1" "2" "3" "4" "5")
declare -a HYBRID_HORIZONS=("1" "2" "3" "4" "5" "6")
declare -a PURE_MC_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
declare K_FACTOR="0.7"
declare ROOT_FOLDER="/home/grmstj001"
declare SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
declare TIME_LIMIT="3-00:00:00"
declare MEMORY_ALLOCATION="4G"

export NUM_SEEDS MEMORY_HORIZONS NO_MEMORY_HORIZONS HYBRID_HORIZONS PURE_MC_HORIZONS K_FACTOR ROOT_FOLDER SCRIPT_PATH TIME_LIMIT MEMORY_ALLOCATION

declare -i MAX_RUNNING=120
declare -i MAX_QUEUED=240
declare -i TOTAL_SUBMITTED=0
declare -i ACTIVE_SUBMISSIONS=0
# Initialize SEED counter
declare -i SEED=1

# Create every experiment combination of the variables. Currently the while loop is only checking seeds, and hybrid horizons and then running something(?) up to 3300. This current logic seems wrong to me. I think the easiest way to do this will be to create a list of every possible experiment combination I want to run, and then run a periodic loop over this datastructure. This way it won't be as hard to track what has actually run. 

# Job submission function
submit_job() {
    export JOB_NAME="Seed${1}_Hor${2}_MCT_${5}"
    export ALGORITHM="known_large_MCT"
    export SEED="$1"
    export HORIZON="$2"
    export K_FACTOR="$3"
    export ROOT_FOLDER="$4"
    export MCT="$5"
    export NUM_MCT="100"
    export AUTO_REST="$6"

    local SLURM_SCRIPT="submit_${JOB_NAME}.sh"
    cp SLURM_Template.sh $SLURM_SCRIPT

    sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT
    echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('$ALGORITHM', '$SEED', '$HORIZON', '$K_FACTOR', '$ROOT_FOLDER', '$MCT', '$NUM_MCT', '$AUTO_REST'); exit;\"" >> $SLURM_SCRIPT
    sbatch $SLURM_SCRIPT
    ((TOTAL_SUBMITTED++))
    ((ACTIVE_SUBMISSIONS++))
}

while (( TOTAL_SUBMITTED < 3300 )); do
    CURRENT_JOBS=$(squeue -u grmstj001 | grep -E "R|PD" | wc -l)
    let "JOBS_TO_SUBMIT = MAX_QUEUED - CURRENT_JOBS - ACTIVE_SUBMISSIONS"

    if (( JOBS_TO_SUBMIT > 0 )); then
        for ((i=1; i<=JOBS_TO_SUBMIT && TOTAL_SUBMITTED < 3300 && SEED <= NUM_SEEDS; i++, SEED++)); do
            for HORIZON in "${HYBRID_HORIZONS[@]}"; do
                declare -i MCT=$((6 - HORIZON))
                submit_job "$SEED" "$HORIZON" "$K_FACTOR" "$ROOT_FOLDER" "$MCT" "0"
                if (( ACTIVE_SUBMISSIONS >= JOBS_TO_SUBMIT )); then
                    break 2
                fi
            done
        done
    fi

    sleep 30s

    # Update ACTIVE_SUBMISSIONS based on the actual state of the job queue
    ACTIVE_SUBMISSIONS=$(squeue -u grmstj001 | grep -c 'PD')
done
echo "Reached the total job submission limit of 3300."


# My older script
# # Load the MATLAB module:
# module load software/matlab-R2022b

# # Define the parameter ranges
# declare -i NUM_SEEDS=100  # Number of seeds to run for each experiment
# declare -a MEMORY_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")
# declare -a NO_MEMORY_HORIZONS=("1" "2" "3" "4" "5")
# declare -a HYBRID_HORIZONS=("1" "2" "3" "4" "5" "6")  # Total horizon shown on x-axis
# declare -a PURE_MC_HORIZONS=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11")  # MCT settings for pure Monte Carlo
# declare K_FACTOR="0.7"
# declare ROOT_FOLDER="/home/grmstj001"
# declare MCT_HYBRID="3"  # Fixed mct for hybrid if not included in x-axis values

# # Base configuration
# export TIME_LIMIT="7-00:00:00"
# export MEMORY_ALLOCATION="4G"
# export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# # Submit job function
# submit_job() {
#     local job_name="$1"
#     local horizon="$2"
#     local mct="$3"
#     local auto_rest="$4"

#     SLURM_SCRIPT="submit_${job_name}.sh"
#     cp SLURM_Template.sh $SLURM_SCRIPT

#     # Replace placeholders with actual configured values
#     sed -i "s|\$JOB_NAME|$job_name|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

#     # Add the MATLAB run line
#     echo "matlab -nodisplay -nosplash -nodesktop -r \"addpath(genpath('${SCRIPT_PATH}')); main('known_large_MCT', '${SEED}', '${horizon}', '${K_FACTOR}', '${ROOT_FOLDER}', '${mct}', '100', '${auto_rest}'); exit;\"" >> $SLURM_SCRIPT

#     # Submit the job
#     sbatch $SLURM_SCRIPT
# }

# # Loop over each seed
# for ((SEED=1; SEED<=$NUM_SEEDS; SEED++))
# do
#     export SEED

#     # Memory Experiments
#     for HORIZON in "${MEMORY_HORIZONS[@]}"
#     do
#         submit_job "Memory_Seed${SEED}_Hor${HORIZON}" "$HORIZON" "0" "0"
#     done

#     # No Memory Experiments
#     for HORIZON in "${NO_MEMORY_HORIZONS[@]}"
#     do
#         submit_job "NoMemory_Seed${SEED}_Hor${HORIZON}" "$HORIZON" "0" "1"
#     done

#     # Hybrid Experiments
#     for HORIZON in "${HYBRID_HORIZONS[@]}"
#     do
#         submit_job "Hybrid_Seed${SEED}_Hor${HORIZON}" "$HORIZON" "$MCT_HYBRID" "0"
#     done

#     # Pure Monte Carlo Experiments
#     for MCT in "${PURE_MC_HORIZONS[@]}"
#     do
#         submit_job "PureMC_Seed${SEED}_MCT${MCT}" "0" "$MCT" "0"
#     done
# done
