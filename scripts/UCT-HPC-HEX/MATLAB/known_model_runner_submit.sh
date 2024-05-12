#!/bin/bash

# Path to the known_model_runner.sh script
RUNNER_SCRIPT="known_model_runner.sh"

# Path to the job tracking file
JOB_TRACKING_FILE="known_model_submitted_jobs.txt"

while true; do
    # Check the number of lines in the submitted jobs tracking file
    if [ -f "$JOB_TRACKING_FILE" ]; then
        LINE_COUNT=$(wc -l < "$JOB_TRACKING_FILE")
        echo "Current number of submitted jobs: $LINE_COUNT"

        # Exit the script if the number of lines exceeds 3300
        if [ "$LINE_COUNT" -gt 3300 ]; then
            echo "The limit of 3300 submitted jobs has been reached. Exiting script."
            exit 0
        fi
    else
        echo "Job tracking file not found: $JOB_TRACKING_FILE"
    fi

    # Count the number of running and pending jobs
    NUM_JOBS=$(squeue -u grmstj001 | grep -E "R|PD" | tail -n +2 | wc -l)

    echo "Current number of running/pending jobs: $NUM_JOBS"

    # Check if the number of jobs is less than 120
    if [ "$NUM_JOBS" -lt 120 ]; then
        echo "Running job submission script..."
        # Call the job submission script
        bash "$RUNNER_SCRIPT"
    else
        echo "No need to submit more jobs. Waiting for 30 seconds..."
    fi

    # Wait for one minute before checking again
    sleep 30
done
