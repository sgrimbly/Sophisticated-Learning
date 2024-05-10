#!/bin/bash

# Configuration settings (set these as needed)
export JOB_NAME="main_run"
export TIME_LIMIT="72:00:00"
export MEMORY_ALLOCATION="2G"
export SCRIPT_PATH="~/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"

# Set environment variables for main.m
export ALGORITHM="SI"  # Specify the algorithm to run
export SEED="1"
export HORIZON="2"
export K_FACTOR="0.7"
export MCT="3"
export NUM_MCT="100"
export ROOT_FOLDER="/media/labs"  # Adjust as necessary

# Create a copy of the SLURM template
cp SLURM_Template.sh submit_$JOB_NAME.sh

# Use sed to replace placeholders with configured values
sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" submit_$JOB_NAME.sh

# Create experiment log directory
output_dir=~/MATLAB-experiments/$(date +'%Y-%m-%d_%H-%M-%S')
mkdir -p "$output_dir"

# Submit the job
sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" submit_$JOB_NAME.sh