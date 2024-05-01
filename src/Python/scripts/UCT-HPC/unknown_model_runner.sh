#!/bin/bash

# Load the Python module
module load python/miniconda3-py39

# Create a virtual environment (do this once, outside of your job script)
export ENV_PATH="${HOME}/my_python_env"
python -m venv $ENV_PATH

# Activate the environment and install packages
source $ENV_PATH/bin/activate
pip install numpy matplotlib scipy

# Define the parameter ranges
declare -a ALGORITHMS=("SI") # "SL" later, maybe add "BA" and "BAUCB" as well
declare -i NUM_SEEDS=30  # Number of seeds to run for each algorithm

# Base configuration
export JOB_NAME="main_run"
export TIME_LIMIT="72:00:00"
export MEMORY_ALLOCATION="8G"
export SCRIPT_PATH="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python"

for ALGORITHM in "${ALGORITHMS[@]}"
do
    for ((SEED=1; SEED<=$NUM_SEEDS; SEED++))
    do
        # Configure the job name to include parameter info for easier tracking
        export ALGORITHM  # Export the ALGORITHM for each job
        export SEED       # Export the SEED for each job
        JOB_NAME="${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}"
        export JOB_NAME   # Ensure JOB_NAME is also exported

        # Prepare the SLURM script
        SLURM_SCRIPT="submit_${JOB_NAME}.sh"
        cp SLURM_Template.sh $SLURM_SCRIPT

        # Replace placeholders with actual configured values
        sed -i "s|\$JOB_NAME|$JOB_NAME|g; s|\$TIME_LIMIT|$TIME_LIMIT|g; s|\$MEMORY_ALLOCATION|$MEMORY_ALLOCATION|g; s|\$SCRIPT_PATH|$SCRIPT_PATH|g" $SLURM_SCRIPT

        # Add the Python run line
        echo "python $SCRIPT_PATH/main.py --algorithm $ALGORITHM --seed $SEED" >> $SLURM_SCRIPT

        # Create experiment log directory with descriptive name
        output_dir=~/Python-experiments/experiments/Python/${ALGORITHM}_Seed${SEED}_Hor${HORIZON}_KF${K_FACTOR}_MCT${MCT}_Num${NUM_MCT}_$(date +'%Y-%m-%d_%H-%M-%S')
        mkdir -p "$output_dir"

        # Submit the job
        sbatch --output="$output_dir/slurm-%j.out" --error="$output_dir/slurm-%j.err" $SLURM_SCRIPT
    done
done
