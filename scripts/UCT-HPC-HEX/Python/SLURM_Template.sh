#!/bin/bash
# SLURM submission script for Python jobs on UCT HPC

# Essential SBATCH directives:
#SBATCH --account=maths                           # Replace with your actual account
#SBATCH --partition=ada                           # Confirm partition based on your access rights and job needs
#SBATCH --time=$TIME_LIMIT                        # Time limit placeholder
#SBATCH --job-name=$JOB_NAME                      # Job name placeholder
#SBATCH --nodes=1                                 # Single node
#SBATCH --ntasks=1                                # Single task since Python script is typically single-threaded by default
#SBATCH --cpus-per-task=1                         # Number of CPU cores per task
#SBATCH --mem=$MEMORY_ALLOCATION                  # Memory placeholder
#SBATCH --output=python_job_%j.out                # Standard output and error log

# Change the nodelist dependeing on availability http://hpc.uct.ac.za/db/
#SBATCH --nodelist=srvcnthpc105,srvcnthpc108,srvcnthpc109,srvcnthpc116,srvcnthpc120,srvcnthpc125,srvcnthpc127,srvcnthpc128  # Specific nodes request

# Email notifications:
#SBATCH --mail-user=GRMSTJ001@myuct.ac.za         # Replace with your UCT email address
#SBATCH --mail-type=END                           # Receive emails on job end and fail

# Load the Python module (ensure the correct version is loaded):
module load python/miniconda3-py39

# Activate the Python virtual environment
source $ENV_PATH/bin/activate

# Display the run configuration for debugging
echo "Running Python script with ALGORITHM=${ALGORITHM} and SEED=${SEED}"

# Run Python script:
python "${SCRIPT_PATH}/main.py" --algorithm $ALGORITHM --seed $SEED

# Deactivate the environment
deactivate