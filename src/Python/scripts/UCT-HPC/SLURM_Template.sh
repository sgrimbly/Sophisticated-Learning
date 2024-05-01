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

# Email notifications:
#SBATCH --mail-user=GRMSTJ001@myuct.ac.za         # Replace with your UCT email address
#SBATCH --mail-type=END                           # Receive emails on job end and fail

# Load the Python module (ensure the correct version is loaded):
module load software/python/3.8.5                 # Example: Load Python 3.8.5, adjust as per available modules

# Display the run configuration for debugging
echo "Running Python script with ALGORITHM=${ALGORITHM} and SEED=${SEED}"

# Run Python script:
python main.py --algorithm ${ALGORITHM} --seed ${SEED}
