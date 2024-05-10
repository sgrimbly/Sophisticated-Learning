#!/bin/bash
# SLURM submission script for MATLAB jobs on UCT HPC

# Essential SBATCH directives:
#SBATCH --account=maths                           # Replace with your actual account
#SBATCH --partition=ada                           # Confirm partition based on your access rights and job needs
#SBATCH --time=$TIME_LIMIT                        # Time limit placeholder
#SBATCH --job-name=$JOB_NAME                      # Job name placeholder
#SBATCH --nodes=1                                 # Single node
#SBATCH --ntasks=1                                # Single task since MATLAB is single-core by default
#SBATCH --cpus-per-task=1                         # Number of CPU cores per task
#SBATCH --mem=$MEMORY_ALLOCATION                  # Memory placeholder
#SBATCH --output=matlab_job_%j.out                # Standard output and error log

# Email notifications:
#SBATCH --mail-user=GRMSTJ001@myuct.ac.za         # Replace with your UCT email address

# Commented out so no emails: SBATCH --mail-type=FAIL #BEGIN,END,FAIL  

# Load the MATLAB module:
module load software/matlab-R2024a

# Display the run configuration for debugging
echo "Running main with ALGORITHM=${ALGORITHM}, SEED=${SEED}, HORIZON=${HORIZON}, K_FACTOR=${K_FACTOR}, ROOT_FOLDER=${ROOT_FOLDER}, MCT=${MCT}, NUM_MCT=${NUM_MCT}, AUTO_REST=${AUTO_REST}"

# Run MATLAB script. This line will be added below here by the job runner that modifies this template. 
# matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', '${SEED}', '${HORIZON}', '${K_FACTOR}', '${ROOT_FOLDER}', '${MCT}', '${NUM_MCT}'); exit;"
