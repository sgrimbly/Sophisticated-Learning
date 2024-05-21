#!/bin/bash
# SLURM submission script for MATLAB jobs on UCT HPC

# Essential SBATCH directives:
#SBATCH --account=maths                           
#SBATCH --partition=ada                           
#SBATCH --time=$TIME_LIMIT                        # Time limit placeholder
#SBATCH --job-name=$JOB_NAME                      # Job name placeholder
#SBATCH --nodes=1                                 # Single node
#SBATCH --ntasks=1                                # Single task since MATLAB is single-core by default
# SBATCH --cpus-per-task=1                         # Number of CPU cores per task
# SBATCH --mem=$MEMORY_ALLOCATION                  # Memory placeholder
#SBATCH --output=matlab_job_%j.out                # Standard output and error log

# Currently excluding these nodes on HEX because there were issues with jobs not starting/ending correctly. MATLAB related?
# SBATCH --nodelist=srvrochpc[108-111]
# SBATCH --exclude=srvrochpc103,srvrochpc107       # Exclude these nodes

# Email notifications:
# SBATCH --mail-user=GRMSTJ001@myuct.ac.za         

# Commented out so no emails: SBATCH --mail-type=FAIL #BEGIN,END,FAIL  

# Load the MATLAB module:
module load software/matlab-R2022b

# Display the run configuration for debugging
echo "Running main with ALGORITHM=${ALGORITHM}, SEED=${SEED}, HORIZON=${HORIZON}, K_FACTOR=${K_FACTOR}, ROOT_FOLDER=${ROOT_FOLDER}, MCT=${MCT}, NUM_MCT=${NUM_MCT}, AUTO_REST=${AUTO_REST}"

# Wait before submitting MATLAB job for random number of seconds to avoid congestion on MATLAB license server. Possible reason for not starting up of MATLAB jobs seen previously. 
sleep $((5 + RANDOM % 25))

# Run MATLAB script. This line will be added below here by the job runner that modifies this template. 
# matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${SCRIPT_PATH}')); main('${ALGORITHM}', '${SEED}', '${HORIZON}', '${K_FACTOR}', '${ROOT_FOLDER}', '${MCT}', '${NUM_MCT}'); exit;"
