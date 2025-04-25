#!/bin/bash

# Directory for results
RESULTS_DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments"
MATLAB_SCRIPT_PATH="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB"
OUTPUT_DIR="$RESULTS_DIR/averaged_results"
CONFIG_FILE="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB/grid_configs.txt"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
echo "Output directory created: $OUTPUT_DIR"

# Read grid configurations from the config file
mapfile -t GRID_CONFIGS < "$CONFIG_FILE"

# Function to submit jobs
submit_jobs() {
    NUM_STATES=100
    for config in "${GRID_CONFIGS[@]}"; do
        # Extract grid-specific parameters
        GRID_ID=$(echo "$config" | grep -oP 'Grid ID: \K[^,]+')
        GRID_SIZE=$(echo "$config" | grep -oP 'Grid Size: \K\d+')
        HORIZON=$(echo "$config" | grep -oP 'Horizon: \K\d+')
        HILL=$(echo "$config" | grep -oP 'Hill: \K\d+')
        START_POS=$(echo "$config" | grep -oP 'Start Position: \K\d+')
        FOOD=$(echo "$config" | grep -oP 'Food\(\K[^\)]+' | tr ',' ' ' | xargs | sed 's/^/[/' | sed 's/$/]/')
        WATER=$(echo "$config" | grep -oP 'Water\(\K[^\)]+' | tr ',' ' ' | xargs | sed 's/^/[/' | sed 's/$/]/')
        SLEEP=$(echo "$config" | grep -oP 'Sleep\(\K[^\)]+' | tr ',' ' ' | xargs | sed 's/^/[/' | sed 's/$/]/')

        for algorithm in SI SL; do
            echo "Processing grid_id: $GRID_ID with algorithm: $algorithm"
            echo "Grid-specific parameters: Grid Size=$GRID_SIZE, Horizon=$HORIZON, Hill=$HILL, Start Position=$START_POS, Food=$FOOD, Water=$WATER, Sleep=$SLEEP"

            # Collect all files for the current algorithm and grid ID
            files=$(find "$RESULTS_DIR" -name "${algorithm}_Seed_*_GridID_${GRID_ID}.mat" 2>/dev/null)
            
            # Skip if no files for this combination
            if [ -z "$files" ]; then
                echo "No files found for grid_id $GRID_ID and algorithm $algorithm"
                continue
            fi

            # Create a list of all the seeds for this grid ID and algorithm, sorted numerically
            seeds=$(echo "$files" | grep -oP '(?<=_Seed_)\d+(?=_)' | sort -n | paste -sd ',')

            # Create job script content
            JOB_SCRIPT=$(cat <<EOT
#!/bin/bash -l
#SBATCH --account=maths  
#SBATCH --partition=ada  
#SBATCH --time=4:00:00                        
#SBATCH --job-name=${algorithm}_${GRID_ID}_kl
#SBATCH --output=${OUTPUT_DIR}/${algorithm}_${GRID_ID}_output.txt
#SBATCH --error=${OUTPUT_DIR}/${algorithm}_${GRID_ID}_error.txt
#SBATCH --ntasks=1

module load software/matlab-R2022b

# Print environment information for debugging
echo "MATLAB Script Path: ${MATLAB_SCRIPT_PATH}"
echo "Results Directory: ${RESULTS_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Food: ${FOOD}, Water: ${WATER}, Sleep: ${SLEEP}, Seeds: ${seeds}"
echo "Running MATLAB..."

matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${MATLAB_SCRIPT_PATH}')); \
    disp('Grid Size: ${GRID_SIZE}, Horizon: ${HORIZON}, Start Pos: ${START_POS}, Hill: ${HILL}'); \
    disp('Food: ${FOOD}, Water: ${WATER}, Sleep: ${SLEEP}'); \
    try, process_all_seeds_for_grid(${NUM_STATES}, ${GRID_SIZE}, ${START_POS}, ${HILL}, ${FOOD}, ${WATER}, ${SLEEP}, [${seeds}], '${algorithm}', '${GRID_ID}', '${RESULTS_DIR}'); \
    catch ME, disp('Error Message:'); disp(ME.message); disp('Error Stack:'); exit(1); end; exit(0);"
EOT
            )

            # Debug: Save job script to file
            echo "$JOB_SCRIPT" > "${OUTPUT_DIR}/job_script_${GRID_ID}_${algorithm}.sh"

            # Submit job and capture job ID
            job_id=$(echo "$JOB_SCRIPT" | sbatch | awk '{print $4}')
            echo "Submitted SLURM job ID: $job_id"
        done
    done
}

# Submit jobs
submit_jobs