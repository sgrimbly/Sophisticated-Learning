#!/bin/bash
#SBATCH --job-name=zip_results
#SBATCH --output=zip_results_%j.out
#SBATCH --error=zip_results_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=ada

# zip_missing_grid_data.sh
# Compresses the required .txt files into missing_grid_data.zip

set -euo pipefail

# --- paths ---------------------------------------------------------------
SRC_DIR='/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_3horizononly_extra'
DEST_ZIP='/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/missing_grid_data.zip'

# --- start ---------------------------------------------------------------
# Remove a pre-existing archive so we don’t append duplicates
rm -f "${DEST_ZIP}"

# Change to the source directory so the archive stores only relative paths
cd "${SRC_DIR}"

# Find matching files and feed the list to zip via standard input (-@)
# file must contain both substrings
find . -type f -name '*SI_Seed_*751499cb304e0b35280d048a2826f0a1.txt' -print | zip -@ "${DEST_ZIP}"

echo "Created ${DEST_ZIP}"
