#!/bin/bash

# The directory containing all experiment directories
cd /path/to/experiment/folders

# Loop over each combination of algorithm and seed
for algorithm in SI SL BA BAUCB; do
    for seed in {1..30}; do
        # Construct the base prefix
        prefix="${algorithm}_Seed${seed}"
        
        # List, sort, and remove all but the most recent directory for this prefix
        ls -d "${prefix}"_* 2>/dev/null | sort | head -n -1 | while read dir; do
            echo "Deleting older directory: $dir"
            rm -rf "$dir"  # Remove the directory safely
        done
    done
done
