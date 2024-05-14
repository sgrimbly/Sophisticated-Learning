#!/bin/bash

# Change to the directory with the files
cd /home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/survival

# Loop over all text files
for file in *.txt; do
    # Extract the parts of the filename
    if [[ $file =~ ([0-9\-]+)_seed_([0-9]+)_(BA|BAUCB|SI|SL)_experiment\.txt ]]; then
        timestamp="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"
        algorithm="${BASH_REMATCH[3]}"

        # Form the new filename
        new_filename="${algorithm}_Seed${seed}_${timestamp}.txt"

        # Rename the file
        mv "$file" "$new_filename"
    fi
done
