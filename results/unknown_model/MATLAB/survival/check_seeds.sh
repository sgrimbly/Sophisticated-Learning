#!/bin/bash

# Define the directory with the files
dir="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/survival"

# Define expected seed range
declare -A seeds
for i in {1..130}; do
    seeds[$i]=1
done

# Arrays to hold the seeds for each algorithm
declare -A BA_seeds BAUCB_seeds SI_seeds SL_seeds

# Loop over all text files in the new format
cd $dir
for file in *.txt; do
    if [[ $file =~ (BA|BAUCB|SI|SL)_Seed([0-9]+)_ ]]; then
        algorithm="${BASH_REMATCH[1]}"
        seed="${BASH_REMATCH[2]}"

        # Store seeds in a corresponding array
        declare -n current_seeds="${algorithm}_seeds"
        ((current_seeds[$seed]++))
    fi
done

# Function to check seed integrity
check_seeds() {
    algorithm=$1
    declare -n seeds_to_check="${algorithm}_seeds"
    missing=()
    duplicates=()
    for seed in "${!seeds[@]}"; do
        if [[ -z "${seeds_to_check[$seed]}" ]]; then
            missing+=($seed)
        elif [[ "${seeds_to_check[$seed]}" -gt 1 ]]; then
            duplicates+=($seed)
        fi
    done

    # Sort missing and duplicate arrays
    IFS=$'\n' missing=($(sort -n <<<"${missing[*]}"))
    IFS=$'\n' duplicates=($(sort -n <<<"${duplicates[*]}"))
    unset IFS

    echo "$algorithm:"
    echo "  Total Seeds: ${#seeds_to_check[@]}"
    if [ ${#missing[@]} -ne 0 ]; then
        echo "  Missing: ${missing[*]}"
    fi
    if [ ${#duplicates[@]} -ne 0 ]; then
        echo "  Duplicates: ${duplicates[*]}"
    fi
    echo
}

# Check each algorithm
check_seeds BA
check_seeds BAUCB
check_seeds SI
check_seeds SL
