#!/bin/bash

# Define the directory with the files
# dir="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/survival"

# Define expected seed range
declare -A seeds
for i in {1..130}; do
    seeds[$i]=1
done

# Arrays to hold the seeds for each algorithm
declare -A BA_seeds BAUCB_seeds SI_seeds SL_seeds

# Loop over all text files in the new format
# cd $dir
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

# Powershell Version
# Define expected seed range
# $seeds = @{}
# for ($i = 1; $i -le 130; $i++) {
#     $seeds[$i] = 1
# }

# # Hashtables to hold the seeds for each algorithm
# $BA_seeds = @{}
# $BAUCB_seeds = @{}
# $SI_seeds = @{}
# $SL_seeds = @{}

# # Loop over all text files in the directory
# Get-ChildItem -Filter *.txt | ForEach-Object {
#     $file = $_.Name
#     if ($file -match "(BA|BAUCB|SI|SL)_Seed([0-9]+)_") {
#         $algorithm = $matches[1]
#         $seed = [int]$matches[2]

#         # Store seeds in a corresponding hashtable
#         $current_seeds = Get-Variable -Name "${algorithm}_seeds" -ValueOnly
#         $current_seeds[$seed]++
#         Set-Variable -Name "${algorithm}_seeds" -Value $current_seeds
#     }
# }

# # Function to check seed integrity
# function Check-Seeds {
#     param ([string]$algorithm)
#     $seeds_to_check = Get-Variable -Name "${algorithm}_seeds" -ValueOnly
#     $missing = @()
#     $duplicates = @()

#     foreach ($seed in $seeds.Keys) {
#         if (-not $seeds_to_check.ContainsKey($seed)) {
#             $missing += $seed
#         } elseif ($seeds_to_check[$seed] -gt 1) {
#             $duplicates += $seed
#         }
#     }

#     # Sort missing and duplicate arrays
#     $missing = $missing | Sort-Object
#     $duplicates = $duplicates | Sort-Object

#     Write-Output "${algorithm}:"
#     Write-Output "  Total Seeds: $($seeds_to_check.Count)"
#     if ($missing.Length -ne 0) {
#         Write-Output "  Missing: $($missing -join ', ')"
#     }
#     if ($duplicates.Length -ne 0) {
#         Write-Output "  Duplicates: $($duplicates -join ', ')"
#     }
#     Write-Output ""
# }

# # Check each algorithm
# Check-Seeds "BA"
# Check-Seeds "BAUCB"
# Check-Seeds "SI"
# Check-Seeds "SL"
