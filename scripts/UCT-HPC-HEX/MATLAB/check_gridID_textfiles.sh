#!/bin/bash

# Target directory
DIR="/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_3horizononly_extra"

# List of grid IDs of interest
GRID_IDS=(
    "751499cb304e0b35280d048a2826f0a1"
    "7b9c3e7c089c1887e7171cf6d8a3d523"
    "5b80f72def8dc45879c4a24db1326f37"
    "9c65a20092568b4e4e722157aaac50ea"
    "0ed346d295ed11b35a2da67eb1b7940e"
    "1dcfc296a6f022e53348ec030c074957"
    "5aca93921cf0b4a73d1b177ae30030a2"
    "c08857a41d8807252bcd7dcbfd9803bb"
    "1e410e8b9dac71549ba84711f85e7490"
    "49b572651fd399dec95ed7f7926e4685"
)

echo "Checking .txt files for correct line count..."

# Track all matching files to exclude them in the second pass
declare -A matched_files

# Check files for correct line counts
for GRID_ID in "${GRID_IDS[@]}"; do
    for FILE in "$DIR"/*"$GRID_ID"*.txt; do
        if [ -f "$FILE" ]; then
            matched_files["$FILE"]=1
            LINE_COUNT=$(wc -l < "$FILE")
            if [ "$LINE_COUNT" -ne 200 ]; then
                echo "Incorrect line count ($LINE_COUNT): $FILE"
            fi
        fi
    done
done

echo ""
echo "Checking for .txt files that do NOT match any known Grid ID..."

# Loop through all .txt files and report those not matched
for FILE in "$DIR"/*.txt; do
    if [ -f "$FILE" ] && [ -z "${matched_files[$FILE]}" ]; then
        echo "Unexpected file (Grid ID not recognised): $FILE"
    fi
done
