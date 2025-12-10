# --- 1. Load Required Packages ---
# Need stringr for extracting seed numbers and algorithm names from filenames
if (!require("stringr", quietly = TRUE)) {
  install.packages("stringr")
}
library(stringr)

# --- 2. Define Data Directory, Expected Line Count, and Algorithms ---
# Ensure this path is correct for your system
data_directory <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/120trials_data_threefry"
# Alternative for Windows paths:
# data_directory <- "C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning\\results\\unknown_model\\MATLAB\\120trials_data_threefry"

expected_line_count <- 120 # The number of lines each file should have
# Define the expected algorithms to initialize the list structure and check against
expected_algorithms <- c("BA", "BAUCB", "SI", "SL") 

print(paste("Checking .txt files in directory:", data_directory))
print(paste("Expecting each file to have exactly", expected_line_count, "lines."))

# --- 3. List Text Files ---
# Get full paths to all .txt files in the specified directory
all_files <- list.files(path = data_directory, pattern = "\\.txt$", full.names = TRUE, recursive = FALSE)

# Check if any files were found
if (length(all_files) == 0) {
  stop("No .txt files found in the specified directory: ", data_directory)
} else {
  print(paste("Found", length(all_files), "text files to check."))
}

# --- 4. Loop, Read, Count Lines, and Extract Problematic Seeds by Algorithm ---
# Initialize a list to store problematic seeds, named by algorithm
problematic_seeds_by_algo <- vector("list", length(expected_algorithms))
names(problematic_seeds_by_algo) <- expected_algorithms
for(algo in expected_algorithms) { # Ensure each element is initialized as a vector
  problematic_seeds_by_algo[[algo]] <- character()
}

checked_count <- 0
problematic_file_count_total <- 0

print("Starting line count check and seed extraction by algorithm...")

for (file_path in all_files) {
  filename <- basename(file_path) # Get just the filename

  tryCatch({
    # Read all lines from the file
    lines <- readLines(con = file_path, warn = FALSE)
    actual_line_count <- length(lines)

    # Check if the count matches the expected count
    if (actual_line_count != expected_line_count) {
      problematic_file_count_total <- problematic_file_count_total + 1 # Count total problematic files

      # Extract Algorithm Name
      algo_match <- stringr::str_match(filename, "^(.+?)_Seed_")
      algorithm_name <- if(!is.na(algo_match[1,1])) algo_match[1,2] else NA

      # Extract Seed Number
      seed_match <- stringr::str_match(filename, "_Seed_(\\d+)_")
      seed_id <- if(!is.na(seed_match[1,1])) seed_match[1,2] else NA

      # Check if extraction worked and if algorithm is one we expect
      if (!is.na(algorithm_name) && !is.na(seed_id) && (algorithm_name %in% expected_algorithms)) {
        # Add the seed ID to the list for the corresponding algorithm
        problematic_seeds_by_algo[[algorithm_name]] <- c(problematic_seeds_by_algo[[algorithm_name]], seed_id)
      } else {
          if (is.na(seed_id)) {
             warning(paste("Could not extract seed number from problematic file:", filename))
          }
          if (is.na(algorithm_name) || !(algorithm_name %in% expected_algorithms)) {
             warning(paste("Could not extract a valid/expected algorithm name from problematic file:", filename))
          }
      }
    }

    checked_count <- checked_count + 1
    # Optional: Print progress
    # if (checked_count %% 500 == 0) { print(paste("Checked", checked_count, "files..."))}

  }, error = function(e) {
    # Handle potential errors during file reading
    warning(paste("Error reading file:", filename, "-", e$message))
  })
}

print("Line count check finished.")

# --- 5. Process and Report Problematic Seeds by Algorithm ---
if (problematic_file_count_total == 0) {
  print(paste("Success! All", checked_count, "checked files have exactly", expected_line_count, "lines."))
  print("No problematic seeds found for any algorithm.")
  # Print empty bash lists for all expected algorithms
  cat("\nBash formatted lists of problematic seeds (by Algorithm):\n")
  for (algo in expected_algorithms) {
      cat(paste0(algo, "_SEEDS=()\n"))
  }
} else {
  print(paste("Found a total of", problematic_file_count_total, "files with line counts different from", expected_line_count, "out of", checked_count, "files checked."))
  
  print("---------------------------------------------------------")
  print("Bash formatted lists of unique problematic seeds (by Algorithm):")
  print("(Copy the relevant list(s) below and paste into your bash script)")
  print("---------------------------------------------------------")

  # Loop through each algorithm
  for (algo in names(problematic_seeds_by_algo)) {
    raw_seeds_for_algo <- problematic_seeds_by_algo[[algo]]
    
    if (length(raw_seeds_for_algo) > 0) {
      # Get unique seed numbers for this algorithm
      unique_problematic_seeds <- unique(raw_seeds_for_algo)
      # Convert to numeric and sort
      unique_problematic_seeds_numeric <- sort(as.numeric(unique_problematic_seeds))
      
      num_unique_seeds <- length(unique_problematic_seeds_numeric)
      print(paste0("Found ", num_unique_seeds, " unique problematic seeds for algorithm '", algo, "'."))

      # --- Format and Print the Bash-Readable List for this Algorithm ---
      # Collapse the sorted numeric seeds into a single space-separated string
      seeds_string <- paste(unique_problematic_seeds_numeric, collapse = " ")
      # Format as a bash array declaration: ALGO_SEEDS=(seed1 seed2 ...)
      bash_output_string <- paste0(algo, "_SEEDS=(", seeds_string, ")")

      # Use cat() to print the exact string to the console without quotes or indices
      cat(bash_output_string)
      cat("\n\n") # Add extra newline for separation
      
    } else {
      print(paste0("Found 0 problematic seeds for algorithm '", algo, "'."))
      # Print empty bash list for this algorithm
      cat(paste0(algo, "_SEEDS=()\n\n"))
    }
  }
   print("---------------------------------------------------------")
}

print("--- Script Finished ---")
