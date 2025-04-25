# --- 1. Package Management ---
# Optional: Detach moments if loaded and causing conflicts
# if ("moments" %in% loadedNamespaces()) detach("package:moments", unload = TRUE, character.only = TRUE)

required_packages <- c(
  "readr",     # For reading text files efficiently (read_delim)
  "reshape2",  # Used in original, might not be strictly needed now but keep for consistency if other parts rely on it elsewhere
  "lme4",      # For Linear Mixed-Effects Models (lmer)
  "multcomp",  # For post-hoc tests (like glht, though commented out below)
  "car",       # For Anova function (Type II/III ANOVA)
  "lmerTest",  # Provides p-values for lmer models and enhances summary/anova
  "emmeans",   # For Estimated Marginal Means and pairwise comparisons
  "stringr"    # For string manipulation (extracting info from filenames)
)
cran_mirror <- "https://cran.rstudio.com/"

# Check and install missing packages
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE, repos = cran_mirror)
    library(pkg, character.only = TRUE)
  }
}

# --- 2. Define Data Directory ---
# Use forward slashes or double backslashes for paths in R
data_directory <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/120trials_data_threefry"
# Alternative for Windows paths:
# data_directory <- "C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning\\results\\unknown_model\\MATLAB\\120trials_data_threefry"


# --- 3. List Data Files ---
# List all .txt files in the directory
all_files <- list.files(path = data_directory, pattern = "\\.txt$", full.names = TRUE)

# Check if files were found
if (length(all_files) == 0) {
  stop("No .txt files found in the specified directory: ", data_directory)
} else {
  print(paste("Found", length(all_files), "text files."))
}


# --- 4. Data Loading and Preparation Loop ---
data_list <- list() # Initialize an empty list to store data frames

# Loop through files, read data, extract info, and prepare data frame
for (file_path in all_files) {
  # Extract filename from the full path
  filename <- basename(file_path)
  
  # Extract Algorithm Name (part before "_Seed_")
  # Uses regex: ^(.+?)_Seed_ captures characters from the start until "_Seed_" non-greedily
  algo_match <- str_extract(filename, "^(.+?)_Seed_") 
  algorithm_name <- sub("_Seed_$", "", algo_match) # Remove trailing "_Seed_"

  # Extract Seed Number (number between "_Seed_" and the next "_")
  # Uses regex: _Seed_(\\d+)_ captures one or more digits after "_Seed_" and before the next "_"
  seed_match <- str_extract(filename, "_Seed_(\\d+)_")
  seed_id <- sub("_Seed_", "", sub("_$", "", seed_match)) # Remove surrounding underscores and "Seed"

  # Check if extraction worked
  if (is.na(algorithm_name) || is.na(seed_id)) {
    warning(paste("Could not extract Algorithm or Seed from filename:", filename, "- Skipping file."))
    next # Skip to the next file
  }
  
  # Read the single column of survival data
  # Assumes only one column, no header. Delimiter doesn't strictly matter if only one column.
  # Using read_table as a robust option for single column.
  tryCatch({
      survival_data <- readr::read_table(file_path, col_names = FALSE, show_col_types = FALSE)
      
      # Check if data read correctly (should have 1 column and 120 rows)
      if (ncol(survival_data) != 1 || nrow(survival_data) != 120) {
         warning(paste("File", filename, "does not contain exactly 1 column and 120 rows. Skipping."))
         next 
      }

      # Create the data frame for this file
      data_list[[filename]] <- data.frame(
        Survival = survival_data[[1]], # Get the first (only) column as a vector
        Iteration = 1:120,             # Trials 1 to 120
        Id = seed_id,                  # Seed number serves as the subject ID
        Algorithm = algorithm_name     # Algorithm name extracted from filename
      )
  }, error = function(e) {
      warning(paste("Error reading file:", filename, "-", e$message, "- Skipping file."))
  })

}

# Check if any data was successfully processed
if (length(data_list) == 0) {
  stop("No data was successfully processed. Check file formats and warnings.")
}

# --- 5. Combine Data ---
combined_df <- do.call(rbind, data_list)
rownames(combined_df) <- NULL # Reset row names

# Convert ID and Algorithm to factors for modeling
combined_df$Id <- factor(combined_df$Id)
combined_df$Algorithm <- factor(combined_df$Algorithm)

print("Data loading and processing complete. First few rows of combined data:")
print(head(combined_df))
print("Summary of combined data:")
summary(combined_df)
print(paste("Number of unique IDs (Seeds):", length(unique(combined_df$Id))))
print("Algorithm levels:")
print(levels(combined_df$Algorithm))


# --- 6. Fit the Linear Mixed-Effects Model ---
# Model Survival based on Iteration, Algorithm, and their interaction,
# with a random intercept for each Id (Seed).
# Make sure variable names match those created above (Survival, Iteration, Algorithm, Id)
model <- lmer(Survival ~ Iteration + Algorithm + Iteration * Algorithm + (1 | Id), data = combined_df)


# --- 7. Analyze and Report Results ---
print("--- Model Summary ---")
# Provides estimates, SEs, t-values for fixed effects, variances for random effects
print(summary(model)) 

# Optional: Use summary from lmerTest for p-values (default uses Satterthwaite approx)
# print(summary(model, ddf="Kenward-Roger")) # Alternative df approximation

# Optional: Tukey's Post Hoc Test using multcomp (useful if >2 Algorithm levels)
# print("--- Tukey's Post Hoc Test for Algorithm ---")
# tryCatch({
#   test <- multcomp::glht(model, linfct = mcp(Algorithm = "Tukey"))
#   print(summary(test))
# }, error = function(e) {
#   print(paste("Could not perform glht post-hoc test:", e$message))
# })


print("--- ANOVA for Fixed Effects (Type II/III) ---")
# Use car::Anova for Type II or III tests with F-statistics
# Default is Type II, which is often appropriate for balanced designs or models without interactions removed
# Specify type=3 if you prefer Type III sums of squares
tryCatch({
    anova_results <- car::Anova(model, test = 'F') 
    print(anova_results)
    # Save ANOVA results
    output_anova_file <- file.path(data_directory, "my_analysis_anova_results.csv")
    write.csv(as.data.frame(anova_results), file = output_anova_file, row.names = TRUE)
    print(paste("ANOVA results saved to:", output_anova_file))
}, error = function(e) {
    print(paste("Could not perform car::Anova:", e$message))
})


print("--- Estimated Marginal Means (EMMs) for Algorithm ---")
# Calculates adjusted means for each Algorithm level and pairwise comparisons
tryCatch({
    em <- emmeans::emmeans(model, pairwise ~ Algorithm)
    print(em)
    # Save EMM results
    output_emmeans_file <- file.path(data_directory, "my_analysis_emmeans_results.csv")
    # emmeans results are complex; save both parts if needed
    write.csv(summary(em$emmeans), file = sub(".csv", "_means.csv", output_emmeans_file), row.names = FALSE)
    write.csv(summary(em$contrasts), file = sub(".csv", "_contrasts.csv", output_emmeans_file), row.names = FALSE)
    print(paste("EMMeans results saved to:", output_emmeans_file, "(and related files)"))
}, error = function(e) {
    print(paste("Could not calculate emmeans:", e$message))
})


print("--- Script Finished ---")