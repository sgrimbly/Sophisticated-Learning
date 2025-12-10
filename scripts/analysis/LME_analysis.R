# --- 1. Package Management ---
# This section ensures all required packages are installed and loaded.
required_packages <- c(
  "readr",     # For reading text files efficiently
  "lme4",      # For Linear Mixed-Effects Models (lmer)
  "car",       # For Anova function (Type II/III ANOVA)
  "lmerTest",  # Provides p-values for lmer models and enhances summary/anova
  "emmeans",   # For Estimated Marginal Means and pairwise comparisons
  "stringr",   # For string manipulation (extracting info from filenames)
  "ggplot2",   # For plotting
  "data.table", # For efficient data handling (rbindlist, subsetting)
  "Hmisc"      # For mean_cl_normal (used by ggplot2 for CIs)
)

cran_mirror <- "https://cran.rstudio.com/" # Using RStudio's CRAN mirror

print("--- Section 1: Checking and Installing Packages ---")
for (pkg in required_packages) {
  if (!paste0("package:", pkg) %in% search()) {
      if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
          print(paste("Installing package:", pkg))
          install.packages(pkg, dependencies = TRUE, repos = cran_mirror)
          if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
              stop(paste("Failed to install or load package:", pkg))
          }
      }
      library(pkg, character.only = TRUE)
      print(paste("Loaded package:", pkg))
  }
}
if("lme4" %in% required_packages && "lmerTest" %in% required_packages){
  print("Note: lmerTest masks lmer from lme4. This is expected.")
}
print("--- Package Check Complete ---")

# --- 2. Define Data Directory & Phase Parameters ---
print("--- Section 2: Defining Data Directory and Phase Parameters ---")
data_directory <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/120trials_data_threefry" # USER: Ensure this path is correct

# Define trial windows for phase analysis
phase1_end_trial <- 20      # End of Initial Ramp-up phase
phase2_start_trial <- 21    # Start of Active Learning phase
phase2_end_trial <- 60      # End of Active Learning phase
phase3_start_trial <- phase2_end_trial + 1 # Start of Taper/Convergence phase
phase3_end_trial <- 120     # End of Taper/Convergence phase (assuming 120 total trials)


print(paste("Data directory set to:", data_directory))
print(paste("Phase 1 (Ramp-up) analysis: Trials 1 to", phase1_end_trial))
print(paste("Phase 2 (Active Learning) analysis: Trials", phase2_start_trial, "to", phase2_end_trial))
print(paste("Phase 3 (Taper/Convergence) analysis: Trials", phase3_start_trial, "to", phase3_end_trial))


# --- 3. Check for Missing Files ---
print("--- Section 3: Checking for Missing Files ---")
expected_algorithms <- c("BA", "BAUCB", "SI", "SL")
expected_seeds <- 1:2000 # Assuming 2000 seeds as per output
actual_filenames <- list.files(path = data_directory, pattern = "\\.txt$", full.names = FALSE)

if (length(actual_filenames) == 0) {
  print("No .txt files found in the data directory.")
  stop("Execution halted: No data files found.")
} else {
  expected_prefixes <- character()
  for (algo in expected_algorithms) {
    for (seed_num in expected_seeds) {
      expected_prefixes <- c(expected_prefixes, paste0(algo, "_Seed_", seed_num, "_"))
    }
  }
  print(paste("Total expected Algorithm/Seed combinations (file prefixes):", length(expected_prefixes)))
  prefix_exists <- sapply(expected_prefixes, function(prefix_to_check) {
      any(startsWith(actual_filenames, prefix_to_check))
  })
  missing_combinations_report <- sub("_$", "", expected_prefixes[!prefix_exists])
  found_count <- sum(prefix_exists)
  print(paste("Number of expected combinations found based on filename prefixes:", found_count))
  print(paste("Number of expected combinations missing:", length(missing_combinations_report)))
  if (length(missing_combinations_report) > 0) {
    print("WARNING: The following Algorithm_Seed combinations appear to be missing files:")
    print(head(missing_combinations_report, n = 50)) # Show up to 50 missing
  } else {
    print("All expected Algorithm_Seed combinations appear to be present based on filename prefixes.")
  }
}
print("--- Finished Checking for Missing Files ---")

# --- 4. List Data Files for Loading ---
print("--- Section 4: Listing Data Files for Loading ---")
all_files <- list.files(path = data_directory, pattern = "\\.txt$", full.names = TRUE)
if (length(all_files) == 0) {
  stop("No .txt files found for loading: ", data_directory)
} else {
  print(paste("Proceeding to load data from", length(all_files), "text files."))
}

# --- 5. Data Loading and Preparation Loop ---
print("--- Section 5: Loading and Preparing Data ---")
data_list <- list()
skipped_files_count <- 0
error_files_details <- list()

for (file_path in all_files) {
  filename <- basename(file_path)
  # Improved regex to be more robust for typical naming like "ALGO_Seed_SEEDNUM_otherstuff.txt"
  algo_match <- stringr::str_match(filename, "^([A-Z]+(?:UCB)?)_Seed_")
  algorithm_name <- if(!is.na(algo_match[1,1])) algo_match[1,2] else NA
 
  seed_match <- stringr::str_match(filename, "_Seed_(\\d+)_")
  seed_id <- if(!is.na(seed_match[1,1])) seed_match[1,2] else NA

  if (is.na(algorithm_name) || is.na(seed_id)) {
    msg <- paste("Could not extract Algorithm or Seed from filename:", filename)
    # Only print warning for actual data-like files, not miscellaneous .txt files
    if (grepl("_Seed_\\d+_", filename) || grepl("^(BA|SI|SL|BAUCB)", filename, ignore.case = TRUE)) { # Made algo check case-insensitive
        warning(msg)
    }
    skipped_files_count <- skipped_files_count + 1
    error_files_details[[filename]] <- msg; next
  }
  tryCatch({
      lines <- readr::read_lines(file_path, progress = FALSE, n_max = 120) # Ensure only 120 lines read for safety
      if (length(lines) != 120) { # Assuming 120 trials per file
          msg <- paste("File", filename, "does not contain 120 lines (found", length(lines), ").")
          warning(msg); skipped_files_count <- skipped_files_count + 1
          error_files_details[[filename]] <- msg; next
      }
      survival_values <- as.numeric(lines)
      if(any(is.na(survival_values))) {
          msg <- paste("File", filename, "contains non-numeric data.")
          warning(msg); skipped_files_count <- skipped_files_count + 1
          error_files_details[[filename]] <- msg; next
      }
      data_list[[filename]] <- data.table( # Use data.table directly
        Survival = survival_values,
        Iteration = 1:120,
        Id = factor(seed_id), # Factorize Id here
        Algorithm = factor(algorithm_name, levels = expected_algorithms) # Factorize Algorithm here with predefined levels
      )
  }, error = function(e) {
      msg <- paste("Error processing file:", filename, "-", e$message)
      warning(msg); skipped_files_count <- skipped_files_count + 1
      error_files_details[[filename]] <- msg
  })
}
if (skipped_files_count > 0) {
    print(paste("WARNING: Skipped", skipped_files_count, "files. First few errors/skipped non-data files:"))
    print(head(error_files_details, 10))
}
if (length(data_list) == 0) stop("No data was successfully processed.")
print("--- Finished Loading and Preparing Data ---")

# --- 6. Combine Data ---
print("--- Section 6: Combining Data ---")

combined_df_untrimmed <- data.table::rbindlist(data_list) # This is the full, untrimmed dataset

# Rename algorithm levels for plot labelling purposes
levels(combined_df_untrimmed$Algorithm) <- c(
  "BARL",      # formerly "BA"
  "BARLUCB",   # formerly "BAUCB"
  "SI",        # unchanged
  "SL"         # unchanged
)

print("Full untrimmed data summary ('combined_df_untrimmed'):")
summary(combined_df_untrimmed)

# Now define subsets
df_phase1_ramp_up <- combined_df_untrimmed[Iteration >= 1 & Iteration <= phase1_end_trial]
df_phase2_active_learning <- combined_df_untrimmed[Iteration >= phase2_start_trial & Iteration <= phase2_end_trial]
df_phase3_taper <- combined_df_untrimmed[Iteration >= phase3_start_trial & Iteration <= phase3_end_trial]

# Print summaries
print(paste("Total observations in untrimmed data:", nrow(combined_df_untrimmed)))
print(paste("Unique IDs found:", length(unique(combined_df_untrimmed$Id))))
print("Algorithm levels and counts:")
print(table(combined_df_untrimmed$Algorithm))

print(paste("Observations in Phase 1 (Ramp-up) data ('df_phase1_ramp_up'):", nrow(df_phase1_ramp_up)))
if(nrow(df_phase1_ramp_up) > 0) summary(df_phase1_ramp_up)
print(paste("Observations in Phase 2 (Active Learning) data ('df_phase2_active_learning'):", nrow(df_phase2_active_learning)))
if(nrow(df_phase2_active_learning) > 0) summary(df_phase2_active_learning)
print(paste("Observations in Phase 3 (Taper/Convergence) data ('df_phase3_taper'):", nrow(df_phase3_taper)))
if(nrow(df_phase3_taper) > 0) summary(df_phase3_taper)


print("--- Finished Data Preparation & Phasing ---")


# --- 7. Plotting Raw Learning Curves (Using combined_df_untrimmed to show full range) with 95% CIs ---
print("--- Section 7: Generating Plot of Raw Learning Curves (Full Original Range, 95% CIs) ---")
plot_title_raw <- "Average Learning Curves by Algorithm" # MODIFIED TITLE
# if (skipped_files_count > 0) {
#     plot_title_raw <- paste0(plot_title_raw, "\nNOTE: Based on data from successfully loaded files (", length(data_list), " files loaded, ", skipped_files_count, " files skipped/had issues).")
# }

if(nrow(combined_df_untrimmed) > 0){
    learning_curve_plot_full_range_CI <- ggplot(data = combined_df_untrimmed, aes(x = Iteration, y = Survival, color = Algorithm)) +
      stat_summary(fun = mean, geom = "line", linewidth = 1) +
      stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") + # Using 95% CIs
      labs(title = plot_title_raw,
          #  subtitle = "Lines represent mean survival, ribbons represent 95% Confidence Intervals for the Mean",
           x = "Trial", y = "Average Survival Steps") +
      theme_bw(base_size = 11) + # Adjusted base_size for typical paper figs
      theme(plot.title = element_text(hjust = 0.5, face = "bold", size=rel(1.1)),
            # plot.subtitle = element_text(hjust = 0.5, size=rel(0.9)),
            legend.position = "bottom",
            panel.grid.major = element_line(colour = "grey90"),
            panel.grid.minor = element_blank()) +
      geom_vline(xintercept = phase1_end_trial + 0.5, linetype = "dashed", color = "blue", linewidth=0.8) +
      annotate("text", x = phase1_end_trial + 0.5, y = max(combined_df_untrimmed$Survival, na.rm=TRUE)*0.95,
               label = paste("End of Phase 1\n(Trial ", phase1_end_trial, ")", sep=""), angle = 90, vjust = -0.3, hjust=1, size=3, color="blue") +
      geom_vline(xintercept = phase2_end_trial + 0.5, linetype = "dashed", color = "red", linewidth=0.8) +
      annotate("text", x = phase2_end_trial + 0.5, y = max(combined_df_untrimmed$Survival, na.rm=TRUE)*0.95, # Adjusted y for label placement
               label = paste("End of Phase 2\n(Trial ", phase2_end_trial, ")", sep=""), angle = 90, vjust = -0.3, hjust=1, size=3, color="red") +
      scale_color_brewer(palette = "Set1") + # Example of using a color palette
      scale_fill_brewer(palette = "Set1")   # Consistent fill palette

    print(learning_curve_plot_full_range_CI)
    plot_filename_raw_full_range_CI_pdf <- file.path(data_directory, "learning_curve_plot_raw_data_full_range_CI_with_phases.pdf") # Appended filename
    tryCatch({
      ggsave(filename = plot_filename_raw_full_range_CI_pdf,
             plot = learning_curve_plot_full_range_CI,
             device = "pdf",
             width = 7, height = 5, units = "in") # Typical full-width figure size
      print(paste("Raw data learning curve plot (full range, 95% CIs, with phase lines) saved to:", plot_filename_raw_full_range_CI_pdf))
    }, error = function(e) {
      print(paste("Could not save raw data CI plot (full range) as PDF:", e$message))
    })
} else {
    print("Skipping main learning curve plot as combined_df_untrimmed is empty.")
}
print("--- Finished Plotting Raw Learning Curves (Full Range, 95% CIs) ---")


# --- 7A. ORIGINAL FULL LINEAR MODEL ANALYSIS (on Untrimmed Data) --- (COMMENTED OUT) ---
# print(paste0("--- Section 7A: [COMMENTED OUT] ORIGINAL Full Linear Model Analysis (All Trials: 1-120, Untrimmed Data) ---"))
# if (nrow(combined_df_untrimmed) > 0) {
#     model_original_linear_untrimmed <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = combined_df_untrimmed)
#     print("--- [COMMENTED OUT] Finished Fitting Original Full Linear LME Model (Untrimmed) ---")

#     print("--- [COMMENTED OUT] Model Summary (Original Full Linear Model - Untrimmed) ---")
#     # print(summary(model_original_linear_untrimmed))
#     print("--- [COMMENTED OUT] ANOVA for Fixed Effects (Original Full Linear Model - Untrimmed) ---")
#     # tryCatch({
#     #     anova_results_original_linear <- car::Anova(model_original_linear_untrimmed, test = 'F', type="III")
#     #     print(anova_results_original_linear)
#     #     output_file <- file.path(data_directory, "anova_results_original_linear_untrimmed.csv")
#     #     write.csv(as.data.frame(anova_results_original_linear), file = output_file, row.names = TRUE)
#     #     print(paste("Original Full Linear ANOVA results (untrimmed) saved to:", output_file))
#     # }, error = function(e) { print(paste("Could not perform car::Anova on original full linear model (untrimmed):", e$message)) })

#     print("--- [COMMENTED OUT] Probing Interaction for Original Full Linear Model (Untrimmed) ---")
#     # iteration_points_orig_untrimmed <- c(15, 50, 75, 110) 
#     # iteration_points_orig_untrimmed <- iteration_points_orig_untrimmed[iteration_points_orig_untrimmed <= max(combined_df_untrimmed$Iteration) &
#     #                                                                  iteration_points_orig_untrimmed >= min(combined_df_untrimmed$Iteration)]
#     # if(length(iteration_points_orig_untrimmed) == 0 && nrow(combined_df_untrimmed) > 0) iteration_points_orig_untrimmed <- median(combined_df_untrimmed$Iteration)


#     print("--- [COMMENTED OUT] Comparing Learning Rates (Slopes) - Original Full Linear Model (Untrimmed) ---")
#     # tryCatch({
#     #     trends_original_linear <- emmeans::emtrends(model_original_linear_untrimmed, pairwise ~ Algorithm, var = "Iteration")
#     #     print(summary(trends_original_linear, infer = TRUE))
#     #     output_base <- file.path(data_directory, "emtrends_results_original_linear_untrimmed")
#     #     write.csv(summary(trends_original_linear$emtrends), file = paste0(output_base, "_trends.csv"), row.names = FALSE)
#     #     write.csv(summary(trends_original_linear$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
#     #     print(paste("Original Full Linear emtrends results (untrimmed) saved to files starting with:", output_base))
#     # }, error = function(e) { print(paste("Could not calculate emtrends for original full linear model (untrimmed):", e$message)) })

#     # if(length(iteration_points_orig_untrimmed) > 0) {
#     #     print("--- [COMMENTED OUT] Comparing Algorithms at Specific Iterations - Original Full Linear Model (Untrimmed) ---")
#     #     tryCatch({
#     #         emmeans_at_iter_orig_linear <- emmeans::emmeans(model_original_linear_untrimmed, pairwise ~ Algorithm | Iteration, at = list(Iteration = iteration_points_orig_untrimmed))
#     #         print(summary(emmeans_at_iter_orig_linear, infer = TRUE))
#     #         output_base <- file.path(data_directory, "emmeans_at_iterations_original_linear_untrimmed")
#     #         write.csv(summary(emmeans_at_iter_orig_linear$emmeans), file = paste0(output_base, "_means.csv"), row.names = FALSE)
#     #         write.csv(summary(emmeans_at_iter_orig_linear$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
#     #         print(paste("Original Full Linear emmeans at specific iterations (untrimmed) saved to files starting with:", output_base))
#     #     }, error = function(e) { print(paste("Could not calculate emmeans for original full linear model (untrimmed):", e$message)) })
#     # } else {
#     #     print("Skipping emmeans at specific iterations for original full linear model as no valid iteration points were found.")
#     # }

#     print("--- [COMMENTED OUT] Plotting Fit for Original Full Linear Model (Untrimmed) ---")
#     # emm_preds_orig_linear <- emmeans(model_original_linear_untrimmed, ~ Algorithm | Iteration, at = list(Iteration = sort(unique(combined_df_untrimmed$Iteration))), type = "response")
#     # predicted_df_orig_linear <- as.data.frame(summary(emm_preds_orig_linear))
#     # names(predicted_df_orig_linear)[names(predicted_df_orig_linear) == "emmean"] <- "PredictedSurvival"
#     # plot_orig_linear_fit <- ggplot(data = combined_df_untrimmed, aes(x = Iteration, y = Survival, color = Algorithm)) +
#     #     stat_summary(fun = mean, geom = "line", linewidth = 0.8, alpha = 0.5, aes(linetype = "Observed Mean")) +
#     #     stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.1, aes(fill = Algorithm), linetype = "blank") + # Or mean_cl_normal for CIs
#     #     geom_line(data = predicted_df_orig_linear, aes(x = Iteration, y = PredictedSurvival, color = Algorithm, linetype = "Model Fit"), linewidth = 1.1) +
#     #     scale_linetype_manual(name = "Data Type", values = c("Observed Mean" = "dashed", "Model Fit" = "solid")) +
#     #     labs(title = "Original Full Linear Model Fit vs. Observed Untrimmed Data (Trials 1-120)",
#     #          subtitle = "Average learning curves with LME model predictions",
#     #          x = "Trial", y = "Average Survival Steps") +
#     #     theme_bw(base_size = 12) + theme(plot.title = element_text(hjust = 0.5, face = "bold"), plot.subtitle = element_text(hjust = 0.5), legend.position = "bottom")
#     # print(plot_orig_linear_fit)
#     # plot_filename_orig_fit <- file.path(data_directory, "learning_curve_original_linear_model_fit_untrimmed.pdf") # to PDF
#     # tryCatch({ ggsave(filename = plot_filename_orig_fit, plot = plot_orig_linear_fit, device="pdf", width = 10, height = 7, dpi = 300)
#     #            print(paste("Plot with original full linear model fit (untrimmed) saved to:", plot_filename_orig_fit)) },
#     #          error = function(e) { print(paste("Could not save original full linear model fit plot:", e$message)) })
#     print("--- [COMMENTED OUT] Finished Original Full Linear Model Analysis (Untrimmed) ---")
# } else {
#     print("[COMMENTED OUT] Skipping Original Full Linear Model Analysis as 'combined_df_untrimmed' is empty.")
# }
print("--- Section 7A (ORIGINAL Full Linear Model) is COMMENTED OUT ---")


# --- Sections 8, 9, 10: Full NON-LINEAR LME Model (COMMENTED OUT) ---
# print(paste0("--- Section 8: [COMMENTED OUT] Fitting Full LME Model (Trials ", phase2_start_trial, "-120 on a 'trimmed_df') with poly(Iteration, 2) ---"))
# # combined_df_trimmed_for_nonlinear <- combined_df_untrimmed[Iteration >= phase2_start_trial] # Example if needed
# # if (nrow(combined_df_trimmed_for_nonlinear) > 0) { 
# #     model_full_nonlinear <- lmer(Survival ~ poly(Iteration, 2, raw = FALSE) * Algorithm + (1 | Id), data = combined_df_trimmed_for_nonlinear)
# #     print("--- [COMMENTED OUT] Finished Fitting Full Non-Linear LME Model on Trimmed Data ---")
# # 
# #     print("--- Section 9: [COMMENTED OUT] Analyzing Full Non-Linear Model on Trimmed Data ---")
# #     # print("--- Model Summary (Full Non-Linear Model) ---")
# #     # print(summary(model_full_nonlinear))
# #     # print("--- ANOVA for Fixed Effects (Full Non-Linear Model) ---")
# #     # tryCatch({
# #     #     anova_results_full_nonlinear <- car::Anova(model_full_nonlinear, test = 'F', type="III")
# #     #     print(anova_results_full_nonlinear)
# #     #     # output_file <- file.path(data_directory, paste0("anova_results_full_nonlinear_trimmed", phase2_start_trial-1, ".csv"))
# #     #     # write.csv(as.data.frame(anova_results_full_nonlinear), file = output_file, row.names = TRUE)
# #     #     # print(paste("Full Non-Linear ANOVA results (trimmed) saved to:", output_file))
# #     # }, error = function(e) { print(paste("Could not perform car::Anova on full non-linear model (trimmed):", e$message)) })
# #     print("--- [COMMENTED OUT] Finished Analyzing Full Non-Linear Model (Trimmed) ---")
# # 
# #     print("--- Section 10: [COMMENTED OUT] Probing Interaction for Full Non-Linear Model (Trimmed Data) ---")
# #     # iteration_points_full_trimmed <- c(phase2_start_trial + 15, phase2_start_trial + 40, phase2_start_trial + 60, 110) 
# #     # iteration_points_full_trimmed <- iteration_points_full_trimmed[iteration_points_full_trimmed <= max(combined_df_trimmed_for_nonlinear$Iteration)] 
# #     # if(length(iteration_points_full_trimmed) == 0 && nrow(combined_df_trimmed_for_nonlinear) > 0) iteration_points_full_trimmed <- median(combined_df_trimmed_for_nonlinear$Iteration)
# # 
# #     # print("--- Comparing Instantaneous Learning Rates (Slopes) - Full Non-Linear Model (Trimmed) ---")
# #     # tryCatch({
# #     #     trends_full_nonlinear <- emmeans::emtrends(model_full_nonlinear, pairwise ~ Algorithm, var = "Iteration",
# #     #                                                at = list(Iteration = iteration_points_full_trimmed))
# #     #     print(summary(trends_full_nonlinear, infer = TRUE))
# #     #     # output_emtrends_base <- file.path(data_directory, paste0("emtrends_results_full_nonlinear_trimmed", phase2_start_trial-1))
# #     #     # write.csv(summary(trends_full_nonlinear$emtrends), file = paste0(output_emtrends_base, "_trends.csv"), row.names = FALSE)
# #     #     # write.csv(summary(trends_full_nonlinear$contrasts), file = paste0(output_emtrends_base, "_contrasts.csv"), row.names = FALSE)
# #     #     # print(paste("Full Non-Linear emtrends results (trimmed) saved to files starting with:", output_emtrends_base))
# #     # }, error = function(e) { print(paste("Could not calculate emtrends for full non-linear model (trimmed):", e$message)) })
# # 
# #     # print("--- Comparing Algorithms at Specific Iterations - Full Non-Linear Model (Trimmed) ---")
# #     # if(length(iteration_points_full_trimmed) > 0){
# #     #    tryCatch({
# #     #        emmeans_at_iterations_full_nonlinear <- emmeans::emmeans(model_full_nonlinear, pairwise ~ Algorithm | Iteration,
# #     #                                                                 at = list(Iteration = iteration_points_full_trimmed))
# #     #        print(summary(emmeans_at_iterations_full_nonlinear, infer = TRUE))
# #     #        # output_emmeans_base <- file.path(data_directory, paste0("emmeans_at_iterations_full_nonlinear_trimmed", phase2_start_trial-1))
# #     #        # write.csv(summary(emmeans_at_iterations_full_nonlinear$emmeans), file = paste0(output_emmeans_base, "_means.csv"), row.names = FALSE)
# #     #        # write.csv(summary(emmeans_at_iterations_full_nonlinear$contrasts), file = paste0(output_emmeans_base, "_contrasts.csv"), row.names = FALSE)
# #     #        # print(paste("Full Non-Linear emmeans at specific iterations (trimmed) saved to files starting with:", output_emmeans_base))
# #     #    }, error = function(e) { print(paste("Could not calculate emmeans at specific iterations for full non-linear model (trimmed):", e$message)) })
# #     # }
# #     print("--- [COMMENTED OUT] Finished Probing Interaction for Full Non-Linear Model (Trimmed) ---")
# # } else {
# #     print(paste0("[COMMENTED OUT] Skipping Full LME Model (Trials ", phase2_start_trial, "-120) as its data subset would be empty or has insufficient data."))
# # }
print("--- Sections 8-10 (Full Non-Linear Model) are COMMENTED OUT ---")


# --- 11A. Analysis of Phase 1: Ramp-up (Trials 1 to `phase1_end_trial`) ---
print(paste0("--- Section 11A: Analyzing Phase 1 (Ramp-up: Trials 1-", phase1_end_trial, ") ---"))
if (nrow(df_phase1_ramp_up) > 0 && length(unique(df_phase1_ramp_up$Iteration)) > 1) {
    model_phase1_ramp_up <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = df_phase1_ramp_up)
    print("--- Finished Fitting Linear LME Model for Phase 1 (Ramp-up) ---")

    print("--- Model Summary (Phase 1 Ramp-up Model) ---")
    print(summary(model_phase1_ramp_up))
    print("--- ANOVA for Fixed Effects (Phase 1 Ramp-up Model) ---")
    tryCatch({
        anova_results_phase1 <- car::Anova(model_phase1_ramp_up, test = 'F', type="III")
        print(anova_results_phase1)
        output_file <- file.path(data_directory, paste0("anova_results_phase1_ramp_up_trials_1_to_", phase1_end_trial, ".csv"))
        write.csv(as.data.frame(anova_results_phase1), file = output_file, row.names = TRUE)
        print(paste("Phase 1 (Ramp-up) ANOVA results saved to:", output_file))
    }, error = function(e) { print(paste("Could not perform car::Anova on Phase 1 model:", e$message)) })

    print("--- Probing Interaction for Phase 1 (Ramp-up) Model ---")
    print("--- Comparing Learning Rates (Slopes) - Phase 1 (Ramp-up) ---")
    tryCatch({
        trends_phase1 <- emmeans::emtrends(model_phase1_ramp_up, pairwise ~ Algorithm, var = "Iteration")
        print(summary(trends_phase1, infer = TRUE))
        output_base <- file.path(data_directory, paste0("emtrends_results_phase1_ramp_up_trials_1_to_", phase1_end_trial))
        write.csv(summary(trends_phase1$emtrends), file = paste0(output_base, "_trends.csv"), row.names = FALSE)
        write.csv(summary(trends_phase1$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
        print(paste("Phase 1 (Ramp-up) emtrends results saved to files starting with:", output_base))
    }, error = function(e) { print(paste("Could not calculate emtrends for Phase 1 model:", e$message)) })

    print(paste0("--- Comparing Algorithms at End of Phase 1 (Trial ", phase1_end_trial, ") ---"))
    if (phase1_end_trial %in% unique(df_phase1_ramp_up$Iteration)) {
        tryCatch({
            emmeans_at_end_phase1 <- emmeans::emmeans(model_phase1_ramp_up, pairwise ~ Algorithm | Iteration,
                                                    at = list(Iteration = phase1_end_trial))
            print(summary(emmeans_at_end_phase1, infer = TRUE))
            output_base <- file.path(data_directory, paste0("emmeans_at_iteration_phase1_ramp_up_trial_", phase1_end_trial))
            write.csv(summary(emmeans_at_end_phase1$emmeans), file = paste0(output_base, "_means.csv"), row.names = FALSE)
            write.csv(summary(emmeans_at_end_phase1$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
            print(paste("Phase 1 (Ramp-up) emmeans at trial ", phase1_end_trial, " saved to files starting with:", output_base))
        }, error = function(e) { print(paste("Could not calculate emmeans at end of Phase 1:", e$message)) })
    } else {
        print(paste("Skipping emmeans at trial", phase1_end_trial, "as it's not in the Phase 1 data."))
    }
    print("--- Finished Analyzing Phase 1 (Ramp-up) ---")
} else {
    print(paste0("Skipping Phase 1 (Ramp-up) analysis (Trials 1-", phase1_end_trial, ") as 'df_phase1_ramp_up' is empty or has insufficient iteration variation."))
}


# --- 11B. Analysis of Phase 2: Active Learning (Trials `phase2_start_trial` to `phase2_end_trial`) ---
print(paste0("--- Section 11B: Analyzing Phase 2 (Active Learning: Trials ", phase2_start_trial, "-", phase2_end_trial, ") ---"))
if (nrow(df_phase2_active_learning) > 0 && length(unique(df_phase2_active_learning$Iteration)) > 1) {
    model_phase2_active_learning <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = df_phase2_active_learning)
    print("--- Finished Fitting Linear LME Model for Phase 2 (Active Learning) ---")

    print("--- Model Summary (Phase 2 Active Learning Model) ---")
    print(summary(model_phase2_active_learning))
    print("--- ANOVA for Fixed Effects (Phase 2 Active Learning Model) ---")
    tryCatch({
        anova_results_phase2 <- car::Anova(model_phase2_active_learning, test = 'F', type="III")
        print(anova_results_phase2)
        output_file <- file.path(data_directory, paste0("anova_results_phase2_active_learning_trials_",phase2_start_trial,"_to_", phase2_end_trial, ".csv"))
        write.csv(as.data.frame(anova_results_phase2), file = output_file, row.names = TRUE)
        print(paste("Phase 2 (Active Learning) ANOVA results saved to:", output_file))
    }, error = function(e) { print(paste("Could not perform car::Anova on Phase 2 model:", e$message)) })

    print("--- Probing Interaction for Phase 2 (Active Learning) Model ---")
    print("--- Comparing Learning Rates (Slopes) - Phase 2 (Active Learning) ---")
    tryCatch({
        trends_phase2 <- emmeans::emtrends(model_phase2_active_learning, pairwise ~ Algorithm, var = "Iteration")
        print(summary(trends_phase2, infer = TRUE))
        output_base <- file.path(data_directory, paste0("emtrends_results_phase2_active_learning_trials_",phase2_start_trial,"_to_", phase2_end_trial))
        write.csv(summary(trends_phase2$emtrends), file = paste0(output_base, "_trends.csv"), row.names = FALSE)
        write.csv(summary(trends_phase2$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
        print(paste("Phase 2 (Active Learning) emtrends results saved to files starting with:", output_base))
    }, error = function(e) { print(paste("Could not calculate emtrends for Phase 2 model:", e$message)) })

    # Compare at a midpoint of this phase
    iteration_point_phase2 <- phase2_start_trial + floor((phase2_end_trial - phase2_start_trial) / 2)
    if (iteration_point_phase2 %in% unique(df_phase2_active_learning$Iteration)) {
        print(paste0("--- Comparing Algorithms at Midpoint of Phase 2 (Trial ", iteration_point_phase2, ") ---"))
        tryCatch({
            emmeans_at_mid_phase2 <- emmeans::emmeans(model_phase2_active_learning, pairwise ~ Algorithm | Iteration,
                                                    at = list(Iteration = iteration_point_phase2))
            print(summary(emmeans_at_mid_phase2, infer = TRUE))
            output_base <- file.path(data_directory, paste0("emmeans_at_iteration_phase2_active_learning_trial_", iteration_point_phase2))
            write.csv(summary(emmeans_at_mid_phase2$emmeans), file = paste0(output_base, "_means.csv"), row.names = FALSE)
            write.csv(summary(emmeans_at_mid_phase2$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
            print(paste("Phase 2 (Active Learning) emmeans at trial ", iteration_point_phase2, " saved to files starting with:", output_base))
        }, error = function(e) { print(paste("Could not calculate emmeans at midpoint of Phase 2:", e$message)) })
    } else {
        print(paste("Skipping emmeans at trial", iteration_point_phase2, "(midpoint) as it's not in the Phase 2 data."))
    }
    print("--- Finished Analyzing Phase 2 (Active Learning) ---")
} else {
    print(paste0("Skipping Phase 2 (Active Learning) analysis (Trials ", phase2_start_trial, "-", phase2_end_trial, ") as 'df_phase2_active_learning' is empty or has insufficient iteration variation."))
}

# --- 11C. Analysis of Phase 3: Taper/Convergence (COMMENTED OUT) ---
# print(paste0("--- Section 11C: [COMMENTED OUT] Analyzing Taper/Convergence Phase ---"))
# taper_phase_start_trial_val <- phase2_end_trial + 1 # Renamed to avoid conflict
# max_trial_val <- max(combined_df_untrimmed$Iteration, na.rm = TRUE) # Renamed
# df_phase3_taper <- combined_df_untrimmed[Iteration >= taper_phase_start_trial_val & Iteration <= max_trial_val]
# if (nrow(df_phase3_taper) > 0 && length(unique(df_phase3_taper$Iteration)) > 1) {
#     model_phase3_linear <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = df_phase3_taper)
#     print("--- [COMMENTED OUT] Finished Fitting Linear LME Model for Taper/Convergence Phase ---")

#     print("--- [COMMENTED OUT] Model Summary (Taper/Convergence Phase Linear Model) ---")
#     # print(summary(model_phase3_linear))
#     print("--- [COMMENTED OUT] ANOVA for Fixed Effects (Taper/Convergence Phase Linear Model) ---")
#     # tryCatch({
#     #     anova_results_phase3 <- car::Anova(model_phase3_linear, test = 'F', type="III")
#     #     print(anova_results_phase3)
#     #     output_file <- file.path(data_directory, paste0("anova_results_phase3_taper_trials_",taper_phase_start_trial_val,"_to_", max_trial_val, ".csv"))
#     #     write.csv(as.data.frame(anova_results_phase3), file = output_file, row.names = TRUE)
#     #     print(paste("Taper/Convergence Phase Linear ANOVA results saved to:", output_file))
#     # }, error = function(e) { print(paste("Could not perform car::Anova on Taper/Convergence Phase model:", e$message)) })

#     print("--- [COMMENTED OUT] Probing Interaction for Taper/Convergence Phase Model ---")
#     print("--- [COMMENTED OUT] Comparing Learning Rates (Slopes) - Taper/Convergence Phase ---")
#     # Slopes might be near zero or negative here
#     # tryCatch({
#     #     trends_phase3 <- emmeans::emtrends(model_phase3_linear, pairwise ~ Algorithm, var = "Iteration")
#     #     print(summary(trends_phase3, infer = TRUE))
#     #     output_base <- file.path(data_directory, paste0("emtrends_results_phase3_taper_trials_",taper_phase_start_trial_val,"_to_", max_trial_val))
#     #     write.csv(summary(trends_phase3$emtrends), file = paste0(output_base, "_trends.csv"), row.names = FALSE)
#     #     write.csv(summary(trends_phase3$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
#     #     print(paste("Taper/Convergence Phase emtrends results saved to files starting with:", output_base))
#     # }, error = function(e) { print(paste("Could not calculate emtrends for Taper/Convergence Phase model:", e$message)) })

#     # iteration_point_phase3 <- max_trial_val - 10 
#     # if (iteration_point_phase3 < taper_phase_start_trial_val && nrow(df_phase3_taper) > 0) {
#     #    iteration_point_phase3 <- taper_phase_start_trial_val + floor((max_trial_val - taper_phase_start_trial_val)/2)
#     # }
    
#     # if (iteration_point_phase3 %in% unique(df_phase3_taper$Iteration)) {
#     #     print(paste0("--- [COMMENTED OUT] Comparing Algorithms Late in Taper Phase (Trial ", iteration_point_phase3, ") ---"))
#     #     tryCatch({
#     #         emmeans_at_late_phase3 <- emmeans::emmeans(model_phase3_linear, pairwise ~ Algorithm | Iteration,
#     #                                                     at = list(Iteration = iteration_point_phase3))
#     #         print(summary(emmeans_at_late_phase3, infer = TRUE))
#     #         output_base <- file.path(data_directory, paste0("emmeans_at_iteration_phase3_taper_trial_", iteration_point_phase3))
#     #         write.csv(summary(emmeans_at_late_phase3$emmeans), file = paste0(output_base, "_means.csv"), row.names = FALSE)
#     #         write.csv(summary(emmeans_at_late_phase3$contrasts), file = paste0(output_base, "_contrasts.csv"), row.names = FALSE)
#     #         print(paste("Taper/Convergence Phase emmeans at trial ", iteration_point_phase3, " saved to files starting with:", output_base))
#     #     }, error = function(e) { print(paste("Could not calculate emmeans late in Taper/Convergence Phase:", e$message)) })
#     # } else {
#     #     print(paste("Skipping emmeans for Phase 3 at trial", iteration_point_phase3, "as it's not in the Phase 3 data."))
#     # }
#     print("--- [COMMENTED OUT] Finished Analyzing Taper/Convergence Phase ---")
# } else {
#     print(paste0("[COMMENTED OUT] Skipping Taper/Convergence Phase analysis as 'df_phase3_taper' would be empty or has insufficient iteration variation."))
# }

# --- Sections 11 (old), 12, 13 (Original Early Linear Model on trimmed data) ---
# print("--- Section 11: [COMMENTED OUT / REPLACED BY PHASE ANALYSES] Preparing Early Trial Data (from Trimmed Data) & Fitting Early Linear LME Model ---")
# n_trials_for_early_analysis_old <- 35 
# start_early_iter_old <- phase1_end_trial + 1 # This was trim_initial_n_trials + 1
# end_early_iter_old <- phase1_end_trial + n_trials_for_early_analysis_old # trim_initial_n_trials + n_trials_for_early_analysis
# 
# early_df_old <- combined_df_untrimmed[Iteration >= start_early_iter_old & Iteration <= end_early_iter_old] 
# 
# if (nrow(early_df_old) > 0 && length(unique(early_df_old$Iteration)) > 1) {
#     print(paste0("--- [COMMENTED OUT] Fitting LME Model (Old Early Data: Original Iterations ", start_early_iter_old, "-", end_early_iter_old, ") - Linear ---"))
#     # model_early_linear_old <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = early_df_old)
#     # ... rest of sections 12, 13 analysis ...
# } else {
#     # print(paste0("[COMMENTED OUT] Skipping old early trial analysis (Original Iterations ", start_early_iter_old, "-", end_early_iter_old, ") as 'early_df_old' would be empty."))
# }
print("--- Sections 11-13 (Original Early Linear Model on data from trial 16 onwards) are COMMENTED OUT / REPLACED by new phase-specific analyses ---")


# --- 14. Plotting Model Fits for Each Analyzed Phase (with 95% CIs) ---
print("--- Section 14: Plotting Phase-Specific Data (with 95% CIs) ---")

# Plot for Phase 1: Ramp-up (Observed Data Only)
if (exists("df_phase1_ramp_up") && nrow(df_phase1_ramp_up) > 0) {
  print("--- Generating Plot for Phase 1: Ramp-up (Observed Data, 95% CIs) ---")
  plot_phase1_observed_CI <- ggplot(data = df_phase1_ramp_up, aes(x = Iteration, y = Survival, color = Algorithm)) +
    stat_summary(fun = mean, geom = "line", linewidth = 1) + # Observed Mean
    stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") + # 95% CIs for the Mean
    labs(title = paste0("Phase 1: Ramp-up (Trials 1-", phase1_end_trial, ") - Observed Data"),
        #  subtitle = "Observed mean survival with 95% Confidence Intervals",
         x = "Trial", y = "Average Survival Steps") +
    theme_bw(base_size = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size=rel(1.1)), legend.position = "bottom",
          # plot.subtitle = element_text(hjust = 0.5, size=rel(0.9)), 
          panel.grid.major = element_line(colour = "grey90"),
          panel.grid.minor = element_blank()) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1")
   
  print(plot_phase1_observed_CI)
  plot_filename_phase1_observed_CI_pdf <- file.path(data_directory, paste0("learning_curve_phase1_ramp_up_observed_trials_1_to_", phase1_end_trial, "_CI.pdf"))
  tryCatch({
    ggsave(filename = plot_filename_phase1_observed_CI_pdf,
           plot = plot_phase1_observed_CI,
           device = "pdf",
           width = 6, height = 4, units="in")
    print(paste("Plot for Phase 1 (Ramp-up) observed data (95% CIs) saved to:", plot_filename_phase1_observed_CI_pdf))
  }, error = function(e) {
    print(paste("Could not save Phase 1 observed data CI plot as PDF:", e$message))
  })
} else {
  print("Phase 1 data not found/empty. Skipping Phase 1 observed data plot.")
}

# NEW Plot for Phase 1: Ramp-up (Observed Data WITH LME Fit)
if (exists("model_phase1_ramp_up") && exists("df_phase1_ramp_up") && nrow(df_phase1_ramp_up) > 0) {
  print("--- Generating Plot for Phase 1: Ramp-up (Observed Data with LME Fit, 95% CIs) ---")
 
  # Calculate predictions for Phase 1 model
  emm_preds_phase1_fit <- NULL
  predicted_df_phase1_fit <- NULL
  tryCatch({
    emm_preds_phase1_fit <- emmeans(model_phase1_ramp_up, ~ Algorithm | Iteration,
                                 at = list(Iteration = sort(unique(df_phase1_ramp_up$Iteration))),
                                 type = "response")
    predicted_df_phase1_fit <- as.data.frame(summary(emm_preds_phase1_fit))
    names(predicted_df_phase1_fit)[names(predicted_df_phase1_fit) == "emmean"] <- "PredictedSurvival"
  }, error = function(e) {
    print(paste("Could not generate emmeans predictions for Phase 1 LME model fit plot:", e$message))
  })

  if (!is.null(predicted_df_phase1_fit)) {
    plot_phase1_LME_fit_CI <- ggplot(data = df_phase1_ramp_up, aes(x = Iteration, y = Survival, color = Algorithm)) +
      stat_summary(fun = mean, geom = "line", linewidth = 0.8, alpha = 0.5, aes(linetype = "Observed Mean")) +
      stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") + # Using 95% CIs
      geom_line(data = predicted_df_phase1_fit, aes(x = Iteration, y = PredictedSurvival, color = Algorithm, linetype = "Model Fit"), linewidth = 1.1) +
      scale_linetype_manual(name = "Data Type", values = c("Observed Mean" = "dashed", "Model Fit" = "solid")) +
      labs(title = paste0("Phase 1: Ramp-up (Trials 1-", phase1_end_trial, ") with LME Model Fit"),
          #  subtitle = "Observed means with 95% CIs, and LME model predictions",
           x = "Trial", y = "Average Survival Steps") +
      theme_bw(base_size = 10) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold", size=rel(1.1)), legend.position = "bottom",
            # plot.subtitle = element_text(hjust = 0.5, size=rel(0.9)), 
            panel.grid.major = element_line(colour = "grey90"),
            panel.grid.minor = element_blank()) +
      scale_color_brewer(palette = "Set1") +
      scale_fill_brewer(palette = "Set1")
     
    print(plot_phase1_LME_fit_CI)
    plot_filename_phase1_LME_fit_CI_pdf <- file.path(data_directory, paste0("learning_curve_phase1_ramp_up_LME_fit_trials_1_to_", phase1_end_trial, "_CI.pdf"))
    tryCatch({
      ggsave(filename = plot_filename_phase1_LME_fit_CI_pdf,
             plot = plot_phase1_LME_fit_CI,
             device = "pdf",
             width = 6, height = 4, units="in")
      print(paste("Plot for Phase 1 (Ramp-up) with LME fit (95% CIs) saved to:", plot_filename_phase1_LME_fit_CI_pdf))
    }, error = function(e) {
      print(paste("Could not save Phase 1 LME fit CI plot as PDF:", e$message))
    })
  } else {
    print("Skipping Phase 1 LME fit plot due to issues generating predictions.")
  }
} else {
  print("Phase 1 model or data not found/empty. Skipping Phase 1 LME fit plot.")
}


# Plot for Phase 2: Active Learning (Observed Data with LME Fit)
if (exists("model_phase2_active_learning") && exists("df_phase2_active_learning") && nrow(df_phase2_active_learning) > 0) {
  print("--- Generating Plot with Model Fit (Phase 2: Active Learning, 95% CIs) ---")
  emm_preds_phase2 <- emmeans(model_phase2_active_learning, ~ Algorithm | Iteration,
                              at = list(Iteration = sort(unique(df_phase2_active_learning$Iteration))),
                              type = "response")
  predicted_df_phase2 <- as.data.frame(summary(emm_preds_phase2))
  names(predicted_df_phase2)[names(predicted_df_phase2) == "emmean"] <- "PredictedSurvival"

  plot_phase2_fit_CI <- ggplot(data = df_phase2_active_learning, aes(x = Iteration, y = Survival, color = Algorithm)) +
    stat_summary(fun = mean, geom = "line", linewidth = 0.8, alpha = 0.5, aes(linetype = "Observed Mean")) +
    stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") + # Using 95% CIs
    geom_line(data = predicted_df_phase2, aes(x = Iteration, y = PredictedSurvival, color = Algorithm, linetype = "Model Fit"), linewidth = 1.1) +
    scale_linetype_manual(name = "Data Type", values = c("Observed Mean" = "dashed", "Model Fit" = "solid")) +
    labs(title = paste0("Phase 2: Active Learning (Trials ",phase2_start_trial,"-", phase2_end_trial, ") Linear Model Fit"),
        #  subtitle = "Observed means with 95% CIs, and LME model predictions",
         x = "Trial", y = "Average Survival Steps") +
    theme_bw(base_size = 10) + # Adjusted base_size
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size=rel(1.1)), legend.position = "bottom",
          # plot.subtitle = element_text(hjust = 0.5, size=rel(0.9)), 
          panel.grid.major = element_line(colour = "grey90"),
          panel.grid.minor = element_blank()) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1")

  print(plot_phase2_fit_CI)
  plot_filename_phase2_fit_CI_pdf <- file.path(data_directory, paste0("learning_curve_phase2_active_learning_fit_trials_",phase2_start_trial,"_to_", phase2_end_trial, "_CI.pdf"))
  tryCatch({
    ggsave(filename = plot_filename_phase2_fit_CI_pdf,
           plot = plot_phase2_fit_CI,
           device = "pdf",
           width = 6, height = 4, units="in") # Half-page or single-column typical size
    print(paste("Plot for Phase 2 (Active Learning) fit (95% CIs) saved to:", plot_filename_phase2_fit_CI_pdf))
  }, error = function(e) {
    print(paste("Could not save Phase 2 CI fit plot as PDF:", e$message))
  })
} else {
  print("Phase 2 model or data not found/empty. Skipping Phase 2 plot.")
}

# Plot for Phase 3: Taper/Convergence (Observed Data Only)
if (exists("df_phase3_taper") && nrow(df_phase3_taper) > 0) {
  print(paste0("--- Generating Plot for Phase 3: Taper/Convergence (Observed Data, Trials ", phase3_start_trial, "-", phase3_end_trial, ", 95% CIs) ---"))
  plot_phase3_observed_CI <- ggplot(data = df_phase3_taper, aes(x = Iteration, y = Survival, color = Algorithm)) +
    stat_summary(fun = mean, geom = "line", linewidth = 1) + # Observed Mean
    stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") + # 95% CIs for the Mean
    labs(title = paste0("Phase 3: Taper/Convergence (Trials ", phase3_start_trial, "-", phase3_end_trial, ") - Observed Data"),
        #  subtitle = "Observed mean survival with 95% Confidence Intervals",
         x = "Trial", y = "Average Survival Steps") +
    theme_bw(base_size = 10) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size=rel(1.1)), legend.position = "bottom",
          # plot.subtitle = element_text(hjust = 0.5, size=rel(0.9)), 
          panel.grid.major = element_line(colour = "grey90"),
          panel.grid.minor = element_blank()) +
    scale_color_brewer(palette = "Set1") +
    scale_fill_brewer(palette = "Set1")
   
  print(plot_phase3_observed_CI)
  plot_filename_phase3_observed_CI_pdf <- file.path(data_directory, paste0("learning_curve_phase3_taper_observed_trials_", phase3_start_trial, "_to_", phase3_end_trial, "_CI.pdf"))
  tryCatch({
    ggsave(filename = plot_filename_phase3_observed_CI_pdf,
           plot = plot_phase3_observed_CI,
           device = "pdf",
           width = 6, height = 4, units="in")
    print(paste("Plot for Phase 3 (Taper/Convergence) observed data (95% CIs) saved to:", plot_filename_phase3_observed_CI_pdf))
  }, error = function(e) {
    print(paste("Could not save Phase 3 observed data CI plot as PDF:", e$message))
  })
} else {
  print("Phase 3 data not found/empty. Skipping Phase 3 observed data plot.")
}


print("--- Finished Plotting Phase-Specific Data ---")

print("--- Script Finished ---")
