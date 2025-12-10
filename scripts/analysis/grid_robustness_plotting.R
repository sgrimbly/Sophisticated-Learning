# 0. Preamble: Load necessary libraries and set parameters
# Ensure these packages are installed: install.packages(c("dplyr", "stringr", "ggplot2", "readr", "purrr", "rstatix", "broom", "rlang", "scales"))
library(dplyr)
library(stringr)
library(ggplot2)
library(readr)
library(purrr)
library(rstatix) # For t_test and effect sizes
library(broom)   # For tidying model outputs
library(rlang)   # For last_trace()
library(scales)  # For pvalue formatting

# --- Configuration ---
# directory_path <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments"
dir_main  <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments"
dir_extra <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_3horizononly_extra"

# Consider a new output folder name to reflect the new trial ranges
output_plot_path <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/R_plots_120_60_trials_selected_analysis"
dir.create(output_plot_path, showWarnings = FALSE, recursive = TRUE)

valid_trial_length <- 200       # Files should contain this many trials initially
max_plot_trial <- 120           # MODIFIED: Max trial for full range plots
expected_seeds_per_group <- 200
# filename_pattern <- "^(SI|SL|[^_]+)_Seed_(\\d+)_GridID_([a-f0-9]+)_\\d{2}-\\d{2}-\\d{2}-\\d{3}\\.txt$"
filename_pattern <- paste0(
  "^(SI|SL|[^_]+)_Seed_(\\d+)_GridID_([a-f0-9]+)",
  "(?:_\\d{2}-\\d{2}-\\d{2}-\\d{3})?",  # optional timestamp
  "\\.txt$"
)

# Phase definitions
phase1_end_trial <- 20
phase2_start_trial <- 21
phase2_end_trial <- 60
phase3_start_trial <- 61
phase3_end_trial <- max_plot_trial 

# Define the end trial for the second set of plots (early/mid range, 1-60 trials)
early_mid_phase_end_trial <- phase2_end_trial # This is 60 - ENSURE THIS LINE IS PRESENT

if (!dir.exists(output_plot_path)) {
  dir.create(output_plot_path, recursive = TRUE)
  message("Created output directory: ", output_plot_path)
} else {
  message("Output directory set to: ", output_plot_path)
}

# 1. Function to parse filename (unchanged)
parse_filename <- function(filename) {
  match <- stringr::str_match(basename(filename), filename_pattern)
  if (is.na(match[1,1])) {
    warning(paste("Failed to parse filename:", filename))
    return(NULL)
  }
  return(list(
    algo_type = match[1, 2],
    seed = as.integer(match[1, 3]),
    grid_id = match[1, 4]
  ))
}

# 2. Function to load and process a single file's data (unchanged for loading 200 trials)
load_single_file_data <- function(filepath) {
  file_info <- parse_filename(filepath)
  if (is.null(file_info)) return(NULL)

  lines <- readr::read_lines(filepath, skip_empty_rows = TRUE)
  data_values <- as.numeric(lines)
  data_values <- data_values[!is.na(data_values)]

  if (length(data_values) == valid_trial_length) {
    return(
      tibble::tibble(
        algo_type = file_info$algo_type,
        seed = file_info$seed,
        grid_id = file_info$grid_id,
        trial = 1:valid_trial_length,
        survival = data_values
      )
    )
  } else {
    warning(paste("File", basename(filepath), "does not have", valid_trial_length, "trials. Found:", length(data_values), "Skipping."))
    return(NULL)
  }
}

# 3. Main script logic
# all_files <- list.files(directory_path, pattern = "\\.txt$", full.names = TRUE, recursive = TRUE)
all_files <- c(
  list.files(dir_main,  pattern = "\\.txt$", full.names = TRUE, recursive = TRUE),
  list.files(dir_extra, pattern = "\\.txt$", full.names = TRUE, recursive = TRUE)
)
if (length(all_files) == 0)
  stop("No .txt files found in either directory.")
message(paste("Found", length(all_files), "'.txt' files to process."))

# all_data_raw <- purrr::map_dfr(all_files, load_single_file_data, .id = NULL)
all_data_raw <- purrr::map_dfr(all_files, load_single_file_data) %>% 
    dplyr::filter(seed > 0 & seed <= expected_seeds_per_group)   # drop seed 0, keep 1-200


if (nrow(all_data_raw) == 0) stop(paste("No valid data (with", valid_trial_length, "trials) could be processed from the files."))
message(paste("Successfully processed", nrow(all_data_raw), "data rows (raw, up to", valid_trial_length, "trials) from valid files."))

# --- Data Integrity Check for Seed Counts ---
message("\n--- Starting Data Integrity Check for Seed Counts (based on full ", valid_trial_length, " trial files) ---")
seed_counts_check <- all_data_raw %>%
  filter(algo_type %in% c("SI", "SL")) %>%
  group_by(grid_id, algo_type) %>%
  summarise(distinct_seeds = n_distinct(seed), .groups = "drop")


# Quick check
# --- parameters -----------------------------------------------------------
problem_id   <- "751499cb304e0b35280d048a2826f0a1"
seeds_needed <- 1:200          # we now ignore Seed 0 everywhere
out_csv      <- file.path(output_plot_path,
                          paste0("missing_SI_seeds_", problem_id, ".csv"))
# -------------------------------------------------------------------------

# 1. collect the SI files that already exist for this grid
si_files <- list.files(dir_extra,                 # or dir_main, or both
                       pattern = paste0("^SI_Seed_\\d+_GridID_", problem_id,
                                         "(?:_|\\.txt$)"),
                       full.names = FALSE)

# 2. extract the seed numbers
present_seeds <- stringr::str_match(si_files, "Seed_(\\d+)")[,2] |>
                 as.integer()

# 3. which seeds are missing?
missing_seeds <- setdiff(seeds_needed, present_seeds)

# 4. show them and (optionally) save to CSV
print(missing_seeds)
readr::write_csv(tibble::tibble(seed = missing_seeds), out_csv)

cat(length(missing_seeds), "SI seeds still missing for grid",
    substr(problem_id, 1, 8), "…\n",
    "CSV written to:", out_csv, "\n")

# End



all_grids_initially_checked <- character(0)

# valid_grid_algo_pairs <- seed_counts_check %>%
#   filter(distinct_seeds == expected_seeds_per_group)
valid_grid_algo_pairs <- seed_counts_check %>%
    filter(distinct_seeds >= expected_seeds_per_group)

grid_summary <- valid_grid_algo_pairs %>%
  group_by(grid_id) %>%
  summarise(algo_count = n_distinct(algo_type), .groups = "drop")

grids_with_both_SI_SL_complete <- grid_summary %>%
  filter(algo_count == 2) %>%
  pull(grid_id)

if(length(grids_with_both_SI_SL_complete) > 0) {
    message(paste("Found", length(grids_with_both_SI_SL_complete), "GridIDs with initially complete (", expected_seeds_per_group, " seeds) data for both SI and SL: ", paste(grids_with_both_SI_SL_complete, collapse=", ")))
    all_grids_initially_checked <- grids_with_both_SI_SL_complete
} else {
    message("No GridIDs found with initially complete data for both SI and SL.")
}

# --- Filter for specific Grid IDs for quantitative analysis ---
# target_prefixes <- c("4bc", "12d", "426", "a83", "b14") # Your 5 selected grid prefixes
target_prefixes <- c(
  "4bc","12d","426","a83","b14",   # existing horizon=3 without horizon=5 grids
  "751","7b9","5b8","9c6","0ed",   # extra grids horizon=3
  "1dc","5ac","c08","1e4","49b"    
)
selected_grid_ids_for_analysis <- purrr::map(target_prefixes, ~all_grids_initially_checked[startsWith(all_grids_initially_checked, .x)]) %>%
  unlist() %>%
  unique()

if(length(selected_grid_ids_for_analysis) > 0) {
    message(paste("\nSelected", length(selected_grid_ids_for_analysis), "GridIDs for focused quantitative analysis based on prefixes:", paste(selected_grid_ids_for_analysis, collapse=", ")))
    if (length(selected_grid_ids_for_analysis) < length(target_prefixes)) {
        message("Note: Not all target prefixes resulted in a selected GridID (some might not have been in the 'initially complete' list).")
    }
} else {
    # Stop execution if no valid grids are selected, as subsequent analysis and plotting depend on this.
    stop("CRITICAL: No GridIDs selected for focused analysis. Please check 'target_prefixes' and data integrity. Halting script.")
}

# --- Report on other issues ---
other_issues_display <- seed_counts_check %>%
    filter(!(grid_id %in% all_grids_initially_checked & distinct_seeds == expected_seeds_per_group & algo_type %in% c("SI", "SL")) |
           (grid_id %in% all_grids_initially_checked & algo_type %in% c("SI", "SL") & distinct_seeds != expected_seeds_per_group) |
           (!(grid_id %in% all_grids_initially_checked) & algo_type %in% c("SI", "SL")))

if(nrow(other_issues_display) > 0){
    actual_problematic_groups <- other_issues_display %>%
        filter(distinct_seeds != expected_seeds_per_group)

    if(nrow(actual_problematic_groups) > 0) {
        message("--- Additional Seed Count Issues Found in Full Dataset (for general awareness): ---")
        for (i in 1:nrow(actual_problematic_groups)) {
            row_issue <- actual_problematic_groups[i, ]
            problem_message <- paste0("Data integrity issue: GridID '", row_issue$grid_id,
                                      "' with Algorithm '", row_issue$algo_type,
                                      "' has ", row_issue$distinct_seeds,
                                      " unique seeds instead of the expected ", expected_seeds_per_group, ".")
            warning(problem_message)
        }
    }
}
message("--- Finished Data Integrity Check ---\n")

# --- Filter data to max_plot_trial (if max_plot_trial < valid_trial_length) ---
# Since max_plot_trial is 120 and valid_trial_length is 200, this step will keep all trials.
all_data_trimmed <- all_data_raw %>%
  filter(trial <= max_plot_trial)
message(paste("Data filtered up to a maximum of", max_plot_trial, "trials. Resulting rows:", nrow(all_data_trimmed)))

if (nrow(all_data_trimmed) == 0) stop("No data remaining after filtering to max_plot_trial.")

# --- Aggregate data ---
summary_data <- all_data_trimmed %>%
  group_by(grid_id, algo_type, trial) %>%
  summarise(
    mean_survival = mean(survival, na.rm = TRUE),
    sd_survival = sd(survival, na.rm = TRUE),
    n_seeds_in_trial = n_distinct(seed), # More accurate if each seed has one row per trial
    .groups = "drop"
  ) %>%
  mutate(sd_survival = ifelse(is.na(sd_survival) | n_seeds_in_trial < 2, 0, sd_survival)) # SD is 0 if only 1 seed

if (nrow(summary_data) == 0) stop("Data aggregation resulted in an empty summary.")
message("Data aggregation complete.")

# --- Quantitative Comparison (Paired t-tests on SELECTED GRIDS) ---
message("\n--- Starting Quantitative Comparison across SELECTED Grid Configurations (SL vs. SI) ---")
comparison_data <- summary_data %>%
  filter(grid_id %in% selected_grid_ids_for_analysis, algo_type %in% c("SL", "SI"))

if(length(selected_grid_ids_for_analysis) >= 2 && nrow(comparison_data) > 0) {
    comparison_points <- list(
        "Trial_20" = phase1_end_trial,
        # Adjusted to be mid-point of Phase 2 (21-60), so around trial 40
        "Trial_MidPhase2" = phase2_start_trial + floor((phase2_end_trial - phase2_start_trial) / 2),
        # Example: A point in Phase 3, e.g., trial 110, or closer to max_plot_trial if shorter
        "Trial_InPhase3" = if ( (phase3_start_trial + 50) <= max_plot_trial) (phase3_start_trial + 49) else max_plot_trial,
        "Avg_Phase1" = seq(1, phase1_end_trial),
        "Avg_Phase2" = seq(phase2_start_trial, phase2_end_trial),
        "Avg_Phase3" = seq(phase3_start_trial, phase3_end_trial) # phase3_end_trial is now 120
    )

    # Ensure Phase 3 is valid and doesn't extend beyond max_plot_trial
    if (phase3_start_trial > max_plot_trial) {
        comparison_points$Avg_Phase3 <- NULL
        comparison_points$Trial_InPhase3 <- NULL # Remove if Phase 3 doesn't exist
        message("Note: Avg_Phase3 and Trial_InPhase3 comparison points removed as Phase 3 start is beyond max_plot_trial.")
    } else if (!is.null(comparison_points$Avg_Phase3) && max(comparison_points$Avg_Phase3, na.rm=TRUE) > max_plot_trial) {
        comparison_points$Avg_Phase3 <- seq(phase3_start_trial, max_plot_trial)
    }
     if (!is.null(comparison_points$Trial_InPhase3) && comparison_points$Trial_InPhase3 > max_plot_trial) {
        comparison_points$Trial_InPhase3 <- max_plot_trial
    }


    paired_test_results_list <- list()

    for (point_name in names(comparison_points)) {
        current_trials <- comparison_points[[point_name]]
        if(is.null(current_trials) || length(current_trials) == 0) {
            message(paste0("\nSkipping ", point_name, " as current_trials is empty or null."))
            next
        }

        metric_data_intermediate <- comparison_data %>%
            filter(trial %in% current_trials) %>%
            group_by(grid_id, algo_type) %>% # Group by grid_id and algo_type
            summarise(metric_value = mean(mean_survival, na.rm = TRUE), .groups = "drop")

        message(paste0("\n--- Debugging: Intermediate data for ", point_name, " (Source Trials: ", min(current_trials),"-",max(current_trials), ") ---"))
        message(paste0("Number of rows in intermediate data for ", point_name, ": ", nrow(metric_data_intermediate)))

        metric_data <- metric_data_intermediate %>%
            tidyr::pivot_wider(names_from = algo_type, values_from = metric_value)

        message(paste0("Number of rows in pivoted data for ", point_name, ": ", nrow(metric_data)))
        if("SL" %in% names(metric_data)) message("Number of NAs in SL column: ", sum(is.na(metric_data$SL))) else message("SL column missing in pivoted data.")
        if("SI" %in% names(metric_data)) message("Number of NAs in SI column: ", sum(is.na(metric_data$SI))) else message("SI column missing in pivoted data.")

        num_complete_pairs <- if("SL" %in% names(metric_data) && "SI" %in% names(metric_data)) sum(complete.cases(metric_data$SL, metric_data$SI)) else 0
        message("Number of complete SL/SI pairs for t-test: ", num_complete_pairs)

        if ("SL" %in% names(metric_data) && "SI" %in% names(metric_data) && num_complete_pairs >= 2) {
            message(paste0("\n--- Paired t-test & Cohen's d for: ", point_name, " (Source Trials: ", min(current_trials),"-",max(current_trials), ") ---"))

            direct_test_result <- NULL
            manual_cohens_d <- NULL

            tryCatch({
                direct_test_result <- stats::t.test(metric_data$SL, metric_data$SI, paired = TRUE)

                differences <- metric_data$SL - metric_data$SI
                # Ensure there are at least 2 non-NA differences and sd is not 0
                valid_diffs <- differences[!is.na(differences)]
                if (length(valid_diffs) >= 2 && !is.na(sd(valid_diffs)) && sd(valid_diffs) != 0) {
                    manual_cohens_d <- mean(valid_diffs) / sd(valid_diffs)
                } else {
                    manual_cohens_d <- NA
                }
            }, error = function(e) {
                message("Error during stats::t.test or manual Cohen's d calculation for '", point_name, "': ", e$message)
            })

            if (!is.null(direct_test_result)) {
                message("Paired t-test results (from stats::t.test):")
                message(sprintf("  t = %.3f, df = %d, p-value = %s",
                                direct_test_result$statistic,
                                direct_test_result$parameter,
                                scales::pvalue(direct_test_result$p.value))) # Using scales::pvalue for formatting
                message(sprintf("  95%% CI for mean difference: [%.3f, %.3f]",
                                direct_test_result$conf.int[1],
                                direct_test_result$conf.int[2]))
                message(sprintf("  Mean of differences (SL - SI): %.3f", direct_test_result$estimate))

                if (!is.na(manual_cohens_d)) {
                    message(paste("  Cohen's d (paired, manual):", round(manual_cohens_d, 3)))
                    abs_d <- abs(manual_cohens_d)
                    magnitude <- ifelse(abs_d < 0.2, "negligible",
                                 ifelse(abs_d < 0.5, "small",
                                 ifelse(abs_d < 0.8, "medium", "large")))
                    message(paste("  Effect size magnitude:", magnitude))
                } else {
                    message("  Cohen's d (paired, manual): Could not be calculated (e.g., sd of differences is 0 or not enough data).")
                }

                paired_test_results_list[[point_name]] <- list(
                    t_statistic = direct_test_result$statistic,
                    df = direct_test_result$parameter,
                    p.value = direct_test_result$p.value,
                    conf.low = direct_test_result$conf.int[1],
                    conf.high = direct_test_result$conf.int[2],
                    mean.diff = direct_test_result$estimate,
                    cohens_d_manual = manual_cohens_d,
                    cohens_d_magnitude = if(!is.na(manual_cohens_d)) magnitude else NA
                )

                mean_sl_val <- mean(metric_data$SL, na.rm = TRUE)
                mean_si_val <- mean(metric_data$SI, na.rm = TRUE)
                message(sprintf("Mean SL at %s: %.2f, Mean SI at %s: %.2f",
                                point_name, mean_sl_val, point_name, mean_si_val))

                if (!is.na(direct_test_result$p.value) && direct_test_result$p.value < 0.05) {
                    message(paste("Significant difference found between SL and SI for", point_name, "(p =", scales::pvalue(direct_test_result$p.value), ")."))
                } else {
                    message(paste("No significant difference found between SL and SI for", point_name, "(p =", scales::pvalue(direct_test_result$p.value), ")."))
                }
            } else {
                message("stats::t.test failed or did not produce a result for '", point_name, "'. Cannot proceed with interpretation for this point.")
            }
        } else {
            message(paste0("\nSkipping t-test for ", point_name, ": Not enough complete pairs of SL/SI data (found ", num_complete_pairs, " complete pairs out of ", nrow(metric_data) ," GridIDs) for selected grids or SL/SI columns missing/all NA after pivoting."))
        }
    } # End of for loop for comparison points
    saveRDS(paired_test_results_list, file.path(output_plot_path, "paired_ttest_SELECTED_grids_results.rds"))
    message("\nDetailed paired t-test results for SELECTED grids object saved as .rds file.")

} else {
    message("Skipping quantitative comparison: Not enough SELECTED GridIDs with complete data for both SI and SL, or no comparison data for selected grids.")
}
message("--- Finished Quantitative Comparison ---")


# --- PLOTTING SECTION ---

# Plot 1: Full Range (1 to max_plot_trial, i.e., 1-120 trials) for SELECTED GRIDS
if (length(selected_grid_ids_for_analysis) == 0) {
  message("No selected grid IDs for analysis found. No full range (1-", max_plot_trial, " trials) plots will be generated.")
} else {
  message(paste("\nGenerating full range (1-", max_plot_trial, " trials) plots for ", length(selected_grid_ids_for_analysis), " selected unique grid ID(s):", paste(selected_grid_ids_for_analysis, collapse=", ")))
}

for (current_grid_id in selected_grid_ids_for_analysis) {
  plot_data_full <- summary_data %>%
    filter(grid_id == current_grid_id, algo_type %in% c("SI", "SL"))

  if (nrow(plot_data_full) == 0) {
    warning(paste("No summary data available for GridID:", current_grid_id, " (for SI/SL) for full range plot (1-", max_plot_trial, " trials) - Skipping plot."))
    next
  }

  num_algos_for_grid <- length(unique(plot_data_full$algo_type))

  first_trial_seed_counts_all_data <- all_data_raw %>%
    filter(grid_id == current_grid_id, algo_type %in% c("SI", "SL"), trial == 1) %>%
    group_by(algo_type) %>%
    summarise(n_seeds = n_distinct(seed), .groups="drop")

  si_seeds_ok <- any(first_trial_seed_counts_all_data$algo_type == "SI" & first_trial_seed_counts_all_data$n_seeds == expected_seeds_per_group)
  sl_seeds_ok <- any(first_trial_seed_counts_all_data$algo_type == "SL" & first_trial_seed_counts_all_data$n_seeds == expected_seeds_per_group)

  subtitle_text <- if (si_seeds_ok && sl_seeds_ok) {
                       paste("Based on", expected_seeds_per_group, "seeds per algorithm.")
                   } else {
                       paste("Note: Original seed counts for this grid may vary from expected", expected_seeds_per_group, "(check console).")
                   }

  p_full <- ggplot(plot_data_full, aes(x = trial, y = mean_survival, color = algo_type, fill = algo_type, linetype = algo_type)) +
    geom_line(linewidth = 0.8) +
    geom_ribbon(aes(ymin = mean_survival - sd_survival,
                    ymax = mean_survival + sd_survival),
                alpha = 0.2, colour = NA) +
    # labs(
    #   title = paste("Average Survival for Selected GridID:", current_grid_id, "(Trials 1-", max_plot_trial, ")"),
    #   subtitle = subtitle_text,
    #   x = "Trial",
    #   y = "Average Survival",
    #   color = "Algorithm", fill = "Algorithm", linetype = "Algorithm"
    # ) +
    labs(
      x = "Trial",
      y = "Average Survival"
    ) +
    scale_x_continuous(limits = c(1, max_plot_trial), expand = c(0, 0.02), breaks = scales::pretty_breaks(n=10)) +
    scale_y_continuous(labels = scales::comma) +
    theme_minimal(base_size = 14) +
    # theme(
    #   plot.title = element_text(hjust = 0.5, face = "bold"),
    #   plot.subtitle = element_text(hjust = 0.5, size = 10),
    #   legend.position = "top",
    #   legend.background = element_rect(fill = "white", colour = "grey"),
    #   axis.text = element_text(size = 11),
    #   axis.title = element_text(size = 12, face = "bold"),
    #   panel.grid.major = element_line(colour = "grey90"),
    #   panel.grid.minor = element_line(colour = "grey95"),
    #   plot.background = element_rect(fill = "white", colour = NA)
    # )
    theme(
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      legend.position = "none"
    ) +
    guides(colour = "none", fill = "none", linetype = "none")
    
  if(num_algos_for_grid <= 1) {
    p_full <- p_full + theme(legend.position = "none")
  }

  plot_filename_full <- file.path(output_plot_path, paste0("Plot_Selected_GridID_", current_grid_id, "_Survival_1to", max_plot_trial, "trials.pdf"))
  ggsave(plot_filename_full, plot = p_full, width = 10, height = 6.5, device = "pdf", bg = "white")
  message(paste("Saved full range (1-", max_plot_trial, " trials) plot for selected grid:", plot_filename_full))
}

# Plot 2: Early/Mid Range (1 to early_mid_phase_end_trial, i.e., 1-60 trials) for SELECTED GRIDS
message(paste("\nGenerating early/mid range (1-", early_mid_phase_end_trial, " trials) plots for ", length(selected_grid_ids_for_analysis), " selected unique grid ID(s):", paste(selected_grid_ids_for_analysis, collapse=", ")))

if (length(selected_grid_ids_for_analysis) > 0) {
  for (current_grid_id_early in selected_grid_ids_for_analysis) {
    plot_data_early <- summary_data %>%
      filter(grid_id == current_grid_id_early,
             algo_type %in% c("SI", "SL"),
             trial <= early_mid_phase_end_trial) # Filter for trials 1-60

    if (nrow(plot_data_early) == 0) {
      warning(paste("No summary data available for GridID:", current_grid_id_early, " (for SI/SL) for early/mid range plot (1-", early_mid_phase_end_trial, " trials) - Skipping plot."))
      next
    }

    num_algos_for_grid_early <- length(unique(plot_data_early$algo_type))

    first_trial_seed_counts_all_data_early <- all_data_raw %>%
        filter(grid_id == current_grid_id_early, algo_type %in% c("SI", "SL"), trial == 1) %>%
        group_by(algo_type) %>%
        summarise(n_seeds = n_distinct(seed), .groups="drop")

    si_seeds_ok_early <- any(first_trial_seed_counts_all_data_early$algo_type == "SI" & first_trial_seed_counts_all_data_early$n_seeds == expected_seeds_per_group)
    sl_seeds_ok_early <- any(first_trial_seed_counts_all_data_early$algo_type == "SL" & first_trial_seed_counts_all_data_early$n_seeds == expected_seeds_per_group)

    subtitle_text_early <- if (si_seeds_ok_early && sl_seeds_ok_early) {
                               paste("Based on", expected_seeds_per_group, "seeds per algorithm.")
                           } else {
                               paste("Note: Original seed counts for this grid may vary from expected", expected_seeds_per_group, "(check console).")
                           }

    p_early <- ggplot(plot_data_early, aes(x = trial, y = mean_survival, color = algo_type, fill = algo_type, linetype = algo_type)) +
      geom_line(linewidth = 0.8) +
      geom_ribbon(aes(ymin = mean_survival - sd_survival,
                      ymax = mean_survival + sd_survival),
                  alpha = 0.2, colour = NA) +
      # labs(
      #   title = paste("Average Survival for Selected GridID:", current_grid_id_early, "(Trials 1-", early_mid_phase_end_trial, ")"),
      #   subtitle = subtitle_text_early,
      #   x = "Trial",
      #   y = "Average Survival",
      #   color = "Algorithm", fill = "Algorithm", linetype = "Algorithm"
      # ) +
      labs(
        x = "Trial",
        y = "Average Survival"
      ) +
      scale_x_continuous(limits = c(1, early_mid_phase_end_trial), breaks = seq(0, early_mid_phase_end_trial, by = 10), expand = c(0, 0.02)) +
      scale_y_continuous(labels = scales::comma) +
      theme_minimal(base_size = 14) +
      # theme(
      #   plot.title = element_text(hjust = 0.5, face = "bold"),
      #   plot.subtitle = element_text(hjust = 0.5, size = 10),
      #   legend.position = "top",
      #   legend.background = element_rect(fill = "white", colour = "grey"),
      #   axis.text = element_text(size = 11),
      #   axis.title = element_text(size = 12, face = "bold"),
      #   panel.grid.major = element_line(colour = "grey90"),
      #   panel.grid.minor = element_line(colour = "grey95"),
      #   plot.background = element_rect(fill = "white", colour = NA)
      # )
      theme(
        plot.title    = element_blank(),
        plot.subtitle = element_blank(),
        legend.position = "none",
        axis.text = element_text(size = 11),
        axis.title = element_text(size = 12, face = "bold"),
        panel.grid.major = element_line(colour = "grey90"),
        panel.grid.minor = element_line(colour = "grey95"),
        plot.background = element_rect(fill = "white", colour = NA)
      ) +
      guides(colour = "none", fill = "none", linetype = "none")

    if(num_algos_for_grid_early <= 1) {
      p_early <- p_early + theme(legend.position = "none")
    }

    plot_filename_early <- file.path(output_plot_path, paste0("Plot_Selected_GridID_", current_grid_id_early, "_Survival_1to", early_mid_phase_end_trial, "trials.pdf"))
    ggsave(plot_filename_early, plot = p_early, width = 10, height = 6.5, device = "pdf", bg = "white")
    message(paste("Saved early/mid range (1-", early_mid_phase_end_trial, " trials) plot for selected grid:", plot_filename_early))
  }
} else {
    message("No selected grid IDs for analysis found. No early/mid range (1-", early_mid_phase_end_trial, " trials) plots will be generated.")
}

message("R script execution finished.")