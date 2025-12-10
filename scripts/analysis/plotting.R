# --- 1. Package Management ---
# This section ensures all required packages are installed and loaded.
required_packages <- c(
  "readr",     # For reading text files efficiently
  "stringr",   # For string manipulation (extracting info from filenames)
  "ggplot2",   # For plotting
  "data.table",# For efficient data handling (rbindlist, subsetting)
  "Hmisc"      # For mean_cl_normal, mean_cl_boot, mean_sdl (used by stat_summary)
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
print("--- Package Check Complete ---")

# --- 2. Define Data Directory ---
# !!! IMPORTANT: SET YOUR DATA DIRECTORY HERE !!!
data_directory <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/120trials_data_threefry"
# Example: data_directory <- "path/to/your/results/folder"
# !!! IMPORTANT: ALSO SET YOUR OUTPUT DIRECTORY FOR PLOTS !!!
output_directory <- file.path(data_directory, "error_band_comparison_plots_pdf") # Changed folder name for PDF outputs

if (!dir.exists(output_directory)) {
  dir.create(output_directory, recursive = TRUE)
  print(paste("Created output directory:", output_directory))
}
print(paste("Data directory set to:", data_directory))
print(paste("Plot output directory set to:", output_directory))


# --- 3. Data Loading and Preparation (Simplified) ---
print("--- Section 3: Loading and Preparing Data ---")
all_files <- list.files(path = data_directory, pattern = "\\.txt$", full.names = TRUE)
if (length(all_files) == 0) {
  stop("No .txt files found for loading in: ", data_directory)
} else {
  print(paste("Found", length(all_files), "text files. Proceeding to load..."))
}

data_list <- list()
skipped_files_count <- 0
error_files_details <- list()

for (file_path in all_files) {
  filename <- basename(file_path)
  algo_match <- stringr::str_match(filename, "^(.+?)_Seed_")
  algorithm_name <- if(!is.na(algo_match[1,1])) algo_match[1,2] else NA
  seed_match <- stringr::str_match(filename, "_Seed_(\\d+)_")
  seed_id <- if(!is.na(seed_match[1,1])) seed_match[1,2] else NA

  if (is.na(algorithm_name) || is.na(seed_id)) {
    msg <- paste("Could not extract Algorithm or Seed from filename:", filename)
    # warning(msg); # Suppressing warnings for brevity in this example script
    skipped_files_count <- skipped_files_count + 1
    error_files_details[[filename]] <- msg; next
  }
  tryCatch({
      lines <- readr::read_lines(file_path, progress = FALSE)
      if (length(lines) != 120) { # Assuming 120 trials as per original script
          msg <- paste("File", filename, "does not contain 120 lines (found", length(lines), ").")
          skipped_files_count <- skipped_files_count + 1
          error_files_details[[filename]] <- msg; next
      }
      survival_values <- as.numeric(lines)
      if(any(is.na(survival_values))) {
          msg <- paste("File", filename, "contains non-numeric data.")
          skipped_files_count <- skipped_files_count + 1
          error_files_details[[filename]] <- msg; next
      }
      data_list[[filename]] <- data.table( # Using data.table directly
        Survival = survival_values, Iteration = 1:120,
        Id = seed_id, Algorithm = algorithm_name
      )
  }, error = function(e) {
      msg <- paste("Error processing file:", filename, "-", e$message)
      skipped_files_count <- skipped_files_count + 1
      error_files_details[[filename]] <- msg
  })
}

if (skipped_files_count > 0) {
    print(paste("WARNING: Skipped", skipped_files_count, "files due to parsing issues or length mismatch. First few errors:"))
    print(head(error_files_details, 5))
}
if (length(data_list) == 0) stop("No data was successfully processed. Check data_directory and file formats.")

combined_df <- data.table::rbindlist(data_list)
combined_df[, Id := factor(Id)]
combined_df[, Algorithm := factor(Algorithm)]

print("Data summary ('combined_df'):")
summary(combined_df)
print(paste("Total observations in combined_df:", nrow(combined_df)))
if (nrow(combined_df) == 0) {
    stop("Combined data frame is empty. Halting script.")
}
print("--- Finished Loading and Preparing Data ---")


# --- 4. Plotting with Different Error Band Methods ---
print("--- Section 4: Generating Plots with Different Error Bands ---")

# Base plot elements to reuse
base_plot_stat_summary <- function(data, title_suffix) { # Renamed for clarity
  ggplot(data = data, aes(x = Iteration, y = Survival, color = Algorithm, fill = Algorithm)) +
    stat_summary(fun = mean, geom = "line", linewidth = 1) + # Mean line based on point-wise mean
    labs(title = paste("Average Learning Curves by Algorithm", title_suffix),
         subtitle = "Lines represent point-wise mean survival",
         x = "Iteration (Trial Number)", y = "Average Survival Steps") +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size=14),
          plot.subtitle = element_text(hjust = 0.5, size=10),
          legend.position = "bottom")
}

# Base plot elements for geom_smooth
base_plot_geom_smooth <- function(data, title_suffix) {
  ggplot(data = data, aes(x = Iteration, y = Survival, color = Algorithm, fill = Algorithm)) +
    # geom_smooth will plot both the smoothed line and the CI band
    labs(title = paste("Smoothed Learning Curves by Algorithm", title_suffix),
         x = "Iteration (Trial Number)", y = "Average Survival Steps") +
    theme_bw(base_size = 12) +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size=14),
          plot.subtitle = element_text(hjust = 0.5, size=10),
          legend.position = "bottom")
}


# Method 1: Standard Error of the Mean (SE)
plot_se <- base_plot_stat_summary(combined_df, "(Error Bands: +/- 1 Standard Error)") +
  stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.2, linetype = "blank") +
  labs(subtitle = "Lines represent point-wise mean survival, ribbons represent +/- 1 Standard Error")
print(plot_se)
ggsave(filename = file.path(output_directory, "learning_curve_plot_SE.pdf"), plot = plot_se, width = 10, height = 7, device = "pdf")
print("Saved PDF plot with Standard Error bands.")

# Method 2: 95% Confidence Interval (Normal Approximation using stat_summary) - MODIFIED
plot_ci_normal_stat <- base_plot_stat_summary(combined_df, "") + # Suffix removed for custom title
  stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, linetype = "blank") +
  labs(title = "Average Learning Curves by Algorithm", # Explicitly set title
       subtitle = "Lines represent point-wise mean survival, ribbons represent 95% CI (normal approx.)") +
  geom_vline(xintercept = 20.5, linetype = "dashed", color = "blue", linewidth = 0.7) +
  annotate("text", x = 20.5, y = Inf, label = "Trial 20", angle = 90, vjust = 1.5, hjust = 1.1, size = 3.5, color = "blue") +
  geom_vline(xintercept = 65.5, linetype = "dashed", color = "red", linewidth = 0.7) +
  annotate("text", x = 65.5, y = Inf, label = "Trial 65", angle = 90, vjust = -0.5, hjust = 1.1, size = 3.5, color = "red") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size=14), # Ensure theme elements are consistent
        plot.subtitle = element_text(hjust = 0.5, size=10),
        legend.position = "bottom")

print(plot_ci_normal_stat)
ggsave(filename = file.path(output_directory, "learning_curve_plot_CI_normal_stat_summary_phases.pdf"), plot = plot_ci_normal_stat, width = 10, height = 7, device = "pdf") # Added _phases to filename
print("Saved PDF plot with 95% CI (Normal via stat_summary) bands and phase lines.")

# Method 3: 95% Confidence Interval (Bootstrap using stat_summary)
plot_ci_boot_stat <- base_plot_stat_summary(combined_df, "(Error Bands: 95% CI - Bootstrap via stat_summary)") +
  stat_summary(fun.data = mean_cl_boot, geom = "ribbon", alpha = 0.2, linetype = "blank") + # Default B is 1000
  labs(subtitle = "Lines represent point-wise mean survival, ribbons represent 95% CI (bootstrap)")
print(plot_ci_boot_stat)
ggsave(filename = file.path(output_directory, "learning_curve_plot_CI_bootstrap_stat_summary.pdf"), plot = plot_ci_boot_stat, width = 10, height = 7, device = "pdf")
print("Saved PDF plot with 95% CI (Bootstrap via stat_summary) bands.")

# Method 4: Standard Deviation (SD)
plot_sd <- base_plot_stat_summary(combined_df, "(Error Bands: +/- 1 Standard Deviation)") +
  stat_summary(fun.data = mean_sdl, fun.args = list(mult = 1), geom = "ribbon", alpha = 0.2, linetype = "blank") +
  labs(subtitle = "Lines represent point-wise mean survival, ribbons represent +/- 1 Standard Deviation")
print(plot_sd)
ggsave(filename = file.path(output_directory, "learning_curve_plot_SD.pdf"), plot = plot_sd, width = 10, height = 7, device = "pdf")
print("Saved PDF plot with Standard Deviation bands.")

# Method 5: Percentile Bands (e.g., 5th - 95th Percentiles)
plot_percentiles_5_95 <- base_plot_stat_summary(combined_df, "(Error Bands: 5th - 95th Percentiles)") +
  stat_summary(fun.data = function(x) {
    mean_val <- mean(x, na.rm = TRUE)
    ymin_val <- quantile(x, probs = 0.05, na.rm = TRUE)
    ymax_val <- quantile(x, probs = 0.95, na.rm = TRUE)
    data.frame(y = mean_val, ymin = ymin_val, ymax = ymax_val)
  }, geom = "ribbon", alpha = 0.2, linetype = "blank") +
  labs(subtitle = "Lines represent point-wise mean survival, ribbons represent 5th - 95th Percentile range")
print(plot_percentiles_5_95)
ggsave(filename = file.path(output_directory, "learning_curve_plot_Percentiles_5_95.pdf"), plot = plot_percentiles_5_95, width = 10, height = 7, device = "pdf")
print("Saved PDF plot with 5th-95th Percentile bands.")

# Method 6: Percentile Bands (Interquartile Range: 25th - 75th Percentiles)
plot_iqr <- base_plot_stat_summary(combined_df, "(Error Bands: Interquartile Range, 25th-75th Percentiles)") +
  stat_summary(fun.data = function(x) {
    mean_val <- mean(x, na.rm = TRUE)
    ymin_val <- quantile(x, probs = 0.25, na.rm = TRUE)
    ymax_val <- quantile(x, probs = 0.75, na.rm = TRUE)
    data.frame(y = mean_val, ymin = ymin_val, ymax = ymax_val)
  }, geom = "ribbon", alpha = 0.2, linetype = "blank") +
  labs(subtitle = "Lines represent point-wise mean survival, ribbons represent Interquartile Range (25th-75th Percentiles)")
print(plot_iqr)
ggsave(filename = file.path(output_directory, "learning_curve_plot_IQR.pdf"), plot = plot_iqr, width = 10, height = 7, device = "pdf")
print("Saved PDF plot with Interquartile Range (IQR) bands.")

# Method 7: Sample Trajectories with 95% CI (Normal Approx. via stat_summary)
n_sample_ids <- min(10, length(unique(combined_df$Id)))
if (n_sample_ids > 0) {
    sample_ids_for_plot <- sample(unique(combined_df$Id), n_sample_ids)
    sample_runs_df <- combined_df[Id %in% sample_ids_for_plot]

    plot_sample_traj_ci_stat <- ggplot(data = combined_df, aes(x = Iteration, y = Survival, color = Algorithm)) +
      geom_line(data = sample_runs_df, aes(group = interaction(Id, Algorithm)), alpha = 0.2, linewidth = 0.4, show.legend = FALSE) +
      stat_summary(fun = mean, geom = "line", linewidth = 1) +
      stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.25, aes(fill = Algorithm), linetype = "blank") +
      labs(title = "Average Learning Curves with Sample Trajectories & 95% CI (stat_summary)",
           subtitle = paste0("Faint lines: ", n_sample_ids, " sample trajectories. Ribbons: 95% CI (normal approx.)"),
           x = "Iteration (Trial Number)", y = "Average Survival Steps") +
      theme_bw(base_size = 12) +
      theme(plot.title = element_text(hjust = 0.5, face = "bold", size=14),
            plot.subtitle = element_text(hjust = 0.5, size=10),
            legend.position = "bottom")
    print(plot_sample_traj_ci_stat)
    ggsave(filename = file.path(output_directory, "learning_curve_plot_SampleTraj_CI_stat_summary.pdf"), plot = plot_sample_traj_ci_stat, width = 10, height = 7, device = "pdf")
    print("Saved PDF plot with Sample Trajectories and 95% CI (stat_summary) bands.")
} else {
    print("Skipping sample trajectories plot (stat_summary) as no valid IDs were found or n_sample_ids was 0.")
}

# --- NEW METHOD ---
# Method 8: Smoothed Confidence Interval (95% CI using geom_smooth)
# geom_smooth uses 'loess' for <1000 obs per group, 'gam' for >=1000 obs by default.
# 'gam' (Generalized Additive Model) might be more appropriate for larger, potentially complex datasets.
# You can specify method = "gam", formula = y ~ s(x, bs = "cs") for a cubic regression spline with GAM.
plot_ci_smooth <- base_plot_geom_smooth(combined_df, "(Smoothed Line and 95% CI via geom_smooth)") +
  geom_smooth(aes(color = Algorithm, fill = Algorithm), method = "gam", formula = y ~ s(x, bs = "cs"), se = TRUE, linewidth = 1, alpha = 0.2) +
  # `se = TRUE` is default to show confidence interval.
  # `linewidth` controls the smoothed line thickness.
  # `alpha` controls the transparency of the CI ribbon.
  labs(subtitle = "Lines and ribbons represent smoothed mean and 95% Confidence Interval (GAM)")
print(plot_ci_smooth)
ggsave(filename = file.path(output_directory, "learning_curve_plot_CI_smooth_gam.pdf"), plot = plot_ci_smooth, width = 10, height = 7, device = "pdf")
print("Saved PDF plot with Smoothed 95% CI (GAM) bands using geom_smooth.")


print(paste("--- All PDF plots generated and saved to:", output_directory, " ---"))
print("--- Script Finished ---")