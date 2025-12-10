# --- 0. Script Preamble ---
# This script analyzes learning curve data, identifies learning phases
# algorithmically using derivatives of smoothed curves, and performs
# Linear Mixed-Effects (LME) modeling on these data-informed phases.

# --- 1. Package Management ---
print("--- Section 1: Checking and Installing Packages ---")
required_packages <- c(
  "readr",      # Efficiently read text files
  "data.table", # Efficient data handling (rbindlist, subsetting)
  "stringr",    # String manipulation
  "dplyr",      # Data manipulation verbs
  "tidyr",      # Tidying data
  "ggplot2",    # Plotting
  "mgcv",       # For GAMs (smooth curves)
  # "gratia",   # We will use manual derivatives instead
  "lme4",       # For LME Models (lmer)
  "lmerTest",   # p-values for lmer models
  "car",        # Anova function (Type II/III ANOVA)
  "emmeans",    # Estimated Marginal Means
  "Hmisc",      # For mean_cl_normal (ggplot2 CIs)
  "cowplot"     # For arranging plots (if needed, good for breakpoint viz)
)
cran_mirror <- "https://cran.rstudio.com/"

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
if ("lme4" %in% required_packages && "lmerTest" %in% required_packages) {
  print("Note: lmerTest masks lmer from lme4. This is expected.")
}
print("--- Package Check Complete ---")


# --- 2. Define Data Directory & Parameters ---
print("--- Section 2: Defining Data Directory and Initial Parameters ---")
data_directory <- "C:/Users/stjoh/Documents/ActiveInference/Sophisticated-Learning/results/unknown_model/MATLAB/120trials_data_threefry" # USER: Ensure this path is correct
output_directory <- file.path(data_directory, "data_driven_phase_analysis_v3_manual_deriv") # New output folder
if (!dir.exists(output_directory)) {
  dir.create(output_directory, recursive = TRUE)
  print(paste("Created output directory:", output_directory))
}

# Parameters for breakpoint detection (USER MAY NEED TO TUNE THESE)
gam_k_knots <- 15 
derivative_n_points <- 200 # Number of points for predicting smooth curve for derivative calculation
taper_threshold_factor <- 0.25 
min_ramp_up_trials <- 5 # Minimum trials for ramp-up phase end
min_linear_phase_duration <- 10 # Minimum duration for the linear phase

print(paste("Data directory set to:", data_directory))
print(paste("Output directory for this analysis:", output_directory))
print(paste("GAM smoother knots (k):", gam_k_knots))
print(paste("Taper start threshold (factor of peak rate):", taper_threshold_factor))

# --- 3. File Checking (Simplified) ---
print("--- Section 3: Checking for Data Files ---")
actual_filenames <- list.files(path = data_directory, pattern = "\\.txt$", full.names = FALSE)
if (length(actual_filenames) < 100) {
  warning("Low number of .txt files found. Check data_directory and file pattern.")
} else {
  print(paste("Found", length(actual_filenames), ".txt files. Proceeding with loading."))
}

# --- 4. Data Loading and Preparation ---
print("--- Section 4: Loading and Preparing Data ---")
all_files <- list.files(path = data_directory, pattern = "\\.txt$", full.names = TRUE)
data_list <- list()
skipped_files_count <- 0
error_files_details <- list()
expected_algorithms_vector <- c("BA", "BAUCB", "SI", "SL") 

for (file_path in all_files) {
  filename <- basename(file_path)
  algo_match <- stringr::str_match(filename, "^([A-Z]+(?:UCB)?)_Seed_")
  algorithm_name <- if (!is.na(algo_match[1, 1])) algo_match[1, 2] else NA
  seed_match <- stringr::str_match(filename, "_Seed_(\\d+)_")
  seed_id <- if (!is.na(seed_match[1, 1])) seed_match[1, 2] else NA

  if (is.na(algorithm_name) || is.na(seed_id)) {
    msg <- paste("Could not extract Algorithm or Seed from filename:", filename)
    if (grepl("_Seed_\\d+_", filename) || grepl("^(BA|SI|SL|BAUCB)", filename, ignore.case = TRUE)) {
      # warning(msg) # Reduce console noise from non-data txt files
    }
    skipped_files_count <- skipped_files_count + 1
    error_files_details[[filename]] <- msg; next
  }
  tryCatch({
    lines <- readr::read_lines(file_path, progress = FALSE, n_max = 120)
    if (length(lines) != 120) {
      msg <- paste("File", filename, "does not contain 120 lines (found", length(lines), ").")
      warning(msg); skipped_files_count <- skipped_files_count + 1
      error_files_details[[filename]] <- msg; next
    }
    survival_values <- as.numeric(lines)
    if (any(is.na(survival_values))) {
      msg <- paste("File", filename, "contains non-numeric data.")
      warning(msg); skipped_files_count <- skipped_files_count + 1
      error_files_details[[filename]] <- msg; next
    }
    data_list[[filename]] <- data.table(
      Survival = survival_values, Iteration = 1:120,
      Id = factor(seed_id),
      Algorithm = factor(algorithm_name, levels = expected_algorithms_vector)
    )
  }, error = function(e) {
    msg <- paste("Error processing file:", filename, "-", e$message)
    warning(msg); skipped_files_count <- skipped_files_count + 1
    error_files_details[[filename]] <- msg
  })
}
if (skipped_files_count > 0) {
  print(paste("INFO: Skipped", skipped_files_count, "files (likely non-data .txt or problematic). First few:"))
  print(head(error_files_details, 5))
}
if (length(data_list) == 0) stop("No data was successfully processed.")
combined_df_untrimmed <- data.table::rbindlist(data_list)
print("--- Finished Loading and Preparing Data ---")

print("Full untrimmed data summary ('combined_df_untrimmed'):")
summary(combined_df_untrimmed)


# --- 5. Data-Driven Breakpoint Identification (Manual Derivatives) ---
print("--- Section 5: Identifying Data-Driven Phase Breakpoints (Manual Derivatives) ---")
mean_survival_by_algo_iter <- combined_df_untrimmed[, .(MeanSurvival = mean(Survival, na.rm=TRUE)), by = .(Algorithm, Iteration)]

breakpoints <- list()
derivative_plots_list <- list() # To store plots of derivatives

for (algo in levels(mean_survival_by_algo_iter$Algorithm)) { # Iterate in defined order
  print(paste("Processing algorithm:", algo))
  algo_data <- mean_survival_by_algo_iter[Algorithm == algo]
  
  actual_k <- min(gam_k_knots, nrow(unique(algo_data[, .(Iteration)])) - 1)
  if (actual_k < 5) {
    print(paste("Warning: Too few unique iterations for", algo, ". Using k =", actual_k))
    if (actual_k < 3) {
      print(paste("Skipping GAM fitting for", algo, "- insufficient points."))
      breakpoints[[as.character(algo)]] <- list(T_ramp_end = NA, T_taper_start = NA, derivatives_df = NULL)
      next
    }
  }

  gam_model <- tryCatch({
    mgcv::gam(MeanSurvival ~ s(Iteration, bs = "cs", k = actual_k), data = algo_data, method = "REML")
  }, error = function(e) { NULL })

  if (is.null(gam_model)) {
    print(paste("GAM fitting failed for", algo))
    breakpoints[[as.character(algo)]] <- list(T_ramp_end = NA, T_taper_start = NA, derivatives_df = NULL)
    next
  }
  
  # Manual derivative calculation
  iter_pred_points <- seq(min(algo_data$Iteration), max(algo_data$Iteration), length.out = derivative_n_points)
  smoothed_survival <- predict(gam_model, newdata = data.frame(Iteration = iter_pred_points), type = "response")
  
  h_step <- iter_pred_points[2] - iter_pred_points[1]
  finite_diff_derivs <- diff(smoothed_survival) / h_step
  deriv_iterations <- iter_pred_points[-length(iter_pred_points)] + h_step / 2 # Midpoints
  
  derivatives_df <- data.frame(Iteration = deriv_iterations, Derivative = finite_diff_derivs)
  
  # Identify T_ramp_end: Iteration with maximum positive derivative
  max_deriv_val <- -Inf
  T_ramp_end <- NA
  if(nrow(derivatives_df) > 0){
      positive_derivs_idx <- which(derivatives_df$Derivative > 0.01) # Small positive threshold
      if(length(positive_derivs_idx) > 0){
          max_deriv_idx <- positive_derivs_idx[which.max(derivatives_df$Derivative[positive_derivs_idx])]
          T_ramp_end <- round(derivatives_df$Iteration[max_deriv_idx])
      } else { # Fallback if no clearly positive derivatives
          max_deriv_idx <- which.max(derivatives_df$Derivative)
          if(length(max_deriv_idx) > 0) T_ramp_end <- round(derivatives_df$Iteration[max_deriv_idx])
      }
  }
  if (is.na(T_ramp_end) || T_ramp_end < min_ramp_up_trials) T_ramp_end <- min_ramp_up_trials # Ensure ramp_end is not too early

  # Identify T_taper_start
  T_taper_start <- NA
  if(nrow(derivatives_df) > 0 && !is.na(T_ramp_end)){
      # Peak rate after identified T_ramp_end (or overall if T_ramp_end is late)
      # Consider derivatives only some distance after T_ramp_end and before very end of trials
      relevant_derivs_for_peak <- derivatives_df[derivatives_df$Iteration >= T_ramp_end & derivatives_df$Iteration < (max(iter_pred_points) - 5), ]
      peak_rate_after_ramp <- if(nrow(relevant_derivs_for_peak) > 0) max(relevant_derivs_for_peak$Derivative, na.rm=TRUE) else 0
      if(peak_rate_after_ramp <= 0.01) peak_rate_after_ramp <- max(derivatives_df$Derivative[derivatives_df$Iteration >= T_ramp_end], 0.01, na.rm=TRUE) # Ensure positive

      # Find where derivative drops to taper_threshold_factor * peak_rate_after_ramp
      # Search in iterations after T_ramp_end
      potential_taper_points <- derivatives_df[derivatives_df$Iteration > (T_ramp_end + 2) & # Give some gap
                                               derivatives_df$Derivative < (peak_rate_after_ramp * taper_threshold_factor), ]
      if (nrow(potential_taper_points) > 0) {
          T_taper_start <- round(min(potential_taper_points$Iteration))
      } else {
          # If never drops that low, search for point where derivative gets very close to zero or slightly negative after peak
          near_zero_derivs <- derivatives_df[derivatives_df$Iteration > (T_ramp_end + min_linear_phase_duration) & abs(derivatives_df$Derivative) < 0.05, ]
          if(nrow(near_zero_derivs) > 0){
              T_taper_start <- round(min(near_zero_derivs$Iteration))
          } else {
              T_taper_start <- round(0.85 * max(algo_data$Iteration)) # Fallback
              print(paste("Warning: Taper start difficult to define for", algo,". Using heuristic:", T_taper_start))
          }
      }
  }
  if (is.na(T_taper_start) || T_taper_start <= T_ramp_end + min_linear_phase_duration) {
      T_taper_start <- T_ramp_end + min_linear_phase_duration + 1
      if (T_taper_start >= max(algo_data$Iteration) - 5) T_taper_start <- max(algo_data$Iteration) - 5 # Ensure taper isn't too late
  }
   # Ensure T_taper_start is not beyond max iterations
  if(T_taper_start >= max(algo_data$Iteration)) T_taper_start <- max(algo_data$Iteration) - 1


  breakpoints[[as.character(algo)]] <- list(
    T_ramp_end = T_ramp_end,
    T_taper_start = T_taper_start,
    derivatives_df = derivatives_df # Store the calculated derivatives
  )

  # Plotting
  p_smooth <- ggplot(algo_data, aes(x = Iteration, y = MeanSurvival)) +
    geom_point(alpha = 0.3, size=0.8) +
    geom_line(data = data.frame(Iteration = iter_pred_points, SmoothedSurvival = smoothed_survival),
              aes(x = Iteration, y = SmoothedSurvival), color = "blue", linewidth = 1) +
    geom_vline(xintercept = T_ramp_end, color = "green3", linetype = "dashed", linewidth = 1) +
    annotate("text", x = T_ramp_end, y = max(algo_data$MeanSurvival)*0.9, label = paste("Ramp End\n~T", T_ramp_end), angle = 90, vjust = -0.4, hjust = 1, color="green3", size=2.5) +
    geom_vline(xintercept = T_taper_start, color = "darkorange", linetype = "dashed", linewidth = 1) +
    annotate("text", x = T_taper_start, y = max(algo_data$MeanSurvival)*0.7, label = paste("Taper Start\n~T", T_taper_start), angle = 90, vjust = -0.4, hjust = 1, color="darkorange", size=2.5) +
    labs(title = paste("Smoothed Curve & Breakpoints -", algo), x = "Iteration", y = "Mean Survival") + theme_bw(base_size=9)

  p_deriv <- ggplot(derivatives_df, aes(x = Iteration, y = Derivative)) +
    geom_line(color = "red", linewidth = 1) +
    geom_vline(xintercept = T_ramp_end, color = "green3", linetype = "dashed", linewidth = 1) +
    geom_vline(xintercept = T_taper_start, color = "darkorange", linetype = "dashed", linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dotted") +
    labs(title = paste("1st Derivative (Learning Rate) -", algo), x = "Iteration", y = "Rate of Change") + theme_bw(base_size=9)
  
  derivative_plots_list[[as.character(algo)]] <- cowplot::plot_grid(p_smooth, p_deriv, ncol = 1)
  ggsave(filename = file.path(output_directory, paste0("breakpoint_viz_manual_", algo, ".pdf")),
         plot = derivative_plots_list[[as.character(algo)]], device = "pdf", width = 6, height = 7, units = "in")
}

print("Algorithm-specific breakpoints (manual derivatives):")
all_ramp_ends <- c()
all_taper_starts <- c()
for (algo_name in names(breakpoints)) {
  bp_info <- breakpoints[[algo_name]]
  if(!is.null(bp_info) && !is.na(bp_info$T_ramp_end) && !is.na(bp_info$T_taper_start)){
    print(paste(algo_name, "- Ramp End:", bp_info$T_ramp_end, "| Taper Start:", bp_info$T_taper_start))
    all_ramp_ends <- c(all_ramp_ends, bp_info$T_ramp_end)
    all_taper_starts <- c(all_taper_starts, bp_info$T_taper_start)
  } else {
    print(paste(algo_name, "- Breakpoint identification incomplete."))
  }
}

# Define Consensus Breakpoints for LME phases
consensus_T_ramp_end <- if (length(all_ramp_ends) > 0) round(median(all_ramp_ends, na.rm = TRUE)) else 15
# Ensure T_ramp_end is reasonable
consensus_T_ramp_end <- max(min_ramp_up_trials, consensus_T_ramp_end)
consensus_T_ramp_end <- min(consensus_T_ramp_end, max(combined_df_untrimmed$Iteration) - (2*min_linear_phase_duration)) # Ensure space for other phases

consensus_T_linear_start <- consensus_T_ramp_end + 1

# Taper start should be after ramp end + min linear duration
valid_taper_starts_for_consensus <- all_taper_starts[all_taper_starts > (consensus_T_ramp_end + min_linear_phase_duration)]
consensus_T_taper_start <- if (length(valid_taper_starts_for_consensus) > 0) round(median(valid_taper_starts_for_consensus, na.rm = TRUE)) else (consensus_T_ramp_end + min_linear_phase_duration + round(0.4 * (max(combined_df_untrimmed$Iteration)- (consensus_T_ramp_end + min_linear_phase_duration) )))

consensus_T_taper_start <- max(consensus_T_taper_start, consensus_T_linear_start + min_linear_phase_duration) # Ensure linear phase has some duration
consensus_T_taper_start <- min(consensus_T_taper_start, max(combined_df_untrimmed$Iteration) - 5) # Ensure taper phase has some data

consensus_T_linear_end <- consensus_T_taper_start - 1

if(consensus_T_linear_end < consensus_T_linear_start){ # Safety check
    consensus_T_linear_end = consensus_T_linear_start + min_linear_phase_duration
    consensus_T_taper_start = consensus_T_linear_end + 1
    print("Warning: Consensus linear phase was too short or inverted, adjusted.")
}


print(paste("Consensus Phase 1 (Ramp-up) End: Trial", consensus_T_ramp_end))
print(paste("Consensus Phase 2 (Sustained Learning) Start: Trial", consensus_T_linear_start, "End: Trial", consensus_T_linear_end))
print(paste("Consensus Phase 3 (Taper) Start: Trial", consensus_T_taper_start))

df_consensus_phase1 <- combined_df_untrimmed[Iteration >= 1 & Iteration <= consensus_T_ramp_end]
df_consensus_phase2 <- combined_df_untrimmed[Iteration >= consensus_T_linear_start & Iteration <= consensus_T_linear_end]
df_consensus_phase3 <- combined_df_untrimmed[Iteration >= consensus_T_taper_start & Iteration <= max(combined_df_untrimmed$Iteration)]

print(paste("Obs in C-Phase 1:", nrow(df_consensus_phase1), "| C-Phase 2:", nrow(df_consensus_phase2), "| C-Phase 3:", nrow(df_consensus_phase3)))

# --- 6. Overall Learning Curve Plot with Consensus Phases ---
# (Code remains largely the same as previous script, just uses new consensus breakpoints)
# ... (ensure this section is complete as in previous version, it was abridged here for brevity) ...
print("--- Section 6: Plotting Overall Learning Curve with Data-Driven Consensus Phases ---")
if(nrow(combined_df_untrimmed) > 0){
    plot_overall_lc_consensus <- ggplot(data = combined_df_untrimmed, aes(x = Iteration, y = Survival, color = Algorithm)) +
      stat_summary(fun = mean, geom = "line", linewidth = 1) +
      stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") +
      labs(title = "Average Learning Curves with Data-Driven Consensus Phases",
           subtitle = "Lines: Mean Survival, Ribbons: 95% CI. Vertical lines mark consensus phase boundaries.",
           x = "Iteration", y = "Average Survival") +
      theme_bw(base_size = 11) +
      theme(legend.position = "bottom", plot.title = element_text(hjust=0.5), plot.subtitle = element_text(hjust=0.5),
            panel.grid.major = element_line(colour = "grey90"), panel.grid.minor = element_blank()) +
      geom_vline(xintercept = consensus_T_ramp_end + 0.5, linetype = "dashed", color = "blue", linewidth = 0.8) +
      annotate("text", x = consensus_T_ramp_end + 0.5, y = max(combined_df_untrimmed$Survival, na.rm = TRUE) * 0.95,
               label = paste("End Ramp-up\n~T", consensus_T_ramp_end), angle = 90, vjust = -0.3, hjust = 1, size = 3, color = "blue") +
      geom_vline(xintercept = consensus_T_linear_end + 0.5, linetype = "dashed", color = "red", linewidth = 0.8) +
      annotate("text", x = consensus_T_linear_end + 0.5, y = max(combined_df_untrimmed$Survival, na.rm = TRUE) * 0.85,
               label = paste("End Sustained\n~T", consensus_T_linear_end), angle = 90, vjust = -0.3, hjust = 1, size = 3, color = "red") +
      scale_color_brewer(palette = "Set1") +
      scale_fill_brewer(palette = "Set1")
      
    print(plot_overall_lc_consensus)
    ggsave(filename = file.path(output_directory, "learning_curve_plot_consensus_phases.pdf"),
           plot = plot_overall_lc_consensus, device = "pdf", width = 7, height = 5.5, units = "in") # Adjusted height slightly
    print(paste("Saved overall learning curve plot with consensus phases to:", output_directory))
} else {
    print("Skipping overall learning curve plot as data is empty.")
}


# --- 7. LME Analysis for Consensus Phases (Structure similar to previous, using new DFs) ---
# Full details for summary, Anova, emtrends, emmeans, and plot saving should be here for each phase.
# Example for Phase 1:
print(paste0("--- Section 7A: Analyzing Consensus Phase 1 (Ramp-up: Trials 1-", consensus_T_ramp_end, ") ---"))
if (nrow(df_consensus_phase1) > 0 && length(unique(df_consensus_phase1$Iteration)) > 1) {
    model_cphase1_ramp_up <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = df_consensus_phase1)
    print(paste("Summary for Consensus Phase 1 (Ramp-up) Model:"))
    tryCatch({print(summary(model_cphase1_ramp_up))}, error=function(e) print(e))
    print(paste("ANOVA for Consensus Phase 1 (Ramp-up) Model (Iteration 1 to", consensus_T_ramp_end,")"))
    tryCatch({print(car::Anova(model_cphase1_ramp_up,test='F', type="III"))}, error=function(e) print(e))
    print(paste("EMTRENDS for Consensus Phase 1 (Ramp-up) Model (Iteration 1 to", consensus_T_ramp_end,")"))
    tryCatch({print(summary(emmeans::emtrends(model_cphase1_ramp_up, pairwise ~ Algorithm, var = "Iteration"), infer=TRUE))}, error=function(e) print(e))
    print(paste("EMMEANS at end of Consensus Phase 1 (Trial", consensus_T_ramp_end,")"))
    if(consensus_T_ramp_end %in% unique(df_consensus_phase1$Iteration)){
        tryCatch({print(summary(emmeans::emmeans(model_cphase1_ramp_up, pairwise ~ Algorithm | Iteration, at = list(Iteration = consensus_T_ramp_end)), infer=TRUE))}, error=function(e) print(e))
    } else { print(paste("Consensus_T_ramp_end", consensus_T_ramp_end, "not in phase 1 data for emmeans (unique iterations:", paste(sort(unique(df_consensus_phase1$Iteration)), collapse=", "), ")"))}
    
    # Plotting LME fit for phase 1
    emm_preds_cphase1 <- emmeans(model_cphase1_ramp_up, ~ Algorithm | Iteration,
                           at = list(Iteration = sort(unique(df_consensus_phase1$Iteration))), type = "response")
    predicted_df_cphase1 <- as.data.frame(summary(emm_preds_cphase1))
    names(predicted_df_cphase1)[names(predicted_df_cphase1) == "emmean"] <- "PredictedSurvival"
    plot_cphase1_fit <- ggplot(data = df_consensus_phase1, aes(x = Iteration, y = Survival, color = Algorithm)) +
        stat_summary(fun = mean, geom = "line", linewidth = 0.8, alpha = 0.5, aes(linetype = "Observed Mean")) +
        stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") +
        geom_line(data = predicted_df_cphase1, aes(x = Iteration, y = PredictedSurvival, color = Algorithm, linetype = "Model Fit"), linewidth = 1.1) +
        scale_linetype_manual(name = "Data Type", values = c("Observed Mean" = "dashed", "Model Fit" = "solid")) +
        labs(title = paste("Consensus Phase 1 (Ramp-up) Fit: Trials 1-", consensus_T_ramp_end),
             subtitle = "Observed (95% CI) vs. LME Predicted Mean Survival", x = "Iteration", y = "Average Survival") +
        theme_bw(base_size=10) + theme(legend.position="bottom", plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5),
                                       panel.grid.major = element_line(colour = "grey90"), panel.grid.minor = element_blank()) +
        scale_color_brewer(palette="Set1") + scale_fill_brewer(palette="Set1")
    print(plot_cphase1_fit)
    ggsave(file.path(output_directory, "lme_fit_consensus_phase1.pdf"), plot_cphase1_fit, device="pdf", width=6, height=4.5, units="in")

} else { print("Skipping LME for Consensus Phase 1 due to insufficient data.") }


print(paste0("--- Section 7B: Analyzing Consensus Phase 2 (Sustained Learning: Trials ", consensus_T_linear_start, "-", consensus_T_linear_end, ") ---"))
if (nrow(df_consensus_phase2) > 0 && length(unique(df_consensus_phase2$Iteration)) > 1) {
    model_cphase2_sustained <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = df_consensus_phase2)
    print(paste("Summary for Consensus Phase 2 (Sustained) Model:"))
    tryCatch({print(summary(model_cphase2_sustained))}, error=function(e) print(e))
    print(paste("ANOVA for Consensus Phase 2 (Sustained) Model (Iteration", consensus_T_linear_start, "to", consensus_T_linear_end,")"))
    tryCatch({print(car::Anova(model_cphase2_sustained,test='F', type="III"))}, error=function(e) print(e))
    print(paste("EMTRENDS for Consensus Phase 2 (Sustained) Model (Iteration", consensus_T_linear_start, "to", consensus_T_linear_end,")"))
    tryCatch({print(summary(emmeans::emtrends(model_cphase2_sustained, pairwise ~ Algorithm, var = "Iteration"), infer=TRUE))}, error=function(e) print(e))
    midpoint_cphase2 <- floor((consensus_T_linear_start + consensus_T_linear_end) / 2)
    print(paste("EMMEANS at midpoint of Consensus Phase 2 (Trial", midpoint_cphase2,")"))
    if(midpoint_cphase2 %in% unique(df_consensus_phase2$Iteration)){
        tryCatch({print(summary(emmeans::emmeans(model_cphase2_sustained, pairwise ~ Algorithm | Iteration, at = list(Iteration = midpoint_cphase2)), infer=TRUE))}, error=function(e) print(e))
    } else { print(paste("Midpoint_cphase2", midpoint_cphase2, "not in phase 2 data for emmeans (unique iterations:", paste(sort(unique(df_consensus_phase2$Iteration)), collapse=", "), ")"))}

    # Plotting LME fit for phase 2
    emm_preds_cphase2 <- emmeans(model_cphase2_sustained, ~ Algorithm | Iteration,
                           at = list(Iteration = sort(unique(df_consensus_phase2$Iteration))), type = "response")
    predicted_df_cphase2 <- as.data.frame(summary(emm_preds_cphase2))
    names(predicted_df_cphase2)[names(predicted_df_cphase2) == "emmean"] <- "PredictedSurvival"
    plot_cphase2_fit <- ggplot(data = df_consensus_phase2, aes(x = Iteration, y = Survival, color = Algorithm)) +
        stat_summary(fun = mean, geom = "line", linewidth = 0.8, alpha = 0.5, aes(linetype = "Observed Mean")) +
        stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") +
        geom_line(data = predicted_df_cphase2, aes(x = Iteration, y = PredictedSurvival, color = Algorithm, linetype = "Model Fit"), linewidth = 1.1) +
        scale_linetype_manual(name = "Data Type", values = c("Observed Mean" = "dashed", "Model Fit" = "solid")) +
        labs(title = paste("Consensus Phase 2 (Sustained) Fit: Trials", consensus_T_linear_start, "-", consensus_T_linear_end),
             subtitle = "Observed (95% CI) vs. LME Predicted Mean Survival", x = "Iteration", y = "Average Survival") +
        theme_bw(base_size=10) + theme(legend.position="bottom", plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5),
                                       panel.grid.major = element_line(colour = "grey90"), panel.grid.minor = element_blank()) +
        scale_color_brewer(palette="Set1") + scale_fill_brewer(palette="Set1")
    print(plot_cphase2_fit)
    ggsave(file.path(output_directory, "lme_fit_consensus_phase2.pdf"), plot_cphase2_fit, device="pdf", width=6, height=4.5, units="in")

} else { print("Skipping LME for Consensus Phase 2 due to insufficient data.") }


print(paste0("--- Section 7C: Analyzing Consensus Phase 3 (Taper: Trials ", consensus_T_taper_start, "-", max(combined_df_untrimmed$Iteration), ") ---"))
if (nrow(df_consensus_phase3) > 0 && length(unique(df_consensus_phase3$Iteration)) > 1) {
    model_cphase3_taper <- lmer(Survival ~ Iteration * Algorithm + (1 | Id), data = df_consensus_phase3)
    print(paste("Summary for Consensus Phase 3 (Taper) Model:"))
    tryCatch({print(summary(model_cphase3_taper))}, error=function(e) print(e))
    print(paste("ANOVA for Consensus Phase 3 (Taper) Model (Iteration", consensus_T_taper_start, "to end)"))
    tryCatch({print(car::Anova(model_cphase3_taper,test='F', type="III"))}, error=function(e) print(e))
    print(paste("EMTRENDS for Consensus Phase 3 (Taper) Model (Iteration", consensus_T_taper_start, "to end)"))
    tryCatch({print(summary(emmeans::emtrends(model_cphase3_taper, pairwise ~ Algorithm, var = "Iteration"), infer=TRUE))}, error=function(e) print(e))
    late_point_cphase3 <- if(nrow(df_consensus_phase3) > 0) floor(median(unique(df_consensus_phase3$Iteration))) else NA # Use median if possible
    if(is.na(late_point_cphase3) && nrow(df_consensus_phase3) > 0) late_point_cphase3 <- min(df_consensus_phase3$Iteration) + 1
    print(paste("EMMEANS at a representative point of Consensus Phase 3 (Trial", late_point_cphase3,")"))
    if(!is.na(late_point_cphase3) && late_point_cphase3 %in% unique(df_consensus_phase3$Iteration)){
        tryCatch({print(summary(emmeans::emmeans(model_cphase3_taper, pairwise ~ Algorithm | Iteration, at = list(Iteration = late_point_cphase3)), infer=TRUE))}, error=function(e) print(e))
    } else { print(paste("Representative point for Phase 3 (", late_point_cphase3,") not valid or not in phase 3 data for emmeans."))}

    # Plotting LME fit for phase 3
    emm_preds_cphase3 <- emmeans(model_cphase3_taper, ~ Algorithm | Iteration,
                           at = list(Iteration = sort(unique(df_consensus_phase3$Iteration))), type = "response")
    predicted_df_cphase3 <- as.data.frame(summary(emm_preds_cphase3))
    names(predicted_df_cphase3)[names(predicted_df_cphase3) == "emmean"] <- "PredictedSurvival"
    plot_cphase3_fit <- ggplot(data = df_consensus_phase3, aes(x = Iteration, y = Survival, color = Algorithm)) +
        stat_summary(fun = mean, geom = "line", linewidth = 0.8, alpha = 0.5, aes(linetype = "Observed Mean")) +
        stat_summary(fun.data = mean_cl_normal, geom = "ribbon", alpha = 0.2, aes(fill = Algorithm), linetype = "blank") +
        geom_line(data = predicted_df_cphase3, aes(x = Iteration, y = PredictedSurvival, color = Algorithm, linetype = "Model Fit"), linewidth = 1.1) +
        scale_linetype_manual(name = "Data Type", values = c("Observed Mean" = "dashed", "Model Fit" = "solid")) +
        labs(title = paste("Consensus Phase 3 (Taper) Fit: Trials", consensus_T_taper_start, "-", max(df_consensus_phase3$Iteration)),
             subtitle = "Observed (95% CI) vs. LME Predicted Mean Survival", x = "Iteration", y = "Average Survival") +
        theme_bw(base_size=10) + theme(legend.position="bottom", plot.title=element_text(hjust=0.5), plot.subtitle=element_text(hjust=0.5),
                                       panel.grid.major = element_line(colour = "grey90"), panel.grid.minor = element_blank()) +
        scale_color_brewer(palette="Set1") + scale_fill_brewer(palette="Set1")
    print(plot_cphase3_fit)
    ggsave(file.path(output_directory, "lme_fit_consensus_phase3.pdf"), plot_cphase3_fit, device="pdf", width=6, height=4.5, units="in")
} else { print("Skipping LME for Consensus Phase 3 due to insufficient data.") }

print("--- SCRIPT FINISHED ---")