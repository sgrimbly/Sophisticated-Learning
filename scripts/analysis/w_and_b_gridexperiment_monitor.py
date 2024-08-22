import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle

def parse_filename(filename):
    pattern = r"^(SI|SL)_Seed_(\d+)_GridID_([a-f0-9]+)_\d{2}-\d{2}-\d{2}-\d{3}.txt"
    match = re.match(pattern, filename)
    if match:
        return match.groups()
    else:
        print(f"Failed to match: {filename}")
    return None

def load_data(filepath):
    data = pd.read_csv(filepath, header=None)
    if data.shape[1] == 1:  # If there is only one column, convert DataFrame to Series
        data = data.iloc[:, 0]
    return data

def main(directory_path):
    results = {}
    max_trials = {}
    grid_colors = {}
    color_palette = cycle(plt.cm.tab10.colors)
    valid_trial_length = 200  # Define the valid trial length as 200

    print(f"Scanning directory: {directory_path}")

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):  # Only process .txt files
            print(f"Processing .txt file: {filename}")
            file_info = parse_filename(filename)
            if file_info:
                print(f"Parsed info: {file_info}")
                algo_type, seed, grid_id = file_info
                key = (algo_type, grid_id)
                data_path = os.path.join(directory_path, filename)
                data = load_data(data_path)

                # Only include data with the correct number of trials (200)
                if len(data) == valid_trial_length:
                    if key not in results:
                        results[key] = []
                    results[key].append(data)

                    if key not in max_trials or len(data) > max_trials[key]:
                        max_trials[key] = len(data)
                
                if grid_id not in grid_colors:
                    grid_colors[grid_id] = next(color_palette)
            else:
                print(f"Failed to parse: {filename}")
        else:
            print(f"Skipping non-txt file: {filename}")

    unique_grid_ids = sorted(set(grid_id for _, grid_id in results.keys()))
    print(f"Unique Grid IDs: {unique_grid_ids}")

    if len(unique_grid_ids) == 0:
        print("No grid IDs found. Exiting...")
        return

    # Individual plots: Modify to 3 columns (Survival, Sample Count, Rate of Change)
    fig, axes = plt.subplots(nrows=len(unique_grid_ids), ncols=3, figsize=(22, 5 * len(unique_grid_ids)), constrained_layout=True)

    linestyles = {'SI': '-', 'SL': '--'}
    survival_limits = [float('inf'), float('-inf')]
    count_limits = [float('inf'), float('-inf')]
    roc_limits = [float('inf'), float('-inf')]  # Limits for rate of change

    # First pass: Calculate data and find limits
    for grid_id in unique_grid_ids:
        for algo_type in ['SI', 'SL']:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            for key in key_list:
                data_list = results[key]
                trial_length = max_trials[key]
                padded_data = np.full((len(data_list), trial_length), np.nan)
                sample_count = np.zeros(trial_length)

                for i, data in enumerate(data_list):
                    valid_length = len(data)
                    padded_data[i, :valid_length] = data
                    sample_count[:valid_length] += 1

                # Calculate mean, standard deviation, and rate of change
                data_avg = np.nanmean(padded_data, axis=0)
                data_std = np.nanstd(padded_data, axis=0)
                roc = np.diff(data_avg, prepend=np.nan)  # Rate of change (using prepend for size alignment)

                survival_limits[0] = min(survival_limits[0], np.nanmin(data_avg - data_std))
                survival_limits[1] = max(survival_limits[1], np.nanmax(data_avg + data_std))
                count_limits[0] = min(count_limits[0], np.min(sample_count))
                count_limits[1] = max(count_limits[1], np.max(sample_count))
                roc_limits[0] = min(roc_limits[0], np.nanmin(roc))
                roc_limits[1] = max(roc_limits[1], np.nanmax(roc))

    # Second pass: Plot using consistent axes limits
    for grid_id in unique_grid_ids:
        ax1 = axes[unique_grid_ids.index(grid_id), 0]  # Column 1: Survival
        ax2 = axes[unique_grid_ids.index(grid_id), 1]  # Column 2: Sample Count
        ax3 = axes[unique_grid_ids.index(grid_id), 2]  # Column 3: Rate of Change (new)

        for algo_type in ['SI', 'SL']:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            for key in key_list:
                data_list = results[key]
                trial_length = max_trials[key]
                padded_data = np.full((len(data_list), trial_length), np.nan)
                sample_count = np.zeros(trial_length)

                for i, data in enumerate(data_list):
                    valid_length = len(data)
                    padded_data[i, :valid_length] = data
                    sample_count[:valid_length] += 1

                # Calculate mean, standard deviation, and rate of change
                data_avg = np.nanmean(padded_data, axis=0)
                data_std = np.nanstd(padded_data, axis=0)
                roc = np.diff(data_avg, prepend=np.nan)  # Rate of change (same size as data_avg)

                # Plot average survival with uncertainty bounds in ax1
                ax1.plot(data_avg, label=f"{key[0]} {key[1]}", color=grid_colors[grid_id], linestyle=linestyles[algo_type])
                ax1.fill_between(range(trial_length), data_avg - data_std, data_avg + data_std, 
                                 color=grid_colors[grid_id], alpha=0.2)

                # Plot sample count in ax2
                ax2.plot(sample_count, label=f"Sample count for {key[0]}", linestyle=':', color=grid_colors[grid_id])

                # Plot rate of change in ax3
                ax3.plot(roc, label=f"{key[0]} {key[1]}", color=grid_colors[grid_id], linestyle=linestyles[algo_type])

        # Set axis limits and labels
        ax1.set_ylim(survival_limits)
        ax1.set_title(f"Average Survival for GridID {grid_id}")
        ax1.set_xlabel("Trial")
        ax1.set_ylabel("Average Survival")
        ax1.legend()
        ax1.grid(True)

        ax2.set_ylim(count_limits)
        ax2.set_title(f"Sample Count for GridID {grid_id}")
        ax2.set_xlabel("Trial")
        ax2.set_ylabel("Count")
        ax2.legend()
        ax2.grid(True)

        ax3.set_ylim(roc_limits)
        ax3.set_title(f"Rate of Change for GridID {grid_id}")
        ax3.set_xlabel("Trial")
        ax3.set_ylabel("Rate of Change")
        ax3.legend()
        ax3.grid(True)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/individual_plots_with_roc.png")
    plt.close(fig)

    # Heatmap plot for SI performance across environments with fixed vmin and vmax
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
    heatmap_data = []
    for grid_id in unique_grid_ids:
        si_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SI']
        max_length = max([max_trials[key] for key in si_key_list])
        si_combined = np.full((len(si_key_list), max_length), np.nan)

        for i, key in enumerate(si_key_list):
            data_list = results[key]
            for data in data_list:
                valid_length = len(data)
                si_combined[i, :valid_length] = data

        si_avg = np.nanmean(si_combined, axis=0)
        heatmap_data.append(si_avg)

        # Debugging check for heatmap data
        print(f"Heatmap data for Grid {grid_id}: {si_avg}")

    heatmap_data = np.array(heatmap_data)
    cax = ax_heatmap.imshow(heatmap_data, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=100)
    ax_heatmap.set_title("Heatmap of SI Performance Across Environments")
    ax_heatmap.set_xlabel("Trial")
    ax_heatmap.set_ylabel("Grid Environment")
    ax_heatmap.set_yticks(np.arange(len(unique_grid_ids)))
    ax_heatmap.set_yticklabels(unique_grid_ids)
    fig_heatmap.colorbar(cax)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/si_heatmap_plot.png")
    plt.close(fig_heatmap)

    # Heatmap plot for SL performance across environments
    fig_sl_heatmap, ax_sl_heatmap = plt.subplots(figsize=(12, 8))
    sl_heatmap_data = []  # Initialize as a list
    for grid_id in unique_grid_ids:
        sl_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SL']
        max_length = max([max_trials[key] for key in sl_key_list])
        sl_combined = np.full((len(sl_key_list), max_length), np.nan)
        
        for i, key in enumerate(sl_key_list):
            data_list = results[key]
            for data in data_list:
                valid_length = len(data)
                sl_combined[i, :valid_length] = data

        sl_avg = np.nanmean(sl_combined, axis=0)
        sl_heatmap_data.append(sl_avg)  # Append to the list

    sl_heatmap_data = np.array(sl_heatmap_data)  # Convert to a NumPy array after the loop
    cax = ax_sl_heatmap.imshow(sl_heatmap_data, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=100)
    ax_sl_heatmap.set_title("Heatmap of SL Performance Across Environments")
    ax_sl_heatmap.set_xlabel("Trial")
    ax_sl_heatmap.set_ylabel("Grid Environment")
    ax_sl_heatmap.set_yticks(np.arange(len(unique_grid_ids)))
    ax_sl_heatmap.set_yticklabels(unique_grid_ids)
    fig_sl_heatmap.colorbar(cax)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/sl_heatmap_plot.png")
    plt.close(fig_sl_heatmap)

    # Difference heatmap (SI - SL) across environments
    fig_diff_heatmap, ax_diff_heatmap = plt.subplots(figsize=(12, 8))
    diff_heatmap_data = []  # Initialize as a list
    for si_data, sl_data in zip(heatmap_data, sl_heatmap_data):
        diff_heatmap_data.append(si_data - sl_data)  # Append the difference to the list

    diff_heatmap_data = np.array(diff_heatmap_data)  # Convert to a NumPy array after the loop
    cax = ax_diff_heatmap.imshow(diff_heatmap_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-100, vmax=100)
    ax_diff_heatmap.set_title("Difference Heatmap (SI - SL) Performance Across Environments")
    ax_diff_heatmap.set_xlabel("Trial")
    ax_diff_heatmap.set_ylabel("Grid Environment")
    ax_diff_heatmap.set_yticks(np.arange(len(unique_grid_ids)))
    ax_diff_heatmap.set_yticklabels(unique_grid_ids)
    fig_diff_heatmap.colorbar(cax)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/difference_heatmap_plot.png")
    plt.close(fig_diff_heatmap)
    
# Cumulative survival plot for SI vs SL across all grid environments
    fig_cumulative, ax_cumulative = plt.subplots(figsize=(12, 8))
    for algo_type in ['SI', 'SL']:
        cumulative_avg = []
        cumulative_var = []
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            trial_lengths = [max_trials[key] for key in key_list]
            max_length = max(trial_lengths) if trial_lengths else 0
            combined_padded = np.full((len(key_list), max_length), np.nan)
            for i, key in enumerate(key_list):
                data_list = results[key]
                trial_length = max_trials[key]
                for data in data_list:
                    valid_length = len(data)
                    combined_padded[i, :valid_length] = data
            
            # Calculate cumulative sum for each grid
            grid_cumulative_avg = np.nancumsum(np.nanmean(combined_padded, axis=0))
            grid_cumulative_var = np.nancumsum(np.nanvar(combined_padded, axis=0))  # Sum variances
            cumulative_avg.append(grid_cumulative_avg)
            cumulative_var.append(grid_cumulative_var)
        
        # Average and standard deviation across all grids
        if cumulative_avg:
            overall_cumulative_avg = np.nanmean(cumulative_avg, axis=0)
            overall_cumulative_std = np.sqrt(np.nanmean(cumulative_var, axis=0))  # Convert variance back to standard deviation
            ax_cumulative.plot(overall_cumulative_avg, label=f"Cumulative {algo_type}", linestyle=linestyles[algo_type])
            ax_cumulative.fill_between(range(len(overall_cumulative_avg)), overall_cumulative_avg - overall_cumulative_std, 
                                       overall_cumulative_avg + overall_cumulative_std, alpha=0.2, color=grid_colors[grid_id])

    ax_cumulative.set_title("Cumulative Survival for SI and SL Across All Grid Environments")
    ax_cumulative.set_xlabel("Trial")
    ax_cumulative.set_ylabel("Cumulative Survival")
    ax_cumulative.legend()
    ax_cumulative.grid(True)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/cumulative_survival_plot.png")
    plt.close(fig_cumulative)

    # Algorithm dominance plot (SI - SL)
    fig_dominance, ax_dominance = plt.subplots(figsize=(12, 8))
    for grid_id in unique_grid_ids:
        si_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SI']
        sl_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SL']

        max_length = max([max_trials[key] for key in si_key_list + sl_key_list])
        si_combined = np.full((len(si_key_list), max_length), np.nan)
        sl_combined = np.full((len(sl_key_list), max_length), np.nan)
        
        for i, key in enumerate(si_key_list):
            data_list = results[key]
            for data in data_list:
                valid_length = len(data)
                si_combined[i, :valid_length] = data
                
        for i, key in enumerate(sl_key_list):
            data_list = results[key]
            for data in data_list:
                valid_length = len(data)
                sl_combined[i, :valid_length] = data
        
        si_avg = np.nanmean(si_combined, axis=0)
        sl_avg = np.nanmean(sl_combined, axis=0)
        dominance = si_avg - sl_avg

        ax_dominance.plot(dominance, label=f"SI - SL {grid_id}", color=grid_colors[grid_id])
    
    ax_dominance.set_title("Algorithm Dominance (SI - SL) Across All Grid Environments")
    ax_dominance.set_xlabel("Trial")
    ax_dominance.set_ylabel("Difference in Survival Rate (SI - SL)")
    ax_dominance.legend()
    ax_dominance.grid(True)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/algorithm_dominance_plot.png")
    plt.close(fig_dominance)
    
    # Relative improvement plot (SI vs SL)
    fig_rel_improvement, ax_rel_improvement = plt.subplots(figsize=(12, 8))
    for grid_id in unique_grid_ids:
        si_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SI']
        sl_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SL']

        max_length = max([max_trials[key] for key in si_key_list + sl_key_list])
        si_combined = np.full((len(si_key_list), max_length), np.nan)
        sl_combined = np.full((len(sl_key_list), max_length), np.nan)
        
        for i, key in enumerate(si_key_list):
            data_list = results[key]
            for data in data_list:
                valid_length = len(data)
                si_combined[i, :valid_length] = data
                
        for i, key in enumerate(sl_key_list):
            data_list = results[key]
            for data in data_list:
                valid_length = len(data)
                sl_combined[i, :valid_length] = data

        si_avg = np.nanmean(si_combined, axis=0)
        sl_avg = np.nanmean(sl_combined, axis=0)
        rel_improvement = (si_avg - sl_avg) / sl_avg * 100  # Calculate relative improvement

        ax_rel_improvement.plot(rel_improvement, label=f"Relative Improvement {grid_id}", color=grid_colors[grid_id])

    ax_rel_improvement.set_title("Relative Improvement (SI vs SL) Across All Grid Environments")
    ax_rel_improvement.set_xlabel("Trial")
    ax_rel_improvement.set_ylabel("Relative Improvement (%)")
    ax_rel_improvement.legend()
    ax_rel_improvement.grid(True)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/relative_improvement_plot.png")
    plt.close(fig_rel_improvement)

    # Boxplot of survival performance across grids for SI and SL
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 8))
    si_data_boxplot = []
    sl_data_boxplot = []
    labels = []

    for grid_id in unique_grid_ids:
        si_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SI']
        sl_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SL']

        for key in si_key_list:
            si_data_boxplot.append(np.concatenate(results[key]))
            labels.append(f"SI {grid_id}")
        
        for key in sl_key_list:
            sl_data_boxplot.append(np.concatenate(results[key]))
            labels.append(f"SL {grid_id}")

    ax_boxplot.boxplot(si_data_boxplot + sl_data_boxplot, labels=labels, vert=False)
    ax_boxplot.set_title("Distribution of Survival Performance Across All Grids (SI vs SL)")
    ax_boxplot.set_xlabel("Survival Performance")
    ax_boxplot.set_ylabel("Algorithm and Grid ID")
    plt.tight_layout()

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/boxplot_performance.png")
    plt.close(fig_boxplot)
    
    # Correlation heatmap of performance across grids for SI and SL
    fig_corr_heatmap, ax_corr_heatmap = plt.subplots(figsize=(12, 8))
    all_data = []

    for algo_type in ['SI', 'SL']:
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            for key in key_list:
                data_list = results[key]
                trial_length = max_trials[key]
                combined_data = np.full((len(data_list), trial_length), np.nan)
                
                for i, data in enumerate(data_list):
                    valid_length = len(data)
                    combined_data[i, :valid_length] = data

                avg_data = np.nanmean(combined_data, axis=0)
                all_data.append(avg_data)

    all_data = np.array(all_data)

    # Calculate the correlation matrix
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress divide-by-zero warnings
        corr_matrix = np.corrcoef(all_data)
        
        # Replace NaN values with 0 (correlation undefined when stddev is zero)
        corr_matrix = np.nan_to_num(corr_matrix)

    cax = ax_corr_heatmap.imshow(corr_matrix, cmap='coolwarm', origin='lower', aspect='auto')
    ax_corr_heatmap.set_title("Correlation Heatmap of Performance Across Grids (SI and SL)")
    fig_corr_heatmap.colorbar(cax)

    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/correlation_heatmap.png")
    plt.close(fig_corr_heatmap)
    
    print(unique_grid_ids)
    print(f"Number of unique grid IDs: {len(unique_grid_ids)}")
    # Check which grid pairs have the most negative correlations
    grid_ids = unique_grid_ids
    for i in range(len(grid_ids)):
        for j in range(i + 1, len(grid_ids)):
            if corr_matrix[i, j] < -0.5:  # Threshold for significant negative correlation
                print(f"Strong negative correlation between Grid {grid_ids[i]} and Grid {grid_ids[j]}: {corr_matrix[i, j]}")
           
    # Check rates of change 
    fig, ax = plt.subplots(figsize=(12, 8))
    for grid_id in unique_grid_ids:
        si_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SI']
        sl_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == 'SL']

        for key in si_key_list:
            si_data = np.nanmean(np.vstack(results[key]), axis=0)
            si_rate_of_change = np.diff(si_data)  # Calculate the rate of change
            ax.plot(si_rate_of_change, label=f"SI ROC {grid_id}", linestyle='-', color=grid_colors[grid_id])

        for key in sl_key_list:
            sl_data = np.nanmean(np.vstack(results[key]), axis=0)
            sl_rate_of_change = np.diff(sl_data)  # Calculate the rate of change
            ax.plot(sl_rate_of_change, label=f"SL ROC {grid_id}", linestyle='--', color=grid_colors[grid_id])

    ax.set_title("Rate of Change in Performance for SI vs SL Across Grids")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Rate of Change (Survival)")
    ax.legend()
    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/rate_of_change.png")
    
if __name__ == "__main__":
    main("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments")
    
    