import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.ndimage import uniform_filter1d

def parse_filename(filename):
    pattern = r"^(SI|SL|BA|BAUCB)_Seed_(\d+)_GridID_([a-f0-9]+)_\d{2}-\d{2}-\d{2}-\d{3}.txt"
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

def plot_grouped_boxplots(results, algo_types, unique_grid_ids, grid_colors, output_dir):
    num_grids = len(unique_grid_ids)
    num_algorithms = len(algo_types)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_grids + cols - 1) // cols  # Calculate rows needed
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18, 5 * rows))
    axes = axes.flatten()  # Flatten axes array for easy indexing
    
    for idx, grid_id in enumerate(unique_grid_ids):
        ax = axes[idx]
        data = []
        labels = []
        for algo_type in sorted(algo_types):
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            algo_data = []
            for key in key_list:
                algo_data.extend(results[key])  # Collect all data arrays for this algorithm-grid combination
            if algo_data:
                # Flatten all arrays into a single list
                flattened_data = np.concatenate(algo_data)
                data.append(flattened_data)
                labels.append(algo_type)
        
        if data:
            # Create the boxplot for this grid
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            # Set colors for each box
            for patch, color in zip(bp['boxes'], [grid_colors[grid_id]] * len(labels)):
                patch.set_facecolor(color)
            ax.set_title(f"Grid {grid_id}")
            ax.set_ylabel("Survival Performance")
            ax.set_xlabel("Algorithm")
            ax.grid(True)
        else:
            ax.set_visible(False)  # Hide subplot if there's no data

    # Hide any unused subplots
    for i in range(idx + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grouped_boxplots_per_grid.png")
    plt.close(fig)

def main(directory_path, include_sample_count=False):
    output_dir = "/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/"

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
    ncols = 3 if include_sample_count else 2
    fig, axes = plt.subplots(nrows=len(unique_grid_ids), ncols=ncols, figsize=(22, 5 * len(unique_grid_ids)), constrained_layout=True)

    algo_types = set(algo for algo, _ in results.keys())

    linestyles = {algo: style for algo, style in zip(algo_types, ['-', '--', '-.', ':'])}
    survival_limits = [float('inf'), float('-inf')]
    count_limits = [float('inf'), float('-inf')]
    roc_limits = [float('inf'), float('-inf')]  # Limits for rate of change
    
    # First pass: Calculate data and find limits
    for grid_id in unique_grid_ids:
        for algo_type in algo_types:
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
        ax3 = axes[unique_grid_ids.index(grid_id), 1 if not include_sample_count else 2]  # Column 2 or 3: Rate of Change

        for algo_type in algo_types:
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
                roc = np.diff(data_avg, prepend=np.nan)

                # Plot average survival with uncertainty bounds in ax1
                ax1.plot(data_avg, label=f"{key[0]} {key[1]}", color=grid_colors[grid_id], linestyle=linestyles[algo_type])
                ax1.fill_between(range(trial_length), data_avg - data_std, data_avg + data_std, 
                                color=grid_colors[grid_id], alpha=0.2)

                # Conditionally plot sample count in ax2
                if include_sample_count:
                    ax2 = axes[unique_grid_ids.index(grid_id), 1]  # Column 2: Sample Count
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

        if include_sample_count:
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

    # Heatmap plot for performance across environments for all algorithms
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 8))
    heatmap_data = {}
    for algo_type in algo_types:
        algo_heatmap_data = []
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            data_arrays = []
            for key in key_list:
                data_arrays.extend(results[key])  # Collect all data arrays for this algorithm-grid combination

            if data_arrays:
                max_length = max(len(data) for data in data_arrays)
                padded_data = np.full((len(data_arrays), max_length), np.nan)
                for i, data in enumerate(data_arrays):
                    padded_data[i, :len(data)] = data

                avg_data = np.nanmean(padded_data, axis=0)
                algo_heatmap_data.append(avg_data)
            else:
                # Append NaNs if no data is available for this grid and algorithm
                algo_heatmap_data.append(np.full(valid_trial_length, np.nan))

        # Stack the data for all grids into a 2D array for the heatmap
        heatmap_array = np.vstack(algo_heatmap_data)
        heatmap_data[algo_type] = heatmap_array

        # Plot individual heatmap for each algorithm
        fig, ax = plt.subplots(figsize=(12, 8))
        cax = ax.imshow(heatmap_array, aspect='auto', cmap='viridis', origin='lower', vmin=0, vmax=100)
        ax.set_title(f"Heatmap of {algo_type} Performance Across Environments")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Grid Environment")
        ax.set_yticks(np.arange(len(unique_grid_ids)))
        ax.set_yticklabels(unique_grid_ids)
        fig.colorbar(cax)
        plt.savefig(f"{output_dir}/{algo_type.lower()}_heatmap_plot.png")
        plt.close(fig)

    # Difference heatmap (pairwise for all algorithms)
    fig_diff_heatmap, ax_diff_heatmap = plt.subplots(figsize=(12, 8))
    for algo1, algo2 in [('SI', 'SL'), ('BA', 'BAUCB')]:  # Adjust pairs as needed
        diff_data = heatmap_data[algo1] - heatmap_data[algo2]
        fig, ax = plt.subplots(figsize=(12, 8))
        cax = ax.imshow(diff_data, aspect='auto', cmap='coolwarm', origin='lower', vmin=-100, vmax=100)
        ax.set_title(f"Difference Heatmap ({algo1} - {algo2}) Performance Across Environments")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Grid Environment")
        ax.set_yticks(np.arange(len(unique_grid_ids)))
        ax.set_yticklabels(unique_grid_ids)
        fig.colorbar(cax)
        plt.savefig(f"/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/difference_heatmap_{algo1.lower()}_{algo2.lower()}.png")
        plt.close(fig)
    
    # Cumulative survival plot across all algorithms
    fig_cumulative, ax_cumulative = plt.subplots(figsize=(12, 8))
    for algo_type in algo_types:
        cumulative_avgs = []
        cumulative_vars = []
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            data_arrays = []
            for key in key_list:
                data_arrays.extend(results[key])

            if data_arrays:
                max_length = max(len(data) for data in data_arrays)
                padded_data = np.full((len(data_arrays), max_length), np.nan)
                for i, data in enumerate(data_arrays):
                    padded_data[i, :len(data)] = data

                avg_over_seeds = np.nanmean(padded_data, axis=0)
                var_over_seeds = np.nanvar(padded_data, axis=0)

                grid_cumulative_avg = np.nancumsum(avg_over_seeds)
                grid_cumulative_var = np.nancumsum(var_over_seeds)

                cumulative_avgs.append(grid_cumulative_avg)
                cumulative_vars.append(grid_cumulative_var)
            else:
                cumulative_avgs.append(np.full(valid_trial_length, np.nan))
                cumulative_vars.append(np.full(valid_trial_length, np.nan))

        if cumulative_avgs:
            cumulative_avgs_array = np.vstack(cumulative_avgs)
            cumulative_vars_array = np.vstack(cumulative_vars)

            overall_cumulative_avg = np.nanmean(cumulative_avgs_array, axis=0)
            overall_cumulative_std = np.sqrt(np.nanmean(cumulative_vars_array, axis=0))

            ax_cumulative.plot(overall_cumulative_avg, label=f"Cumulative {algo_type}", linestyle=linestyles.get(algo_type, '-'))
            ax_cumulative.fill_between(range(len(overall_cumulative_avg)),
                                        overall_cumulative_avg - overall_cumulative_std,
                                        overall_cumulative_avg + overall_cumulative_std, alpha=0.2)
    ax_cumulative.set_title("Cumulative Survival Across All Algorithms and Grids")
    ax_cumulative.set_xlabel("Trial")
    ax_cumulative.set_ylabel("Cumulative Survival")
    ax_cumulative.legend()
    ax_cumulative.grid(True)
    plt.savefig(f"{output_dir}/cumulative_survival_all_algorithms.png")
    plt.close(fig_cumulative)
    
    # Algorithm dominance plot for all algorithm pairs
    fig_dominance, ax_dominance = plt.subplots(figsize=(12, 8))
    for grid_id in unique_grid_ids:
        for algo1, algo2 in [('SI', 'SL'), ('BA', 'BAUCB')]:  # Adjust pairs as needed
            algo1_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo1]
            algo2_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo2]

            max_length = max([max_trials[key] for key in algo1_key_list + algo2_key_list]) if algo1_key_list + algo2_key_list else 0
            algo1_combined = np.full((len(algo1_key_list), max_length), np.nan)
            algo2_combined = np.full((len(algo2_key_list), max_length), np.nan)

            for i, key in enumerate(algo1_key_list):
                data_list = results[key]
                for data in data_list:
                    valid_length = len(data)
                    algo1_combined[i, :valid_length] = data

            for i, key in enumerate(algo2_key_list):
                data_list = results[key]
                for data in data_list:
                    valid_length = len(data)
                    algo2_combined[i, :valid_length] = data

            algo1_avg = np.nanmean(algo1_combined, axis=0)
            algo2_avg = np.nanmean(algo2_combined, axis=0)
            dominance = algo1_avg - algo2_avg

            ax_dominance.plot(dominance, label=f"{algo1} - {algo2} {grid_id}", color=grid_colors[grid_id])

    ax_dominance.set_title("Algorithm Dominance Across All Grid Environments")
    ax_dominance.set_xlabel("Trial")
    ax_dominance.set_ylabel("Difference in Survival Rate")
    ax_dominance.legend()
    ax_dominance.grid(True)
    plt.savefig("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/algorithm_dominance_plot.png")
    plt.close(fig_dominance)
    
    # Relative improvement plot for all algorithm pairs
    fig_rel_improvement, ax_rel_improvement = plt.subplots(figsize=(12, 8))
    for grid_id in unique_grid_ids:
        for algo1, algo2 in [('SI', 'SL'), ('BA', 'BAUCB')]:  # Adjust pairs as needed
            algo1_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo1]
            algo2_key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo2]

            algo1_data_arrays = []
            for key in algo1_key_list:
                algo1_data_arrays.extend(results[key])

            algo2_data_arrays = []
            for key in algo2_key_list:
                algo2_data_arrays.extend(results[key])

            if algo1_data_arrays and algo2_data_arrays:
                max_length = max(
                    max(len(data) for data in algo1_data_arrays),
                    max(len(data) for data in algo2_data_arrays)
                )
                algo1_combined = np.full((len(algo1_data_arrays), max_length), np.nan)
                for i, data in enumerate(algo1_data_arrays):
                    algo1_combined[i, :len(data)] = data

                algo2_combined = np.full((len(algo2_data_arrays), max_length), np.nan)
                for i, data in enumerate(algo2_data_arrays):
                    algo2_combined[i, :len(data)] = data

                algo1_avg = np.nanmean(algo1_combined, axis=0)
                algo2_avg = np.nanmean(algo2_combined, axis=0)

                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_improvement = np.where(
                        algo2_avg != 0,
                        ((algo1_avg - algo2_avg) / algo2_avg) * 100,
                        np.nan
                    )

                ax_rel_improvement.plot(rel_improvement, label=f"{algo1} vs {algo2} {grid_id}",
                                        color=grid_colors[grid_id])
            else:
                print(f"No data for {algo1} or {algo2} on Grid {grid_id}. Skipping...")

    ax_rel_improvement.set_title("Relative Improvement Across All Grid Environments")
    ax_rel_improvement.set_xlabel("Trial")
    ax_rel_improvement.set_ylabel("Relative Improvement (%)")
    ax_rel_improvement.legend()
    ax_rel_improvement.grid(True)
    plt.savefig(f"{output_dir}/relative_improvement_plot.png")
    plt.close(fig_rel_improvement)
    
    # Boxplot of survival performance across grids for all algorithms
    plot_grouped_boxplots(results, algo_types, unique_grid_ids, grid_colors, output_dir)
    
    # Correlation heatmap of performance across all algorithms and grids
    fig_corr_heatmap, ax_corr_heatmap = plt.subplots(figsize=(12, 8))
    all_data = []
    labels = []

    for algo_type in algo_types:
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            data_arrays = []
            for key in key_list:
                data_arrays.extend(results[key])

            if data_arrays:
                max_length = max(len(data) for data in data_arrays)
                padded_data = np.full((len(data_arrays), max_length), np.nan)
                for i, data in enumerate(data_arrays):
                    padded_data[i, :len(data)] = data

                avg_data = np.nanmean(padded_data, axis=0)
                all_data.append(avg_data)
                labels.append(f"{algo_type} {grid_id}")
            else:
                print(f"No data for {algo_type} on Grid {grid_id}. Skipping...")

    all_data = np.array(all_data)

    # Calculate the correlation matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        corr_matrix = np.corrcoef(all_data)
        # Replace NaN values with 0 (correlation undefined when stddev is zero)
        corr_matrix = np.nan_to_num(corr_matrix)

    cax = ax_corr_heatmap.imshow(corr_matrix, cmap='coolwarm', origin='lower', aspect='auto')
    ax_corr_heatmap.set_title("Correlation Heatmap of Performance Across All Algorithms and Grids")
    ax_corr_heatmap.set_xticks(np.arange(len(labels)))
    ax_corr_heatmap.set_yticks(np.arange(len(labels)))
    ax_corr_heatmap.set_xticklabels(labels, rotation=90)
    ax_corr_heatmap.set_yticklabels(labels)
    fig_corr_heatmap.colorbar(cax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap_all_algorithms.png")
    plt.close(fig_corr_heatmap)
    
    print(unique_grid_ids)
    print(f"Number of unique grid IDs: {len(unique_grid_ids)}")

    # Check which grid pairs have the most negative correlations
    grid_ids = unique_grid_ids
    for i in range(len(grid_ids)):
        for j in range(i + 1, len(grid_ids)):
            if corr_matrix[i, j] < -0.5:  # Threshold for significant negative correlation
                print(f"Strong negative correlation between Grid {grid_ids[i]} and Grid {grid_ids[j]}: {corr_matrix[i, j]}")

    # Check rates of change for all algorithms
    num_algorithms = len(algo_types)
    fig, axes = plt.subplots(nrows=num_algorithms, figsize=(12, 4 * num_algorithms), sharex=True)

    for idx, algo_type in enumerate(sorted(algo_types)):
        ax = axes[idx]
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            data_arrays = []
            for key in key_list:
                data_arrays.extend(results[key])

            if data_arrays:
                max_length = max(len(data) for data in data_arrays)
                padded_data = np.full((len(data_arrays), max_length), np.nan)
                for i, data in enumerate(data_arrays):
                    padded_data[i, :len(data)] = data

                if not np.isnan(padded_data).all():
                    data_avg = np.nanmean(padded_data, axis=0)
                    rate_of_change = np.diff(data_avg, prepend=np.nan)
                    smoothed_rate_of_change = uniform_filter1d(rate_of_change, size=5)
                    ax.plot(smoothed_rate_of_change, label=f"{grid_id}", color=grid_colors[grid_id])
                else:
                    print(f"No valid data for {algo_type} on Grid {grid_id}. Skipping...")
            else:
                print(f"No data available for {algo_type} on Grid {grid_id}. Skipping...")

        ax.set_title(f"Rate of Change for {algo_type}")
        ax.set_xlabel("Trials")
        ax.set_ylabel("Rate of Change")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/rate_of_change_faceted_by_algorithm.png")
    plt.close(fig)

    # Prepare data for heatmap
    # Ensure heatmap_data and labels are declared in the main scope
    heatmap_data = []
    labels = []

    # Prepare data for heatmap
    for algo_type in sorted(algo_types):
        for grid_id in unique_grid_ids:
            key_list = [(algo, grid) for (algo, grid) in results if grid == grid_id and algo == algo_type]
            data_arrays = []
            for key in key_list:
                data_arrays.extend(results[key])

            if data_arrays:
                max_length = max(len(data) for data in data_arrays)
                padded_data = np.full((len(data_arrays), max_length), np.nan)
                for i, data in enumerate(data_arrays):
                    padded_data[i, :len(data)] = data

                if not np.isnan(padded_data).all():
                    avg_data = np.nanmean(padded_data, axis=0)
                    rate_of_change = np.diff(avg_data, prepend=np.nan)
                    smoothed_rate_of_change = uniform_filter1d(rate_of_change, size=5)
                    heatmap_data.append(smoothed_rate_of_change)
                    labels.append(f"{algo_type} {grid_id}")
                else:
                    print(f"No valid data for heatmap: {algo_type} on Grid {grid_id}. Skipping...")
            else:
                print(f"No data available for heatmap: {algo_type} on Grid {grid_id}. Skipping...")

    # Only create the heatmap if there is valid data
    if heatmap_data:  # Check if the list is not empty
        heatmap_array = np.array(heatmap_data)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        cax = ax.imshow(heatmap_array, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title("Rate of Change Heatmap")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Algorithm and Grid")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        fig.colorbar(cax)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rate_of_change_heatmap.png")
        plt.close(fig)
    else:
        print("No valid data available for the rate of change heatmap. Skipping plot generation.")
        
if __name__ == "__main__":
    main("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments")
    