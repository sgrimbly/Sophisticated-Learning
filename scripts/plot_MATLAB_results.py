# TODO: Update logic of script to match the Python version

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import os
import re
from datetime import datetime

# Specify the output folder here
OUTPUT_FOLDER = 'path_to_data_folder'

# Function to load data from a given file path
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

# Function to get all files in the specified directory that match the pattern
def get_files(directory):
    file_pattern = re.compile(r"(.+)_Seed(\d+)_Hor(\d+)_KF(\d+)_MCT(\d+)_Num(\d+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
    for file_name in os.listdir(directory):
        if file_pattern.match(file_name):
            yield os.path.join(directory, file_name), file_pattern.match(file_name).groups()

# Load data from files and sort them by algorithm and seed
data_dict = {}
for file_path, details in get_files(OUTPUT_FOLDER):
    algorithm, seed, horizon, k_factor, mct, num_mct, date_str = details
    seed = int(seed)
    if algorithm not in data_dict:
        data_dict[algorithm] = {}
    if seed not in data_dict[algorithm]:
        data_dict[algorithm][seed] = []
    data_dict[algorithm][seed].append(load_data(file_path))

# Process data: average across seeds for each algorithm
results = {}
for algorithm, seeds_data in data_dict.items():
    all_seeds = list(seeds_data.values())
    concatenated_data = np.concatenate(all_seeds)
    average_performance = np.mean(concatenated_data, axis=0)
    results[algorithm] = average_performance

# Plot the average performance over iterations for each algorithm
plt.figure(figsize=(12, 6))
for algorithm, performance in results.items():
    iterations = np.arange(len(performance))
    spl = make_interp_spline(iterations, performance, k=3)  # Smooth spline interpolation
    smooth_iterations = np.linspace(iterations.min(), iterations.max(), 300)
    smooth_performance = spl(smooth_iterations)

    plt.plot(smooth_iterations, smooth_performance, label=f'{algorithm} (Smoothed)')
    plt.scatter(iterations, performance, s=10, label=f'{algorithm} (Data)')

plt.title('Average Performance Over Iterations by Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Average Time Steps Survived')
plt.legend()
plt.show()
