import numpy as np
import matplotlib.pyplot as plt
import os
import re
import time
import wandb
from datetime import datetime, timedelta

# Constants
OUTPUT_FOLDER = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB/scripts/UCT-HPC'
POLLING_INTERVAL = 60  # Time in seconds between checks for new files
FILE_PATTERN = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{3})_(seed_\d+)_(\w+)_experiment.txt")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines()]
    return data

def get_latest_files(directory):
    files = {}
    for file_name in os.listdir(directory):
        match = FILE_PATTERN.match(file_name)
        if match:
            timestamp, seed, algorithm = match.groups()
            key = (seed, algorithm)
            if key not in files or timestamp > files[key][0]:
                files[key] = (timestamp, os.path.join(directory, file_name))
    return [info[1] for info in files.values()]

def plot_polynomial_regression(ax, x_data, y_data, degree, label):
    coeffs = np.polyfit(x_data, y_data, degree)
    p = np.poly1d(coeffs)
    x_line = np.linspace(min(x_data), max(x_data), 300)
    y_line = p(x_line)
    ax.plot(x_line, y_line, label=label, color='black')
    ax.scatter(x_data, y_data, color='red', s=10)

def main():
    wandb.init(project="HPC_Monitoring_MATLAB", entity="shocklab")
    # Send an alert when the wandb tracking starts
    wandb.alert(
        title='Experiment Monitoring Started',
        text=f'Experiment monitoring has started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Tracking new runs and data under the project "HPC_Monitoring_MATLAB" in the "shocklab" entity. This W&B script automatically updates with new results from HPC jobs and replicates Figure 5A in SL paper.',
        level=wandb.AlertLevel.INFO,
        wait_duration=timedelta(minutes=10)  # Prevent alert from being sent more than once every 10 minutes
    )
    
    processed_files = set()
    try:
        while True:
            current_files = set(get_latest_files(OUTPUT_FOLDER))
            new_files = current_files - processed_files
            if new_files:
                data_dict = {}
                for file_path in new_files:
                    details = FILE_PATTERN.search(os.path.basename(file_path)).groups()
                    algorithm, seed = details[2], details[1]
                    data = load_data(file_path)
                    if algorithm not in data_dict:
                        data_dict[algorithm] = {}
                    if seed not in data_dict[algorithm]:
                        data_dict[algorithm][seed] = []
                    data_dict[algorithm][seed].append(data)

                for algorithm, seeds_data in data_dict.items():
                    fig, ax = plt.subplots()
                    all_trials_data = []
                    max_length = max(len(np.concatenate(data_list)) for data_list in seeds_data.values() if data_list)
                    
                    for trial in range(max_length):
                        trial_data = [np.concatenate(data_list)[trial] for data_list in seeds_data.values()
                                      if len(np.concatenate(data_list)) > trial]
                        if trial_data:
                            all_trials_data.append(np.mean(trial_data))
                    
                    if all_trials_data:
                        iterations = np.arange(len(all_trials_data))
                        plot_polynomial_regression(ax, iterations, all_trials_data, 2, f'Polynomial Fit - {algorithm}')

                    ax.set_title(f'Performance Over Time: {algorithm}')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Average Time Steps Survived')
                    ax.legend()
                    wandb.log({f"{algorithm} Performance Overview": wandb.Image(fig)})
                    plt.close(fig)
                
                processed_files.update(new_files)
            time.sleep(POLLING_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping monitoring...")

if __name__ == "__main__":
    main()
