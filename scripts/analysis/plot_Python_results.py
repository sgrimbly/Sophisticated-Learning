import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Constants
OUTPUT_FOLDER = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/Python/scripts/UCT-HPC'
FILE_PATTERN = re.compile(r"SI_Seed_\d+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}.txt")
SAVE_PATH = '/home/grmstj001/Python-experiments/experiments/plots/average_survival_plot.png'

def extract_survival_times(file_path):
    with open(file_path, 'r') as file:
        survival_times = []
        for line in file:
            if 'Total time steps survived:' in line:
                survival_time = int(line.split(': ')[1])
                survival_times.append(survival_time)
    return survival_times

def calculate_differences(survival_times):
    return np.diff(survival_times)

def plot_polynomial_regression(ax, x_data, y_data, degree, label):
    coeffs = np.polyfit(x_data, y_data, degree)
    p = np.poly1d(coeffs)
    x_line = np.linspace(min(x_data), max(x_data), 300)
    y_line = p(x_line)
    ax.plot(x_line, y_line, label=label, color='black')
    ax.scatter(x_data, y_data, color='red', s=10)

def main():
    survival_time_differences_by_trial = []
    
    # Collecting data
    for file_name in os.listdir(OUTPUT_FOLDER):
        if FILE_PATTERN.match(file_name):
            file_path = os.path.join(OUTPUT_FOLDER, file_name)
            survival_times = extract_survival_times(file_path)
            survival_time_differences = calculate_differences(survival_times)
            survival_time_differences_by_trial.append(survival_time_differences)
            print(f"File: {file_name} -> Trials: {len(survival_time_differences)} -> Survival Time Differences: {survival_time_differences}")
    
    # Averaging data
    max_trials = max(len(times) for times in survival_time_differences_by_trial)
    avg_survival_times = []
    for i in range(max_trials):
        trial_times = [times[i] for times in survival_time_differences_by_trial if len(times) > i]
        avg_time = np.mean(trial_times)
        avg_survival_times.append(avg_time)
        print(f"Trial {i+1}: Computed average time steps survived = {avg_time} from {len(trial_times)} files")
    
    # Plotting
    fig, ax = plt.subplots()
    plot_polynomial_regression(ax, list(range(1, max_trials+1)), avg_survival_times, 2, 'Polynomial Fit - Avg Time Survived')
    ax.set_title('Average Time Steps Survived Per Trial')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Average Time Steps Survived')
    ax.legend()
    plt.show()
    plt.savefig(SAVE_PATH)  # Save the figure

if __name__ == "__main__":
    main()
