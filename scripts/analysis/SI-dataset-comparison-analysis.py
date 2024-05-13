import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import logging
from scipy.stats import ttest_ind, sem
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

BASE_PATH = 'C:\\Users\\micro\\Documents\\ActiveInference_Work\\Sophisticated-Learning\\'

# Constants
EXCEL_FILE_PATH = BASE_PATH + 'results\\unknown_model\\SI_final_results_RowanOriginal.xlsx'
SURVIVAL_FOLDER = BASE_PATH + 'results\\unknown_model\\MATLAB\\survival'
PYTHON_SURVIVAL_FOLDER = BASE_PATH + 'results\\unknown_model\\python-SI'

FILE_PATTERN = re.compile(r"SI_Seed(\d+)_\d{2}-\d{2}-\d{2}-\d{3}\.txt")
PYTHON_FILE_PATTERN = re.compile(r"slurm-\d+\.out") 

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_excel_data():
    """Loads and preprocesses Excel data."""
    df = pd.read_excel(EXCEL_FILE_PATH)
    df_mean = df.mean(axis=1).reset_index()
    df_mean.columns = ['Trial', 'SurvivalTime']
    df_mean['Source'] = 'Excel'
    return df_mean

def load_log_data():
    """Loads and preprocesses log data for the SI algorithm, averaging over seeds."""
    data_dict = {}
    for file_name in os.listdir(SURVIVAL_FOLDER):
        match = FILE_PATTERN.match(file_name)
        if match:
            seed = match.group(1)
            file_path = os.path.join(SURVIVAL_FOLDER, file_name)
            with open(file_path, 'r') as file:
                data = [float(line.strip()) for line in file.readlines() if line.strip()]
            if seed not in data_dict:
                data_dict[seed] = data

    # Average the data for each trial across seeds
    trial_data = {}
    for seed, values in data_dict.items():
        for index, value in enumerate(values):
            if index not in trial_data:
                trial_data[index] = []
            trial_data[index].append(value)

    # Create a DataFrame from the averaged data
    averaged_data = {trial: np.mean(values) for trial, values in trial_data.items()}
    trials_df = pd.DataFrame(list(averaged_data.items()), columns=['Trial', 'SurvivalTime'])
    trials_df['Source'] = 'Log'
    return trials_df

def load_python_data():
    """Loads and preprocesses Python data, averaging over runs within each experiment directory."""
    survival_times_by_experiment = {}

    for exp_dir in os.listdir(PYTHON_SURVIVAL_FOLDER):
        exp_path = os.path.join(PYTHON_SURVIVAL_FOLDER, exp_dir)
        if os.path.isdir(exp_path):
            survival_times_by_seed = []  # Storage for one experiment

            for file_name in os.listdir(exp_path):
                if PYTHON_FILE_PATTERN.match(file_name):
                    file_path = os.path.join(exp_path, file_name)
                    survival_times = extract_survival_times(file_path)
                    if survival_times:
                        survival_times_by_seed.append(survival_times)
                    else:
                        logging.warning(f"No survival times extracted from {file_name}")

            if survival_times_by_seed:
                survival_times_by_experiment[exp_dir] = survival_times_by_seed 
            else:
                logging.warning(f"No valid survival data found in experiment directory {exp_dir}")

    # Process the collected data
    if not survival_times_by_experiment:
        logging.error("No survival data extracted from any files")
        return pd.DataFrame(columns=['Trial', 'SurvivalTime', 'Source'])

    trials_df = process_experiment_data(survival_times_by_experiment)
    trials_df['Source'] = 'Python'
    return trials_df

def process_experiment_data(data):
    """Helper function to calculate average survival times across seeds/runs per experiment."""
    # Determine max trials across all experiments
    max_trials = 0
    for exp_data in data.values():
        max_trials = max(max_trials, max(len(times) for times in exp_data))

    # Calculate averages for each trial
    averaged_data = {}
    for i in range(max_trials):
        trial_data = [times[i] for exp_data in data.values() for times in exp_data if len(times) > i]
        averaged_data[i] = np.mean(trial_data)

    return pd.DataFrame(list(averaged_data.items()), columns=['Trial', 'SurvivalTime'])

def plot_regression(ax, x_data, y_data, algorithm, data_color, line_color):
    """Plots different types of regression models on given axes with specific colors."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # Convert x_data and y_data to NumPy arrays for manipulation
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(x_data[:, np.newaxis], y_data)
    y_lin_pred = lin_model.predict(x_data[:, np.newaxis])
    coef_lin = lin_model.coef_[0]
    intercept_lin = lin_model.intercept_
    lin_label = f'{algorithm} Linear: $y = {intercept_lin:.2f} + {coef_lin:.2f}x$'
    ax.plot(x_data, y_lin_pred, label=lin_label, color=line_color)

    # Polynomial Regression (Degree 2)
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(x_data[:, np.newaxis], y_data)
    y_poly_pred = poly_model.predict(x_data[:, np.newaxis])
    coef_poly = poly_model.named_steps['linearregression'].coef_
    intercept_poly = poly_model.named_steps['linearregression'].intercept_
    
    poly_label = f'{algorithm} Polynomial: $y = {intercept_poly:.2f} + {coef_poly[1]:.2f}x + {coef_poly[2]:.3f}x^2$'
    ax.plot(x_data, y_poly_pred, label=poly_label, color=line_color, linestyle='dashed')

    ax.scatter(x_data, y_data, color=data_color, s=10, label=f'{algorithm} Data')  # Plot data points
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average Survival Time')

    ax.legend()

def plot_data_comparison(excel_df, log_df, python_df):
    """Plots regression and comparison of all three data sources on the same axes."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plotting Excel data with specific colors
    plot_regression(ax, excel_df['Trial'], excel_df['SurvivalTime'], 'Excel Data', 'blue', 'green')

    # Plotting Log data with different colors
    plot_regression(ax, log_df['Trial'], log_df['SurvivalTime'], 'Log Data', 'red', 'purple')

    # Plotting Python Data
    plot_regression(ax, python_df['Trial'], python_df['SurvivalTime'], 'Python Data', 'orange', 'brown') 

    plt.title('Comparison of Excel, Log, and Python Data for the SI Algorithm')
    plt.legend()
    plt.show()


def perform_statistical_comparison(excel_df, log_df, python_df):
    """Performs statistical tests to compare all three datasets and logs additional statistics."""

    # Excel vs. Log 
    t_stat_el, p_value_el = ttest_ind(excel_df['SurvivalTime'], log_df['SurvivalTime'])
    ci_excel = stats.t.interval(0.95, len(excel_df['SurvivalTime'])-1, loc=np.mean(excel_df['SurvivalTime']), scale=sem(excel_df['SurvivalTime']))
    ci_log = stats.t.interval(0.95, len(log_df['SurvivalTime'])-1, loc=np.mean(log_df['SurvivalTime']), scale=sem(log_df['SurvivalTime']))
    logging.info(f"Excel vs. Log:")
    logging.info(f"T-test result: T-statistic = {t_stat_el}, P-value = {p_value_el}")
    logging.info(f"Excel Data 95% CI: {ci_excel}")
    logging.info(f"Log Data 95% CI: {ci_log}")

    # Excel vs. Python
    t_stat_ep, p_value_ep = ttest_ind(excel_df['SurvivalTime'], python_df['SurvivalTime'])
    ci_python = stats.t.interval(0.95, len(python_df['SurvivalTime'])-1, loc=np.mean(python_df['SurvivalTime']), scale=sem(python_df['SurvivalTime']))
    logging.info(f"\nExcel vs. Python:")
    logging.info(f"T-test result: T-statistic = {t_stat_ep}, P-value = {p_value_ep}")
    logging.info(f"Python Data 95% CI: {ci_python}")

    # Log vs. Python
    t_stat_lp, p_value_lp = ttest_ind(log_df['SurvivalTime'], python_df['SurvivalTime'])
    logging.info(f"\nLog vs. Python:")
    logging.info(f"T-test result: T-statistic = {t_stat_lp}, P-value = {p_value_lp}")


def extract_survival_times(file_path):
    """Extracts survival times from the Python-generated .out files."""
    with open(file_path, 'r') as file:
        survival_times = []
        for line in file:
            match = re.search(r"At time (\d+) the agent is dead", line)
            if match:
                survival_time = int(match.group(1))
                survival_times.append(survival_time)
    return survival_times

def main():
    excel_df = load_excel_data()
    log_df = load_log_data()
    python_df = load_python_data()  # Load the new Python data

    plot_data_comparison(excel_df, log_df, python_df)  # Update function to handle three datasets
    perform_statistical_comparison(excel_df, log_df, python_df)  # Update to compare all three datasets

if __name__ == "__main__":
    main()
