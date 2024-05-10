import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import re
import time
import wandb
from scipy.stats import f_oneway
import statsmodels.formula.api as smf
import logging
from datetime import datetime, timedelta
import sys
from PIL import Image

# Configuration
ENABLE_WANDB = False  # Set this to True to enable wandb logging

# Constants
OUTPUT_FOLDER = 'C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning'
POLLING_INTERVAL = 600  # Time in seconds between checks for new files
FILE_PATTERN = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{3})_seed_(\d+)_([A-Z]+)_experiment.txt")

# Set up logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("debug.log"),  # Log to a file named debug.log
                        logging.StreamHandler()            # Log to the standard system output
                    ])

# Weights & Biases Initialization
if ENABLE_WANDB:
    wandb.init(project="PC_ParallelExperiment_Monitoring", entity="shocklab")
    wandb.alert(
        title='PC Parallel Experiment Monitoring Started',
        text=f'Monitoring started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        level=wandb.AlertLevel.INFO,
        wait_duration=timedelta(minutes=30)
    )

def load_data(file_path):
    """Loads numerical data from a given file."""
    with open(file_path, 'r') as file:
        data = [float(line.strip()) for line in file.readlines() if line.strip()]
    return data

def get_latest_files(directory):
    """Finds the latest files for each algorithm-seed combination."""
    files = {}
    for file_name in os.listdir(directory):
        match = FILE_PATTERN.match(file_name)
        if match:
            timestamp, seed, algorithm = match.groups()
            key = (seed, algorithm)
            if key not in files or timestamp > files[key][0]:
                files[key] = (timestamp, os.path.join(directory, file_name))
    return [info[1] for info in files.values()]

def plot_regression(ax, x_data, y_data, algorithm):
    """Plots different types of regression models on given axes."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(x_data[:, np.newaxis], y_data)
    y_lin_pred = lin_model.predict(x_data[:, np.newaxis])
    coef_lin = lin_model.coef_[0]
    intercept_lin = lin_model.intercept_
    lin_label = f'Linear: $y = {intercept_lin:.2f} + {coef_lin:.2f}x$'
    ax.plot(x_data, y_lin_pred, label=lin_label, color='blue')

    # Polynomial Regression (Degree 2)
    poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_model.fit(x_data[:, np.newaxis], y_data)
    y_poly_pred = poly_model.predict(x_data[:, np.newaxis])
    coef_poly = poly_model.named_steps['linearregression'].coef_
    intercept_poly = poly_model.named_steps['linearregression'].intercept_
    
    # Correctly access the coefficients; note that coef_poly[0] is for the bias term added by PolynomialFeatures
    poly_label = f'Polynomial: $y = {intercept_poly:.2f} + {coef_poly[1]:.2f}x {coef_poly[2]:.3f}x^2$'
    ax.plot(x_data, y_poly_pred, label=poly_label, color='green')

    ax.scatter(x_data, y_data, color='red', s=10)  # Plot data points
    ax.set_title(f'Performance Over Time: {algorithm}')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average Survival Time')
    
    lin_eq = f'Linear equation: y = {intercept_lin} + {coef_lin}x'
    poly_eq = f'Polynomial equation: y = {intercept_poly} + {coef_poly[1]}x + {coef_poly[2]}x^2'
    logging.info(f"{algorithm} - {lin_eq}")
    logging.info(f"{algorithm} - {poly_eq}")

    # Adding the equation text to the plot
    # ax.text(0.95, 0.05, lin_label, verticalalignment='bottom', horizontalalignment='right',
    #         transform=ax.transAxes, color='blue', fontsize=10)
    # ax.text(0.95, 0.01, poly_label, verticalalignment='bottom', horizontalalignment='right',
    #         transform=ax.transAxes, color='green', fontsize=10)

    ax.legend()
    if ENABLE_WANDB:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        wandb.log({f"{algorithm} Regression Plot": wandb.Image(image)})
        plt.close()
    else:
        plt.show()

def cohens_d(group1, group2):
    """Compute Cohen's d for independent samples."""
    diff = group1.mean() - group2.mean()
    pooled_sd = np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)
    return diff / pooled_sd

def perform_analyses(data_df):
    """Enhanced analysis including interactions and post-hoc tests."""
    logging.info("Starting mixed-effects model analysis...")
    
    from statsmodels.formula.api import mixedlm
    import statsmodels.api as sm

    # Mixed effects model with interaction between Algorithm and Trial
    model = mixedlm("SurvivalTime ~ C(Algorithm, Treatment(reference='SI')) * Trial", data_df, groups=data_df["Seed"])
    result = model.fit()
    logging.info(f"Linear Mixed Effects Model results:\n{result.summary()}")

    # Extracting the model summary to log or display
    model_summary = result.summary()
    logging.info(model_summary)

    # Correct import for the pairwise Tukey HSD test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Post-hoc analysis using pairwise comparisons
    tukey = pairwise_tukeyhsd(endog=data_df['SurvivalTime'], groups=data_df['Algorithm'], alpha=0.05)
    logging.info(f"Tukey HSD results:\n{tukey.summary()}")

    # Optionally log the results to Weights & Biases
    if ENABLE_WANDB:
        wandb.log({"Model Summary": model_summary.as_text(), "Tukey HSD Results": str(tukey.summary())})

def gather_and_analyze_data():
    """Function to gather data, perform statistical analysis, and plot results."""
    data_dict = {}
    files = get_latest_files(OUTPUT_FOLDER)
    for file_path in files:
        details = FILE_PATTERN.search(os.path.basename(file_path)).groups()
        _, seed, algorithm = details
        data = load_data(file_path)
        if algorithm not in data_dict:
            data_dict[algorithm] = {}
        if seed not in data_dict[algorithm]:
            data_dict[algorithm][seed] = []
        data_dict[algorithm][seed].extend(data)

    df_list = []
    for algorithm, seeds in data_dict.items():
        for seed, values in seeds.items():
            trials_df = pd.DataFrame({'SurvivalTime': values})
            trials_df['Trial'] = trials_df.index
            trials_df['Algorithm'] = algorithm
            trials_df['Seed'] = seed
            df_list.append(trials_df)
    data_df = pd.concat(df_list)

    perform_analyses(data_df)
    return data_df

# def perform_analyses(data_df):
#     """Performs statistical analyses including ANOVA, Cohen's d, and Linear Mixed Effects Model."""
#     algorithms = data_df['Algorithm'].unique()
#     grouped_data = [data_df[data_df['Algorithm'] == alg]['SurvivalTime'] for alg in algorithms]
#     f_val, p_val = f_oneway(*grouped_data)
#     logging.info(f"ANOVA results: F = {f_val}, p = {p_val}")

#     max_diff = 0
#     for i in range(len(algorithms)):
#         for j in range(i + 1, len(algorithms)):
#             d = cohens_d(data_df[data_df['Algorithm'] == algorithms[i]]['SurvivalTime'],
#                          data_df[data_df['Algorithm'] == algorithms[j]]['SurvivalTime'])
#             if abs(d) > abs(max_diff):
#                 max_diff = d
#                 pair = (algorithms[i], algorithms[j])
#     logging.info(f"Max Cohen's d = {max_diff} between {pair[0]} and {pair[1]}")

#     model = smf.mixedlm("SurvivalTime ~ Algorithm", data_df, groups=data_df["Algorithm"])
#     result = model.fit()
#     logging.info(f"Linear Mixed Effects Model results:\n{result.summary()}")
    
# def gather_and_analyze_data():
#     """Gathers data, performs statistical analysis, plots results, and logs with wandb if enabled."""
#     data_dict = {}
#     files = get_latest_files(OUTPUT_FOLDER)
#     for file_path in files:
#         details = FILE_PATTERN.search(os.path.basename(file_path)).groups()
#         _, seed, algorithm = details
#         data = load_data(file_path)
#         if algorithm not in data_dict:
#             data_dict[algorithm] = {}
#         if seed not in data_dict[algorithm]:
#             data_dict[algorithm][seed] = []
#         data_dict[algorithm][seed].extend(data)

#     df_list = []
#     for algorithm, seeds in data_dict.items():
#         for seed, values in seeds.items():
#             trials_df = pd.DataFrame({'SurvivalTime': values})
#             trials_df['Trial'] = trials_df.index  # Create a Trial index
#             trials_df['Algorithm'] = algorithm    # Add algorithm name for each row
#             trials_df['Seed'] = seed              # Add seed identifier
#             df_list.append(trials_df)
#     data_df = pd.concat(df_list)

#     perform_analyses(data_df)

#     # Plotting results
#     for algorithm in data_dict.keys():
#         fig, ax = plt.subplots()
#         # Only consider numeric data for mean calculation
#         alg_data = data_df[data_df['Algorithm'] == algorithm]
#         # Make sure to group by 'Trial' and calculate the mean of 'SurvivalTime'
#         trial_means = alg_data.groupby('Trial')['SurvivalTime'].mean().reset_index()
#         plot_regression(ax, trial_means['Trial'].values, trial_means['SurvivalTime'].values, algorithm)

#     return data_df

def plot_violin(data_df):
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Create two subplots vertically

    # Plot for all data
    sns.violinplot(x='Algorithm', y='SurvivalTime', data=data_df, ax=axs[0])
    axs[0].set_title("Survival Time Distribution by Algorithm")
    axs[0].set_xlabel("Algorithm")
    axs[0].set_ylabel("Survival Time")

    # Filtering data to remove early deaths (SurvivalTime <= 25)
    filtered_data = data_df[data_df['SurvivalTime'] > 25]

    # Plot for filtered data
    sns.violinplot(x='Algorithm', y='SurvivalTime', data=filtered_data, ax=axs[1])
    axs[1].set_title("Survival Time Distribution by Algorithm (Excluding Survival Time <= 25)")
    axs[1].set_xlabel("Algorithm")
    axs[1].set_ylabel("Survival Time")

    if ENABLE_WANDB:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)  # Create an image object from the buffer
        wandb.log({"Violin Plots": wandb.Image(image)})
        plt.close(fig)
    else:
        plt.show()

def main():
    try:
        while True:
            data_df = gather_and_analyze_data()
            plot_violin(data_df)
            if not ENABLE_WANDB:
                plt.show()
                break
            time.sleep(POLLING_INTERVAL)
    except KeyboardInterrupt:
        print("Stopping monitoring...")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--enable-wandb':
        ENABLE_WANDB = True
    main()

# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import re
# import time
# import wandb
# from datetime import datetime, timedelta

# # Constants
# OUTPUT_FOLDER = 'C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning'
# POLLING_INTERVAL = 60  # Time in seconds between checks for new files
# FILE_PATTERN = re.compile(r"(\d{2}-\d{2}-\d{2}-\d{3})_seed_(\d+)_([A-Z]+)_experiment.txt")

# def load_data(file_path):
#     """Loads numerical data from a given file."""
#     with open(file_path, 'r') as file:
#         data = [float(line.strip()) for line in file.readlines()]
#     return data

# def get_latest_files(directory):
#     """Finds the latest files for each algorithm-seed combination."""
#     files = {}
#     for file_name in os.listdir(directory):
#         match = FILE_PATTERN.match(file_name)
#         if match:
#             timestamp, seed, algorithm = match.groups()
#             key = (seed, algorithm)
#             if key not in files or timestamp > files[key][0]:
#                 files[key] = (timestamp, os.path.join(directory, file_name))
#     return [info[1] for info in files.values()]

# def plot_polynomial_regression(ax, x_data, y_data, degree, label):
#     """Plots polynomial regression of a specified degree on given axes."""
#     coeffs = np.polyfit(x_data, y_data, degree)
#     p = np.poly1d(coeffs)
#     x_line = np.linspace(min(x_data), max(x_data), 300)
#     y_line = p(x_line)
#     ax.plot(x_line, y_line, label=label, color='black')
#     ax.scatter(x_data, y_data, color='red', s=10)

# def main():
#     """Main function to run the monitoring and analysis."""
#     wandb.init(project="PC_ParallelExperiment_Monitoring", entity="shocklab")
#     print(f'Experiment monitoring has started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')

#     wandb.alert(
#         title='PC Parallel Experiment Monitoring Started',
#         text=f'Monitoring started at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
#         level=wandb.AlertLevel.INFO,
#         wait_duration=timedelta(minutes=10)
#     )

#     processed_files = set()
#     try:
#         while True:
#             current_files = set(get_latest_files(OUTPUT_FOLDER))
#             new_files = current_files - processed_files
#             if new_files:
#                 data_dict = {}
#                 for file_path in new_files:
#                     details = FILE_PATTERN.search(os.path.basename(file_path)).groups()
#                     seed, algorithm = details[1], details[2]
#                     data = load_data(file_path)
#                     if algorithm not in data_dict:
#                         data_dict[algorithm] = {}
#                     if seed not in data_dict[algorithm]:
#                         data_dict[algorithm][seed] = []
#                     data_dict[algorithm][seed].extend(data)

#                 for algorithm, seeds_data in data_dict.items():
#                     fig, ax = plt.subplots()
#                     all_trials_data = []
#                     max_length = max(len(data_list) for data_list in seeds_data.values())
                    
#                     for trial in range(max_length):
#                         trial_data = [data_list[trial] for data_list in seeds_data.values() if len(data_list) > trial]
#                         if trial_data:
#                             all_trials_data.append(np.mean(trial_data))
                    
#                     if all_trials_data:
#                         iterations = np.arange(len(all_trials_data))
#                         plot_polynomial_regression(ax, iterations, all_trials_data, 2, f'Polynomial Fit - {algorithm}')

#                     ax.set_title(f'Performance Over Time: {algorithm}')
#                     ax.set_xlabel('Iteration')
#                     ax.set_ylabel('Average Performance Metric')
#                     ax.legend()
#                     plt.close(fig)
#                     wandb.log({f"{algorithm} Performance Overview": wandb.Image(fig)})
                
#                 processed_files.update(new_files)
#             time.sleep(POLLING_INTERVAL)
#     except KeyboardInterrupt:
#         print("Stopping monitoring...")

# if __name__ == "__main__":
#     main()
