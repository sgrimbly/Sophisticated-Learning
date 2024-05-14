import numpy as np
import matplotlib.pyplot as plt
import os
import re
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import ttest_ind, t, sem
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, filename='algorithm_comparison_2.log', filemode='w',
                    format='%(levelname)s:%(message)s')

# Specify the output folder and regex pattern
SAVE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/'
BASE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/'
SURVIVAL_FOLDER = BASE_PATH + 'results/unknown_model/MATLAB/300trials_data'
# BASE_PATH = 'C:\\Users\\micro\\Documents\\ActiveInference_Work\\Sophisticated-Learning\\'
# SURVIVAL_FOLDER = BASE_PATH + 'results\\unknown_model\\MATLAB\\120trials_data'

# Refined regular expression pattern
file_pattern = re.compile(r"([A-Za-z]+)_Seed(\d+)_(\d{2}-\d{2}-\d{2}-\d{3})\.txt")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    return data

def get_files(directory):
    for file_name in os.listdir(directory):
        match = file_pattern.match(file_name)
        if match:
            yield os.path.join(directory, file_name), match.groups()
        else:
            print(f"Did not match: {file_name}")
            print(f"Expected pattern: {file_pattern.pattern}")

def perform_statistical_comparison(results):
    algorithms = list(results.keys())
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            data1, data2 = results[algo1], results[algo2]

            # Perform t-test
            t_stat, p_value = ttest_ind(data1, data2)
            ci1 = t.interval(0.95, len(data1)-1, loc=np.mean(data1), scale=sem(data1))
            ci2 = t.interval(0.95, len(data2)-1, loc=np.mean(data2), scale=sem(data2))

            # Log the results
            logging.info(f"\n{algo1} vs. {algo2}:")
            logging.info(f"T-test result: T-statistic = {t_stat}, P-value = {p_value}")
            logging.info(f"{algo1} Data 95% CI: {ci1}")
            logging.info(f"{algo2} Data 95% CI: {ci2}")

def perform_statistical_comparison_polynomial(results, x_values):
    algorithms = list(results.keys())
    predictions = {}
    for algorithm in algorithms:
        model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        model.fit(x_values[:, np.newaxis], results[algorithm])
        predictions[algorithm] = model.predict(x_values[:, np.newaxis])
    
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            pred1, pred2 = predictions[algo1], predictions[algo2]

            # Perform t-test on the predictions
            t_stat, p_value = ttest_ind(pred1, pred2)
            ci1 = t.interval(0.95, len(pred1)-1, loc=np.mean(pred1), scale=sem(pred1))
            ci2 = t.interval(0.95, len(pred2)-1, loc=np.mean(pred2), scale=sem(pred2))

            # Log the results
            logging.info(f"\n{algo1} vs. {algo2}:")
            logging.info(f"T-test result: T-statistic = {t_stat}, P-value = {p_value}")
            logging.info(f"{algo1} Predictions 95% CI: {ci1}")
            logging.info(f"{algo2} Predictions 95% CI: {ci2}")

def plot_regression(ax, x_data, y_data, algorithm, data_color, line_color):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    n = len(y_data)
    
    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(x_data[:, np.newaxis], y_data)
    y_lin_pred = lin_model.predict(x_data[:, np.newaxis])
    residuals = y_data - y_lin_pred
    std_residuals = np.std(residuals)
    
    # Confidence intervals for linear regression
    t_value_lin = t.ppf(0.975, df=n-2)
    ci_lin = t_value_lin * std_residuals * np.sqrt(1/n + (x_data - np.mean(x_data))**2 / np.sum((x_data - np.mean(x_data))**2))
    ax.fill_between(x_data, y_lin_pred - ci_lin, y_lin_pred + ci_lin, color=line_color, alpha=0.2, label=f'{algorithm} Linear 95% CI')
    ax.plot(x_data, y_lin_pred, label=f'{algorithm} Linear', color=line_color)
    
    # Polynomial Regression
    degree = 2
    poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    poly_model.fit(x_data[:, np.newaxis], y_data)
    y_poly_pred = poly_model.predict(x_data[:, np.newaxis])
    residuals_poly = y_data - y_poly_pred
    std_residuals_poly = np.std(residuals_poly)
    
    # Confidence intervals for polynomial regression
    t_value_poly = t.ppf(0.975, df=n-(degree+1))
    X_poly = PolynomialFeatures(degree).fit_transform(x_data[:, np.newaxis])
    leverage = np.sum(X_poly * np.linalg.pinv(X_poly).T, axis=1)
    ci_poly = t_value_poly * std_residuals_poly * np.sqrt(1/n + leverage)
    ax.fill_between(x_data, y_poly_pred - ci_poly, y_poly_pred + ci_poly, color=line_color, alpha=0.1, label=f'{algorithm} Polynomial 95% CI')
    ax.plot(x_data, y_poly_pred, label=f'{algorithm} Polynomial', color=line_color, linestyle='dashed')
    
    # Scatter original data points
    ax.scatter(x_data, y_data, color=data_color, s=10, label=f'{algorithm} Data')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average Survival Time')
    ax.legend()

# Load, process, and plot data
data_dict = {}
for file_path, (algorithm, seed, _) in get_files(SURVIVAL_FOLDER):
    if algorithm not in data_dict:
        data_dict[algorithm] = []
    data_dict[algorithm].append(load_data(file_path))

# Ensure data_dict has data for at least one algorithm
if not data_dict:
    raise ValueError("No matching files found for any algorithm.")

# Average performances
results = {alg: np.mean(np.array(data), axis=0) for alg, data in data_dict.items()}

# Perform and log statistical comparisons
x_data = np.linspace(0, 120, 120)
perform_statistical_comparison(results)
perform_statistical_comparison_polynomial(results, x_data)

# Plot all algorithms on one plot with regressions
fig, ax = plt.subplots(figsize=(12, 8))
colors = {'BA': 'red', 'BAUCB': 'blue', 'SI': 'green', 'SL': 'orange'}
markers = {'BA': 'o', 'BAUCB': '^', 'SI': 's', 'SL': 'x'}

for algorithm, performance in results.items():
    iterations = np.arange(len(performance))
    plot_regression(ax, iterations, performance, algorithm, colors[algorithm], colors[algorithm])

ax.set_title('Comparison of Algorithm Performance Over Trials')
ax.set_xlabel('Trial')
ax.set_ylabel('Average Survival Time')
ax.legend(title='Algorithm', loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_PATH+'MATLAB_1.png')

fig, ax = plt.subplots(figsize=(12, 6))
for algorithm, performance in results.items():
    iterations = np.arange(len(performance))
    ax.plot(iterations, performance, label=f'{algorithm}', color=colors[algorithm], marker=markers[algorithm])
    ax.scatter(iterations, performance, color=colors[algorithm], s=10)

ax.set_title('Comparison of Algorithm Performance Over Trials')
ax.set_xlabel('Trial')
ax.set_ylabel('Average Survival Time')
ax.legend(title='Algorithm', loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_PATH+'MATLAB_2.png')