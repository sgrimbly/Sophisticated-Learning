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
import imageio

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, filename='algorithm_comparison_2.log', filemode='w',
                    format='%(levelname)s:%(message)s')

# Specify the output folder and regex pattern
SAVE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/'
BASE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/'
SURVIVAL_FOLDER = BASE_PATH + 'results/unknown_model/MATLAB/300trials_data'
# BASE_PATH = 'C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning\\'
# SURVIVAL_FOLDER = BASE_PATH + 'results\\unknown_model\\MATLAB\\300trials_data'

# Refined regular expression pattern
file_pattern = re.compile(r"([A-Za-z]+)_Seed_(\d+)_(\d{2}-\d{2}-\d{2}-\d{3})\.txt")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    if len(data) == 300:
        return data
    else:
        print(f"File {file_path} does not have 300 lines.")
        return None  # Return None for files that do not have 300 lines

def get_files(directory):
    print(f"Reading directory: {directory}")
    for file_name in os.listdir(directory):
        print(f"File found: {file_name}")
        file_name = file_name.strip()  # Remove any leading/trailing whitespace
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

# def moving_average(data, window_size=10):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def moving_average_with_cumulative_start(data, window_size=10):
    if window_size <= 0:
        raise ValueError("Window size must be positive")

    # Initialize the result array
    result = np.empty(len(data))
    
    # Cumulative averaging for the initial part
    for i in range(window_size):
        result[i] = np.mean(data[:i+1])
    
    # Moving average for the rest
    moving_avg = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Adjust the index where the moving average is assigned
    result[window_size-1:] = moving_avg
    
    return result

def find_convergence_point(data, window_size=20, threshold=0.005):
    # mov_avg = moving_average(data, window_size)
    mov_avg = moving_average_with_cumulative_start(data, window_size)
    for i in range(len(mov_avg) - window_size):
        if np.max(np.abs(mov_avg[i:i+window_size] - mov_avg[i+window_size])) < threshold:
            return i + window_size
    return len(data) - 1

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

def plot_trimmed_regression(ax, x_data, y_data, algorithm, data_color, line_color):
    # Find convergence point
    conv_point = find_convergence_point(y_data, window_size=40, threshold=2)
    trimmed_x = x_data[conv_point:]
    trimmed_y = y_data[conv_point:]

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(trimmed_x[:, np.newaxis], trimmed_y)
    y_lin_pred = lin_model.predict(trimmed_x[:, np.newaxis])
    residuals = trimmed_y - y_lin_pred
    std_residuals = np.std(residuals)

    # Confidence intervals for linear regression
    n = len(trimmed_y)
    t_value_lin = t.ppf(0.975, df=n-2)
    ci_lin = t_value_lin * std_residuals * np.sqrt(1/n + (trimmed_x - np.mean(trimmed_x))**2 / np.sum((trimmed_x - np.mean(trimmed_x))**2))
    ax.fill_between(trimmed_x, y_lin_pred - ci_lin, y_lin_pred + ci_lin, color=line_color, alpha=0.2, label=f'{algorithm} Linear 95% CI')
    ax.plot(trimmed_x, y_lin_pred, label=f'{algorithm} Linear', color=line_color)

    # Scatter original data points
    ax.scatter(trimmed_x, trimmed_y, color=data_color, s=10, label=f'{algorithm} Data')
    ax.set_xlabel('Trial (after convergence)')
    ax.set_ylabel('Average Survival Time')
    ax.legend()

def plot_moving_average(ax, x_data, y_data, algorithm, color, window_size):
    mov_avg = moving_average_with_cumulative_start(y_data, window_size)
    ax.plot(x_data[:len(mov_avg)], mov_avg, label=f'{algorithm} Moving Average (window={window_size})', color=color)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average Survival Time')
    ax.legend()

def plot_log_transformed(ax, x_data, y_data, algorithm, data_color, line_color):
    x_data = np.array(x_data)
    y_data = np.log(np.array(y_data))
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
    ax.set_ylabel('Log(Average Survival Time)')
    ax.legend()

# Load, process, and plot data
data_dict = {}
for file_path, (algorithm, seed, _) in get_files(SURVIVAL_FOLDER):
    data = load_data(file_path)
    if data is not None:  # Only add data if it has 300 lines
        if algorithm not in data_dict:
            data_dict[algorithm] = []
        data_dict[algorithm].append(data)

# Ensure data_dict has data for at least one algorithm
if not data_dict:
    raise ValueError("No matching files found for any algorithm.")

# Average performances
results = {alg: np.mean(np.array(data), axis=0) for alg, data in data_dict.items()}

# Perform and log statistical comparisons
x_data = np.linspace(0, 300, 300)
perform_statistical_comparison(results)
perform_statistical_comparison_polynomial(results, x_data)

# Plot all algorithms on one plot with regressions (before and after convergence)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))
colors = {'BA': 'red', 'BAUCB': 'blue', 'SI': 'green', 'SL': 'orange'}

for algorithm, performance in results.items():
    iterations = np.arange(len(performance))
    plot_regression(axes[0], iterations, performance, algorithm, colors[algorithm], colors[algorithm])
    plot_trimmed_regression(axes[1], iterations, performance, algorithm, colors[algorithm], colors[algorithm])

axes[0].set_title('Comparison of Algorithm Performance Over Trials')
axes[1].set_title('Data Comparison of Algorithm Performance After Convergence')
axes[0].set_xlabel('Trial')
axes[0].set_ylabel('Average Survival Time')
axes[1].set_xlabel('Trial (after convergence)')
axes[1].set_ylabel('Average Survival Time')
axes[0].legend(title='Algorithm', loc='upper left')
axes[1].legend(title='Algorithm', loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_PATH+'MATLAB_combined.png')
# plt.show()

# Plot Polynomial Fits on Log-Transformed Data with Normal Axes
fig, ax = plt.subplots(figsize=(12, 8))

for algorithm, performance in results.items():
    iterations = np.arange(len(performance))
    plot_log_transformed(ax, iterations, performance, algorithm, colors[algorithm], colors[algorithm])

ax.set_title('Polynomial Fits on Log-Transformed Data with Normal Axes')
ax.set_xlabel('Trial')
ax.set_ylabel('Log(Average Survival Time)')
ax.legend(title='Algorithm', loc='upper left')
plt.tight_layout()
plt.savefig(SAVE_PATH+'MATLAB_logged.png')
# plt.show()

# Generate GIF of moving average with sliding window size
gif_frames = []

for window_size in range(1, 51):
    fig, ax = plt.subplots(figsize=(12, 8))
    for algorithm, performance in results.items():
        iterations = np.arange(len(performance))
        plot_moving_average(ax, iterations, performance, algorithm, colors[algorithm], window_size)
    
    ax.set_title(f'Moving Average of Algorithm Performance (Window Size={window_size})')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average Survival Time')
    ax.legend(title='Algorithm', loc='upper left')
    plt.tight_layout()
    
    # Save frame as PNG file
    frame_path = f'/tmp/frame_{window_size}.png'
    plt.savefig(frame_path)
    plt.close(fig)
    
    # Append frame to list
    gif_frames.append(imageio.imread(frame_path))

# Save frames as GIF
gif_path = SAVE_PATH + 'moving_average.gif'
imageio.mimsave(gif_path, gif_frames, duration=0.2)

print(f"GIF saved to {gif_path}")
