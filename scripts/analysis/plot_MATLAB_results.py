"""
Statistical Analysis Script for Algorithm Performance

This script is designed to analyze, compare, and visualize algorithm performance data using statistical techniques and various types of plots.
It uses command-line arguments to control which sections of the code are executed, making it easier to run specific parts of the analysis without 
manually commenting and uncommenting sections of the code.

Usage:
    python plot_MATLAB_results.py [OPTIONS]

Options:
    --stat_compare       Perform a standard statistical comparison (t-tests) between algorithm performances.
    --poly_compare       Perform a polynomial regression comparison between algorithms.
    --plot_regression    Plot the regression curves for all algorithms, both full and after convergence.
    --plot_log           Plot polynomial fits on log-transformed data.
    --make_gif           Create a GIF of the moving averages for the performance of each algorithm across different window sizes.

Examples:
    1. To perform a standard statistical comparison:
       python plot_MATLAB_results.py --stat_compare

    2. To perform a polynomial regression comparison:
       python plot_MATLAB_results.py --poly_compare

    3. To generate regression plots for algorithm performance:
       python plot_MATLAB_results.py --plot_regression

    4. To plot polynomial fits on log-transformed data:
       python plot_MATLAB_results.py --plot_log

    5. To create a GIF showing moving averages:
       python plot_MATLAB_results.py --make_gif
       
    6. To plot the Kolmogorov-Smirnov test:
       python plot_MATLAB_results.py --plot_ks
       
    7. To run all options
       python plot_MATLAB_results.py --stat_compare --poly_compare --plot_regression --plot_log --make_gif --plot_ks

Notes:
    - The script expects the data files to be located in the specified directories (adjust the SAVE_PATH and BASE_PATH as needed).
    - Results and figures are saved in the output folder defined by the SAVE_PATH variable.
    - Multiple options can be combined in one run. For example:
       python script.py --stat_compare --plot_regression
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import re
import argparse
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import ttest_ind, t, sem, mannwhitneyu, ks_2samp
from scipy.optimize import curve_fit
import logging
import imageio


# Specify the output folder and regex pattern
SAVE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/'
BASE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/'
SURVIVAL_FOLDER = BASE_PATH + 'results/unknown_model/MATLAB/200trials_data_threefry'
# BASE_PATH = 'C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning\\'
# SURVIVAL_FOLDER = BASE_PATH + 'results\\unknown_model\\MATLAB\\300trials_data'

# Setup basic configuration for logging
LOGFILE_PATH = SAVE_PATH + 'algorithm_comparison_2.log'
logging.basicConfig(level=logging.INFO, filename=LOGFILE_PATH, filemode='w',
                    format='%(levelname)s:%(message)s')

# Refined regular expression pattern
file_pattern = re.compile(r"([A-Za-z]+)_Seed_(\d+)_(\d{2}-\d{2}-\d{2}-\d{3})\.txt")

def logistic(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def load_data(file_path, num_lines):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    if len(data) == num_lines: # NOTE: Should be 300 for 300 trial data
        return data
    else:
        # print(f"File {file_path} does not have {num_lines} lines.")
        return None  # Return None for files that do not have 300 lines

def get_files(directory):
    # print(f"Reading directory: {directory}")
    for file_name in os.listdir(directory):
        # print(f"File found: {file_name}")
        file_name = file_name.strip()  # Remove any leading/trailing whitespace
        match = file_pattern.match(file_name)
        if match:
            yield os.path.join(directory, file_name), match.groups()
        # else:
            # print(f"Did not match: {file_name}")
            # print(f"Expected pattern: {file_pattern.pattern}")

def perform_statistical_comparison(results):
    algorithms = list(results.keys())
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            algo1, algo2 = algorithms[i], algorithms[j]
            data1, data2 = results[algo1], results[algo2]

            # Perform t-test
            t_stat, p_value_t = ttest_ind(data1, data2)
            ci1 = t.interval(0.95, len(data1)-1, loc=np.mean(data1), scale=sem(data1))
            ci2 = t.interval(0.95, len(data2)-1, loc=np.mean(data2), scale=sem(data2))

            # Perform Mann–Whitney U test
            u_stat, p_value_u = mannwhitneyu(data1, data2, alternative='two-sided')

            # Perform Kolmogorov–Smirnov test
            ks_stat, p_value_ks = ks_2samp(data1, data2)

            # Log the results
            logging.info(f"\n{algo1} vs. {algo2}:")
            logging.info(f"T-test result: T-statistic = {t_stat}, P-value = {p_value_t}")
            logging.info(f"Mann-Whitney U test result: U-statistic = {u_stat}, P-value = {p_value_u}")
            logging.info(f"Kolmogorov-Smirnov test result: KS-statistic = {ks_stat}, P-value = {p_value_ks}")
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

def moving_average_standard(data, window_size=10):
    if window_size <= 0:
        raise ValueError("Window size must be positive")
    
    # Apply the moving average (valid mode ensures no artificial difference at the start)
    mov_avg = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    
    # Padding the beginning to maintain the same length as the input
    padding = np.full(window_size-1, np.nan)  # Fill with NaNs or zeros if preferred
    result = np.concatenate((padding, mov_avg))
    
    return result

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

def plot_regression(ax, x_data, y_data, algorithm, data_color, line_color, trim_range=None):
    # Convert input data to numpy arrays for processing
    x_data_array = np.array(x_data)
    y_data_array = np.array(y_data)
    
    # Trim data to focus analysis on a specific range if provided
    if trim_range and len(trim_range) == 2:
        indices = (x_data_array >= trim_range[0]) & (x_data_array <= trim_range[1])
        x_data_array = x_data_array[indices]
        y_data_array = y_data_array[indices]

    # # Linear regression fitting
    # lin_model = LinearRegression()
    # lin_model.fit(x_data_array[:, np.newaxis], y_data_array)
    # y_lin_pred = lin_model.predict(x_data_array[:, np.newaxis])

    # # Calculate and plot confidence intervals for linear regression
    # t_value_lin = t.ppf(0.975, df=len(y_data_array)-2)
    # ci_lin = t_value_lin * np.std(y_data_array - y_lin_pred) * np.sqrt(1/len(y_data_array) + (x_data_array - np.mean(x_data_array))**2 / np.sum((x_data_array - np.mean(x_data_array))**2))
    # ax.fill_between(x_data_array, y_lin_pred - ci_lin, y_lin_pred + ci_lin, color=line_color, alpha=0.2, label=f'{algorithm} Linear 95% CI')
    # ax.plot(x_data_array, y_lin_pred, label=f'{algorithm} Linear', color=line_color)

    # # Polynomial regression fitting
    # degree = 2
    # poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    # poly_model.fit(x_data_array[:, np.newaxis], y_data_array)
    # y_poly_pred = poly_model.predict(x_data_array[:, np.newaxis])

    # # Calculate and plot confidence intervals for polynomial regression
    # t_value_poly = t.ppf(0.975, df=len(y_data_array)-(degree+1))
    # ci_poly = t_value_poly * np.std(y_data_array - y_poly_pred) * np.sqrt(1/len(y_data_array) + (x_data_array - np.mean(x_data_array))**2 / np.sum((x_data_array - np.mean(x_data_array))**2))
    # ax.fill_between(x_data_array, y_poly_pred - ci_poly, y_poly_pred + ci_poly, color=line_color, alpha=0.1, label=f'{algorithm} Polynomial 95% CI')
    # ax.plot(x_data_array, y_poly_pred, label=f'{algorithm} Polynomial', color=line_color, linestyle='dashed')

    # Logistic regression fitting and confidence interval calculation
    initial_guess = [max(y_data_array), 1, np.median(x_data_array)]
    try:
        popt, pcov = curve_fit(logistic, x_data_array, y_data_array, p0=initial_guess, maxfev=10000)
        x_model = np.linspace(min(x_data_array), max(x_data_array), 120)
        y_model = logistic(x_model, *popt)
        
        # Calculate the Jacobian matrix for variance of logistic model predictions
        J = np.zeros((len(x_model), 3))
        J[:, 0] = 1 / (1 + np.exp(-popt[1] * (x_model - popt[2])))  # Partial derivative with respect to L
        J[:, 1] = popt[0] * (x_model - popt[2]) * np.exp(-popt[1] * (x_model - popt[2])) / (1 + np.exp(-popt[1] * (x_model - popt[2])))**2  # Partial derivative with respect to k
        J[:, 2] = popt[0] * popt[1] * np.exp(-popt[1] * (x_model - popt[2])) / (1 + np.exp(-popt[1] * (x_model - popt[2])))**2  # Partial derivative with respect to x0
        
        # Calculate prediction variance and standard errors
        y_var = np.dot(J, np.dot(pcov, J.T)).diagonal()
        y_std = np.sqrt(y_var)
        ci_lower = y_model - t.ppf(0.975, df=len(y_data_array)-len(popt)) * y_std
        ci_upper = y_model + t.ppf(0.975, df=len(y_data_array)-len(popt)) * y_std
        
        # Plot logistic model predictions and confidence intervals
        ax.plot(x_model, y_model, color=line_color, linestyle=':', label=f'{algorithm} Logistic Fit [20,120]')
        ax.fill_between(x_model, ci_lower, ci_upper, color=line_color, alpha=0.2, label=f'{algorithm} 95% CI')
    except Exception as e:
        print(f"Error in fitting logistic model for {algorithm}: {e}")

    # Plot original data points
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
    # mov_avg = moving_average_with_cumulative_start(y_data, window_size)
    mov_avg = moving_average_standard(y_data, window_size)
    
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

def plot_ks_test_with_confidence(ax, data1, data2, algo1, algo2, alpha=0.05):
    # Sort the data for ECDF
    data1_sorted = np.sort(data1)
    data2_sorted = np.sort(data2)
    
    # Calculate ECDF
    ecdf1 = np.arange(1, len(data1_sorted)+1) / len(data1_sorted)
    ecdf2 = np.arange(1, len(data2_sorted)+1) / len(data2_sorted)
    
    # Perform KS test
    ks_stat, p_value = ks_2samp(data1, data2)
    
    # Plot ECDFs
    ax.step(data1_sorted, ecdf1, label=f'{algo1} ECDF', color='blue', where='post')
    ax.step(data2_sorted, ecdf2, label=f'{algo2} ECDF', color='orange', where='post')
    
    # Find the point where the maximum difference occurs
    idx1 = np.searchsorted(data1_sorted, np.median(data1_sorted))
    idx2 = np.searchsorted(data2_sorted, np.median(data2_sorted))
    max_diff = np.abs(ecdf1[idx1] - ecdf2[idx2])
    
    # Highlight the KS statistic
    ax.vlines(x=data1_sorted[idx1], ymin=ecdf1[idx1], ymax=ecdf2[idx2], colors='red', linestyles='dashed', label=f'KS Statistic = {ks_stat:.3f}')
    
    # Add Confidence Bands using DKW inequality
    n1 = len(data1_sorted)
    n2 = len(data2_sorted)
    
    # Calculate epsilon for the DKW confidence bands
    epsilon1 = np.sqrt(np.log(2 / alpha) / (2 * n1))
    epsilon2 = np.sqrt(np.log(2 / alpha) / (2 * n2))
    
    # Plot confidence bands for data1
    ax.fill_between(data1_sorted, np.maximum(0, ecdf1 - epsilon1), np.minimum(1, ecdf1 + epsilon1), 
                    color='blue', alpha=0.2, label=f'{algo1} 95% Confidence Band')
    
    # Plot confidence bands for data2
    ax.fill_between(data2_sorted, np.maximum(0, ecdf2 - epsilon2), np.minimum(1, ecdf2 + epsilon2), 
                    color='orange', alpha=0.2, label=f'{algo2} 95% Confidence Band')

    # Add labels and legend
    ax.set_title(f'Kolmogorov-Smirnov Test: {algo1} vs {algo2}')
    ax.set_xlabel('Data Value')
    ax.set_ylabel('ECDF')
    ax.legend()
    ax.grid(True)

def bootstrap_ecdf(data, num_bootstrap=1000):
    """Generate bootstrap ECDFs for the given data."""
    n = len(data)
    ecdf_samples = []
    for _ in range(num_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        resample_sorted = np.sort(resample)
        ecdf = np.arange(1, n+1) / n
        ecdf_samples.append((resample_sorted, ecdf))
    return ecdf_samples

def plot_ks_test_with_bootstrap_confidence(ax, data1, data2, algo1, algo2, num_bootstrap=1000, alpha=0.05):
    # Sort the data for ECDF
    data1_sorted = np.sort(data1)
    data2_sorted = np.sort(data2)
    
    # Calculate ECDF
    ecdf1 = np.arange(1, len(data1_sorted)+1) / len(data1_sorted)
    ecdf2 = np.arange(1, len(data2_sorted)+1) / len(data2_sorted)
    
    # Perform KS test
    ks_stat, p_value = ks_2samp(data1, data2)
    
    # Plot ECDFs
    ax.step(data1_sorted, ecdf1, label=f'{algo1} ECDF', color='blue', where='post')
    ax.step(data2_sorted, ecdf2, label=f'{algo2} ECDF', color='orange', where='post')
    
    # Generate bootstrap ECDFs
    ecdf_bootstrap1 = bootstrap_ecdf(data1, num_bootstrap)
    ecdf_bootstrap2 = bootstrap_ecdf(data2, num_bootstrap)
    
    # Define a common set of points for interpolation
    common_points = np.linspace(min(data1.min(), data2.min()), max(data1.max(), data2.max()), 1000)
    
    # Interpolate bootstrap ECDFs
    boot_ecdf1 = [np.interp(common_points, sorted(sample[0]), sample[1]) for sample in ecdf_bootstrap1]
    boot_ecdf2 = [np.interp(common_points, sorted(sample[0]), sample[1]) for sample in ecdf_bootstrap2]
    
    # Convert to numpy arrays for percentile calculation
    boot_ecdf1 = np.array(boot_ecdf1)
    boot_ecdf2 = np.array(boot_ecdf2)
    
    # Calculate percentiles
    lower_percentile1 = np.percentile(boot_ecdf1, 100 * (alpha / 2), axis=0)
    upper_percentile1 = np.percentile(boot_ecdf1, 100 * (1 - alpha / 2), axis=0)
    lower_percentile2 = np.percentile(boot_ecdf2, 100 * (alpha / 2), axis=0)
    upper_percentile2 = np.percentile(boot_ecdf2, 100 * (1 - alpha / 2), axis=0)
    
    # Plot confidence bands
    ax.fill_between(common_points, lower_percentile1, upper_percentile1, color='blue', alpha=0.2, label=f'{algo1} 95% Bootstrap CI')
    ax.fill_between(common_points, lower_percentile2, upper_percentile2, color='orange', alpha=0.2, label=f'{algo2} 95% Bootstrap CI')
    
    # Highlight KS statistic
    ax.vlines(x=data1_sorted[np.argmax(np.abs(ecdf1 - ecdf2))], ymin=min(ecdf1[np.argmax(np.abs(ecdf1 - ecdf2))], ecdf2[np.argmax(np.abs(ecdf1 - ecdf2))]), 
              ymax=max(ecdf1[np.argmax(np.abs(ecdf1 - ecdf2))], ecdf2[np.argmax(np.abs(ecdf1 - ecdf2))]),
              colors='red', linestyles='dashed', label=f'KS Statistic = {ks_stat:.3f}')
    
    # Add labels and legend
    ax.set_title(f'Kolmogorov-Smirnov Test: {algo1} vs {algo2}')
    ax.set_xlabel('Data Value')
    ax.set_ylabel('ECDF')
    ax.legend()
    ax.grid(True)

def bootstrap_ks_stat(data1, data2, num_bootstrap=1000):
    ks_stats = []
    for _ in range(num_bootstrap):
        resample1 = np.random.choice(data1, size=len(data1), replace=True)
        resample2 = np.random.choice(data2, size=len(data2), replace=True)
        ks_stat, _ = ks_2samp(resample1, resample2)
        ks_stats.append(ks_stat)
    
    # Calculate 2.5th and 97.5th percentiles for 95% confidence interval
    lower_bound = np.percentile(ks_stats, 2.5)
    upper_bound = np.percentile(ks_stats, 97.5)
    return lower_bound, upper_bound

def parse_args():
    parser = argparse.ArgumentParser(description="Statistical analysis script for algorithms.")
    
    # Add arguments to enable or disable different features
    parser.add_argument('--stat_compare', action='store_true', help="Perform statistical comparison.")
    parser.add_argument('--poly_compare', action='store_true', help="Perform polynomial comparison.")
    parser.add_argument('--plot_regression', action='store_true', help="Plot regression curves.")
    parser.add_argument('--plot_log', action='store_true', help="Plot log-transformed data.")
    parser.add_argument('--make_gif', action='store_true', help="Create GIF of moving average.")
    parser.add_argument('--plot_ks', action='store_true', help="Plot Kolmogorov-Smirnov test results between algorithms.")  # New KS test option

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # Load and process the data
    data_dict = {}
    for file_path, (algorithm, seed, _) in get_files(SURVIVAL_FOLDER):
        data = load_data(file_path, num_lines=200)
        if data is not None:
            if algorithm not in data_dict:
                data_dict[algorithm] = []
            data_dict[algorithm].append(data)

    if not data_dict:
        raise ValueError("No matching files found for any algorithm.")

    results = {alg: np.mean(np.array(data), axis=0) for alg, data in data_dict.items()}
    
    x_data = np.linspace(0, 200, 200)

    # Perform statistical comparison
    if args.stat_compare:
        perform_statistical_comparison(results)

    # Perform polynomial comparison
    if args.poly_compare:
        perform_statistical_comparison_polynomial(results, x_data)

    # Plot regressions
    if args.plot_regression:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))
        colors = {'BA': 'red', 'BAUCB': 'blue', 'SI': 'green', 'SL': 'orange'}
        
        for algorithm, performance in results.items():
            iterations = np.arange(len(performance))
            plot_regression(
                axes[0], 
                iterations, 
                performance, 
                algorithm, 
                colors[algorithm], 
                colors[algorithm], 
                trim_range=[20,200])
            plot_trimmed_regression(axes[1], iterations, performance, algorithm, colors[algorithm], colors[algorithm])

        # Set titles, labels, and legends as requested
        axes[0].set_title('Comparison of Algorithm Performance Over Trials')
        axes[1].set_title('Data Comparison of Algorithm Performance After Convergence')
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('Average Survival Time')
        axes[1].set_xlabel('Trial (after convergence)')
        axes[1].set_ylabel('Average Survival Time')
        axes[0].legend(title='Algorithm', loc='upper left')
        axes[1].legend(title='Algorithm', loc='upper left')
        plt.tight_layout()
        plt.savefig(SAVE_PATH+'test_SI_data.png')

    # Plot log-transformed data
    if args.plot_log:
        fig, ax = plt.subplots(figsize=(12, 8))
        for algorithm, performance in results.items():
            iterations = np.arange(len(performance))
            plot_log_transformed(ax, iterations, performance, algorithm, colors[algorithm], colors[algorithm])

        # Set titles, labels, and legends as requested
        ax.set_title('Polynomial Fits on Log-Transformed Data with Normal Axes')
        ax.set_xlabel('Trial')
        ax.set_ylabel('Log(Average Survival Time)')
        ax.legend(title='Algorithm', loc='upper left')
        plt.tight_layout()
        plt.savefig(SAVE_PATH+'MATLAB_logged.png')

    # Generate GIF of moving averages
    if args.make_gif:
        gif_frames = []
        for window_size in range(1, 51):
            fig, ax = plt.subplots(figsize=(12, 8))
            for algorithm, performance in results.items():
                iterations = np.arange(len(performance))
                plot_moving_average(ax, iterations, performance, algorithm, colors[algorithm], window_size)

            # Set titles, labels, and legends as requested
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

    # Plot Kolmogorov-Smirnov test results
    if args.plot_ks:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))

        # Plot DKW Confidence Bands
        algo1 = 'SI'
        algo2 = 'SL'

        if algo1 in results and algo2 in results:
            lower_bound, upper_bound = bootstrap_ks_stat(results[algo1], results[algo2])
            print(f'Bootstrap Confidence Interval for KS Statistic between {algo1} and {algo2}: [{lower_bound}, {upper_bound}]')

            # Now visualize this on the ECDF plot
            fig, ax = plt.subplots(figsize=(12, 8))

            # Get ECDFs for both datasets
            data1_sorted = np.sort(results[algo1])
            data2_sorted = np.sort(results[algo2])
            
            # Calculate ECDFs
            ecdf1 = np.arange(1, len(data1_sorted)+1) / len(data1_sorted)
            ecdf2 = np.arange(1, len(data2_sorted)+1) / len(data2_sorted)

            # Perform KS test
            ks_stat, p_value = ks_2samp(results[algo1], results[algo2])

            # Plot ECDFs
            ax.step(data1_sorted, ecdf1, label=f'{algo1} ECDF', color='blue', where='post')
            ax.step(data2_sorted, ecdf2, label=f'{algo2} ECDF', color='orange', where='post')

            # Highlight the KS statistic as a vertical dashed line
            max_diff_idx = np.argmax(np.abs(ecdf1 - ecdf2))
            max_diff_x = data1_sorted[max_diff_idx]  # The point of max difference
            ax.vlines(max_diff_x, ymin=min(ecdf1[max_diff_idx], ecdf2[max_diff_idx]),
                    ymax=max(ecdf1[max_diff_idx], ecdf2[max_diff_idx]),
                    colors='red', linestyles='dashed', label=f'KS Statistic = {ks_stat:.3f}')

            # Plot the bootstrap confidence interval for the KS statistic
            # Draw the confidence interval as horizontal lines at the KS statistic's max_diff_x
            ax.hlines(y=[min(ecdf1[max_diff_idx] - upper_bound, 1), 
                        min(ecdf1[max_diff_idx] - lower_bound, 1)], 
                    xmin=max_diff_x - 0.5, xmax=max_diff_x + 0.5, 
                    colors='gray', linestyles='solid', label=f'Bootstrap CI for KS: [{lower_bound:.2f}, {upper_bound:.2f}]')
            
            # Add labels, legend, and title
            ax.set_title(f'Kolmogorov-Smirnov Test: {algo1} vs {algo2} with Bootstrap CI')
            ax.set_xlabel('Data Value')
            ax.set_ylabel('ECDF')
            ax.legend()
            ax.grid(True)

            # Save the plot
            plt.tight_layout()
            plt.savefig(SAVE_PATH + f'ks_statistic_bootstrap_{algo1}_vs_{algo2}.png')
            plt.show()
