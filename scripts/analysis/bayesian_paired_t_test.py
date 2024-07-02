import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
from jax import random
import logging
import netCDF4 as nc
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

# Basic logging setup
logging.basicConfig(level=logging.INFO, filename='algorithm_comparison_2.log', filemode='w',
                    format='%(levelname)s:%(message)s')

# Define paths and file pattern
SAVE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/'
BASE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/'
SURVIVAL_FOLDER = BASE_PATH + 'results/unknown_model/MATLAB/300trials_data'
file_pattern = re.compile(r"([A-Za-z]+)_Seed_(\d+)_(\d{2}-\d{2}-\d{2}-\d{3})\.txt")

def load_data(file_path):
    """Loads data from a specified path and returns a numpy array if the data has exactly 300 lines."""
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    if len(data) == 300:
        return data
    else:
        logging.warning(f"File {file_path} does not have 300 lines.")
        return None

def get_files(directory):
    """Yield file paths and groups for files matching the defined regex within a directory."""
    for file_name in os.listdir(directory):
        match = file_pattern.match(file_name)
        if match:
            yield os.path.join(directory, file_name), match.groups()

def bayesian_t_tests_by_time(data1, data2, num_timesteps=10, num_chains=1, num_warmup=500, num_samples=1000):
    results = {}
    for t in range(num_timesteps):
        def model():
            mu = numpyro.sample('mu', dist.Normal(0, 10))
            sigma = numpyro.sample('sigma', dist.HalfNormal(10))
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=data1[:, t] - data2[:, t])

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
        rng_keys = random.split(random.PRNGKey(0), num_chains)
        mcmc.run(rng_keys)
        results[t] = mcmc.get_samples()

        logging.info(f"Mu samples shape: {results[t]['mu'].shape}, Sigma samples shape: {results[t]['sigma'].shape}")
        if results[t]['mu'].shape[0] > 0:  # Check if not empty
            logging.info(f"Sample Mu values: {results[t]['mu'][:5]}")

    return results

def save_results_directly(time_point_results, algo1, algo2):
    for time_point, samples in time_point_results.items():
        filename = f"Direct_Bayesian_Test_{algo1}_vs_{algo2}_time_{time_point}.nc"
        file_path = os.path.join(SAVE_PATH, filename)
        try:
            with nc.Dataset(file_path, 'w', format='NETCDF4') as ds:
                ds.createDimension('sample_dim', samples['mu'].shape[0])

                mu_var = ds.createVariable('mu', np.float32, ('sample_dim',))
                sigma_var = ds.createVariable('sigma', np.float32, ('sample_dim',))

                mu_var[:] = samples['mu'].flatten()  # Flatten to match dimension
                sigma_var[:] = samples['sigma'].flatten()  # Flatten to match dimension

                logging.info(f"Successfully saved directly: {file_path}")
        except Exception as e:
            logging.error(f"Failed to save directly {filename}: {str(e)}")

def plot_time_series_density():
    files = [f for f in os.listdir(SAVE_PATH) if f.startswith('Direct_Bayesian_Test') and f.endswith('.nc')]
    files.sort(key=lambda x: int(x.split('time_')[-1].split('.')[0]))

    # Prepare data collection
    densities = []
    time_points = []

    # Collect data
    for file in files:
        full_file_path = os.path.join(SAVE_PATH, file)
        with nc.Dataset(full_file_path, 'r') as ds:
            values = ds.variables['mu'][:]  # Assuming 'mu' is your parameter of interest
            time_point = int(file.split('time_')[-1].split('.')[0])
            time_points.append(time_point)

            # Compute the density
            kde_values = np.linspace(values.min(), values.max(), 100)
            density, _ = np.histogram(values, bins=kde_values, density=True)
            densities.append(np.interp(np.linspace(values.min(), values.max(), 100), kde_values[:-1], density))

    # Convert list to numpy array
    densities = np.array(densities)
    time_points = np.array(time_points)

    # Optionally smooth the densities
    densities = gaussian_filter(densities, sigma=1)

    # Plotting
    plt.figure(figsize=(12, 6))
    extent = [time_points.min(), time_points.max(), values.min(), values.max()]
    plt.imshow(densities.T, aspect='auto', origin='lower', extent=extent, norm=Normalize(vmin=densities.min(), vmax=densities.max()), cmap='viridis')
    plt.colorbar(label='Density')
    plt.xlabel('Time Point')
    plt.ylabel('Mu Values')
    plt.title('Time Series of Density Distributions')
    plt.savefig(SAVE_PATH + 'timeseries_density_plot.png')

def plot_time_series_density_with_summary():
    files = [f for f in os.listdir(SAVE_PATH) if f.startswith('Direct_Bayesian_Test') and f.endswith('.nc')]
    files.sort(key=lambda x: int(x.split('time_')[-1].split('.')[0]))

    # Prepare data collection
    densities = []
    time_points = []
    means = []
    medians = []
    percentiles_25 = []
    percentiles_75 = []

    # Collect data
    for file in files:
        full_file_path = os.path.join(SAVE_PATH, file)
        with nc.Dataset(full_file_path, 'r') as ds:
            values = ds.variables['mu'][:]  # Assuming 'mu' is your parameter of interest
            time_point = int(file.split('time_')[-1].split('.')[0])
            time_points.append(time_point)

            # Compute the density
            kde_values = np.linspace(values.min(), values.max(), 100)
            density, _ = np.histogram(values, bins=kde_values, density=True)
            densities.append(np.interp(np.linspace(values.min(), values.max(), 100), kde_values[:-1], density))

            # Calculate summary statistics
            means.append(np.mean(values))
            medians.append(np.median(values))
            percentiles_25.append(np.percentile(values, 25))
            percentiles_75.append(np.percentile(values, 75))

    # Convert list to numpy array
    densities = np.array(densities)
    time_points = np.array(time_points)

    # Optionally smooth the densities
    densities = gaussian_filter(densities, sigma=1)

    # Plotting the density heatmap
    plt.figure(figsize=(12, 6))
    extent = [time_points.min(), time_points.max(), values.min(), values.max()]
    plt.imshow(densities.T, aspect='auto', origin='lower', extent=extent, norm=Normalize(vmin=densities.min(), vmax=densities.max()), cmap='viridis')
    plt.colorbar(label='Density')
    
    # Overlay summary statistics
    plt.plot(time_points, means, color='white', label='Mean')
    plt.plot(time_points, medians, color='red', label='Median')
    plt.plot(time_points, percentiles_25, color='orange', linestyle='--', label='25th Percentile')
    plt.plot(time_points, percentiles_75, color='orange', linestyle='--', label='75th Percentile')

    plt.xlabel('Time Point')
    plt.ylabel('Mu Values')
    plt.title('Time Series of Density Distributions with Summary Statistics')
    plt.legend()
    plt.savefig(SAVE_PATH + 'timeseries_density_plot_with_summary.png')
    
def check_nc_file(file_path):
    try:
        with nc.Dataset(file_path, 'r') as dataset:
            print("Variables after saving:", list(dataset.variables.keys()))
            print("Dimensions after saving:", list(dataset.dimensions.keys()))
            for var in dataset.variables.keys():
                print(f"Details for {var} after saving:")
                print("Shape:", dataset[var].shape)
                print("Data:", dataset[var][:])
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")

def test_create_nc():
    sample_data = np.random.rand(10, 10)
    with nc.Dataset('test_file.nc', 'w', format='NETCDF4') as ds:
        ds.createDimension('dim1', 10)
        ds.createDimension('dim2', 10)
        var = ds.createVariable('random_data', np.float32, ('dim1', 'dim2'))
        var[:] = sample_data
        print("Data saved directly via netCDF4:")
        print(ds.variables['random_data'][:])

# Example usage (commented out for safety; uncomment for actual use)
data_dict = {}
for file_path, (algorithm, seed, timestamp) in get_files(SURVIVAL_FOLDER):
    data = load_data(file_path)
    if data is not None:
        if algorithm not in data_dict:
            data_dict[algorithm] = []
        data_dict[algorithm].append(data)

algorithms = list(data_dict.keys())
if len(algorithms) > 1:
    algo1, algo2 = algorithms[0], algorithms[1]
    data1 = np.array(data_dict[algo1])
    data2 = np.array(data_dict[algo2])
    time_point_results = bayesian_t_tests_by_time(data1, data2, num_timesteps=300, num_chains=1, num_warmup=1500, num_samples=5000)
    save_results_directly(time_point_results, algo1, algo2)
else:
    logging.warning("Not enough algorithms found for comparison.")

plot_time_series_density()
plot_time_series_density_with_summary()

# plt.savefig(SAVE_PATH+'timeseries_paired_test_improved.png')
# check_nc_file("/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/Direct_Bayesian_Test_SI_vs_SL_time_0.nc")
# test_create_nc()


def bayesian_statistical_test(alpha=0.05):
    files = [f for f in os.listdir(SAVE_PATH) if f.startswith('Direct_Bayesian_Test') and f.endswith('.nc')]
    files.sort(key=lambda x: int(x.split('time_')[-1].split('.')[0]))

    significant_time_points = []
    posterior_probabilities = []

    for file in files:
        full_file_path = os.path.join(SAVE_PATH, file)
        with nc.Dataset(full_file_path, 'r') as ds:
            mu_values = ds.variables['mu'][:]
            time_point = int(file.split('time_')[-1].split('.')[0])
            
            # Calculate the posterior probability that the difference is greater than zero
            prob_greater_than_zero = np.mean(mu_values > 0)
            posterior_probabilities.append(prob_greater_than_zero)
            
            # Check if this probability is significant (e.g., less than alpha or greater than 1 - alpha)
            if prob_greater_than_zero > 1 - alpha or prob_greater_than_zero < alpha:
                significant_time_points.append((time_point, prob_greater_than_zero))
    
    return significant_time_points, posterior_probabilities

def plot_posterior_probabilities(posterior_probabilities, alpha=0.05):
    time_points = range(len(posterior_probabilities))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_points, posterior_probabilities, label='Posterior Probability')
    plt.axhline(1 - alpha, color='red', linestyle='--', label=f'Significance Threshold (1 - {alpha})')
    plt.axhline(alpha, color='red', linestyle='--')
    
    plt.xlabel('Time Point')
    plt.ylabel('Posterior Probability')
    plt.title('Posterior Probability that Difference is Greater than Zero')
    plt.legend()
    plt.savefig(SAVE_PATH + 'posterior_probabilities_plot.png')

# Run the test and plot the results
significant_time_points, posterior_probabilities = bayesian_statistical_test(alpha=0.05)
plot_posterior_probabilities(posterior_probabilities, alpha=0.05)

# Print significant time points
print("Significant Time Points (Time Point, Posterior Probability):")
for tp in significant_time_points:
    print(tp)