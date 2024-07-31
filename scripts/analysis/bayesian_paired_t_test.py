import numpy as np
import matplotlib.pyplot as plt
import os
import re
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import jax.random
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, filename='algorithm_comparison_2.log', filemode='w',
                    format='%(levelname)s:%(message)s')

# Specify the output folder and regex pattern
SAVE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/'
BASE_PATH = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/'
SURVIVAL_FOLDER = BASE_PATH + 'results/unknown_model/MATLAB/300trials_data'
file_pattern = re.compile(r"([A-Za-z]+)_Seed_(\d+)_(\d{2}-\d{2}-\d{2}-\d{3})\.txt")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = np.array([float(line.strip()) for line in file.readlines()])
    if len(data) == 300:
        return data
    else:
        print(f"File {file_path} does not have 300 lines.")
        return None

def get_files(directory):
    for file_name in os.listdir(directory):
        match = file_pattern.match(file_name)
        if match:
            yield os.path.join(directory, file_name), match.groups()

from jax import random

def bayesian_t_tests_by_time(data1, data2, num_chains=1):
    assert data1.shape == data2.shape, "Data arrays must have the same shape"
    time_steps = data1.shape[1]
    results = {}
    
    for t in range(time_steps):
        def model():
            mu = numpyro.sample('mu', dist.Normal(0, 10))
            sigma = numpyro.sample('sigma', dist.HalfNormal(10))
            numpyro.sample('obs', dist.Normal(mu, sigma), obs=data1[:, t] - data2[:, t])

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=num_chains)
        # Initialize random keys for each chain
        rng_keys = random.split(random.PRNGKey(0), num_chains)
        mcmc.run(rng_keys, extra_fields=('potential_energy',))
        results[t] = mcmc.get_samples()
    
    return results

def plot_posterior_time_series(algo1, algo2, num_time_points):
    means = []
    hdi_lowers = []
    hdi_uppers = []

    for t in range(num_time_points):
        filename = f"{SAVE_PATH}Bayesian_Test_{algo1}_vs_{algo2}_time_{t}.nc"
        data = az.load_arviz_data(filename)
        mu_posterior = data.posterior["mu"].values.flatten()
        hdi = az.hdi(mu_posterior, hdi_prob=0.95)
        means.append(np.mean(mu_posterior))
        hdi_lowers.append(hdi[0])
        hdi_uppers.append(hdi[1])

    plt.figure(figsize=(10, 5))
    plt.plot(range(num_time_points), means, label='Posterior Mean Difference')
    plt.fill_between(range(num_time_points), hdi_lowers, hdi_uppers, color='gray', alpha=0.5, label='95% HDI')
    plt.title(f'Time Series of Posterior Differences Between {algo1} and {algo2}')
    plt.xlabel('Time Point')
    plt.ylabel('Difference in Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(SAVE_PATH+'Bayesian_paired_test_posteriors.png')

def plot_density_over_time(parameter='mu'):
    files = [f for f in os.listdir(SAVE_PATH) if os.path.isfile(os.path.join(SAVE_PATH, f)) and f.endswith('.nc')]
    
    files.sort(key=lambda x: int(x.split('time_')[-1].split('.')[0]))

    plt.figure(figsize=(12, 8))
    for file in files:
        full_file_path = os.path.join(SAVE_PATH, file)
        try:
            data = az.load_arviz_data(full_file_path)
            param_values = data.posterior[parameter].values.flatten()
            time_point = int(file.split('time_')[-1].split('.')[0])
            plt.hist(param_values, bins=30, density=True, alpha=0.6, label=f'Time {time_point}')
        except ValueError as e:
            print(f"Error loading data from {full_file_path}: {e}")
            continue

    plt.xlabel('Parameter Value')
    plt.ylabel('Density')
    plt.title(f'Density Plots of Bayesian Estimates Over Time for {parameter}')
    plt.legend()
    plt.savefig(SAVE_PATH+'timeseries_paired_test.png')

data_dict = {}
for file_path, (algorithm, seed, _) in get_files(SURVIVAL_FOLDER):
    data = load_data(file_path)
    if data is not None:
        if algorithm not in data_dict:
            data_dict[algorithm] = []
        data_dict[algorithm].append(data)

algorithms = list(data_dict.keys())
results = {}



# for i in range(len(algorithms)):
#     for j in range(i + 1, len(algorithms)):
#         algo1, algo2 = algorithms[i], algorithms[j]
#         data1 = np.array(data_dict[algo1])
#         data2 = np.array(data_dict[algo2])

#         time_point_results = bayesian_t_tests_by_time(data1, data2)

#         for time_point, samples in time_point_results.items():
#             arviz_data = az.from_dict(posterior={"mu": samples["mu"], "sigma": samples["sigma"]})
#             filename = f"Bayesian_Test_{algo1}_vs_{algo2}_time_{time_point}.nc"
#             az.to_netcdf(arviz_data, os.path.join(SAVE_PATH, filename))

# plot_posterior_time_series('SI', 'SL', 300)

plot_density_over_time('mu')

print("Bayesian analysis complete and results saved.")
