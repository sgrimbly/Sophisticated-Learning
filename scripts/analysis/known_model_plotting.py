# import os
# import re
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# # Define the directory where the .out files are located
# directory = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB/scripts/UCT-HPC'
# tracking_file_path = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB/scripts/UCT-HPC/known_model_submitted_jobs.txt'

# # Read the tracking file and create a configuration mapping
# configurations = []
# with open(tracking_file_path, 'r') as file:
#     configurations = [line.strip() for line in file if line.strip()]

# # Map job_ids to configurations
# job_to_config = {}
# for index, config in enumerate(configurations):
#     job_to_config[(index % 33) + 1] = config  # Assuming job_id starts from 1 and resets every 33

# # Initialize a dictionary to hold the data
# data = {'Memory': [], 'NoMemory': [], 'Hybrid': [], 'PureMC': []}

# # Regular expression to extract job_id from filename
# filename_pattern = re.compile(r'matlab_job_(\d+)\.out')

# # Function to parse configuration details from the config line
# def parse_details(config_line):
#     # Assume config line format: "{exp_type}_Seed{seed}_Hor{horizon}_MCT_{mct} type={type}, horizon={horizon}, mct={mct}, seed={seed}"
#     description = config_line.split()[0]
#     details = config_line.split(', ')
#     exp_type = description.split('_')[0]
#     seed = details[3].split('=')[1]
#     horizon = details[1].split('=')[1]
#     mct = details[2].split('=')[1]
#     return exp_type, seed, horizon, mct

# # Correctly classify experiments based on detailed conditions
# def classify_experiment(mct, horizon, auto_rest):
#     if mct == 0 and horizon == 6:
#         # Specific logic to determine if this should be 'Hybrid' or 'Memory'
#         # This may need additional data or flags to decide correctly
#         return 'Hybrid'  # Assuming you want to prioritize 'Hybrid' for this edge case
#     elif mct == 0 and horizon != 0:
#         return 'Memory' if auto_rest == '0' else 'NoMemory'
#     elif mct != 0 and horizon != 0:
#         return 'Hybrid'
#     elif mct != 0 and horizon == 0:
#         return 'PureMC'
#     else:
#         return None  # For cases that do not match any configuration

# for filename in os.listdir(directory):
#     match = filename_pattern.search(filename)
#     if match:
#         job_id = int(match.group(1))
#         sequence_number = (job_id - 1) % 33 + 1  # Calculate which configuration this job_id corresponds to

#         if sequence_number in job_to_config:
#             config_line = job_to_config[sequence_number]
#             exp_type, seed, horizon, mct = parse_details(config_line)

#             # Classify the experiment type based on mct, horizon, and possibly other factors
#             exp_type = classify_experiment(mct, horizon, exp_type)

#             if exp_type:
#                 filepath = os.path.join(directory, filename)
#                 with open(filepath, 'r') as file:
#                     content = file.read()
#                     survival_matches = re.findall(r'^\s*(\d+)\s*$', content, re.MULTILINE)
#                     if survival_matches:
#                         survival_time = int(survival_matches[-1])
#                         data[exp_type].append({
#                             'Seed': int(seed),
#                             'Horizon': int(horizon),
#                             'MCT': int(mct),
#                             'Survival': survival_time
#                         })


# # Convert lists to pandas DataFrames and plot data
# for exp_type, exp_data in data.items():
#     if exp_data:
#         df = pd.DataFrame(exp_data)
#         plt.figure(figsize=(12, 6))
#         grouped = df.groupby('Horizon' if exp_type != 'PureMC' else 'MCT')['Survival']
#         mean = grouped.mean()
#         sem = grouped.std().div(np.sqrt(grouped.count())).replace(0, np.nan)  # Avoid zero division, replace SEM=0 with NaN

#         ci_lower = mean - 1.96 * sem
#         ci_upper = mean + 1.96 * sem

#         plt.plot(mean.index, mean, marker='o', label='Mean Survival', color='blue')
#         plt.fill_between(mean.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

#         plt.xlabel('MCT' if exp_type == 'PureMC' else 'Horizon')
#         plt.ylabel('Survival Time')
#         plt.title(f'{exp_type} Survival Analysis with 95% CI')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(f'{exp_type}_survival_CI_plot.png')
#         # plt.show()
#     else:
#         print(f"No data found for {exp_type}")


import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# Define the directory where the .out files are located
directory = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/src/MATLAB/scripts/UCT-HPC'

# Initialize a dictionary to hold the data
data = {'Memory': [], 'NoMemory': [], 'Hybrid': [], 'PureMC': []}

# Regular expression to extract the necessary information from the first line of each file
info_pattern = re.compile(r'ALGORITHM=(.*), SEED=(\d+), HORIZON=(\d+), K_FACTOR=(.*), ROOT_FOLDER=(.*), MCT=(\d+), NUM_MCT=(\d+), AUTO_REST=(\d+)')

# Regular expression to find the survival time
survival_pattern = re.compile(r'^\s*(\d+)\s*$', re.MULTILINE)

# Process each file in the directory that matches the filename pattern
for filename in os.listdir(directory):
    if filename.startswith('matlab_job_') and filename.endswith('.out'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            content = file.read()  # Read the whole file content at once

            # Extract experiment info from the first line
            match = info_pattern.search(content.split('\n')[0])
            if match:
                algorithm, seed, horizon, k_factor, root_folder, mct, num_mct, auto_rest = match.groups()
                seed = int(seed)
                horizon = int(horizon)
                mct = int(mct)

                # Determine the type of experiment based on the file content
                if mct == 0 and horizon != 0:
                    exp_type = 'Memory' if auto_rest == '0' else 'NoMemory'
                elif mct != 0 and horizon != 0:
                    exp_type = 'Hybrid'
                elif mct != 0 and horizon == 0:
                    exp_type = 'PureMC'
                else:
                    continue  # Ignore other configurations

                # Find the survival time using the updated pattern
                survival_matches = survival_pattern.findall(content)
                if survival_matches:
                    survival_time = int(survival_matches[-1])  # Get the last matched survival time
                    data[exp_type].append({'Seed': seed, 'Horizon': horizon, 'MCT': mct, 'Survival': survival_time})

# Convert the lists to pandas DataFrames for easier manipulation
for exp_type in data:
    if data[exp_type]:  # Check if there's data in the list before creating DataFrame
        data[exp_type] = pd.DataFrame(data[exp_type])
    else:
        print(f"No data found for {exp_type}")

# Create line plots with confidence intervals for each experiment type
for exp_type, df in data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        # plt.figure(figsize=(12, 6))
        # # Group data to calculate mean and confidence interval
        # grouped = df.groupby('Horizon' if exp_type != 'PureMC' else 'MCT')['Survival']
        # mean = grouped.mean()
        # sem = grouped.std().div(np.sqrt(grouped.count())).replace(0, np.nan)  # Avoid zero division, replace SEM=0 with NaN

        # ci_lower = mean - 1.96 * sem
        # ci_upper = mean + 1.96 * sem

        # # Plotting the line plot with fill_between for confidence interval
        # plt.plot(mean.index, mean, marker='o', label='Mean Survival', color='blue')
        # plt.fill_between(mean.index, ci_lower, ci_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

        # plt.xlabel('MCT' if exp_type == 'PureMC' else 'Horizon')
        # plt.ylabel('Survival Time')
        # plt.title(f'{exp_type} Survival Analysis with 95% CI')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f'{exp_type}_survival_CI_plot.png')
        # # plt.show()
        # Create one figure for all plots
        plt.figure(figsize=(12, 6))

        # Colors for each experiment type
        colors = {'Memory': 'blue', 'NoMemory': 'orange', 'Hybrid': 'green', 'PureMC': 'red'}

        # Create line plots with confidence intervals for each experiment type
        for exp_type, df in data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # Group data to calculate mean and confidence interval
                grouped = df.groupby('Horizon' if exp_type != 'PureMC' else 'MCT')['Survival']
                mean = grouped.mean()
                sem = grouped.std().div(np.sqrt(grouped.count())).replace(0, np.nan)  # Avoid zero division, replace SEM=0 with NaN

                ci_lower = mean - 1.96 * sem
                ci_upper = mean + 1.96 * sem

                # Plotting the line plot with fill_between for confidence interval
                plt.plot(mean.index, mean, marker='o', label=f'{exp_type} Mean Survival', color=colors[exp_type])
                plt.fill_between(mean.index, ci_lower, ci_upper, color=colors[exp_type], alpha=0.2, label=f'{exp_type} 95% CI')

        plt.xlabel('Horizon/MCT')  # Adjust x-label to cover both cases
        plt.ylabel('Survival Time')
        plt.title('Survival Analysis with 95% CI (All Experiments)')
        plt.legend()
        plt.grid(True)
        plt.savefig('all_experiments_survival_CI_plot.png')
        # plt.show()