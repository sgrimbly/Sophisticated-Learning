import time
import re
import matplotlib.pyplot as plt
import os
import numpy as np

def read_and_extract_numbers(directory_path):
    pattern = re.compile(r'time_steps_survived: (\d+)')
    data = {}
    for filename in os.listdir(directory_path):
        if filename.startswith("model_free_results_seed") and filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            numbers = []
            with open(file_path, 'r') as file:
                for line in file:
                    match = pattern.search(line)
                    if match:
                        number = int(match.group(1))
                        numbers.append(number)
            if numbers:
                seed = filename.split('seed')[1].split('.')[0]
                data[int(seed)] = numbers
    return data

def average_data(numbers, bin_size=1000):
    return np.mean(np.reshape(numbers[:len(numbers)//bin_size*bin_size], (-1, bin_size)), axis=1)

def aggregate_data(data, bin_size=1000):
    aggregated_data = []
    all_data = list(data.values())
    min_length = min(len(d) for d in all_data)
    min_length = min_length - min_length % bin_size
    for i in range(0, min_length, bin_size):
        bin_data = [d[i:i+bin_size] for d in all_data]
        aggregated_data.append([np.mean(b) for b in bin_data])
    return np.array(aggregated_data)

def compute_aggregated_batch_stats(data, bin_size=1000):
    all_batches = []
    min_length = min(len(d) for d in data.values())
    min_length = min_length - min_length % bin_size

    # Collect all batch data
    for i in range(0, min_length, bin_size):
        batches = [d[i:i+bin_size] for d in data.values()]
        concatenated = np.concatenate(batches)
        reshaped = concatenated.reshape(-1, bin_size)
        all_batches.append(reshaped)

    # Calculate global statistics
    means = [np.mean(b) for b in all_batches]
    q1 = [np.percentile(b, 25) for b in all_batches]
    q3 = [np.percentile(b, 75) for b in all_batches]
    mins = [np.min(b) for b in all_batches]
    maxs = [np.max(b) for b in all_batches]

    return means, q1, q3, mins, maxs

def update_plot(data, aggregated_data):
    plt.cla()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # First plot (existing line chart)
    for seed, numbers in data.items():
        averaged_numbers = average_data(numbers)
        ax1.plot(averaged_numbers, label=f'Seed {seed}', alpha=0.5)

    mean_values = np.mean(aggregated_data, axis=1)
    std_dev = np.std(aggregated_data, axis=1)
    x_values = np.arange(len(mean_values))
    ax1.plot(x_values, mean_values, 'k-', label='Mean across seeds')
    ax1.fill_between(x_values, mean_values - std_dev, mean_values + std_dev, color='gray', alpha=0.5, label='Std Dev')
    ax1.set_title('Time Steps Survived (Averaged over 1000 datapoints with Uncertainty)')
    ax1.set_xlabel('Batch Index (each representing 1000 datapoints)')
    ax1.set_ylabel('Time Steps Survived')
    ax1.legend()

    # Second plot (aggregated batch stats across all seeds)
    means, q1, q3, mins, maxs = compute_aggregated_batch_stats(data)
    batch_indices = np.arange(len(means))
    ax2.plot(batch_indices, means, 'r-', label='Mean across all seeds')
    ax2.fill_between(batch_indices, q1, q3, color='blue', alpha=0.3, label='IQR across all seeds')
    ax2.plot(batch_indices, mins, 'k.', markersize=2, label='Min across all seeds')
    ax2.plot(batch_indices, maxs, 'k.', markersize=2, label='Max across all seeds')
    ax2.set_title('Aggregated Batch Statistics Across All Seeds')
    ax2.set_xlabel('Batch Index (each representing 1000 datapoints)')
    ax2.set_ylabel('Time Steps Survived')
    ax2.legend()

    plt.tight_layout()
    plt.draw()
    plt.pause(0.1)

plt.ion()
fig = plt.figure()

directory_path = 'C:\\Users\\stjoh\\Documents\\ActiveInference\\Sophisticated-Learning\\results\\model_free_results'
continuous_update = False

if continuous_update:
    while True:
        data = read_and_extract_numbers(directory_path)
        update_plot(data, aggregated_data)
        time.sleep(60)
else:
    data = read_and_extract_numbers(directory_path)
    aggregated_data = aggregate_data(data)
    update_plot(data, aggregated_data)
    plt.ioff()
    plt.show()
