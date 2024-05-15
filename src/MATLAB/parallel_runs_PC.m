% Flag to determine whether to use Weights & Biases
USE_WEIGHTS_AND_BIASES = false;

% Ensure a parallel pool is open with a specified number of workers
if isempty(gcp('nocreate'))
    parpool(2); 
end

% Directory and path setup
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir, '../..');
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);

% Define the new algorithms and grid sizes
algorithms = {'model_free_RL', 'SL', 'SI', 'BA', 'BAUCB'}; % other algorithms can be added
grid_sizes = [3, 5, 7, 10];
horizons = {1:2, 1:2:5, 3:2:7, 1:2:7};

% Set experiment parameters
k_factor = 1.5;
root_folder = 'defaultFolder';
mct = 500;
num_mct = 10;
num_trials = 300;
weights = struct('novelty', 10, 'learning', 40, 'epistemic', 1, 'preference', 10);

% Initialize Weights & Biases if required
if USE_WEIGHTS_AND_BIASES
    try
        py.importlib.import_module('wandb');
        wb = py.wandb.init(pyargs('project','active_inference_agent', 'entity', 'sgrimbly', 'resume', 'allow'));
    catch e
        disp('Failed to load wandb module');
        disp(e.message);
    end
end

% Ensure output log directory exists
outputLogDir = 'C:\Users\stjoh\Documents\ActiveInference\Sophisticated-Learning\results\output_logs';
if ~exist(outputLogDir, 'dir')
    mkdir(outputLogDir);
end

% Construct list of experiments
experimentsToRun = {};
for i = 1:length(algorithms)
    for gs_idx = 1:length(grid_sizes)
        grid_size = grid_sizes(gs_idx);
        num_states = grid_size^2;
        possible_horizons = horizons{gs_idx};
        for horizon = possible_horizons
            [hill_pos, food_sources, water_sources, sleep_sources] = generate_resources(grid_size);
            for s = 1:30
                experimentsToRun{end+1} = struct('algorithm', algorithms{i}, 'seed', s, 'grid_size', grid_size, 'horizon', horizon, ...
                                                 'hill_pos', hill_pos, 'food_sources', food_sources, 'water_sources', water_sources, ...
                                                 'sleep_sources', sleep_sources, 'num_states', num_states, 'num_trials', num_trials, ...
                                                 'weights', weights);
            end
        end
    end
end

% Print the intended order of experiments
fprintf('Intended order of experiments:\n');
for idx = 1:length(experimentsToRun)
    expData = experimentsToRun{idx};
    fprintf('Experiment %d: %s with seed %d, grid_size %d, horizon %d, hill_pos %d, food_sources %d, water_sources %d, sleep_sources %d\n', ...
            idx, expData.algorithm, expData.seed, expData.grid_size, expData.horizon, ...
            expData.hill_pos, expData.food_sources, expData.water_sources, expData.sleep_sources);
end

% Total number of experiments
totalExperiments = numel(experimentsToRun);



% Use parfor for parallel execution over all experiments
% parfor idx = 1:totalExperiments
%     expData = experimentsToRun{idx};
%     algorithm = expData.algorithm;
%     seed = expData.seed;
%     grid_size = expData.grid_size;
%     horizon = expData.horizon;
%     hill_pos = expData.hill_pos;
%     food_sources = expData.food_sources;
%     water_sources = expData.water_sources;
%     sleep_sources = expData.sleep_sources;
%     num_states = expData.num_states;
%     num_trials = expData.num_trials;
%     weights = expData.weights;
%     seedStr = num2str(seed);  % Convert seed to string if necessary

%     % Construct output file name
%     outputFileName = fullfile(outputLogDir, sprintf('output_%s_seed%d_gridsize%d_horizon%d.txt', algorithm, seed, grid_size, horizon));

%     % Start capturing output to file
%     diary(outputFileName);
%     diary on;

%     fprintf('Running %s with seed %d, grid_size %d, horizon %d...\n', algorithm, seed, grid_size, horizon);
%     try
%         % Periodic memory check before execution
%         check_memory();

%         % Call the main function
%         survived = main(algorithm, seedStr, horizon, k_factor, root_folder, mct, num_mct, false, '', grid_size, hill_pos, food_sources, water_sources, sleep_sources, weights, num_states, num_trials);

%         % Log survived data to W&B if needed
%         if USE_WEIGHTS_AND_BIASES
%             for j = 1:length(survived)
%                 step_data = py.dict(pyargs('algorithm', algorithm, 'seed', seed, 'trial', j, 'survived', double(survived(j))));
%                 py.wandb.log(step_data);
%             end
%         end

%         % Periodic memory check after execution
%         check_memory();
%     catch e
%         fprintf('Error occurred in %s with seed %d, grid_size %d, horizon %d: %s\n', algorithm, seed, grid_size, horizon, e.message);
%         check_memory();  % Check memory also in case of an error
%     finally
%         % Stop capturing output to file
%         diary off;
%     end
% end

% Close the W&B session if initialized
if USE_WEIGHTS_AND_BIASES
    wb.finish();
end

fprintf('All experiments completed.\n');

% Function to check memory usage
function check_memory()
    [usermem, sysmem] = memory;
    fprintf('Memory used by MATLAB: %g GB\n', usermem.MemUsedMATLAB / 1e9);
    fprintf('Total system memory available: %g GB\n', sysmem.PhysicalMemory.Total / 1e9);
    fprintf('Memory available for data: %g GB\n', sysmem.PhysicalMemory.Available / 1e9);
end

% Helper function to generate non-overlapping resource positions
function [hill_pos, food_sources, water_sources, sleep_sources] = generate_resources(grid_size)
    num_cells = grid_size^2;
    num_resources = 4; % Total resources (1 hill + 1 food + 1 water + 1 sleep)
    
    if num_resources > num_cells
        error('Grid size is too small to place all resources without overlap');
    end
    
    positions = randperm(num_cells, num_resources); % Generate unique positions for resources
    
    hill_pos = positions(1);
    food_sources = positions(2);
    water_sources = positions(3);
    sleep_sources = positions(4);
end
