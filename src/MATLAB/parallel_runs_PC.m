% Main script
% Flag to determine whether to use Weights & Biases
USE_WEIGHTS_AND_BIASES = false;

% Ensure a parallel pool is open with a specified number of workers
if isempty(gcp('nocreate'))
    pool = parpool(2); 
else
    pool = gcp();
end

% Attach necessary files to the parallel pool
% addAttachedFiles(pool, {'src/MATLAB/grid_configs.txt', 'src/MATLAB/main.m'}); % Add more files if needed

% Directory and path setup
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir);
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);

% Define the new algorithms and grid sizes
algorithms = {'SI', 'SL'}; 
grid_sizes = [3, 5, 7, 10];
horizons = {[1, 2], [2, 4], [3, 5], [3, 5, 7]};

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

% Read configurations from the file
configurations = read_configurations('src/MATLAB/grid_configs.txt');

% Construct list of experiments using configurations from the file
experimentsToRun = construct_experiments_from_configs(configurations, algorithms, num_trials, weights);
length(experimentsToRun)
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
parfor idx = 1:totalExperiments
    run_experiment(experimentsToRun{idx}, outputLogDir, k_factor, root_folder, mct, num_mct, num_trials, USE_WEIGHTS_AND_BIASES);
end

% Close the W&B session if initialized
if USE_WEIGHTS_AND_BIASES
    wb.finish();
end

fprintf('All experiments completed.\n');

% Function to check memory usage
% function check_memory()
%     [usermem, sysmem] = memory;
%     fprintf('Memory used by MATLAB: %g GB\n', usermem.MemUsedMATLAB / 1e9);
%     fprintf('Total system memory available: %g GB\n', sysmem.PhysicalMemory.Total / 1e9);
%     fprintf('Memory available for data: %g GB\n', sysmem.PhysicalMemory.Available / 1e9);
% end

% Function to read configurations from a file
function configs = read_configurations(filename)
    configs = {};
    fid = fopen(filename, 'r');
    if fid == -1
        error('Failed to open configuration file.');
    end
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Grid Size')
            gridSizeData = textscan(line, 'Grid Size: %d, Horizon: %d, Hill: %d');
            grid_size = gridSizeData{1};
            horizon = gridSizeData{2};
            hill_pos = gridSizeData{3};
            
            seasons = cell(1, 4);
            for k = 1:4
                line = fgetl(fid);
                seasonData = textscan(line, 'Season %d: Food(%d), Water(%d), Sleep(%d)');
                seasons{k} = [seasonData{2}, seasonData{3}, seasonData{4}];
            end
            
            configs{end+1} = struct('grid_size', grid_size, 'horizon', horizon, 'hill_pos', hill_pos, 'seasons', {seasons});
        end
    end
    fclose(fid);
end

% Function to construct experiments from configurations
function expList = construct_experiments_from_configs(configs, algorithms, num_trials, weights)
    expList = {};
    for i = 1:numel(configs)
        config = configs{i};
        grid_size = config.grid_size;
        num_states = grid_size^2;
        for alg = 1:numel(algorithms)
            for seed = 1:30
                for season = 1:4
                    seasonData = config.seasons{season};
                    expList{end+1} = struct('algorithm', algorithms{alg}, 'seed', seed, 'grid_size', grid_size, 'horizon', config.horizon, ...
                        'hill_pos', config.hill_pos, ...
                        'food_sources', seasonData(1), ...
                        'water_sources', seasonData(2), ...
                        'sleep_sources', seasonData(3), ...
                        'num_states', num_states, 'num_trials', num_trials, 'weights', weights);
                end
            end
        end
    end
end

% Function to run a single experiment
function run_experiment(expData, logDir, k_factor, root_folder, mct, num_mct, num_trials, USE_WEIGHTS_AND_BIASES)
    outputFileName = fullfile(logDir, sprintf('output_%s_seed%d_gridsize%d_horizon%d.txt', expData.algorithm, expData.seed, expData.grid_size, expData.horizon));
    diary(outputFileName);
    diary on;
    fprintf('Running %s with seed %d, grid size %d, horizon %d...\n', expData.algorithm, expData.seed, expData.grid_size, expData.horizon);

    try
        % Periodic memory check before execution
        % check_memory();

        % Ensure all inputs are of the correct type
        algorithm = char(expData.algorithm);
        seed = double(expData.seed);
        horizon = double(expData.horizon);
        grid_size = double(expData.grid_size);
        hill_pos = double(expData.hill_pos);
        food_sources = double(expData.food_sources);
        water_sources = double(expData.water_sources);
        sleep_sources = double(expData.sleep_sources);
        weights = [expData.weights.novelty, expData.weights.learning, expData.weights.epistemic, expData.weights.preference];
        num_states = double(expData.num_states);

        % Call the main function with correctly typed arguments
        survived = main(algorithm, seed, horizon, k_factor, root_folder, mct, num_mct, false, '', grid_size, ...
                        hill_pos, food_sources, water_sources, sleep_sources, weights, num_states, num_trials);

        % Log survived data to W&B if needed
        if USE_WEIGHTS_AND_BIASES
            for j = 1:length(survived)
                step_data = py.dict(pyargs('algorithm', algorithm, 'seed', seed, 'trial', j, 'survived', double(survived(j))));
                py.wandb.log(step_data);
            end
        end
    catch e
        fprintf('Error occurred in %s with seed %d, grid size %d, horizon %d: %s\n', expData.algorithm, expData.seed, expData.grid_size, expData.horizon, e.message);
    end

    % Periodic memory check after execution
    % try
    %     check_memory();
    % catch e
    %     fprintf('Memory check error after %s with seed %d, grid size %d, horizon %d: %s\n', expData.algorithm, expData.seed, expData.grid_size, expData.horizon, e.message);
    % end

    % Stop capturing output to file
    diary off;
end
