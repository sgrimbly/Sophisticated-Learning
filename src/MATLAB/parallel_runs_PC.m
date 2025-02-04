% Main script
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

% Define the algorithms
% make global num seeds variable
NUM_SEEDS = 100;
algorithms = {'SI', 'SL'}; 

% Set experiment parameters
k_factor = 1.5;
root_folder = 'defaultFolder';
mct = 0;
num_mct = 100;
num_trials = 100;
weights = struct('novelty', 10, 'learning', 40, 'epistemic', 1, 'preference', 10);

% Ensure output log directory exists
% outputLogDir = 'C:\Users\stjoh\Documents\ActiveInference\Sophisticated-Learning\results\output_logs';
outputLogDir = '/Users/stjohngrimbly/Documents/Sophisticated-Learning/results/output_logs';
if ~exist(outputLogDir, 'dir')
    mkdir(outputLogDir);
end

% Read configurations from the file
configurations = read_configurations('src/MATLAB/grid_configs.txt');
fprintf('Read %d configurations from file.\n', numel(configurations));
% Print configurations
for i = 1:numel(configurations)
    cfg = configurations{i};
    fprintf('Grid Size: %d, Horizon: %d, Hill: %d, Start Position: %d, Food(%s), Water(%s), Sleep(%s)\n', ...
            cfg.grid_size, cfg.horizon, cfg.hill_pos, cfg.start_position, mat2str(cfg.food_sources), mat2str(cfg.water_sources), mat2str(cfg.sleep_sources));
end


% Construct list of experiments using configurations from the file
experimentsToRun = construct_experiments_from_configs(configurations, algorithms, num_trials, weights, NUM_SEEDS);
% Print out experiment details
fprintf('Constructed %d experiments from configurations.\n', numel(experimentsToRun));

% fprintf('Intended order of experiments:\n');
% for idx = 1:length(experimentsToRun)
%     expData = experimentsToRun{idx};
%     fprintf('Experiment %d: %s with seed %d, grid_size %d, horizon %d, hill_pos %d, food_sources %d, water_sources %d, sleep_sources %d\n', ...
%             idx, expData.algorithm, expData.seed, expData.grid_size, expData.horizon, ...
%             expData.hill_pos, expData.food_sources, expData.water_sources, expData.sleep_sources);
% end

% Total number of experiments
totalExperiments = numel(experimentsToRun);

% Use parfor for parallel execution over all experiments
% parfor idx = 1:totalExperiments
%     run_experiment(experimentsToRun{idx}, outputLogDir, k_factor, root_folder, mct, num_mct, num_trials);
% end

fprintf('All experiments completed.\n');

function configs = read_configurations(filename)
    configs = {};
    fid = fopen(filename, 'r');
    if fid == -1
        error('Failed to open configuration file.');
    end
    while ~feof(fid)
        line = fgetl(fid);
        if contains(line, 'Grid ID')
            cfgData = textscan(line, 'Grid ID: %s, Grid Size: %d, Horizon: %d, Hill: %d, Start Position: %d, Food(%[^)]), Water(%[^)]), Sleep(%[^)])');
            grid_id = cfgData{1}{1};
            grid_size = cfgData{2};
            horizon = cfgData{3};
            hill_pos = cfgData{4};
            start_position = cfgData{5};
            food_sources = str2num(cfgData{6}{1});
            water_sources = str2num(cfgData{7}{1});
            sleep_sources = str2num(cfgData{8}{1});
            configs{end+1} = struct('grid_id', grid_id, 'grid_size', grid_size, 'horizon', horizon, 'hill_pos', hill_pos, 'start_position', start_position, ...
                                    'food_sources', food_sources, 'water_sources', water_sources, 'sleep_sources', sleep_sources);
        end
    end
    fclose(fid);
end

function expList = construct_experiments_from_configs(configs, algorithms, num_trials, weights, num_seeds)
    expList = {};
    for i = 1:numel(configs)
        config = configs{i};
        grid_size = config.grid_size;
        num_states = grid_size^2;
        for alg = 1:numel(algorithms)
            for seed = 1:num_seeds
                expList{end+1} = struct('algorithm', algorithms{alg}, 'seed', seed, 'grid_size', grid_size, 'horizon', config.horizon, ...
                                        'hill_pos', config.hill_pos, 'start_position', config.start_position, ...
                                        'food_sources', config.food_sources, 'water_sources', config.water_sources, ...
                                        'sleep_sources', config.sleep_sources, 'num_states', num_states, 'num_trials', num_trials, 'weights', weights);
            end
        end
    end
end

function run_experiment(expData, logDir, k_factor, root_folder, mct, num_mct, num_trials)
    outputFileName = fullfile(logDir, sprintf('output_%s_seed%d_gridsize%d_horizon%d.txt', expData.algorithm, expData.seed, expData.grid_size, expData.horizon));
    diary(outputFileName);
    diary on;

    fprintf('Running %s with seed %d, grid size %d, horizon %d, start position %d...\n', ...
            expData.algorithm, expData.seed, expData.grid_size, expData.horizon, expData.start_position);
    
    % Calling main with all required parameters
    main(expData.algorithm, expData.seed, expData.horizon, k_factor, root_folder, mct, num_mct, false, '', ...
         expData.grid_size, expData.start_position, expData.hill_pos, expData.food_sources, expData.water_sources, expData.sleep_sources, ...
         expData.weights, expData.num_states, expData.num_trials);
    
    diary off;
end

