% Flag to determine whether to use Weights & Biases
USE_WEIGHTS_AND_BIASES = false;

% Ensure a parallel pool is open with a specified number of workers
if isempty(gcp('nocreate'))
    parpool(24); 
end

% Directory and path setup
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir, '../../../MATLAB');
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);

% Define the algorithms and their currently running or completed seeds
algorithms = {'SL', 'SI', 'BA', 'BAUCB'};
completedOrRunningSeeds = struct(...
    'SL', [1:16 18:30], ...       % SL from 1 to 16 and 18 to 30, except seed 17
    'SI', [1:30], ...             % SI from 1 to 30 (no issues reported for SI)
    'BA', [1:10 13:18 21 22 29 30], ...   % BA from 1 to 10, 13 to 18, 21, 22, 29, 30
    'BAUCB', [1:5 8] ...          % BAUCB from 1 to 5, 8
);

% Parameters
horizon = 1000;  % Example parameter
k_factor = 1.5;  % Example parameter
root_folder = 'defaultFolder';  % Example parameter
mct = 500;  % Example parameter
num_mct = 10;  % Example parameter

if USE_WEIGHTS_AND_BIASES
    % Initialize Weights & Biases using Python
    try
        py.importlib.import_module('wandb');
        wb = py.wandb.init(pyargs('project','active_inference_agent', 'entity', 'sgrimbly', 'resume', 'allow'));
    catch e
        disp('Failed to load wandb module');
        disp(e.message);
    end
end

% Make sure the directory for output logs exists
outputLogDir = fullfile(currentDir, '../output_logs');
if ~exist(outputLogDir, 'dir')
    mkdir(outputLogDir);
end

% Construct list of experiments to run based on remaining seeds
experimentsToRun = {};
for i = 1:length(algorithms)
    remainingSeeds = setdiff(1:30, completedOrRunningSeeds.(algorithms{i}));
    for s = remainingSeeds
        experimentsToRun{end+1} = struct('algorithm', algorithms{i}, 'seed', s);
    end
end

% Print the intended order of experiments
fprintf('Intended order of experiments:\n');
for idx = 1:length(experimentsToRun)
    fprintf('Experiment %d: %s with seed %d\n', idx, experimentsToRun{idx}.algorithm, experimentsToRun{idx}.seed);
end

% Total number of experiments
totalExperiments = numel(experimentsToRun);

% Use parfor for parallel execution over all experiments
parfor idx = 1:totalExperiments
    expData = experimentsToRun{idx};
    algorithm = expData.algorithm;
    seed = expData.seed;
    seedStr = num2str(seed);  % Convert seed to string if necessary

    % Construct output file name
    outputFileName = fullfile(outputLogDir, sprintf('output_%s_seed%d.txt', algorithm, seed));

    % Start capturing output to file
    diary(outputFileName);
    diary on;

    fprintf('Running %s with seed %d...\n', algorithm, seed);
    try
        survived = main(algorithm, seedStr, horizon, k_factor, root_folder, mct, num_mct);
        
        if USE_WEIGHTS_AND_BIASES
            % Log survived data to W&B
            for j = 1:length(survived)
                step_data = py.dict(pyargs('algorithm', algorithm, 'seed', seed, ...
                                           'trial', j, 'survived', double(survived(j))));
                py.wandb.log(step_data);
            end
        end
    catch e
        fprintf('Error occurred in %s with seed %d: %s\n', algorithm, seed, e.message);
    end

    % Stop capturing output to file
    diary off;
end

if USE_WEIGHTS_AND_BIASES
    % Close the W&B session
    wb.finish();
end

fprintf('All experiments completed.\n');
