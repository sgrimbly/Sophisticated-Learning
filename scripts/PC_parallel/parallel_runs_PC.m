% Flag to determine whether to use Weights & Biases
USE_WEIGHTS_AND_BIASES = false;

% Ensure a parallel pool is open with a specified number of workers
if isempty(gcp('nocreate'))
    parpool(16); 
end

% Directory and path setup
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir, '../..');
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);

% Define the new algorithms
algorithms = {'model_free_RL'}; % other algorithms are commented out for clarity

% Set experiment parameters
horizon = 1000;
k_factor = 1.5;
root_folder = 'defaultFolder';
mct = 500;
num_mct = 10;

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
    for s = 1:30
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

% Function to check memory usage
function check_memory()
    [usermem, sysmem] = memory;
    fprintf('Memory used by MATLAB: %g GB\n', usermem.MemUsedMATLAB / 1e9);
    fprintf('Total system memory available: %g GB\n', sysmem.PhysicalMemory.Total / 1e9);
    fprintf('Memory available for data: %g GB\n', sysmem.PhysicalMemory.Available / 1e9);
end

% Use parfor for parallel execution over all experiments
parfor idx = 1:totalExperiments
    expData = experimentsToRun{idx};
    algorithm = expData.algorithm;
    seed = expData.seed;
    seedStr = num2str(seed);  % Convert seed to string if necessary

    % Construct output file name
    outputFileName = fullfile(outputLogDir, fprintf('output_%s_seed%d.txt', algorithm, seed));

    % Start capturing output to file
    diary(outputFileName);
    diary on;

    fprintf('Running %s with seed %d...\n', algorithm, seed);
    try
        % Periodic memory check before execution
        check_memory();

        % Call the main function or specific model function here
        survived = main(algorithm, seedStr, horizon, k_factor, root_folder, mct, num_mct);

        % Log survived data to W&B if needed
        if USE_WEIGHTS_AND_BIASES
            for j = 1:length(survived)
                step_data = py.dict(pyargs('algorithm', algorithm, 'seed', seed, 'trial', j, 'survived', double(survived(j))));
                py.wandb.log(step_data);
            end
        end

        % Periodic memory check after execution
        check_memory();
    catch e
        fprintf('Error occurred in %s with seed %d: %s\n', algorithm, seed, e.message);
        check_memory();  % Check memory also in case of an error
    finally
        % Stop capturing output to file
        diary off;
    end
end

% Close the W&B session if initialized
if USE_WEIGHTS_AND_BIASES
    wb.finish();
end

fprintf('All experiments completed.\n');
