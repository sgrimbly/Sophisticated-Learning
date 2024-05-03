% Ensure a parallel pool is open with a specified number of workers
if isempty(gcp('nocreate'))
    parpool(28);  % Start the parallel pool with 28 workers
end

% Directory and path setup
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir, '../../../MATLAB');
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);

% Define the algorithms and the range of seeds
algorithms = {'SL', 'SI', 'BA', 'BAUCB'};  % Example set of algorithms
numSeeds = 30;  % Total number of seeds
horizon = 1000;  % Example parameter
k_factor = 1.5;  % Example parameter
root_folder = 'defaultFolder';  % Example parameter
mct = 500;  % Example parameter
num_mct = 10;  % Example parameter

% Make sure the directory for output logs exists
outputLogDir = fullfile(currentDir, 'output_logs');
if ~exist(outputLogDir, 'dir')
    mkdir(outputLogDir);
end

% Total number of experiments
totalExperiments = numel(algorithms) * numSeeds;

% Use parfor for parallel execution over all experiments
parfor i = 1:totalExperiments
    % Calculate algorithm index and seed
    [a, seed] = ind2sub([length(algorithms), numSeeds], i);
    algorithm = algorithms{a};
    seedStr = num2str(seed);  % Convert seed to string if necessary

    % Construct output file name
    outputFileName = fullfile(outputLogDir, sprintf('output_%s_seed%d.txt', algorithm, seed));

    % Start capturing output to file
    diary(outputFileName);
    diary on;

    fprintf('Running %s with seed %d...\n', algorithm, seed);
    main(algorithm, seedStr, horizon, k_factor, root_folder, mct, num_mct);

    % Stop capturing output to file
    diary off;
end

fprintf('All experiments completed.\n');
