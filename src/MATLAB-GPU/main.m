function [survived] = main(algorithm, seed, results_file_name, num_trials)
% MAIN - Simplified GPU-accelerated version for running SI
%
% Inputs:
%   algorithm - the algorithm to run ('SI' supported)
%   seed - random seed for reproducibility
%   results_file_name - optional filename for results logging
%   num_trials - number of trials to run
%
% Output:
%   survived - results from SI simulation

% Default arguments with validation
arguments
    algorithm char {mustBeMember(algorithm, {'SI'})} = 'SI';
    seed (1,1) double {mustBeInteger} = 120;
    results_file_name char = '';
    num_trials (1,1) double {mustBeInteger, mustBePositive} = 100;
end

% Set results_file_name if not provided
if isempty(results_file_name)
    results_file_name = sprintf('results_SI_GPU_Seed%d.txt', seed);
end

% Display execution settings
fprintf('Running %s with Seed: %d, Trials: %d\n', algorithm, seed, num_trials);
fprintf('Results will be saved in: %s\n', results_file_name);

% Ensure GPU usage
gpuDevice(); % Initialize GPU device

% Run the selected algorithm
survived = 0;
if strcmp(algorithm, 'SI')
    disp('Starting SI on GPU...');
    tic;
    survived = SI_GPU(seed, num_trials);
    toc;
    disp('SI execution complete.');
end

end
