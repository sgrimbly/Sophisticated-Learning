function [survived] = main(algorithm, seed, horizon, k_factor, root_folder, mct, num_mct, auto_rest, results_file_name)
    % Check the number of arguments and set default values if necessary
    arguments
        algorithm char {mustBeMember(algorithm,{'model_mixed_RL','model_free_RL','SL','SI','BA','BAUCB','known_large_MCT'})} = 'model_mixed_RL';
        seed (1,1) double {mustBeInteger} = 1;  % Seed MUST be an integer
        horizon (1,1) double {mustBeInteger, mustBePositive} = 1000;
        k_factor (1,1) double = 1.5;
        root_folder char = '/home/grmstj001';
        mct (1,1) double {mustBeInteger, mustBePositive} = 500;
        num_mct (1,1) double {mustBeInteger, mustBePositive} = 10;
        auto_rest (1,1) logical = false; % Default is false, meaning memory is usually enabled
        results_file_name char = ''; % Will be set later if empty
    end

    % Print out the values of the arguments
    fprintf('Algorithm: %s\n', algorithm);
    fprintf('Seed: %d\n', seed);
    fprintf('Horizon: %d\n', horizon);
    fprintf('K-factor: %.2f\n', k_factor);
    fprintf('Root folder: %s\n', root_folder);
    fprintf('MCT: %d\n', mct);
    fprintf('Number of MCT: %d\n', num_mct);
    fprintf('Auto restore: %d\n', auto_rest);
    fprintf('Results file name: %s\n', results_file_name);

    % Set results_file_name if it was not provided, now incorporating the seed
    if isempty(results_file_name) 
        switch algorithm
            case 'model_free_RL'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/results_model_free_RL_Seed%d.txt', seed);
            case 'model_mixed_RL'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/results_model_mixed_RL_Seed%d.txt', seed);
            otherwise
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/results_Seed%d.txt', seed); 
        end
    end
    
    % Directory and path setup (unchanged)
    currentDir = fileparts(mfilename('fullpath'));
    srcPath = fullfile(currentDir, '../MATLAB');
    fullSrcPath = genpath(srcPath);
    addpath(fullSrcPath);

    % Execute based on the selected algorithm
    survived = 0;
    switch algorithm
        case 'SL'
            survived = SL(seed);
            disp('SL run complete');
        case 'SI'
            survived = SI(seed);
            disp('SI run complete');
        case 'BA'
            survived = BA(seed);
            disp('BA run complete');
        case 'BAUCB'
            survived = BA_UCB(seed);
            disp('BA_UCB run complete');
        case 'known_large_MCT'
            known_large_MCT(seed, horizon, k_factor, root_folder, mct, num_mct, auto_rest);
            disp('Known large MCT run complete');
        case 'model_free_RL'
            survived = model_free_RL(seed, results_file_name);
            disp('Model free RL run complete');
        case 'model_mixed_RL'
            disp('Starting model mixed.')
            survived = model_mixed_RL(seed, results_file_name);
            disp('Model mixed RL run complete');
        otherwise
            error('Unknown algorithm specified');
    end
end
