function [survived] = main(algorithm, seed, horizon, k_factor, root_folder, mct, num_mct, auto_rest, results_file_name, ...
        grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weights, num_states, num_trials)
    % Check the number of arguments and set default values if necessary
    arguments
        algorithm char {mustBeMember(algorithm, {'model_mixed_RL', 'model_free_RL', 'SL', 'SI', 'BA', 'BAUCB', 'known_large_MCT'})} = 'SI';
        seed (1, 1) double {mustBeInteger} = 1;
        horizon (1, 1) double {mustBeInteger, mustBePositive} = 6;
        k_factor (1, 1) double = 1.5;
        root_folder char = '/home/grmstj001';
        mct (1, 1) double {mustBeInteger} = 0;
        num_mct (1, 1) double {mustBeInteger, mustBePositive} = 100;
        auto_rest (1, 1) logical = false;
        results_file_name char = '';
        grid_size (1, 1) double {mustBeInteger, mustBePositive} = 10;
        start_position (1, 1) double {mustBeInteger, mustBePositive} = 51;
        hill_pos (1, 1) double {mustBeInteger, mustBePositive} = 55;
        food_sources (1, :) double = [71, 43, 57, 78];
        water_sources (1, :) double = [73, 33, 48, 67];
        sleep_sources (1, :) double = [64, 44, 49, 59];
        weights struct = struct('novelty', 10, 'learning', 40, 'epistemic', 1, 'preference', 10);  % Default values for weights
        num_states (1, 1) double {mustBeInteger, mustBePositive} = grid_size^2;
        num_trials (1, 1) double {mustBeInteger, mustBePositive} = 120;
    end
    weight_info = '';
    % Set results_file_name if it was not provided, now incorporating the seed and environment setup
    if isempty(results_file_name)
        weight_info = sprintf('novelty_%d-learning_%d-epistemic_%d-preference_%d', ...
            weights.novelty, weights.learning, weights.epistemic, weights.preference);
        env_info = sprintf('_GS%d_HP%d_FS%s_WS%s_SS%s_W%s_NS%d_NT%d', ...
            grid_size, hill_pos, mat2str(food_sources), mat2str(water_sources), ...
            mat2str(sleep_sources), weight_info, num_states, num_trials);
        env_info = strrep(env_info, ' ', ''); % Remove spaces from the string

        switch algorithm
            case 'model_free_RL'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/results_model_free_RL_Seed%d%s.txt', seed, env_info);
            case 'model_mixed_RL'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/results_model_mixed_RL_Seed%d%s.txt', seed, env_info);
            case 'SL'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/SL-runs/results_SL_Seed%d%s.txt', seed, env_info);
            case 'SI'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/SI-runs/results_SI_Seed%d%s.txt', seed, env_info);
            case 'BA'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/BA-runs/results_BA_Seed%d%s.txt', seed, env_info);
            case 'BAUCB'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/BAUCB-runs/results_BAUCB_Seed%d%s.txt', seed, env_info);
            case 'known_large_MCT'
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/MCT-runs/results_known_large_MCT_Seed%d%s.txt', seed, env_info);
            otherwise
                results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/results_Seed%d%s.txt', seed, env_info);
        end

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
    fprintf('Grid size: %d\n', grid_size);
    fprintf('Hill position: %d\n', hill_pos);
    fprintf('Food sources: %s\n', mat2str(food_sources));
    fprintf('Water sources: %s\n', mat2str(water_sources));
    fprintf('Sleep sources: %s\n', mat2str(sleep_sources));
    fprintf('Weights: %s\n', weight_info);
    fprintf('Number of states: %d\n', num_states);
    fprintf('Number of trials: %d\n', num_trials);

    % Directory and path setup (unchanged)
    currentDir = fileparts(mfilename('fullpath'));
    srcPath = fullfile(currentDir, '../MATLAB');
    fullSrcPath = genpath(srcPath);
    addpath(fullSrcPath);

    % Execute based on the selected algorithm
    survived = 0;

    switch algorithm
        case 'SI'
            disp('Starting SI.');
            survived = SI_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, [weights.novelty, weights.learning, weights.epistemic, weights.preference], num_states, num_trials);
            % survived = SI(seed);
            % SI_rowan(seed);
            disp('SI run complete');
        case 'SL'
            disp('Starting SL.');
            survived = SL_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources,  [weights.novelty, weights.learning, weights.epistemic, weights.preference], num_states, num_trials);
            % survived = SL(seed);
            % SL_rowan(seed);
            disp('SL run complete');
        case 'BA'
            disp('Starting BA.');
            survived = BA_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, num_states, num_trials);
            disp('BA run complete');
        case 'BAUCB'
            disp('Starting BAUCB.');
            survived = BAUCB_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, num_states, num_trials);
            disp('BA_UCB run complete');
        case 'known_large_MCT'
            disp('Starting known_large_MCT.');
            known_large_MCT(seed, horizon, k_factor, root_folder, mct, num_mct, auto_rest);
            disp('Known large MCT run complete');
        case 'model_free_RL'
            disp('Starting model_free_RL.');
            survived = model_free_RL(seed, results_file_name);
            disp('Model free RL run complete');
        case 'model_mixed_RL'
            disp('Starting model mixed.');
            survived = model_mixed_RL(seed, results_file_name);
            disp('Model mixed RL run complete');
        otherwise
            error('Unknown algorithm specified');
    end

end
