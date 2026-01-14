function [survived] = main(algorithm, seed, horizon, k_factor, root_folder, mct, num_mct, auto_rest, results_file_name, ...
    grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weights, num_states, num_trials, grid_id)
    % Check the number of arguments and set default values if necessary
	    arguments
	        algorithm char {mustBeMember(algorithm, { ...
	            'model_mixed_RL', 'model_free_RL', ...
	            'SL', 'SL_noSmooth', 'SL_noNovelty', 'SL_noNovelty_noSmooth', ...
	            'SI', 'SI_smooth', 'SI_novelty', 'SI_novelty_smooth', ...
	            'BA', 'BAUCB', 'known_large_MCT' ...
	        })} = 'SL_noNovelty_noSmooth';
        seed (1, 1) double {mustBeInteger} = 1;
        horizon (1, 1) double {mustBeInteger, mustBePositive} = 9;
        k_factor (1, 1) double = 1.5;
        root_folder char = '/home/grmstj001';
        mct (1, 1) double {mustBeInteger} = 0;
        num_mct (1, 1) double {mustBeInteger, mustBePositive} = 100;
        auto_rest (1, 1) logical = false; % Auto restore
        results_file_name char = '';
        grid_size (1, 1) double {mustBeInteger, mustBePositive} = 10;
        start_position (1, 1) double {mustBeInteger, mustBePositive} = 51;
        hill_pos (1, 1) double {mustBeInteger, mustBePositive} = 55;
        food_sources (1, :) double = [71, 43, 57, 78];
        water_sources (1, :) double = [73, 33, 48, 67];
        sleep_sources (1, :) double = [64, 44, 49, 59];
	        weights struct = struct(...
	            'novelty', 10, ...
	            'learning', 40, ...
	            'epistemic', 1, ...
	            'preference', 10, ...
	            'ucb_scale', 5, ...
	            'state_selection', 'sample', ...
	            'preference_param', 'weight', ...
	            'baucb_variant', 'legacy', ...
	            'real_smoothing', true, ...
	            'adaptive_likelihood_in_plan', false ...
	        );
	        num_states (1, 1) double {mustBeInteger, mustBePositive} = grid_size ^ 2;
	        num_trials (1, 1) double {mustBeInteger, mustBePositive} = 120;
	        grid_id char = '';  % Default empty string if not provided
		    end

	    % Directory and path setup (keep early so helpers like config_hash are on path)
	    currentDir = fileparts(mfilename('fullpath'));
	    srcPath = fullfile(currentDir, '../MATLAB');
	    fullSrcPath = genpath(srcPath);
	    addpath(fullSrcPath);

	    if ~isfield(weights, 'ucb_scale')
	        weights.ucb_scale = 5;
	    end

    if ~isfield(weights, 'state_selection')
        weights.state_selection = 'sample';
    end

    if ~isfield(weights, 'preference_param')
        weights.preference_param = 'weight';
    end

	    if ~isfield(weights, 'baucb_variant')
	        weights.baucb_variant = 'legacy';
	    end

	    if ~isfield(weights, 'real_smoothing')
	        weights.real_smoothing = true;
	    end

	    if ~isfield(weights, 'adaptive_likelihood_in_plan')
	        weights.adaptive_likelihood_in_plan = false;
	    end

		    novelty_for_run = weights.novelty;
			    if ismember(algorithm, {'SI', 'SI_smooth', 'SL_noNovelty', 'SL_noNovelty_noSmooth'})
			        novelty_for_run = 0;
			    end

    % Preference parameterisation:
    %   - preference_param='weight' : larger values strengthen the extrinsic term
    %   - preference_param='inverse_precision' : larger values weaken the extrinsic term
    if strcmp(weights.preference_param, 'inverse_precision')
        if isfield(weights, 'preference_inverse_precision')
            preference_value = weights.preference_inverse_precision;
        else
            preference_value = weights.preference;
        end
	    else
	        preference_value = weights.preference;
	    end

	    weight_vector = [novelty_for_run, weights.learning, weights.epistemic, preference_value];

		    weight_info = '';
		    % Set results_file_name if it was not provided, now incorporating the seed and environment setup
			    if isempty(results_file_name)
			        weight_info = sprintf('novelty_%g-learning_%g-epistemic_%g-preference_%g-prefParam_%s-stateSel_%s-realSmooth_%d-adaptPlan_%d', ...
			            novelty_for_run, weights.learning, weights.epistemic, preference_value, weights.preference_param, weights.state_selection, ...
			            double(logical(weights.real_smoothing)), double(logical(weights.adaptive_likelihood_in_plan)));
			        if strcmp(algorithm, 'BAUCB')
			            weight_info = sprintf('%s-baucbVar_%s-ucb_%g', weight_info, weights.baucb_variant, weights.ucb_scale);
			        end
		        is_unknown_model = ismember(algorithm, {'SI','SI_smooth','SI_novelty','SI_novelty_smooth','SL','SL_noSmooth','SL_noNovelty','SL_noNovelty_noSmooth','BA','BAUCB'});
	        if is_unknown_model
	            grid_id_safe = sanitize_file_component(grid_id);
	            run_config = struct(...
	                'algorithm', algorithm, ...
	                'seed', seed, ...
	                'grid_size', grid_size, ...
	                'start_position', start_position, ...
	                'hill_pos', hill_pos, ...
	                'food_sources', food_sources, ...
	                'water_sources', water_sources, ...
	                'sleep_sources', sleep_sources, ...
		                'weights', weight_vector, ...
		                'state_selection', weights.state_selection, ...
		                'preference_param', weights.preference_param, ...
		                'real_smoothing', logical(weights.real_smoothing), ...
		                'adaptive_likelihood_in_plan', logical(weights.adaptive_likelihood_in_plan), ...
		                'num_states', num_states, ...
		                'num_trials', num_trials, ...
		                'grid_id', grid_id, ...
		                'max_horizon', horizon ...
		            );
	            if strcmp(algorithm, 'BAUCB')
	                run_config.baucb_variant = weights.baucb_variant;
	                run_config.ucb_scale = weights.ucb_scale;
	            end
	            config_id = config_hash(run_config);

	            env_info = sprintf('_GridID_%s_Cfg_%s_SP%d_GS%d_Hor%d_HP%d_FS%s_WS%s_SS%s_W%s_NS%d_NT%d', ...
	                grid_id_safe, config_id, start_position, grid_size, horizon, hill_pos, mat2str(food_sources), mat2str(water_sources), ...
	                mat2str(sleep_sources), weight_info, num_states, num_trials);
	        else
	            env_info = sprintf('_GS%d_Hor%d_HP%d_FS%s_WS%s_SS%s_W%s_NS%d_NT%d', ...
	                grid_size, horizon, hill_pos, mat2str(food_sources), mat2str(water_sources), ...
	                mat2str(sleep_sources), weight_info, num_states, num_trials);
	        end
	        env_info = strrep(env_info, ' ', ''); % Remove spaces from the string

	        % ---------- PORTABLE RESULTS PATH (macOS/Linux) ----------
        % Allow override via environment variable if you want:
        %   export SL_RESULTS_ROOT="/some/path/results"
        results_root = getenv('SL_RESULTS_ROOT');
        if isempty(results_root)
            % Default: put results under the repo root (one level above src/MATLAB)
            thisFileDir  = fileparts(mfilename('fullpath'));         % .../src/MATLAB
            projectRoot  = fullfile(thisFileDir, '..', '..');        % .../ (repo root)
            results_root = fullfile(projectRoot, 'results');
        end
        
	        % Map algorithm -> subfolder + filename prefix
		        switch algorithm
		            case {'model_free_RL','model_mixed_RL'}
		                run_folder = 'RL-runs';
		                file_prefix = ['results_' algorithm];
	            case {'SI','SI_smooth','SI_novelty','SI_novelty_smooth'}
	                run_folder = 'SI-runs';
	                file_prefix = ['results_' algorithm];
	            case {'SL','SL_noSmooth','SL_noNovelty','SL_noNovelty_noSmooth'}
	                run_folder = 'SL-runs';
	                file_prefix = ['results_' algorithm];
            case 'BA'
                run_folder = 'BA-runs';
                file_prefix = 'results_BA';
            case 'BAUCB'
                run_folder = 'BAUCB-runs';
                file_prefix = 'results_BAUCB';
            case 'known_large_MCT'
                run_folder = 'MCT-runs';
                file_prefix = 'results_known_large_MCT';
            otherwise
                run_folder = 'misc-runs';
                file_prefix = ['results_' algorithm];
        end
        
        results_dir = fullfile(results_root, run_folder);
        if ~exist(results_dir, 'dir')
            mkdir(results_dir);
        end
        
        results_file_name = fullfile(results_dir, sprintf('%s_Seed%d%s.txt', file_prefix, seed, env_info));
        % --------------------------------------------------------

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
	    fprintf('Start position: %d\n', start_position);
	    fprintf('Hill position: %d\n', hill_pos);
	    fprintf('Food sources: %s\n', mat2str(food_sources));
	    fprintf('Water sources: %s\n', mat2str(water_sources));
	    fprintf('Sleep sources: %s\n', mat2str(sleep_sources));
    fprintf('Weights: %s\n', weight_info);
    fprintf('Number of states: %d\n', num_states);
    fprintf('Number of trials: %d\n', num_trials);

		    % Execute based on the selected algorithm
		    survived = 0;
			    run_options = struct(...
			        'state_selection', weights.state_selection, ...
			        'preference_param', weights.preference_param, ...
			        'baucb_variant', weights.baucb_variant, ...
			        'real_smoothing', logical(weights.real_smoothing), ...
			        'adaptive_likelihood_in_plan', logical(weights.adaptive_likelihood_in_plan), ...
		        'algorithm_label', algorithm ...
		    );

    switch algorithm
	        case 'SI'
	            disp('Starting SI.');
	            survived = SI_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            % survived = SI(seed);
	            % SI_rowan(seed);
	            disp('SI run complete');
	        case 'SI_novelty'
	            disp('Starting SI_novelty.');
	            survived = SI_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('SI_novelty run complete');
	        case 'SL'
	            disp('Starting SL.');
	            survived = SL_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources,  weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            % survived = SL(seed);
	            % SL_rowan(seed);
	            disp('SL run complete');
	        case 'SL_noNovelty'
	            disp('Starting SL_noNovelty.');
	            survived = SL_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources,  weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('SL_noNovelty run complete');
	        case 'SL_noSmooth'
	            disp('Starting SL_noSmooth.');
	            survived = SL_noSmooth_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('SL_noSmooth run complete');
	        case 'SL_noNovelty_noSmooth'
	            disp('Starting SL_noNovelty_noSmooth.');
	            survived = SL_noSmooth_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('SL_noNovelty_noSmooth run complete');
	        case 'BA'
	            disp('Starting BA.');
	            survived = BA_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('BA run complete');
        case 'BAUCB'
            disp('Starting BAUCB.');
            survived = BAUCB_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, weights.ucb_scale, results_file_name, horizon, run_options);
            disp('BA_UCB run complete');
	        case 'SI_smooth'
	            disp('Starting SI_smooth.');
	            survived = SI_smooth_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('SI_smooth run complete');
	        case 'SI_novelty_smooth'
	            disp('Starting SI_novelty_smooth.');
	            survived = SI_smooth_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weight_vector, num_states, num_trials, grid_id, results_file_name, horizon, run_options);
	            disp('SI_novelty_smooth run complete');
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
