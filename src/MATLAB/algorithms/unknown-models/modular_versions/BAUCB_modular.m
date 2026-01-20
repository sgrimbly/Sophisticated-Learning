function [survived] = BAUCB_modular(seed, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, weights, num_states, num_trials, grid_id, ucb_scale, results_file_override, max_horizon, run_options)
    % Set default values if not provided
    if nargin < 2, grid_size = 10; end
    if nargin < 3, start_position = 51; end  % Default start position set to 51
    if nargin < 4, hill_pos = 55; end
    if nargin < 5, food_sources = [71, 43, 57, 78]; end
    if nargin < 6, water_sources = [73, 33, 48, 67]; end
    if nargin < 7, sleep_sources = [64, 44, 49, 59]; end
    if nargin < 8, weights = [10, 40, 1, 10]; end
    if nargin < 9, num_states = 100; end  % Assumes grid size 10x10, aligns with default grid_size
    if nargin < 10, num_trials = 200; end
    if nargin < 11, grid_id = ''; end
    if nargin < 12, ucb_scale = 5; end
    if nargin < 13, results_file_override = ''; end
    if nargin < 14, max_horizon = 9; end
    if nargin < 15, run_options = struct(); end

    if ~isstruct(run_options)
        error('run_options must be a struct.');
    end
    if ~isfield(run_options, 'state_selection')
        run_options.state_selection = 'sample';
    end
    if ~isfield(run_options, 'preference_param')
        run_options.preference_param = 'weight';
    end
    if ~isfield(run_options, 'real_smoothing')
        run_options.real_smoothing = true;
    end
    if ~isfield(run_options, 'adaptive_likelihood_in_plan')
        run_options.adaptive_likelihood_in_plan = false;
    end
    if ~isfield(run_options, 'learning_prune_threshold')
        run_options.learning_prune_threshold = 0.2;
    end
    if ~isfield(run_options, 'baucb_variant')
        run_options.baucb_variant = 'legacy';
    end
    if ~isfield(run_options, 'progress_queue')
        run_options.progress_queue = [];
    end

    current_time = char(datetime('now', 'Format', 'HH-mm-ss-SSS'));  % This should be safe, ensure there are no colons

    results_root = getenv('SL_RESULTS_ROOT');
    if isempty(results_root)
        thisFileDir = fileparts(mfilename('fullpath')); % .../src/MATLAB/algorithms/unknown-models/modular_versions
        projectRoot = fullfile(thisFileDir, '..', '..', '..', '..', '..');
        results_root = fullfile(projectRoot, 'results');
    end

    directory_path = fullfile(results_root, 'unknown_model', 'MATLAB', 'grid_config_experiments');
    if ~exist(directory_path, 'dir')
        [ok, msg] = mkdir(directory_path);
        if ~ok
            error('Unable to create results directory: %s', msg);
        end
    end
    food_str = strjoin(arrayfun(@num2str, food_sources, 'UniformOutput', false), '-');
    water_str = strjoin(arrayfun(@num2str, water_sources, 'UniformOutput', false), '-');
    sleep_str = strjoin(arrayfun(@num2str, sleep_sources, 'UniformOutput', false), '-');

    % Initialize environment once, outside of any saved state check
    [A, a, B, b, D, T, num_modalities] = initialiseEnvironment(num_states, start_position, grid_size, hill_pos, food_sources, water_sources, sleep_sources);

    num_contexts = numel(D{2});
    num_joint_states = num_states * num_contexts;

    time_since_food = 0;
    time_since_water = 0;
    time_since_sleep = 0;

    grid_id_safe = sanitize_file_component(grid_id);
    run_config = struct(...
        'algorithm', 'BAUCB', ...
        'seed', seed, ...
        'grid_size', grid_size, ...
        'start_position', start_position, ...
        'hill_pos', hill_pos, ...
        'food_sources', food_sources, ...
        'water_sources', water_sources, ...
        'sleep_sources', sleep_sources, ...
        'weights', weights, ...
        'state_selection', run_options.state_selection, ...
        'preference_param', run_options.preference_param, ...
        'real_smoothing', logical(run_options.real_smoothing), ...
        'adaptive_likelihood_in_plan', logical(run_options.adaptive_likelihood_in_plan), ...
        'learning_prune_threshold', run_options.learning_prune_threshold, ...
        'baucb_variant', run_options.baucb_variant, ...
        'num_states', num_states, ...
        'num_trials', num_trials, ...
        'grid_id', grid_id, ...
        'ucb_scale', ucb_scale, ...
        'max_horizon', max_horizon ...
    );
    config_id = config_hash(run_config);
    run_meta = struct('config_id', config_id, 'run_config', run_config);

    % Define file path for state and results
    result_file = fullfile(directory_path, sprintf('BAUCB_Seed_%d_GridID_%s_Cfg_%s_%s.txt', seed, grid_id_safe, config_id, current_time));
    stateFile = fullfile(directory_path, sprintf('BAUCB_Seed_%d_GridID_%s_Cfg_%s.mat', seed, grid_id_safe, config_id));

    if ~isempty(results_file_override)
        result_file = results_file_override;
        [results_dir, result_base, result_ext] = fileparts(result_file);
        if isempty(results_dir)
            results_dir = pwd;
            result_file = fullfile(results_dir, [result_base, result_ext]);
        end
        if ~exist(results_dir, 'dir')
            [ok, msg] = mkdir(results_dir);
            if ~ok
                error('Unable to create results directory: %s', msg);
            end
        end
        stateFile = fullfile(results_dir, sprintf('BAUCB_Seed_%d_GridID_%s_Cfg_%s.mat', seed, grid_id_safe, config_id));
    end

    log_metrics = ~isempty(getenv('SL_LOG_METRICS')) && ~strcmp(getenv('SL_LOG_METRICS'), '0');
    metrics_file = '';
    if log_metrics
        [metrics_dir, metrics_base, ~] = fileparts(result_file);
        metrics_file = fullfile(metrics_dir, sprintf('%s_metrics.csv', metrics_base));
    end

    [loadedState, isNew] = load_state(stateFile);

    t = 1;
    surety = 1;
    simulated_time = 0;
    preference_value = weights(4);
    switch run_options.preference_param
        case 'inverse_precision'
            preference_inverse_precision = preference_value;
        case 'weight'
            if preference_value == 0
                preference_inverse_precision = Inf;
            else
                preference_inverse_precision = 1 / preference_value;
            end
        otherwise
            error('Unknown run_options.preference_param "%s". Expected ''weight'' or ''inverse_precision''.', run_options.preference_param);
    end
    use_fixed_joint_counts = strcmp(run_options.baucb_variant, 'fixed_joint_counts');

    if ~isNew
        % Load variables from the saved state, using indices to access the cell array
        rng(loadedState{1});       % Restore the RNG state
        trial = loadedState{2} + 1; % Start from the next trial to ensure continuity
        
        a_history = loadedState{3}; % Retrieve a_history
        a = a_history{trial-1};          % Access the last valid entry in a_history
        
        b_history = loadedState{4}; % Retrieve b_history
        b = b_history{trial-1};          % Access the last valid entry in b_history
        
        Q = loadedState{5};          % Retrieve Q
        P = loadedState{6};          % Retrieve P
        
        true_states = loadedState{7}; % Retrieve true_states
        
        chosen_action = loadedState{8}; % Retrieve chosen_action
        memory_resets = loadedState{9}; % Retrieve memory_resets
        pe_memory_resets = loadedState{10}; % Retrieve pe_memory_resets
        hill_memory_resets = loadedState{11}; % Retrieve hill_memory_resets
        
        total_search_depth = loadedState{12}; % Retrieve total_search_depth
        total_memory_accessed = loadedState{13}; % Retrieve total_memory_accessed
        total_t = loadedState{14}; % Retrieve total_t
        survived = loadedState{15}; % Retrieve survived
        
        t_at_25 = loadedState{16};  % Retrieve t_at_25
        t_at_50 = loadedState{17};  % Retrieve t_at_50
        t_at_75 = loadedState{18};  % Retrieve t_at_75
        t_at_100 = loadedState{19}; % Retrieve t_at_100

        result_file = loadedState{20}

        if numel(loadedState) >= 21
            Nt = loadedState{21};
        elseif use_fixed_joint_counts
            Nt = ones(num_states, num_contexts);
        else
            % Legacy mode: match original code (ones(400)) to preserve linear-indexing behaviour.
            Nt = ones(num_joint_states);
        end

        if ~use_fixed_joint_counts && isvector(Nt) && numel(Nt) == num_joint_states
            Nt_legacy = Nt(:);
            Nt = ones(num_joint_states);
            Nt(1:num_joint_states) = Nt_legacy;
        end

        if numel(loadedState) >= 22
            loaded_meta = loadedState{22};
            if isstruct(loaded_meta) && isfield(loaded_meta, 'config_id') && ~strcmp(loaded_meta.config_id, config_id)
                error('Loaded state config_id does not match current run_config.');
            end
        end
    else
        % Initialization of variables for a new simulation
        rng(seed, 'twister') % Set the initial random state
        trial = 1;
    
        a_history = cell(1, num_trials);
        b_history = cell(1, num_trials);
        if use_fixed_joint_counts
            Nt = ones(num_states, num_contexts);
        else
            % Legacy mode: match original code (ones(400)) to preserve linear-indexing behaviour.
            Nt = ones(num_joint_states);
        end
        chosen_action = zeros(1, T - 1);
        memory_resets = zeros(num_trials, 1);
        pe_memory_resets = zeros(num_trials, 1);
        hill_memory_resets = zeros(num_trials, 1);
        total_search_depth = 0;
        total_memory_accessed = 0;
        total_t = 0;
        survived = zeros(1, num_trials);
        t_at_25 = 0;
        t_at_50 = 0;
        t_at_75 = 0;
        t_at_100 = 0;
    end

    total_startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    for trial = trial:num_trials
        startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        fprintf('\n----------------------------------------\n');
        fprintf('TRIAL %d STARTED\n', trial);
        fprintf('----------------------------------------\n');
        fprintf('Start Time: %s\n', startTime);

        short_term_memory = zeros(35, 35, 35, num_joint_states, 5);
        search_depth = 0;
        memory_accessed = 0;
        param_update_kl_sum = 0;

        for factor = 1:2
            Q{1, factor} = D{factor}';
            P{1, factor} = D{factor}';
            true_states{trial}(1, t) = start_position;
            true_states{trial}(2, t) = find(cumsum(D{2}) >= rand, 1);
        end

        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
            bb{2} = normalise_matrix(b{2});

            if t ~= 1
                [P, Q, true_states] = updateEnvironmentStates(P, Q, true_states, trial, t, chosen_action, B, bb);
            end

            % NOTE: ismember is probably not working here
            % if ismember(true_states{trial}(2, t), 1:4) && ismember(true_states{trial}(1, t), food_sources)
            %     time_since_food = 0;
            %     time_since_water = time_since_water + 1;
            %     time_since_sleep = time_since_sleep + 1;
            % elseif ismember(true_states{trial}(2, t), 1:4) && ismember(true_states{trial}(1, t), water_sources)
            %     time_since_water = 0;
            %     time_since_food = time_since_food + 1;
            %     time_since_sleep = time_since_sleep + 1;
            % elseif ismember(true_states{trial}(2, t), 1:4) && ismember(true_states{trial}(1, t), sleep_sources)
            %     time_since_sleep = 0;
            %     time_since_food = time_since_food + 1;
            %     time_since_water = time_since_water + 1;
            % else
            %     if t > 1
            %         time_since_food = time_since_food + 1;
            %         time_since_water = time_since_water + 1;
            %         time_since_sleep = time_since_sleep + 1;
            %     end
            % end
            if (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == food_sources(1)) || (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == food_sources(2)) || (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == food_sources(3)) || (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == food_sources(4))
                time_since_food = 0;
                time_since_water = time_since_water + 1;
                time_since_sleep = time_since_sleep + 1;

            elseif (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == water_sources(1)) || (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == water_sources(2)) || (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == water_sources(3)) || (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == water_sources(4))
                time_since_water = 0;
                time_since_food = time_since_food + 1;
                time_since_sleep = time_since_sleep + 1;

            elseif (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == sleep_sources(1)) || (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == sleep_sources(2)) || (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == sleep_sources(3)) || (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == sleep_sources(4))
                time_since_sleep = 0;
                time_since_food = time_since_food + 1;
                time_since_water = time_since_water + 1;

            else
                if t > 1
                    time_since_food = time_since_food + 1;
                    time_since_water = time_since_water + 1;
                    time_since_sleep = time_since_sleep + 1;
                end
            end

            for modality = 1:num_modalities
                ob = A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t));
                observations(modality, t) = find(cumsum(A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t))) >= rand, 1);
                vec = zeros(1, size(A{modality}, 1));
                vec(1, observations(modality, t)) = 1;
                O{modality, t} = vec;
            end

            true_t = t;

            if t > 1
                bb{2} = normalise_matrix(b{2});
                y{2} = normalise_matrix(a{2});
                qs = spm_cross(Q{t, :});
                predictive_observations_posterior{2, t} = normalise(y{2}(:, :) * qs(:))';
                predictive_observations_posterior{3, t} = normalise(y{3}(:, :) * qs(:))';
                predicted_posterior = calculate_posterior(Q, y, predictive_observations_posterior, t);
                a_prior_step = a{2};

                if run_options.real_smoothing
                    start = t - 6;
                    if start <= 0, start = 1; end

                    for timey = start:t
                        L = spm_backwards(O, Q, A, bb, chosen_action, timey, t);
                        LL{2} = L;
                        LL{1} = Q{timey, 1};

                        if (timey > start && ~isequal(round(L, 3), round(Q{timey, 2}, 3)')) || (timey == t)
                            for modality = 2:2
                                a_learning = O(modality, timey)';
                                for factor = 1:2
                                    a_learning = spm_cross(a_learning, LL{factor});
                                end
                                a_learning = a_learning .* (a{modality} > 0);
                                proportion = 0.3;
                                for i = 1:size(a_learning, 3)
                                    for j = 1:size(a_learning, 2)
                                        max_value = max(a_learning(2:end, j, i));
                                        amount_to_subtract = proportion * max_value;
                                        a_learning(a_learning(1, j, i) == 0, j, i) = a_learning(a_learning(1, j, i) == 0, j, i) - amount_to_subtract;
                                    end
                                end
                                a{modality} = a{modality} + 0.7 * a_learning;
                                a{modality}(a{modality} <= 0.05) = 0.05;
                            end
                        end
                    end
                else
                    timey = t;
                    LL{2} = Q{timey, 2}';
                    LL{1} = Q{timey, 1};

                    for modality = 2:2
                        a_learning = O(modality, timey)';
                        for factor = 1:2
                            a_learning = spm_cross(a_learning, LL{factor});
                        end
                        a_learning = a_learning .* (a{modality} > 0);
                        proportion = 0.3;
                        for i = 1:size(a_learning, 3)
                            for j = 1:size(a_learning, 2)
                                max_value = max(a_learning(2:end, j, i));
                                amount_to_subtract = proportion * max_value;
                                a_learning(a_learning(1, j, i) == 0, j, i) = a_learning(a_learning(1, j, i) == 0, j, i) - amount_to_subtract;
                            end
                        end
                        a{modality} = a{modality} + 0.7 * a_learning;
                        a{modality}(a{modality} <= 0.05) = 0.05;
                    end
                end

                param_update_kl_sum = param_update_kl_sum + kldir(normalise(a{2}(:)), normalise(a_prior_step(:)));
            end

           if true_states{trial}(2, t) == 1
                food = food_sources(1);
                water = water_sources(1);
                sleep = sleep_sources(1);
            elseif true_states{trial}(2, t) == 2
                food = food_sources(2);
                water = water_sources(2);
                sleep = sleep_sources(2);
            elseif true_states{trial}(2, t) == 3
                food = food_sources(3);
                water = water_sources(3);
                sleep = sleep_sources(3);
            else
                food = food_sources(4);
                water = water_sources(4);
                sleep = sleep_sources(4);
            end

            y{2} = normalise_matrix(a{2});
            y{1} = A{1};
            y{3} = A{3};
            % prefs = determineObservationPreference(time_since_food, time_since_water, time_since_sleep);
            % time_since_resources = [time_since_food, time_since_water, time_since_sleep];
            horizon = min([max_horizon, min([22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep])]);
            if horizon == 0, horizon = 1; end

            temp_Q = Q;
            temp_Q{t, 2} = temp_Q{t, 2}';
            P = calculate_posterior(temp_Q, y, O, t);
            long_term_memory = 0;
            % trajectory = [];
            a_complexity = 0;
            current_pos(t) = select_from_posterior(P{t, 1}, run_options.state_selection);
            optimal_traj = [];

            if t > 1 && ~isequal(round(predicted_posterior{t, 2}, 1), round(P{t, 2}, 1))
                short_term_memory(:, :, :, :, :) = 0;
                memory_resets(trial) = memory_resets(trial) + 1;
                pe_memory_resets(trial) = pe_memory_resets(trial) + 1;
            end

            if current_pos(t) == hill_pos
                short_term_memory(:, :, :, :, :) = 0;
                memory_resets(trial) = memory_resets(trial) + 1;
                hill_memory_resets(trial) = hill_memory_resets(trial) + 1;
            end

            best_actions = [];
            if use_fixed_joint_counts
                current_context = select_from_posterior(P{t, 2}, run_options.state_selection);
                current_joint_state = sub2ind([num_states, num_contexts], current_pos(t), current_context);
                [G, Q, D, short_term_memory, long_term_memory, optimal_traj, best_actions, Nt, memory_accessed] = tree_search_frwd_UCB(long_term_memory, short_term_memory, O, Q, a, A, y, D, B, B, t, T, t + horizon, time_since_food, time_since_water, time_since_sleep, time_since_food, time_since_water, time_since_sleep, current_joint_state, true_t, chosen_action, a_complexity, surety, simulated_time, time_since_food, time_since_water, time_since_sleep, 0, optimal_traj, best_actions, Nt, memory_accessed, preference_inverse_precision, ucb_scale);
            else
                cur_state_for_Nt = select_from_posterior(P{t, 1}, run_options.state_selection);
                Nt(cur_state_for_Nt) = Nt(cur_state_for_Nt) + 1;
                [G, Q, D, short_term_memory, long_term_memory, optimal_traj, best_actions, memory_accessed] = tree_search_frwd_UCB_legacy(long_term_memory, short_term_memory, O, Q, a, A, y, D, B, B, t, T, t + horizon, time_since_food, time_since_water, time_since_sleep, time_since_food, time_since_water, time_since_sleep, current_pos(t), true_t, chosen_action, a_complexity, surety, simulated_time, time_since_food, time_since_water, time_since_sleep, 0, optimal_traj, best_actions, Nt, memory_accessed, ucb_scale, run_options.state_selection);
            end

            chosen_action(t) = best_actions(1);
            t = t + 1;
            search_depth = search_depth + length(best_actions);
        end

        total_search_depth = total_search_depth + search_depth;
        total_memory_accessed = total_memory_accessed + memory_accessed;
        total_t = total_t + t;

        if t >= 25 && t < 50
            t_at_25 = t_at_25 + 1;
        elseif t >= 50 && t < 75
            t_at_50 = t_at_50 + 1;
        elseif t >= 75 && t < 100
            t_at_75 = t_at_75 + 1;
        elseif t >= 100
            t_at_100 = t_at_100 + 1;
        end

        fid = fopen(result_file, 'a+');
        if fid < 0
            error('Failed to open results file: %s', result_file);
        end
        fprintf(fid, '%f\n', t);
        fclose(fid);

        survived(trial) = t;
        append_trial_metrics(metrics_file, run_meta, trial, t, param_update_kl_sum, a{2}, food_sources, water_sources, sleep_sources);
        if ~isempty(run_options.progress_queue)
            try
                send(run_options.progress_queue, struct(...
                    'algorithm', 'BAUCB', ...
                    'trial', trial, ...
                    'survival', t, ...
                    'param_update_kl', param_update_kl_sum, ...
                    'efe_novelty_term_sum', 0, ...
                    'efe_epistemic_term_sum', 0, ...
                    'efe_extrinsic_term_sum', 0, ...
                    'efe_future_term_sum', 0, ...
                    'efe_node_count', 0, ...
                    'search_depth', search_depth, ...
                    'memory_accessed', memory_accessed, ...
                    'memory_resets', memory_resets(trial), ...
                    'pe_memory_resets', pe_memory_resets(trial), ...
                    'hill_memory_resets', hill_memory_resets(trial) ...
                ));
            catch
            end
        end

        endTime = datestr(now + 1/24/60/60, 'yyyy-mm-dd HH:MM:SS');
        totalRuntimeInSeconds = etime(datevec(endTime), datevec(startTime));
        minutes = floor(mod(totalRuntimeInSeconds, 3600) / 60);
        seconds = mod(totalRuntimeInSeconds, 60);

        fprintf('At time step %d the agent is dead\n', t - 1);
        fprintf('The agent had %d food, %d water, and %d sleep.\n', 22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep);
        fprintf('The total tree search depth for this trial was %d. \n', search_depth);
        fprintf('The agent accessed its memory %d times. \n', memory_accessed);
        fprintf('The agent cleared its short-term memory %d times. \n', memory_resets(trial));
        fprintf('     State prediction error memory resets: %d. \n', pe_memory_resets(trial));
        fprintf('     Hill memory resets: %d. \n', hill_memory_resets(trial));
        fprintf('TRIAL %d COMPLETE ?\n', trial);
        fprintf('End Time: %s\n', endTime);
        fprintf('Total runtime for this trial (minutes/seconds): %02d:%02d\n', minutes, seconds);
        fprintf('----------------------------------------\n');
        fprintf('Total hill visits: %d. \n', sum(hill_memory_resets(:)));
        fprintf('Total prediction errors: %d. \n', sum(pe_memory_resets(:)));
        fprintf('Total search depth: %d. \n', sum(total_search_depth));
        fprintf('Total times memory accessed: %d. \n', total_memory_accessed);
        fprintf('Total times 25 >= t <= 50: %d. \n', t_at_25);
        fprintf('Total times 50 >= t <= 75: %d. \n', t_at_50);
        fprintf('Total times 75 >= t <= 100: %d. \n', t_at_75);
        fprintf('Total times t == 100: %d. \n', t_at_100);
        fprintf('Total time steps survived: %d. \n', total_t);
        totalRuntimeInSeconds = etime(datevec(endTime), datevec(total_startTime));
        hours = floor(totalRuntimeInSeconds / 3600);
        minutes = floor(mod(totalRuntimeInSeconds, 3600) / 60);
        seconds = mod(totalRuntimeInSeconds, 60);
        fprintf('Total runtime so far (hours/minutes/seconds): %02d:%02d:%02d\n', hours, minutes, seconds);
        fprintf('----------------------------------------\n');

        % Update variable histories at the end of each trial
        a_history{trial} = a;
        b_history{trial} = b;
        
        % Prepare the state to save (end of each trial)
        currentState = {rng, trial, a_history, b_history, Q, P, true_states, ...
                        chosen_action, memory_resets, pe_memory_resets, hill_memory_resets, ...
                        total_search_depth, total_memory_accessed, total_t, survived, ...
                        t_at_25, t_at_50, t_at_75, t_at_100, result_file, Nt, run_meta};

        % Save the state at the end of each trial
        save_state(stateFile, currentState);

        t = 1;
        time_since_food = 0;
        time_since_water = 0;
        time_since_sleep = 0;
    end

    total_endTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    totalRuntimeInSeconds = etime(datevec(total_endTime), datevec(total_startTime));
    hours = floor(totalRuntimeInSeconds / 3600);
    minutes = floor(mod(totalRuntimeInSeconds, 3600) / 60);
    seconds = mod(totalRuntimeInSeconds, 60);
    fprintf('EXPERIMENT COMPLETE ?\n');
    fprintf('End Time: %s\n', total_endTime);
    fprintf('TOTAL RUNTIME (hours/minutes/seconds): %02d:%02d:%02d\n', hours, minutes, seconds);
    fprintf('AVERAGE RUNTIME PER TIME STEP: %.3f seconds\n', totalRuntimeInSeconds / total_t);
    fprintf('Average hill visits per time step: %.3f. \n', sum(hill_memory_resets(:)) / total_t);
    fprintf('Average prediction errors per time step: %.3f. \n', sum(pe_memory_resets(:)) / total_t);
    fprintf('Average search depth per time step: %.0f. \n', sum(total_search_depth(:)) / total_t);
    fprintf('Average times memory accessed per time step: %.0f. \n', total_memory_accessed / total_t);
    fprintf('----------------------------------------\n');
end
