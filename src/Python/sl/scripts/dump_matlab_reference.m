function dump_matlab_reference(out_path, algorithm, seed, num_trials, max_horizon, grid_size, ...
                                start_position, hill_pos, food_sources, water_sources, sleep_sources, weights)
% DUMP_MATLAB_REFERENCE  Run a tiny MATLAB simulation and save a reference trace.
%
% Produces a .mat file with per-step ground-truth state for the Python port to
% validate against. Uses the canonical initialiseEnvironment / SI_modular /
% SL_modular pipeline, but instruments only enough to capture per-step values
% without modifying those files.
%
% USAGE
%     dump_matlab_reference()                               % defaults
%     dump_matlab_reference('refs/si_seed1.mat', 'SI', 1, 2, 4)
%
% OUTPUTS (saved to out_path)
%   meta            : config struct (seed, algorithm, grid params, weights)
%   B1, B2, A1, A2, A3 : initial generative tensors (1-based MATLAB convention)
%   a1_init, a2_init, a3_init : initial agent Dirichlet counts
%   D1, D2          : initial state priors
%   per_trial(trial).true_states  : 2 x t array, 1-based positions and contexts
%   per_trial(trial).observations : 3 x t array of 1-based observation indices
%   per_trial(trial).chosen_action: 1 x t array, 1-based action indices
%   per_trial(trial).Q_pos        : t x num_states (predictive position posterior)
%   per_trial(trial).Q_ctx        : t x num_contexts (predictive context posterior)
%   per_trial(trial).P_pos        : t x num_states (post-observation position posterior)
%   per_trial(trial).P_ctx        : t x num_contexts (post-observation context posterior)
%   per_trial(trial).a2_snapshots : (t+1) x 4 x num_states x 4 (a{2} after each step)
%   per_trial(trial).horizon      : 1 x t planning horizons
%   per_trial(trial).best_actions : cell{t} of vectors, the planner's best_actions list
%   per_trial(trial).memory_resets_pe / _hill / _total
%   per_trial(trial).t_terminal   : terminal timestep
%   per_trial(trial).survived     : did agent survive trial
%
% The Python validator loads this via scipy.io.loadmat and asserts
% statistical / structural parity (not bit-exactness across RNG streams).

    if nargin < 1 || isempty(out_path),     out_path     = 'matlab_reference.mat'; end
    if nargin < 2 || isempty(algorithm),    algorithm    = 'SI'; end
    if nargin < 3 || isempty(seed),         seed         = 1; end
    if nargin < 4 || isempty(num_trials),   num_trials   = 2; end
    if nargin < 5 || isempty(max_horizon),  max_horizon  = 4; end
    if nargin < 6 || isempty(grid_size),    grid_size    = 10; end
    if nargin < 7 || isempty(start_position), start_position = 51; end
    if nargin < 8 || isempty(hill_pos),     hill_pos     = 55; end
    if nargin < 9 || isempty(food_sources), food_sources = [71, 43, 57, 78]; end
    if nargin < 10 || isempty(water_sources), water_sources = [73, 33, 48, 67]; end
    if nargin < 11 || isempty(sleep_sources), sleep_sources = [64, 44, 49, 59]; end
    if nargin < 12 || isempty(weights),     weights      = [10, 40, 1, 10]; end

    % Add the canonical MATLAB tree to path
    thisDir = fileparts(mfilename('fullpath'));
    matlabRoot = fullfile(thisDir, '..', '..', '..', 'MATLAB');
    if exist(matlabRoot, 'dir')
        addpath(genpath(matlabRoot));
    else
        error('MATLAB source tree not found at %s', matlabRoot);
    end

    num_states = grid_size ^ 2;
    rng(seed, 'twister');

    % --- Initial environment & beliefs (1-based, MATLAB-native) -------------
    [A, a, B, b, D, T, num_modalities] = initialiseEnvironment( ...
        num_states, start_position, grid_size, hill_pos, ...
        food_sources, water_sources, sleep_sources);

    meta = struct( ...
        'seed', seed, ...
        'algorithm', algorithm, ...
        'num_trials', num_trials, ...
        'max_horizon', max_horizon, ...
        'grid_size', grid_size, ...
        'num_states', num_states, ...
        'start_position', start_position, ...
        'hill_pos', hill_pos, ...
        'food_sources', food_sources, ...
        'water_sources', water_sources, ...
        'sleep_sources', sleep_sources, ...
        'weights', weights, ...
        'T', T, ...
        'num_modalities', num_modalities);

    save_struct.meta       = meta;
    save_struct.B1         = B{1};
    save_struct.B2         = B{2};
    save_struct.A1         = A{1};
    save_struct.A2         = A{2};
    save_struct.A3         = A{3};
    save_struct.a1_init    = a{1};
    save_struct.a2_init    = a{2};
    save_struct.a3_init    = a{3};
    save_struct.D1         = D{1};
    save_struct.D2         = D{2};

    novelty_weight   = weights(1);
    learning_weight  = weights(2);
    epistemic_weight = weights(3);
    preference_value = weights(4);
    if preference_value == 0
        preference_inverse_precision = Inf;
    else
        preference_inverse_precision = 1 / preference_value;
    end

    per_trial = struct([]);

    for trial = 1:num_trials
        time_since_food = 0; time_since_water = 0; time_since_sleep = 0;
        short_term_memory = zeros(35, 35, 35, num_states * 4);
        memory_resets = 0; pe_memory_resets = 0; hill_memory_resets = 0;
        chosen_action = zeros(1, T - 1);
        true_states = cell(1, num_trials);
        true_states{trial} = zeros(2, 0);
        Q = cell(T, 2); P = cell(T, 2);
        observations = zeros(num_modalities, 0);
        O = cell(num_modalities, T);
        predicted_posterior = cell(T, 2);

        for factor = 1:2
            Q{1, factor} = D{factor}';
            P{1, factor} = D{factor}';
        end
        true_states{trial}(1, 1) = start_position;
        true_states{trial}(2, 1) = find(cumsum(D{2}) >= rand, 1);

        Q_pos_log = zeros(0, num_states);
        Q_ctx_log = zeros(0, 4);
        P_pos_log = zeros(0, num_states);
        P_ctx_log = zeros(0, 4);
        horizon_log = [];
        best_actions_log = {};
        a2_snapshots = zeros(0, 4, num_states, 4);
        a2_snapshots(end+1, :, :, :) = a{2};

        t = 1;
        survived = 0;
        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
            bb{2} = normalise_matrix(b{2});

            if t ~= 1
                [P, Q, true_states] = updateEnvironmentStates(P, Q, true_states, trial, t, chosen_action, B, bb);
            end

            % Need-timer update
            ts = true_states{trial}(2, t);
            ps = true_states{trial}(1, t);
            if any(arrayfun(@(i) ts == i && ps == food_sources(i), 1:4))
                time_since_food = 0;
                if t > 1, time_since_water = time_since_water + 1; time_since_sleep = time_since_sleep + 1; end
            elseif any(arrayfun(@(i) ts == i && ps == water_sources(i), 1:4))
                time_since_water = 0;
                if t > 1, time_since_food = time_since_food + 1; time_since_sleep = time_since_sleep + 1; end
            elseif any(arrayfun(@(i) ts == i && ps == sleep_sources(i), 1:4))
                time_since_sleep = 0;
                if t > 1, time_since_food = time_since_food + 1; time_since_water = time_since_water + 1; end
            else
                if t > 1
                    time_since_food = time_since_food + 1;
                    time_since_water = time_since_water + 1;
                    time_since_sleep = time_since_sleep + 1;
                end
            end

            for modality = 1:num_modalities
                obs_idx = find(cumsum(A{modality}(:, ps, ts)) >= rand, 1);
                observations(modality, t) = obs_idx;
                vec = zeros(1, size(A{modality}, 1));
                vec(1, obs_idx) = 1;
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
            end

            y{2} = normalise_matrix(a{2}); y{1} = A{1}; y{3} = A{3};
            horizon = min([max_horizon, min([22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep])]);
            if horizon == 0, horizon = 1; end

            temp_Q = Q; temp_Q{t, 2} = temp_Q{t, 2}';
            P = calculate_posterior(temp_Q, y, O, t);

            if t > 1 && ~isequal(round(predicted_posterior{t, 2}, 1), round(P{t, 2}, 1))
                short_term_memory(:, :, :, :) = 0;
                memory_resets = memory_resets + 1;
                pe_memory_resets = pe_memory_resets + 1;
            end
            cur_pos = ps;
            if cur_pos == hill_pos
                short_term_memory(:, :, :, :) = 0;
                memory_resets = memory_resets + 1;
                hill_memory_resets = hill_memory_resets + 1;
            end

            best_actions = [];
            switch upper(algorithm)
                case 'SI'
                    [G, Q, short_term_memory, best_actions, ~] = tree_search_frwd_SI( ...
                        short_term_memory, O, Q, a, A, y, B, B, ...
                        t, T, t + horizon, ...
                        time_since_food, time_since_water, time_since_sleep, ...
                        true_t, chosen_action, ...
                        time_since_food, time_since_water, time_since_sleep, ...
                        best_actions, learning_weight, novelty_weight, epistemic_weight, ...
                        preference_inverse_precision, 0);
                case 'SL'
                    [G, Q, short_term_memory, best_actions, ~] = tree_search_frwd_SL( ...
                        short_term_memory, O, Q, a, A, y, B, B, ...
                        t, T, t + horizon, ...
                        time_since_food, time_since_water, time_since_sleep, ...
                        true_t, chosen_action, ...
                        time_since_food, time_since_water, time_since_sleep, ...
                        best_actions, learning_weight, novelty_weight, epistemic_weight, ...
                        preference_inverse_precision, 0);
                otherwise
                    error('Unsupported algorithm in dump: %s', algorithm);
            end

            % Logs
            Q_pos_log(end+1, :) = Q{t, 1}(:)'; %#ok<AGROW>
            Q_ctx_log(end+1, :) = Q{t, 2}(:)'; %#ok<AGROW>
            P_pos_log(end+1, :) = P{t, 1}(:)'; %#ok<AGROW>
            P_ctx_log(end+1, :) = P{t, 2}(:)'; %#ok<AGROW>
            horizon_log(end+1) = horizon; %#ok<AGROW>
            best_actions_log{end+1} = best_actions; %#ok<AGROW>
            a2_snapshots(end+1, :, :, :) = a{2}; %#ok<AGROW>

            chosen_action(t) = best_actions(1);
            t = t + 1;
        end

        if t >= 100, survived = 1; end

        per_trial(trial).true_states     = true_states{trial};
        per_trial(trial).observations    = observations;
        per_trial(trial).chosen_action   = chosen_action(1:t-1);
        per_trial(trial).Q_pos           = Q_pos_log;
        per_trial(trial).Q_ctx           = Q_ctx_log;
        per_trial(trial).P_pos           = P_pos_log;
        per_trial(trial).P_ctx           = P_ctx_log;
        per_trial(trial).horizon         = horizon_log;
        per_trial(trial).best_actions    = best_actions_log;
        per_trial(trial).a2_snapshots    = a2_snapshots;
        per_trial(trial).memory_resets   = memory_resets;
        per_trial(trial).pe_memory_resets = pe_memory_resets;
        per_trial(trial).hill_memory_resets = hill_memory_resets;
        per_trial(trial).t_terminal      = t;
        per_trial(trial).survived        = survived;
    end

    save_struct.per_trial = per_trial;
    save(out_path, '-struct', 'save_struct', '-v7');
    fprintf('Reference trace saved to %s\n', out_path);
end
