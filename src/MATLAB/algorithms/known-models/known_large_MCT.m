function [] = known_large_MCT(seed, horizon, k_factor, root, mct, num_mct, auto_rest)
    %KNOWN_LARGE_MCT Runs Monte Carlo Tree Search (MCTS) experiments for a known large model.
    %
    % This function initializes the random number generator with a given seed,
    % sets up the experiment's directory structure, and simulates survival times
    % under various conditions controlled by the mct, num_mct, and auto_rest variables.
    %
    % Parameters:
    %   seed (string): Seed for the random number generator to ensure reproducibility.
    %   horizon (string): The horizon up to which sophisticated inference is performed.
    %   k_factor (string): A scaling factor influencing some aspect of the model (details should be specified).
    %   root (string): Root directory path where results are to be saved.
    %   mct (string): Specifies the length of Monte Carlo rollouts. If 'mct' is '0', standard
    %                 sophisticated inference is used up to 'horizon'. If 'mct' >= '1', the model
    %                 performs hybrid inference with both sophisticated inference and Monte Carlo rollouts.
    %   num_mct (string): The number of Monte Carlo simulations to run.
    %   auto_rest (string): Controls the inclusion of memory in the model. '0' indicates memory is included, '1' indicates no memory.
    %
    % Usage:
    %   [] = known_large_MCT('42', '20', '1.5', '/user/path/', '10', '100', '0')
    %
    % This sets up an experiment using seed '42', a horizon of 20, k-factor of 1.5, in the directory
    % '/user/path/', with Monte Carlo rollouts of length 10, running 100 simulations, and including memory in the model.
    %
    % Outputs:
    %   The function does not return any values. It saves the simulation results in .mat files
    %   within the specified directory path.
    %
    % Files Created:
    %   - Survival time data file: Contains the simulated survival times under different conditions.
    %   - Agent state file: Stores the state of the agent at various points, which can be used for further analysis.
    %
    % Note:
    %   Ensure that the input parameters are in string format as the function converts them into double
    %   for computational purposes within the script.

    rng(str2double(seed))
    rng
    %file_name = strcat(seed,'_hor',horizon,'.txt');
    path = [root '/MATLAB-experiments/experiments/known_model/'];
    file_name = strcat(path, horizon, 'hor_', k_factor, 'kfactor_', mct, 'MCT_', num_mct, 'num_mct_', seed, 'seed_survival_time', '.mat');
    matfilename = strcat(path, horizon, 'hor_', k_factor, 'kfactor_', mct, 'MCT_', num_mct, 'num_mct_', seed, 'seed', '.mat');

    len_each = 1;

    k_factor = str2double(k_factor);
    horizon = str2double(horizon);
    mct = str2double(mct);
    num_mct = str2double(num_mct);
    auto_rest = str2double(auto_rest);  % Convert auto_rest to numeric for internal logic

    % Conditional logic based on auto_rest
    if auto_rest == 1
        disp('Running without memory...');
        % Implement the functionality for running the experiment without memory
    else
        disp('Running with memory...');
        % Implement the functionality for running the experiment with memory
    end

    previous_positions = [];

    hill_1 = 55;
    true_food_source_1 = 71;
    true_food_source_2 = 43;
    true_food_source_3 = 57;
    true_food_source_4 = 78;
    true_water_source_1 = 73;
    true_water_source_2 = 33;
    true_water_source_3 = 48;
    true_water_source_4 = 67;
    true_sleep_source_1 = 64;
    true_sleep_source_2 = 44;
    true_sleep_source_3 = 49;
    true_sleep_source_4 = 59;

    num_states = 100;
    num_states_low = 25;
    % why number num_states_low = 25;

    A{1}(:, :, :) = zeros(num_states, num_states, 4);
    a{1}(:, :, :) = zeros(num_states, num_states, 4);

    for i = 1:num_states
        A{1}(i, i, :) = 1;
        a{1}(i, i, :) = 1;
    end

    A{2}(:, :, :) = zeros(2, num_states, 4);
    A{2}(1, :, :) = 1;
    A{2}(2, true_food_source_1, 1) = 1;
    A{2}(1, true_food_source_1, 1) = 0;
    A{2}(2, true_food_source_2, 2) = 1;
    A{2}(1, true_food_source_2, 2) = 0;
    A{2}(2, true_food_source_3, 3) = 1;
    A{2}(1, true_food_source_3, 3) = 0;
    A{2}(2, true_food_source_4, 4) = 1;
    A{2}(1, true_food_source_4, 4) = 0;
    A{2}(3, true_water_source_1, 1) = 1;
    A{2}(1, true_water_source_1, 1) = 0;
    A{2}(3, true_water_source_2, 2) = 1;
    A{2}(1, true_water_source_2, 2) = 0;
    A{2}(3, true_water_source_3, 3) = 1;
    A{2}(1, true_water_source_3, 3) = 0;
    A{2}(3, true_water_source_4, 4) = 1;
    A{2}(1, true_water_source_4, 4) = 0;
    A{2}(4, true_sleep_source_1, 1) = 1;
    A{2}(1, true_sleep_source_1, 1) = 0;
    A{2}(4, true_sleep_source_2, 2) = 1;
    A{2}(1, true_sleep_source_2, 2) = 0;
    A{2}(4, true_sleep_source_3, 3) = 1;
    A{2}(1, true_sleep_source_3, 3) = 0;
    A{2}(4, true_sleep_source_4, 4) = 1;
    A{2}(1, true_sleep_source_4, 4) = 0;
    A{3}(:, :, :) = zeros(5, num_states, 4);
    A{3}(5, :, :) = 1;
    A{3}(1, hill_1, 1) = 1;
    A{3}(5, hill_1, 1) = 0;
    A{3}(2, hill_1, 2) = 1;
    A{3}(5, hill_1, 2) = 0;
    A{3}(3, hill_1, 3) = 1;
    A{3}(5, hill_1, 3) = 0;
    A{3}(4, hill_1, 4) = 1;
    A{3}(5, hill_1, 4) = 0;
    a{3} = A{3};
    a{2}(:, :, :) = zeros(4, num_states, 4);
    a{2} = a{2} + 0.1;
    a{2} = A{2};

    %% Setup Agent

    D{1} = zeros(1, num_states)'; %position in environment
    D{2} = [0.25, 0.25, 0.25, 0.25]';

    D{1}(51) = 1; % start position

    survival(:) = zeros(1, 70);

    %why these? wtf is happening?

    D{1} = normalise(D{1});
    num_factors = 1;
    T = 27; % what is this

    num_modalities = 3;
    num_states = 100;
    food_locations = [true_food_source_1, true_food_source_2, true_food_source_3];
    resource_locations = [food_locations];
    short_term_memory(:, :, :, :, :) = zeros(40, 40, 40, 400);

    %%% Distributions %%%

    for action = 1:5
        B{1}(:, :, action) = eye(num_states);
        B{2}(:, :, action) = zeros(4);
        B{2}(:, :, action) = [0.95, 0, 0 0.05;
                            0.05, 0.95, 0, 0;
                            0, 0.05, 0.95, 0;
                            0, 0, 0.05 0.95];

    end

    b = B;

    for i = 1:num_states

        if i ~= [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
            B{1}(:, i, 2) = circshift(B{1}(:, i, 2), -1); % move left
        end

    end

    for i = 1:num_states

        if i ~= [10, 20, 30, 40, 50, 60, 70, 80, 90]
            B{1}(:, i, 3) = circshift(B{1}(:, i, 3), 1); % move right
        end

    end

    for i = 1:num_states

        if i ~= [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
            B{1}(:, i, 4) = circshift(B{1}(:, i, 4), 10); % move rup
        end

    end

    for i = 1:num_states

        if i ~= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            B{1}(:, i, 5) = circshift(B{1}(:, i, 5), -10); % move down
        end

    end

    b{1} = B{1};

    time_since_food = 0;
    time_since_water = 0;
    time_since_sleep = 0;
    t = 1;
    surety = 1;
    simulated_time = 0;
    action_history = zeros(100, 99);
    states_history = zeros(100, 99);
    season_history = zeros(100, 99);

    global TREE_SEARCH_HISTORY;
    TREE_SEARCH_HISTORY = cell(100, 99);
    survival_times = [];
    %% Main Loop
    for trial = 1:len_each
        chosen_action = zeros(1, 99);

        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
            disp(t)
            bb{2} = normalise_matrix(b{2}); % what is the difference between b and bb

            for factor = 1:2

                if t == 1
                    P{t, factor} = D{factor}';
                    Q{t, factor} = D{factor}';
                    true_states{trial}(1, t) = 51;
                    true_states{trial}(2, t) = find(cumsum(D{2}) >= rand, 1);
                else

                    if factor == 1
                        Q{t, factor} = (B{1}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')';
                        true_states{trial}(factor, t) = find(cumsum(B{1}(:, true_states{trial}(factor, t - 1), chosen_action(t - 1))) >= rand, 1);
                    else
                        %b = B{2}(:,:,:);
                        Q{t, factor} = (bb{2}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')'; %(B{2}(:,:)'
                        true_states{trial}(factor, t) = find(cumsum(B{2}(:, true_states{trial}(factor, t - 1), 1)) >= rand, 1);

                    end

                end

                states_history(trial, t) = true_states{trial}(1, t);
                season_history(trial, t) = true_states{trial}(2, t);
            end

            if (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == true_food_source_1) || (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == true_food_source_2) || (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == true_food_source_3) || (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == true_food_source_4)
                time_since_food = 0;
                time_since_water = time_since_water +1;
                time_since_sleep = time_since_sleep +1;

            elseif (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == true_water_source_1) || (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == true_water_source_2) || (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == true_water_source_3) || (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == true_water_source_4)
                time_since_water = 0;
                time_since_food = time_since_food +1;
                time_since_sleep = time_since_sleep +1;

            elseif (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == true_sleep_source_1) || (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == true_sleep_source_2) || (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == true_sleep_source_3) || (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == true_sleep_source_4)
                time_since_sleep = 0;
                time_since_food = time_since_food +1;
                time_since_water = time_since_water +1;

            else

                if t > 1
                    time_since_food = time_since_food +1;
                    time_since_water = time_since_water +1;
                    time_since_sleep = time_since_sleep +1;
                end

            end

            % sample the next observation. Same technique as sampling states

            for modality = 1:num_modalities
                ob = A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t));
                observations(modality, t) = find(cumsum(A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t))) >= rand, 1);
                %create a temporary vectore of 0s
                vec = zeros(1, size(A{modality}, 1));
                % set the index of the vector matching the observation index to 1
                vec(1, observations(modality, t)) = 1;
                O{modality, t} = vec;
            end

            true_t = t;

            if true_states{trial}(2, t) == 1
                food = true_food_source_1;
                water = true_water_source_1;
                sleep = true_sleep_source_1;
            elseif true_states{trial}(2, t) == 2
                food = true_food_source_2;
                water = true_water_source_2;
                sleep = true_sleep_source_2;
            elseif true_states{trial}(2, t) == 3
                food = true_food_source_3;
                water = true_water_source_3;
                sleep = true_sleep_source_3;
            else
                food = true_food_source_4;
                water = true_water_source_4;
                sleep = true_sleep_source_4;
            end

            %   displayGridWorld(true_states{trial}(1,t),food,water,sleep, hill_1, 1, trial, t, time_since_resources, prefs{2}, states_history)
            g = {};

            y{2} = normalise_matrix(a{2});
            y{1} = A{1};
            y{3} = A{3};

            if horizon == 0
                horizon = 1;
            end

            temp_Q = Q;
            temp_Q{t, 2} = temp_Q{t, 2}';
            P = calculate_posterior(temp_Q, y, O, t);
            long_term_memory = 0;
            a_complexity = 0;
            current_pos = find(cumsum(P{t, 1}) >= rand, 1);

            optimal_traj = [];
            % Start tree search from current time point
            best_actions = [];

            global trajectory;
            trajectory = {};

            history = '';
            global tree_history;
            tree_history = {};

            global searches;
            searches = 0;

            global post_calcs;
            post_calcs = 0;

            global auto_rest;
            auto_rest = 0;

            [G, Q, D, short_term_memory, long_term_memory, optimal_traj, best_actions, Tree] = tree_search_frwd_knownMCT(long_term_memory, short_term_memory, O, Q, a, A, y, D, B, B, t, T, t + horizon, time_since_food, time_since_water, time_since_sleep, resource_locations, current_pos, true_t, chosen_action, a_complexity, surety, simulated_time, time_since_food, time_since_water, time_since_sleep, 0, optimal_traj, best_actions, history, k_factor, mct, num_mct);
            %TREE_SEARCH_HISTORY{trial,t} = tree_history;
            TREE_SEARCH_HISTORY{trial, t} = [nnz(short_term_memory), searches, numel(fieldnames(tree_history)), G, post_calcs];
            short_term_memory(:, :, :, :, :) = 0; %reseting over and over
            chosen_action(t) = best_actions(1);
            t = t + 1;
            % end loop over time points

        end

        survival(trial) = t;
        action_history(trial, :) = chosen_action;
        save(file_name, 't');
        save(matfilename, 'TREE_SEARCH_HISTORY');

        %save(matfilename, 'TREE_SEARCH_HISTORY');

        t = 1;
        time_since_food = 0;
        time_since_water = 0;
        time_since_sleep = 0;
    end

end
