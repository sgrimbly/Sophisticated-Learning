function [survival] = model_free_RL_modular(seed, results_file_name)
    % rng(seed) % Seed now set in main.m

    % Define resource locations
    hill_1 = 55;
    true_food_sources = [71, 43, 57, 78];
    true_water_sources = [73, 33, 48, 67];
    true_sleep_sources = [64, 44, 49, 59];

    % Initialize variables
    replay_memory = 0;
    num_states = 100;
    global Q_table ep real_observations;
    real_observations = zeros(10000000, 5);
    ep = 1; % Start with a high exploration rate
    Q_table = zeros(5, 35, 35, 35, 100, 5);
    A = initializeObservationMatrices(num_states, true_food_sources, true_water_sources, true_sleep_sources, hill_1);
    D = initializePriorBeliefs(num_states);
    survival = zeros(1, 70);
    T = 27;
    num_modalities = 3;
    B = initializeTransitionMatrices(num_states);

    % Simulation parameters
    total_trials = 1000000;
    percent_interval = total_trials * 0.01;
    time_since_food = 0;
    time_since_water = 0;
    time_since_sleep = 0;
    t = 1;
    observation_count = 0;

    fprintf('Seed value: %d\n', seed);

    % Open file once for writing
    fid = fopen(results_file_name, 'a+');
    batch_size = 1000; % Write results in batches
    batch_results = zeros(batch_size, 1);
    batch_index = 1;

    true_states = cell(1, total_trials); % Initialize true_states
    chosen_action = zeros(1, 100); % Initialize chosen_action

    for trial = 1:total_trials
        if mod(trial, percent_interval) == 0
            percent_complete = (trial / total_trials) * 100;
            fprintf('Completed %.0f%% of the trials.\n', percent_complete);
        end

        % Decrease ep gradually
        ep = max(0.1, 1 - (trial / total_trials)); % Ensure ep does not go below 0.1

        while t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25
            observation_count = observation_count + 1;

            if t == 1
                true_states{trial}(1, t) = 51;
                true_states{trial}(2, t) = find(cumsum(D{2}) >= rand, 1);
            else
                true_states = updateTrueStates(true_states, B, trial, t, chosen_action);
            end

            % Update time since resource variables
            if t > 1
                t_food_prev = time_since_food;
                t_water_prev = time_since_water;
                t_sleep_prev = time_since_sleep;
            end

            [time_since_food, time_since_water, time_since_sleep] = updateTimesSinceResources(true_states{trial}, t, true_food_sources, true_water_sources, true_sleep_sources, time_since_food, time_since_water, time_since_sleep);

            for modality = 1:num_modalities
                ob = A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t));
                observations(modality, t) = find(cumsum(ob) >= rand, 1);
                vec = zeros(1, size(A{modality}, 1));
                vec(1, observations(modality, t)) = 1;
                O{modality, t} = vec;
            end

            context = find(cumsum(O{3, t}) >= rand, 1);
            observation = find(cumsum(O{1, t}) >= rand, 1);

            if replay_memory == 1
                real_observations(observation_count, :) = [observation, context, time_since_food + 1, time_since_water + 1, time_since_sleep + 1];
            end

            preference = determineObservationPreference(time_since_food, time_since_water, time_since_sleep);
            reward = O{2, t} * preference{2}';

            if t > 1
                updateQValues(observation_prev, context_prev, t_food_prev + 1, t_water_prev + 1, t_sleep_prev + 1, observation, context, time_since_food + 1, time_since_water + 1, time_since_sleep + 1, reward, chosen_action(t - 1));

                if replay_memory == 1
                    Model(chosen_action(t - 1), observation_prev, context_prev, t_food_prev + 1, t_water_prev + 1, t_sleep_prev + 1, :) = [time_since_food + 1, time_since_water + 1, time_since_sleep + 1, observation, context, reward];
                end
            end

            chosen_action(t) = selectAction(observation, context, time_since_food + 1, time_since_water + 1, time_since_sleep + 1);
            observation_prev = observation;
            context_prev = context;

            t = t + 1;
        end

        survival(trial) = t;
        batch_results(batch_index) = t;
        batch_index = batch_index + 1;

        if batch_index > batch_size
            fprintf(fid, 'time_steps_survived: %g\n', batch_results);
            batch_index = 1;
        end

        t = 1;
        time_since_food = 0;
        time_since_water = 0;
        time_since_sleep = 0;
    end

    % Write remaining results
    if batch_index > 1
        fprintf(fid, 'time_steps_survived: %g\n', batch_results(1:batch_index-1));
    end

    fclose(fid);
    save('07k_Q_tab.mat', 'Q_table');
end

function A = initializeObservationMatrices(num_states, food_sources, water_sources, sleep_sources, hill)
    A = cell(1, 3);
    A{1} = eye(num_states);

    A{2} = ones(2, num_states, 4);
    A{2}(1, :, :) = 1; % empty area cell

    for i = 1:4
        A{2}(2, food_sources(i), i) = 1;
        A{2}(1, food_sources(i), i) = 0;
        A{2}(3, water_sources(i), i) = 1;
        A{2}(1, water_sources(i), i) = 0;
        A{2}(4, sleep_sources(i), i) = 1;
        A{2}(1, sleep_sources(i), i) = 0;
    end

    A{3} = ones(5, num_states, 4);
    for i = 1:4
        A{3}(i, hill, i) = 1;
        A{3}(5, hill, i) = 0;
    end
end

function D = initializePriorBeliefs(num_states)
    D = cell(1, 2);
    D{1} = zeros(num_states, 1);
    D{1}(51) = 1;
    D{1} = D{1} / sum(D{1}); % Normalize
    D{2} = [0.25, 0.25, 0.25, 0.25]';
end

function B = initializeTransitionMatrices(num_states)
    B = cell(2, 5);
    for action = 1:5
        B{1}(:, :, action) = eye(num_states);
        B{2}(:, :, action) = [0.95, 0, 0, 0.05;
                              0.05, 0.95, 0, 0;
                              0, 0.05, 0.95, 0;
                              0, 0, 0.05, 0.95];
    end

    for i = 1:num_states
        if ~ismember(i, [1, 11, 21, 31, 41, 51, 61, 71, 81, 91])
            B{1}(:, i, 2) = circshift(B{1}(:, i, 2), -1); % move left
        end
        if ~ismember(i, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            B{1}(:, i, 3) = circshift(B{1}(:, i, 3), 1); % move right
        end
        if ~ismember(i, [91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
            B{1}(:, i, 4) = circshift(B{1}(:, i, 4), 10); % move up
        end
        if ~ismember(i, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            B{1}(:, i, 5) = circshift(B{1}(:, i, 5), -10); % move down
        end
    end
end

function [time_since_food, time_since_water, time_since_sleep] = updateTimesSinceResources(states, t, food_sources, water_sources, sleep_sources, time_since_food, time_since_water, time_since_sleep)
    if t > 1
        t_food_prev = time_since_food;
        t_water_prev = time_since_water;
        t_sleep_prev = time_since_sleep;
    end

    if any(states(1, t) == food_sources)
        time_since_food = 0;
        time_since_water = time_since_water + 1;
        time_since_sleep = time_since_sleep + 1;
    elseif any(states(1, t) == water_sources)
        time_since_water = 0;
        time_since_food = time_since_food + 1;
        time_since_sleep = time_since_sleep + 1;
    elseif any(states(1, t) == sleep_sources)
        time_since_sleep = 0;
        time_since_food = time_since_food + 1;
        time_since_water = time_since_water + 1;
    else
        time_since_food = time_since_food + 1;
        time_since_water = time_since_water + 1;
        time_since_sleep = time_since_sleep + 1;
    end
end

function true_states = updateTrueStates(true_states, B, trial, t, chosen_action)
    for factor = 1:2
        if factor == 1
            true_states{trial}(factor, t) = find(cumsum(B{1}(:, true_states{trial}(factor, t - 1), chosen_action(t - 1))) >= rand, 1);
        else
            true_states{trial}(factor, t) = find(cumsum(B{2}(:, true_states{trial}(factor, t - 1), 1)) >= rand, 1);
        end
    end
end

function writeResultsToFile(fid, t)
    fprintf(fid, 'time_steps_survived: %g\n', t);
end

function action = selectAction(observation, context, t_food, t_water, t_sleep)
    global Q_table;
    global ep;
    actions = [1, 2, 3, 4, 5];
    max_action = max(Q_table(:, t_food, t_water, t_sleep, observation, context));
    actions1 = find(Q_table(:, t_food, t_water, t_sleep, observation, context) == max_action);
    epsilon = rand;

    if epsilon < ep
        action = actions(randsample(numel(actions), 1));
    else
        action = actions1(randsample(numel(actions1), 1));
    end
end

function updateQValues(observation_prev, context_prev, t_food_prev, t_water_prev, t_sleep_prev, observation, context, t_food, t_water, t_sleep, reward, action)
    global Q_table;
    current_Q_value = Q_table(action, t_food_prev, t_water_prev, t_sleep_prev, observation_prev, context_prev);
    next_Q_value = max(Q_table(:, t_food, t_water, t_sleep, observation, context));
    Q_table(action, t_food_prev, t_water_prev, t_sleep_prev, observation_prev, context_prev) = current_Q_value + 0.2 * (reward + 0.7 * next_Q_value - current_Q_value);
end

% Run this script
% Directory and path setup (unchanged)
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir, '../../../MATLAB');
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);
model_free_RL_modular(1, 'results.txt');

% Run this script
% Directory and path setup (unchanged)
currentDir = fileparts(mfilename('fullpath'));
srcPath = fullfile(currentDir, '../../../MATLAB');
fullSrcPath = genpath(srcPath);
addpath(fullSrcPath);
model_free_RL_modular(1, 'results.txt');
