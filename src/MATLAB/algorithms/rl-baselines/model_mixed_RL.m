function [survival] = model_mixed_RL(seed, results_file_name)
    %clear
    %class(seed)
    rng(seed);
    % rng(seed)
    %file_name = strcat(num2str(seed),'.txt');
    file_name = '07k_mixed.txt';
    %%% Hyper Params %%%
    % true_food_source_1 = 13;
    % true_food_source_2 = 91;
    % true_food_source_3 = 92;
    % true_water_source_1 = 28;
    % true_water_source_2 = 93;
    % true_sleep_source_1 = 15;
    % true_sleep_source_2 = 100;

    %%%%%%%%%%%%%% node class %%%%%%%%%%%%%%%
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
    food_locations = [true_food_source_1, true_food_source_2, true_food_source_3];
    water_locations = [true_water_source_1, true_water_source_2];
    sleep_locations = [true_sleep_source_1, true_sleep_source_2];
    resource_locations = [food_locations];
    replay_memory = 1;
    traj_count(1) = 0;

    num_states = 100;
    num_states_low = 25;
    global Model
    global Q_table
    global ep
    global real_observations;
    real_observations = zeros(10000000, 5);
    ep = 20;
    Q_table = zeros(5, 35, 35, 35, 100, 5);
    Model = zeros(5, 100, 5, 35, 35, 35, 6);
    A{1}(:, :, :) = zeros(num_states, num_states, 4);
    a{1}(:, :, :) = zeros(num_states, num_states, 4);

    for i = 1:num_states
        A{1}(i, i, :) = 1;
        a{1}(i, i, :) = 1;
    end

    A{2}(:, :, :) = zeros(2, num_states, 4);
    A{2}(1, :, :) = 1; % empty area cell
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
    %A{3} = A{3}/numel(A{3}(:,1,1));
    A{3}(5, :, :) = 1;
    A{3}(1, hill_1, 1) = 1;
    A{3}(5, hill_1, 1) = 0;
    A{3}(2, hill_1, 2) = 1;
    A{3}(5, hill_1, 2) = 0;
    A{3}(3, hill_1, 3) = 1;
    A{3}(5, hill_1, 3) = 0;
    A{3}(4, hill_1, 4) = 1;
    A{3}(5, hill_1, 4) = 0;

    D{1} = zeros(1, num_states)'; %position in environment
    D{2} = [0.25, 0.25, 0.25, 0.25]';
    D{1}(51) = 1;
    survival(:) = zeros(1, 70);
    D{1} = normalise(D{1});
    num_factors = 1;
    T = 27;
    num_modalities = 3;
    num_states = 100;

    %%% Distributions %%%

    for action = 1:5
        B{1}(:, :, action) = eye(num_states);
        B{2}(:, :, action) = zeros(4);
        B{2}(:, :, action) = [0.95, 0, 0 0.05;
                            0.05, 0.95, 0, 0;
                            0, 0.05, 0.95, 0;
                            0, 0, 0.05 0.95];

        % Uniform prior over season transitions. This is what the agent must
        % learn
        b{2}(:, :, action) = [0.25, 0.25, 0.25 0.25;
                            0.25, 0.25, 0.25, 0.25;
                            0.25, 0.25, 0.25, 0.25;
                            0.25, 0.25, 0.25 0.25];
    end

    b = B;

    for i = 1:num_states

        if i ~= [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]
            B{1}(:, i, 2) = circshift(B{1}(:, i, 2), -1); % move left
        end

    end

    for i = 1:num_states

        if i ~= [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            B{1}(:, i, 3) = circshift(B{1}(:, i, 3), 1); % move right
        end

    end

    for i = 1:num_states

        if i ~= [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
            B{1}(:, i, 4) = circshift(B{1}(:, i, 4), 10); % move up
        end

    end

    for i = 1:num_states

        if i ~= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            B{1}(:, i, 5) = circshift(B{1}(:, i, 5), -10); % move down
        end

    end

    b{1} = B{1};
    C{1} = ones(11, 9); % preference for positional observation. Uniform.
    C_overall{1} = zeros(T, 9);

    chosen_action = zeros(1, T - 1);
    preference_values = zeros(4, T);

    time_since_food = 0;
    time_since_water = 0;
    time_since_sleep = 0;
    %file_name = strcat(seed,'.txt');
    t = 1;
    observation_count = 0;

    % results_file_name = sprintf('/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/RL-runs/model_mixed_results_seed%d.txt', seed);    
    total_trials = 1000000;  % Total number of trials
    percent_interval = total_trials * 0.01;  % Calculate 1% of the total number of trials
    
    for trial = 1:total_trials
        fprintf('Runnning trials.');
        if mod(trial, percent_interval) == 0  % Check if the current trial is at 1% interval
            percent_complete = (trial / total_trials) * 100;  % Calculate the percentage completed
            fprintf('Completed %.0f%% of the trials.\n', percent_complete);  % Print progress to the console
        end
        [bound, ] = min(trial, 5);

        if trial > 1 && replay_memory == 1

            for replay = 1:bound
                %sample from observations received up until the current time point
                random_index = randi([1, observation_count], 1);
                random_samples = real_observations(random_index, :);
                actions = Model(:, random_samples(1), random_samples(2), random_samples(3), random_samples(4), random_samples(5), 1);

                if any(actions ~= 0)
                    possible_actions = find(actions ~= 0);
                    randomAction = possible_actions(randi(length(possible_actions)));
                    values = squeeze(Model(randomAction, random_samples(1), random_samples(2), random_samples(3), random_samples(4), random_samples(5), :));
                    time_food = values(1);
                    time_water = values(2);
                    time_sleep = values(3);
                    observation = values(4);
                    context = values(5);
                    reward = values(6);
                    updateQValues(random_samples(1), random_samples(2), random_samples(3), random_samples(4), random_samples(5), observation, context, time_food, time_water, time_sleep, reward, randomAction)
                end

            end

        end

        if trial > 70000
            ep = 40;
        elseif trial > 130000
            ep = 80;
        elseif trial > 200000
            ep = 1000;
        end

        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
            observation_count = observation_count + 1;
            %bb{2} = normalise_matrix(b{2});
            for factor = 1:2

                if t == 1
                    %P{t,factor} = D{factor}';
                    %Q{t,factor} = D{factor}';
                    true_states{trial}(1, t) = 51;
                    true_states{trial}(2, t) = find(cumsum(D{2}) >= rand, 1);
                else

                    if factor == 1
                        %Q{t,factor} = (B{1}(:,:,chosen_action(t-1))*Q{t-1,factor}')';
                        true_states{trial}(factor, t) = find(cumsum(B{1}(:, true_states{trial}(factor, t - 1), chosen_action(t - 1))) >= rand, 1);
                    else
                        %b = B{2}(:,:,:);
                        %Q{t,factor} = (bb{2}(:,:,chosen_action(t-1))*Q{t-1,factor}')';%(B{2}(:,:)'
                        true_states{trial}(factor, t) = find(cumsum(B{2}(:, true_states{trial}(factor, t - 1), 1)) >= rand, 1);

                    end

                end

            end

            if t > 1
                t_food_prev = time_since_food;
                t_water_prev = time_since_water;
                t_sleep_prev = time_since_sleep;
            end

            prefs = determineObservationPreference(time_since_food, time_since_water, time_since_sleep);

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

            context = find(cumsum(O{3, t}) >= rand, 1);
            observation = find(cumsum(O{1, t}) >= rand, 1);

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

            %displayGridWorld(true_states{trial}(1,t),food,water,sleep, hill_1, 1)

            time_since_resources = [time_since_food, time_since_water, time_since_sleep];

            % Generate the key from current observation data

            % Optionally store the observation data if needed for other purposes
            if replay_memory == 1
                real_observations(observation_count, :) = [observation, context, time_since_food + 1, time_since_water + 1, time_since_sleep + 1];
            end

            reward = O{2, t} * prefs{2}';

            if t > 1
                updateQValues(observation_prev, context_prev, t_food_prev + 1, t_water_prev + 1, t_sleep_prev + 1, observation, context, time_since_food + 1, time_since_water + 1, time_since_sleep + 1, reward, chosen_action(t - 1))
            end

            if t > 1 && replay_memory == 1
                Model(chosen_action(t - 1), observation_prev, context_prev, t_food_prev + 1, t_water_prev + 1, t_sleep_prev + 1, :) = [time_since_food + 1, time_since_water + 1, time_since_sleep + 1, observation, context, reward];
            end

            chosen_action(t) = selectAction(observation, context, time_since_food + 1, time_since_water + 1, time_since_sleep + 1);
            observation_prev = observation;
            context_prev = context;

            t = t + 1;
            % end loop over time points

        end

        survival(trial) = t;

        if (numel(true_states{trial}) == 18)
            alive_status = 1;
        else
            alive_status = 0;
        end

        fid = fopen(results_file_name, 'a+');
        if fid == -1
            error('Error opening file: %s', results_file_name);
        end
        
        try
            fprintf(fid, 'time_steps_survived: %g\n', t);
        catch ME
            fclose(fid);
            rethrow(ME); % Re-throw the error after closing the file
        end
        
        fclose(fid); % Close the file in the normal case

        % t_pref_mv_av = movmean(pref_match, 9);
        % t_food_mv_av = movmean(t_food_plot,9);
        % t_water_mv_av = movmean(t_water_plot,9);
        % t_sleep_mv_av = movmean(t_sleep_plot,9);
        % fid =fopen('results.txt', 'w' );
        % file_name = strcat(seed,'.txt');
        % file_name = 'model_free_results.txt';
        % fid = fopen(file_name, 'a+');
        % fprintf(fid, '%f\n', t);
        % fwrite(fid, 'time_steps_survived: ');
        % fprintf(fid, '%g,', t);
        % printf(fid, '%g\n','');
        % fprintf(fid, '%g\n','');
        % fwrite(fid, 'food_mov_av: ');
        % fprintf(fid, '%g,', t_food_mv_av);
        % fprintf(fid, '%g\n','');
        % fwrite(fid, 'water_mov_av: ');
        % fprintf(fid, '%g,', t_water_mv_av);
        % fprintf(fid, '%g\n','');
        % fwrite(fid, 'sleep_mov_av: ');
        % fprintf(fid, '%g,', t_sleep_mv_av);
        % fprintf(fid, '%g\n','');
        % fwrite(fid, 'overall_mov_av: ');
        % fprintf(fid, '%g,', t_pref_mv_av);
        % fprintf(fid, '%g\n','');
        t = 1;
        time_since_food = 0;
        time_since_water = 0;
        time_since_sleep = 0;
    end

    %end
    %save(strcat(seed,'_Q_tab.mat'), 'Q_table');
    save('07k_mixed_Q_tab.mat', 'Q_table');
    %fprintf(fid, '%f\n', 'targetted forgetting');

end

function action = selectAction(observation, context, t_food, t_water, t_sleep)
    global Q_table;
    global ep;
    actions = [1, 2, 3, 4, 5];
    max_action = max(Q_table(:, t_food, t_water, t_sleep, observation, context));
    actions1 = find(Q_table(:, t_food, t_water, t_sleep, observation, context) == max_action);
    epsilon = randi([1, ep]);

    if epsilon == ep
        % Select a random action from all available actions
        action = actions(randi(numel(actions)));
    else
        % Select a random action from the best actions
        action = actions1(randi(numel(actions1)));
    end

end

function updateQValues(observation_prev, context_prev, t_food_prev, t_water_prev, t_sleep_prev, observation, context, t_food, t_water, t_sleep, reward, action, Q_table)
    global Q_table;
    current_Q_value = Q_table(action, t_food_prev, t_water_prev, t_sleep_prev, observation_prev, context_prev);
    next_Q_value = max(Q_table(:, t_food, t_water, t_sleep, observation, context));
    Q_table(action, t_food_prev, t_water_prev, t_sleep_prev, observation_prev, context_prev) = current_Q_value + 0.2 * (reward + 0.7 * next_Q_value - current_Q_value);
end

function C = determineObservationPreference(t_food, t_water, t_sleep)

    empty = -1;

    if t_water > 19
        t_food = -500;
        t_sleep = -500;
        empty = -500;
    end

    if t_food > 21
        t_water = -500;
        t_sleep = -500;
        empty = -500;

    end

    if t_sleep > 24
        t_food = -500;
        t_water = -500;
        empty = -500;
    end

    C{2} = [empty, t_food, t_water, t_sleep];

end
