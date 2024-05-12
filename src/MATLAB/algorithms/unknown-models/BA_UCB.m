function [survived] = BA_UCB(seed)
    rng(seed);
    rng
    %%% Hyper Params %%%
    % clear
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
    A{1}(:, :, :) = zeros(num_states, num_states, 4);
    a{1}(:, :, :) = zeros(num_states, num_states, 4);

    for i = 1:num_states
        A{1}(i, i, :) = 1;
        a{1}(i, i, :) = 1;
    end

    A{2}(:, :, :) = zeros(4, num_states, 4);
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
    a{3} = A{3};
    a{2}(:, :, :) = zeros(4, num_states, 4);
    a{2} = a{2} + 0.1;
    % a{2}(2, true_food_source_1, 1) = 0.3;
    % a{2}(2, true_food_source_1, 2) = 0.3;
    % a{2}(2, true_food_source_1, 3) = 0.3;
    % a{2}(2, true_food_source_1, 4) = 0.3;
    % a{2}(3, true_water_source_1, 1) = 0.3;
    % a{2}(3, true_water_source_1, 2) = 0.3;
    % a{2}(3, true_water_source_1, 3) = 0.3;
    % a{2}(3, true_water_source_1, 4) = 0.3;
    % a{2}(4, true_sleep_source_1, 1) = 0.3;
    % a{2}(4, true_sleep_source_1, 2) = 0.3;
    % a{2}(4, true_sleep_source_1, 3) = 0.3;
    % a{2}(4, true_sleep_source_1, 4) = 0.3;
    % a{2}(2, true_food_source_2, 1) = 0.3;
    % a{2}(2, true_food_source_2, 2) = 0.3;
    % a{2}(2, true_food_source_2, 3) = 0.3;
    % a{2}(2, true_food_source_2, 4) = 0.3;
    % a{2}(3, true_water_source_2, 1) = 0.3;
    % a{2}(3, true_water_source_2, 2) = 0.3;
    % a{2}(3, true_water_source_2, 3) = 0.3;
    % a{2}(3, true_water_source_2, 4) = 0.3;
    % a{2}(4, true_sleep_source_2, 1) = 0.3;
    % a{2}(4, true_sleep_source_2, 2) = 0.3;
    % a{2}(4, true_sleep_source_2, 3) = 0.3;
    % a{2}(4, true_sleep_source_2, 4) = 0.3;
    % a{2}(2, true_food_source_3, 1) = 0.3;
    % a{2}(2, true_food_source_3, 2) = 0.3;
    % a{2}(2, true_food_source_3, 3) = 0.3;
    % a{2}(2, true_food_source_3, 4) = 0.3;
    % a{2}(3, true_water_source_3, 1) = 0.3;
    % a{2}(3, true_water_source_3, 2) = 0.3;
    % a{2}(3, true_water_source_3, 3) = 0.3;
    % a{2}(3, true_water_source_3, 4) = 0.3;
    % a{2}(4, true_sleep_source_4, 1) = 0.3;
    % a{2}(4, true_sleep_source_4, 2) = 0.3;
    % a{2}(4, true_sleep_source_4, 3) = 0.3;
    % a{2}(4, true_sleep_source_4, 4) = 0.3;

    D{1} = zeros(1, num_states)'; %position in environment
    D{2} = [0.25, 0.25, 0.25, 0.25]';
    D{1}(51) = 1;

    survival(:) = zeros(1, 70);
    D{1} = normalise(D{1});
    T = 27;
    num_modalities = 3;
    num_states = 100;

    short_term_memory(:, :, :, :, :) = zeros(35, 35, 35, 400, 5);

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

    Nt = ones(400);
    b{1} = B{1};

    chosen_action = zeros(1, T - 1);

    time_since_food = 0;
    time_since_water = 0;
    time_since_sleep = 0;

    % Format the current date and time
    current_time = char(datetime('now', 'Format', 'HH-mm-ss-SSS'));

    % Convert seed to string
    seed_str = num2str(seed);
    file_name = strcat(current_time, '_seed_', seed_str, '_BAUCB_experiment.txt');

    t = 1;
    surety = 1;
    simulated_time = 0;

    num_trials = 120;
    memory_resets = zeros(num_trials, 1);
    pe_memory_resets = zeros(num_trials, 1);
    hill_memory_resets = zeros(num_trials, 1);
    total_search_depth = 0;
    total_memory_accessed = 0;
    total_t = 0;
    survived(1:num_trials) = 0;

    t_at_25 = 0;
    t_at_50 = 0;
    t_at_75 = 0;
    t_at_100 = 0;

    total_startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    for trial = 1:num_trials
        startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        fprintf('\n----------------------------------------\n');
        fprintf('TRIAL %d STARTED\n', trial);
        fprintf('----------------------------------------\n');
        fprintf('Start Time: %s\n', startTime);

        short_term_memory(:, :, :, :) = 0;
        search_depth = 0;
        memory_accessed = 0;

        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)

            bb{2} = normalise_matrix(b{2});

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
                        Q{t, factor} = (bb{2}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')'; %(B{2}(:,:)'
                        true_states{trial}(factor, t) = find(cumsum(B{2}(:, true_states{trial}(factor, t - 1), 1)) >= rand, 1);

                    end

                end

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

            if t > 1
                start = t - 6;

                if start <= 0
                    start = 1;
                end

                bb{2} = normalise_matrix(b{2});
                y{2} = normalise_matrix(a{2});

                qs = spm_cross(Q{t, :});
                predictive_observations_posterior{2, t} = normalise(y{2}(:, :) * qs(:))';
                predictive_observations_posterior{3, t} = normalise(y{3}(:, :) * qs(:))';
                predicted_posterior = calculate_posterior(Q, y, predictive_observations_posterior, t);
                %     %Backwards pass to calculate retrospective model (propagated parameter
                %     %belief search. In this implementation, the agent does it over transition only)

                for timey = t:t
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

                            %Define the proportion to subtract
                            proportion = 0.3;

                            for i = 1:size(a_learning, 3)
                                % Subtract an amount proportional to the maximum value from each zero entry in each column
                                for j = 1:size(a_learning, 2)
                                    max_value = max(a_learning(2:end, j, i)); % find the maximum value in column j
                                    amount_to_subtract = proportion * max_value; % calculate the amount to subtract
                                    a_learning(a_learning(1, j, i) == 0, j, i) = a_learning(a_learning(1, j, i) == 0, j, i) - amount_to_subtract;

                                end

                            end

                            a{modality} = a{modality} + 0.7 * a_learning;
                            a{modality}(a{modality} <= 0.1) = 0.05;

                        end

                    end

                end

            end

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
            g = {};
            % Unused in this iteration, as the agent does not need to learn
            % likelihood
            y{2} = normalise_matrix(a{2});
            y{1} = A{1};
            y{3} = A{3};

            prefs = determineObservationPreference(time_since_food, time_since_water, time_since_sleep);
            time_since_resources = [time_since_food, time_since_water, time_since_sleep];

            horizon = min([9, min([22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep])]); %round(calculateLookAhead(0.24,2.7,time_since_resources, resource_cutoffs));

            if horizon == 0
                horizon = 1;
            end

            temp_Q = Q;
            temp_Q{t, 2} = temp_Q{t, 2}';
            P = calculate_posterior(temp_Q, y, O, t);
            long_term_memory = 0;
            trajectory = [];
            a_complexity = 0;
            current_pos(t) = find(cumsum(P{t, 1}) >= rand, 1);
            optimal_traj = [];

            if t > 1 && ~isequal(round(predicted_posterior{t, 2}, 1), round(P{t, 2}, 1)) %~all(a1 == a2)
                short_term_memory(:, :, :, :, :) = 0;
                memory_resets(trial) = memory_resets(trial) + 1;
                pe_memory_resets(trial) = pe_memory_resets(trial) + 1;
            end

            if current_pos(t) == hill_1
                short_term_memory(:, :, :, :) = 0;
                memory_resets(trial) = memory_resets(trial) + 1;
                hill_memory_resets(trial) = hill_memory_resets(trial) + 1;
            end

            cur_state = spm_cross(P{t});
            cur_state = find(cumsum(cur_state(:)) >= rand, 1);
            Nt(cur_state) = Nt(cur_state) + 1;
            best_actions = [];
            % Start tree search from current time point
            [G, Q, D, short_term_memory, long_term_memory, optimal_traj, best_actions, memory_accessed] = tree_search_frwd_UCB(long_term_memory, short_term_memory, O, Q, a, A, y, D, B, B, t, T, t + horizon, time_since_food, time_since_water, time_since_sleep, time_since_food, time_since_water, time_since_sleep, current_pos(t), true_t, chosen_action, a_complexity, surety, simulated_time, time_since_food, time_since_water, time_since_sleep, 0, optimal_traj, best_actions, Nt, memory_accessed);

            chosen_action(t) = best_actions(1);
            t = t + 1;
            % end loop over time points

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

        fid = fopen(file_name, 'a+');
        fprintf(fid, '%f\n', t);

        survived(trial) = t;

        % Sample data for demonstration

        % Calculating total runtime for this trial
        endTime = datestr(now +1/24/60/60, 'yyyy-mm-dd HH:MM:SS');
        totalRuntimeInSeconds = etime(datevec(endTime), datevec(startTime));
        minutes = floor(mod(totalRuntimeInSeconds, 3600) / 60);
        seconds = mod(totalRuntimeInSeconds, 60);

        % Subtract 1 from t, because in Python trial data is presented
        % before t = t + 1, within the while loop
        fprintf('At time step %d the agent is dead\n', t - 1);
        fprintf('The agent had %d food, %d water, and %d sleep.\n', 22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep);
        fprintf('The total tree search depth for this trial was %d. \n', search_depth);
        fprintf('The agent accessed its memory %d times. \n', memory_accessed);
        fprintf('The agent cleared its short-term memory %d times. \n', memory_resets(trial));
        fprintf('     State prediction error memory resets: %d. \n', pe_memory_resets(trial));
        fprintf('     Hill memory resets: %d. \n', hill_memory_resets(trial));
        fprintf('TRIAL %d COMPLETE ✔\n', trial);
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

        % reset for next iteration
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
    fprintf('EXPERIMENT COMPLETE ✔.\n');
    fprintf('End Time: %s\n', total_endTime);
    fprintf('TOTAL RUNTIME (hours/minutes/seconds): %02d:%02d:%02d\n', hours, minutes, seconds);
    fprintf('AVERAGE RUNTIME PER TIME STEP: %.3f seconds\n', totalRuntimeInSeconds / total_t);
    fprintf('Average hill visits per time step: %.3f. \n', sum(hill_memory_resets(:)) / total_t);
    fprintf('Average predition errors per time step: %.3f. \n', sum(pe_memory_resets(:)) / total_t);
    fprintf('Average search depth per time step: %.0f. \n', sum(total_search_depth(:)) / total_t);
    fprintf('Average times memory accessed per time step: %.0f. \n', total_memory_accessed / total_t);
    fprintf('----------------------------------------\n');
end
