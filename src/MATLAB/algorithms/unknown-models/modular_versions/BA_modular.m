function [survived] = BA_modular(seed, grid_size, hill_pos, food_sources, water_sources, sleep_sources, num_states, num_trials)
    % Set default values if not provided
    if nargin < 2, grid_size = 10; end
    if nargin < 3, hill_pos = 55; end
    if nargin < 4, food_sources = [71, 43, 57, 78]; end
    if nargin < 5, water_sources = [73, 33, 48, 67]; end
    if nargin < 6, sleep_sources = [64, 44, 49, 59]; end
    if nargin < 7, num_states = 100; end
    if nargin < 8, num_trials = 300; end

    rng(seed)
    rng

    % Call the utility function to initialise the environment
    [A, a, B, b, D, T, num_modalities] = initialiseEnvironment(num_states, grid_size, hill_pos, food_sources, water_sources, sleep_sources);

    chosen_action = zeros(1, T - 1);
    time_since_food = 0;
    time_since_water = 0;
    time_since_sleep = 0;

    current_time = char(datetime('now', 'Format', 'HH-mm-ss-SSS'));
    seed_str = num2str(seed);
    % directory_path = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/300trials_data';
    directory_path = 'C:\Users\micro\Documents\ActiveInference_Work\Sophisticated-Learning'
    file_name = strcat(directory_path, '\BA_Seed_', seed_str, '_', current_time, '.txt');

    t = 1;
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

    total_startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    for trial = 1:num_trials
        startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        fprintf('\n----------------------------------------\n');
        fprintf('TRIAL %d STARTED\n', trial);
        fprintf('----------------------------------------\n');
        fprintf('Start Time: %s\n', startTime);

        short_term_memory(:, :, :, :, :) = zeros(35, 35, 35, 400, 5);
        search_depth = 0;
        memory_accessed = 0;

        for factor = 1:2
            Q{1, factor} = D{factor}';
            P{1, factor} = D{factor}';
            true_states{trial}(1, t) = 51;
            true_states{trial}(2, t) = find(cumsum(D{2}) >= rand, 1);
        end

        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)
            bb{2} = normalise_matrix(b{2});

            if t ~= 1
                for factor = 1:2
                    Q_prev = Q{t - 1, factor}';
                    
                    if factor == 1
                        Q{t, factor} = (B{1}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')';
                        true_states{trial}(factor, t) = find(cumsum(B{1}(:, true_states{trial}(factor, t - 1), chosen_action(t - 1))) >= rand, 1);
                    else
                        Q{t, factor} = (bb{2}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')'; %(B{2}(:,:)'
                        true_states{trial}(factor, t) = find(cumsum(B{2}(:, true_states{trial}(factor, t - 1), 1)) >= rand, 1);

                    end
                end            
            end
            if ismember(true_states{trial}(2, t), 1:4) && ismember(true_states{trial}(1, t), food_sources)
                time_since_food = 0;
                time_since_water = time_since_water + 1;
                time_since_sleep = time_since_sleep + 1;
            elseif ismember(true_states{trial}(2, t), 1:4) && ismember(true_states{trial}(1, t), water_sources)
                time_since_water = 0;
                time_since_food = time_since_food + 1;
                time_since_sleep = time_since_sleep + 1;
            elseif ismember(true_states{trial}(2, t), 1:4) && ismember(true_states{trial}(1, t), sleep_sources)
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
                start = t - 6;
                if start <= 0, start = 1; end

                bb{2} = normalise_matrix(b{2});
                y{2} = normalise_matrix(a{2});
                qs = spm_cross(Q{t, :});
                predictive_observations_posterior{2, t} = normalise(y{2}(:, :) * qs(:))';
                predictive_observations_posterior{3, t} = normalise(y{3}(:, :) * qs(:))';
                predicted_posterior = calculate_posterior(Q, y, predictive_observations_posterior, t);

                for timey = start:t
                    L = spm_backwards(O, Q, A, bb, chosen_action, timey, t);
                    LL{2} = L;
                    LL{1} = Q{timey, 1};

                    if (timey > start && ~isequal(round(L, 3), round(Q{timey, 2}, 3)')) || (timey == t)
                        a_prior = a{2};

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
            horizon = min([9, min([22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep])]);
            if horizon == 0, horizon = 1; end

            temp_Q = Q;
            temp_Q{t, 2} = temp_Q{t, 2}';
            P = calculate_posterior(temp_Q, y, O, t);
            current_pos(t) = find(cumsum(P{t, 1}) >= rand, 1);

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
            [G, Q, short_term_memory, best_actions, memory_accessed] = tree_search_frwd(short_term_memory, O, Q, a, A, y, B, B, t, T, t + horizon, time_since_food, time_since_water, time_since_sleep, true_t, chosen_action, 0, time_since_food, time_since_water, time_since_sleep, best_actions, memory_accessed);
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

        fid = fopen(file_name, 'a+');
        fprintf(fid, '%f\n', t);

        survived(trial) = t;

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
