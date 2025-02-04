function [survived] = SI_GPU(seed)
% SI_GPU: A GPU-accelerated version of the original SI function.
% =============================================================
% This script preserves the original code structure and logic but replaces
% CPU-based operations with GPU-based equivalents. Large arrays (A, a, B, short_term_memory, etc.)
% are allocated on the GPU using gpuArray. The random number generator is set to
% parallel.gpu.RandStream('Threefry4x64_20'), preserving the original "threefry" stream.
%
% NOTE: The sub-functions (normalise_matrix, spm_cross, spm_backwards, tree_search_frwd_SI,
%       calculate_posterior, etc.) must also be adapted for GPU data or gather/scatter
%       internally if they operate on cpuArrays. Here, we assume they can handle gpuArrays.

% NOTE: I think we can still split some workers over CPUs. 
%       "You could wrap the entire trials loop (for trial=1:num_trials) in a parfor to run each 
%       trial on a separate worker (CPU parallelism). If you do, ensure that file logging is carefully 
%       handled (e.g., each worker writes to its own file, or use parfeval).


    % 1) Set the GPU-based random stream (replacing rng(seed, 'threefry')).
    RandStream.setGlobalStream(parallel.gpu.RandStream('Threefry4x64_20','Seed',seed));
    % If you'd like to check the RNG state, you could do: rng

    % -----------------------
    % Original constants
    % -----------------------
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

    novelty_weight = 10;
    learning_weight = 40;
    epistemic_weight = 1;
    preference_weight = 10;

    num_states = 100;

    % -----------------------
    % GPU-accelerated creation of A and a
    % -----------------------
    % Instead of CPU-based zeros, use gpuArray.zeros for large matrices:
    A = cell(3,1);
    a = cell(3,1);

    A{1} = gpuArray.zeros(num_states, num_states, 4);
    a{1} = gpuArray.zeros(num_states, num_states, 4);

    % Replace the diagonal for-loop:
    % for i = 1:num_states
    %     A{1}(i, i, :) = 1;
    %     a{1}(i, i, :) = 1;
    % end
    % with sub2ind-based vectorisation on the GPU:
    idx = gpuArray(1:num_states);
    linIdx = sub2ind([num_states, num_states, 4], idx, idx, ones(1,num_states));
    A{1}(linIdx) = 1;
    a{1}(linIdx) = 1;

    % Construct A{2}, A{3} on the GPU:
    A{2} = gpuArray.zeros(4, num_states, 4);
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

    A{3} = gpuArray.zeros(5, num_states, 4);
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

    % For a{2}, originally: a{2}(:, :, :) = zeros(4, num_states, 4); a{2} = a{2} + 0.1;
    a{2} = gpuArray.zeros(4, num_states, 4) + 0.1;

    % -----------------------
    % Construct D on the GPU
    % -----------------------
    D = cell(2,1);
    D{1} = gpuArray.zeros(num_states,1);   % position in environment
    D{2} = gpuArray([0.25; 0.25; 0.25; 0.25]);

    D{1}(51) = 1; % starting position
    % normalise(D{1}) => do inline:
    D{1} = D{1} ./ sum(D{1});

    T = 27; % As in original code

    num_modalities = 3;

    % short_term_memory
    % In original: short_term_memory(:, :, :, :) = zeros(35, 35, 35, 400);
    % We'll allocate it on GPU but inside the trial loop if it’s reset each trial.
    % For clarity, we show how to do it once, but each trial does its own reset:

    % -----------------------
    % B, b construction on GPU
    % -----------------------
    B = cell(2,1);
    for action = 1:5
        % B{1}(:, :, action) = eye(num_states);
        % B{2}(:, :, action) = [0.95, 0, 0, 0.05; ...];
        % b{2}(:, :, action) = [0.25, 0.25, 0.25, 0.25; ...];
        % We'll build them after the loop to store in GPU arrays.
    end

    % Preallocate B{1}, B{2} as GPU arrays:
    B{1} = gpuArray.zeros(num_states, num_states, 5);
    B{2} = gpuArray.zeros(4, 4, 5);
    for action = 1:5
        B{1}(:,:,action) = gpuArray.eye(num_states);
        B{2}(:,:,action) = gpuArray([0.95, 0,    0,    0.05; ...
                                     0.05, 0.95, 0,    0;    ...
                                     0,    0.05, 0.95, 0;    ...
                                     0,    0,    0.05, 0.95]);
    end

    % b is just set to B, but the code also tries uniform prior in b{2}; we'll replicate:
    b = B;

    % Now replicate the circshift logic for B{1} with i = 1:num_states on GPU:
    % Instead of looping i=1:num_states, we can do:
    leftMask  = setdiff(gpuArray(1:num_states), [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]);
    B{1}(:, leftMask, 2) = circshift(B{1}(:, leftMask, 2), -1);

    rightMask = setdiff(gpuArray(1:num_states), [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    B{1}(:, rightMask, 3) = circshift(B{1}(:, rightMask, 3), 1);

    upMask    = setdiff(gpuArray(1:num_states), 91:100);
    B{1}(:, upMask, 4)   = circshift(B{1}(:, upMask, 4), 10);

    downMask  = setdiff(gpuArray(1:num_states), 1:10);
    B{1}(:, downMask, 5) = circshift(B{1}(:, downMask, 5), -10);

    % Then set b{1} = B{1}:
    b{1} = B{1};

    % chosen_action array
    % (Though the original code uses chosen_action(1, T-1), we’ll keep it on CPU for convenience):
    % chosen_action = zeros(1, T-1); (We’ll do it inside the loop for each trial)

    time_since_food  = 0;
    time_since_water = 0;
    time_since_sleep = 0;

    % -------------
    % Prepare file I/O
    % -------------
    current_time = char(datetime('now', 'Format', 'HH-mm-ss-SSS'));
    seed_str = num2str(seed);
    directory_path = '/Users/stjohngrimbly/Documents/Sophisticated-Learning';
    file_name = strcat(directory_path, '/SI_Seed_', seed_str, '_', current_time, '.txt');

    t = 1;
    num_trials = 300;
    memory_resets      = zeros(num_trials, 1);
    pe_memory_resets   = zeros(num_trials, 1);
    hill_memory_resets = zeros(num_trials, 1);
    total_search_depth = 0;
    total_memory_accessed = 0;
    total_t = 0;
    survived = zeros(1, num_trials);  % eventually we gather or keep on CPU?

    t_at_25  = 0;
    t_at_50  = 0;
    t_at_75  = 0;
    t_at_100 = 0;

    total_startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');

    % -----------------------
    % Main loop over trials
    % -----------------------
    % We can accelerate or parallelize this with parfor, but to preserve the EXACT order
    % of output logs, we’ll keep it a standard for-loop. If you want to parallelize:
    %   parfor trial = 1:num_trials
    % (then handle concurrency for file writes carefully).
    % For correctness equivalence, we do it one trial at a time:
    for trial = 1:num_trials
        startTime = datestr(now, 'yyyy-mm-dd HH:MM:SS');
        fprintf('\n----------------------------------------\n');
        fprintf('TRIAL %d STARTED\n', trial);
        fprintf('----------------------------------------\n');
        fprintf('Start Time: %s\n', startTime);

        % short_term_memory in GPU
        short_term_memory = gpuArray.zeros(35, 35, 35, 400);

        search_depth = 0;
        memory_accessed = 0;

        % Reset time counters
        t = 1;
        time_since_food  = 0;
        time_since_water = 0;
        time_since_sleep = 0;

        % We keep the data structures Q, P, O, etc. from the original code on CPU or GPU:
        % Because the logic uses many cell arrays, we can store them in GPU or CPU. We'll do partial GPU usage.
        % The user references them intensively, so we must do minimal changes except where beneficial.

        chosen_action = zeros(1, T-1);    % store on CPU
        true_states = cell(num_trials, 1); % storing states as in original code

        while (t < 100 && time_since_food < 22 && time_since_water < 20 && time_since_sleep < 25)

            % b, bb, B logic
            bb{2} = normalise_matrix(b{2});  %#ok<*NASGU> 
            % we assume normalise_matrix is GPU-ready or does gather/spread inside.

            for factor = 1:2
                if t == 1
                    P{t, factor} = D{factor}';
                    Q{t, factor} = D{factor}';
                    % The original code sets initial true_states{trial}(1, t) = 51
                    % but also picks a random for factor=2
                    true_states{trial}(1, t) = 51;  %#ok<*AGROW> 
                    true_states{trial}(2, t) = find(cumsum(D{2}) >= rand, 1);
                else
                    if factor == 1
                        Q{t, factor} = (B{1}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')';
                        % sample next true state
                        old_state = true_states{trial}(factor, t - 1);
                        true_states{trial}(factor, t) = find( ...
                            cumsum(B{1}(:, old_state, chosen_action(t - 1))) >= rand, 1);
                    else
                        Q{t, factor} = (bb{2}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')';
                        old_state = true_states{trial}(factor, t - 1);
                        true_states{trial}(factor, t) = find( ...
                            cumsum(B{2}(:, old_state, 1)) >= rand, 1);
                    end
                end
            end

            % Check resources
            if (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == true_food_source_1) || ...
               (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == true_food_source_2) || ...
               (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == true_food_source_3) || ...
               (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == true_food_source_4)
                time_since_food  = 0;
                time_since_water = time_since_water + 1;
                time_since_sleep = time_since_sleep + 1;

            elseif (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == true_water_source_1) || ...
                   (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == true_water_source_2) || ...
                   (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == true_water_source_3) || ...
                   (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == true_water_source_4)
                time_since_water = 0;
                time_since_food  = time_since_food + 1;
                time_since_sleep = time_since_sleep + 1;

            elseif (true_states{trial}(2, t) == 1 && true_states{trial}(1, t) == true_sleep_source_1) || ...
                   (true_states{trial}(2, t) == 2 && true_states{trial}(1, t) == true_sleep_source_2) || ...
                   (true_states{trial}(2, t) == 3 && true_states{trial}(1, t) == true_sleep_source_3) || ...
                   (true_states{trial}(2, t) == 4 && true_states{trial}(1, t) == true_sleep_source_4)
                time_since_sleep = 0;
                time_since_food  = time_since_food + 1;
                time_since_water = time_since_water + 1;
            else
                if t > 1
                    time_since_food  = time_since_food + 1;
                    time_since_water = time_since_water + 1;
                    time_since_sleep = time_since_sleep + 1;
                end
            end

            % sample next observation
            for modality = 1:num_modalities
                ob = A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t)); %#ok<NASGU> 
                observations(modality, t) = find( ...
                    cumsum(A{modality}(:, true_states{trial}(1, t), true_states{trial}(2, t))) >= rand, 1);
                % One-hot encode
                vec = zeros(1, size(A{modality}, 1));
                vec(observations(modality, t)) = 1;
                O{modality, t} = vec; %#ok<AGROW> 
            end

            true_t = t;

            if t > 1
                start_t = t - 6;
                if start_t <= 0
                    start_t = 1;
                end
                bb{2} = normalise_matrix(b{2});
                y{2}  = normalise_matrix(a{2});
                % spm_cross, normalise, etc. must be GPU-compatible or gather/scatter
                qs = spm_cross(Q{t, :});
                predictive_observations_posterior{2, t} = normalise(y{2} * qs(:))'; %#ok<AGROW> 
                predictive_observations_posterior{3, t} = normalise(y{3} * qs(:))'; %#ok<AGROW> 
                predicted_posterior = calculate_posterior(Q, y, predictive_observations_posterior, t);

                for timey = start_t : t
                    L = spm_backwards(O, Q, A, bb, chosen_action, timey, t);
                    LL{2} = L;
                    LL{1} = Q{timey, 1};

                    if (timey > start_t && ~isequal(round(L, 3), round(Q{timey, 2}, 3)')) || (timey == t)
                        for mod2 = 2:2
                            a_learning = O{mod2, timey};

                            for factor = 1:2
                                a_learning = spm_cross(a_learning, LL{factor});
                            end

                            a_learning = a_learning .* (a{mod2} > 0);
                            proportion = 0.3;

                            for iLearn = 1:size(a_learning, 3)
                                for j = 1:size(a_learning, 2)
                                    max_value = max(a_learning(2:end, j, iLearn));
                                    amount_to_subtract = proportion * max_value;
                                    zero_mask = (a_learning(1, j, iLearn) == 0);
                                    a_learning(zero_mask, j, iLearn) = ...
                                        a_learning(zero_mask, j, iLearn) - amount_to_subtract;
                                end
                            end

                            a{mod2} = a{mod2} + 0.7 * a_learning;
                            a{mod2}(a{mod2} <= 0.05) = 0.05;
                        end
                    end
                end
            end

            % determine which season
            if true_states{trial}(2, t) == 1
                food  = true_food_source_1;
                water = true_water_source_1;
                sleep = true_sleep_source_1;
            elseif true_states{trial}(2, t) == 2
                food  = true_food_source_2;
                water = true_water_source_2;
                sleep = true_sleep_source_2;
            elseif true_states{trial}(2, t) == 3
                food  = true_food_source_3;
                water = true_water_source_3;
                sleep = true_sleep_source_3;
            else
                food  = true_food_source_4;
                water = true_water_source_4;
                sleep = true_sleep_source_4;
            end

            y{2} = normalise_matrix(a{2});
            y{1} = A{1};
            y{3} = A{3};

            horizon = min([9, (22 - time_since_food), (20 - time_since_water), (25 - time_since_sleep)]);
            if horizon == 0, horizon = 1; end

            temp_Q = Q;
            temp_Q{t, 2} = temp_Q{t, 2}';
            P = calculate_posterior(temp_Q, y, O, t);
            current_pos(t) = find(cumsum(P{t, 1}) >= rand, 1);

            if (t > 1) && ~isequal(round(predicted_posterior{t, 2}, 1), round(P{t, 2}, 1))
                short_term_memory(:) = 0;
                memory_resets(trial)    = memory_resets(trial) + 1;
                pe_memory_resets(trial) = pe_memory_resets(trial) + 1;
            end

            if current_pos(t) == hill_1
                short_term_memory(:) = 0;
                memory_resets(trial)    = memory_resets(trial) + 1;
                hill_memory_resets(trial) = hill_memory_resets(trial) + 1;
            end

            best_actions = [];
            [G, Q, short_term_memory, best_actions, memory_accessed] = tree_search_frwd_SI( ...
                short_term_memory, O, Q, a, A, y, B, B, t, T, ...
                t + horizon, time_since_food, time_since_water, time_since_sleep, ...
                true_t, chosen_action, time_since_food, time_since_water, time_since_sleep, ...
                best_actions, learning_weight, novelty_weight, epistemic_weight, ...
                preference_weight, memory_accessed);

            chosen_action(t) = best_actions(1);

            t = t + 1;
            search_depth = search_depth + length(best_actions);
        end

        total_search_depth      = total_search_depth + search_depth;
        total_memory_accessed   = total_memory_accessed + memory_accessed;
        total_t                 = total_t + t;

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
        fclose(fid);

        survived(trial) = t;

        % Log trial summary to console
        endTime = datestr(now + 1/24/60/60, 'yyyy-mm-dd HH:MM:SS');
        totalRuntimeInSeconds = etime(datevec(endTime), datevec(startTime));
        minutes = floor(mod(totalRuntimeInSeconds, 3600) / 60);
        seconds = mod(totalRuntimeInSeconds, 60);

        fprintf('At time step %d the agent is dead\n', t - 1);
        fprintf('The agent had %d food, %d water, and %d sleep.\n', ...
            22 - time_since_food, 20 - time_since_water, 25 - time_since_sleep);
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
    end

    % Final experiment summary
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

    % Return 'survived' on CPU
    survived = gather(survived);
end