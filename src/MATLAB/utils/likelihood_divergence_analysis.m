function likelihood_divergence_analysis(num_states, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, seed, algorithm, grid_id, directory_path)
    % Set default values if parameters are not provided
    if nargin < 1 || isempty(num_states)
        num_states = 100;
    end
    if nargin < 2 || isempty(grid_size)
        grid_size = 10;
    end
    if nargin < 3 || isempty(start_position)
        start_position = 51;
    end
    if nargin < 4 || isempty(hill_pos)
        hill_pos = 55;
    end
    if nargin < 5 || isempty(food_sources)
        food_sources = [71, 43, 57, 78];
    end
    if nargin < 6 || isempty(water_sources)
        water_sources = [73, 33, 48, 67];
    end
    if nargin < 7 || isempty(sleep_sources)
        sleep_sources = [64, 44, 49, 59];
    end
    if nargin < 8 || isempty(seed)
        seed = 120;
    end
    if nargin < 9 || isempty(algorithm)
        algorithm = 'SI'; % Default algorithm to SI if not provided
    end
    if nargin < 10 || isempty(grid_id)
        error('Grid ID is required');
    end
    if nargin < 11 || isempty(directory_path)
        directory_path = '/home/grmstj001/MATLAB-experiments/Sophisticated-Learning/results/unknown_model/MATLAB/grid_config_experiments';
    end

    disp(['num_states: ', num2str(num_states)]);

    % Initialize the environment matrices
    [A, ~, ~, ~, ~, ~, ~] = initialiseEnvironment(num_states, start_position, grid_size, hill_pos, food_sources, water_sources, sleep_sources);

    % Create subdirectory for saving plots if it doesn't exist
    save_dir = strcat(directory_path, '/', algorithm, '_GridID_', grid_id);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Construct the filename based on the algorithm, seed, and grid ID
    stateFile = strcat(directory_path, '/', algorithm, '_Seed_', num2str(seed), '_GridID_', grid_id, '.mat');
    
    % Check if the state file exists
    if ~isfile(stateFile)
        error('State file %s does not exist.', stateFile);
    end
    
    % Load state file
    [loadedState, isNew] = load_state(stateFile);
    a_history = loadedState{3};  % Beliefs over time
    true_states = loadedState{7}; % Retrieve true_states
    
    % Display the structure of a_history
    disp('Structure of a_history:');
    for trial = 1:length(a_history)
        disp(['Trial ', num2str(trial), ':']);
        for modality = 1:length(a_history{trial})
            disp(['  a_history{', num2str(trial), '}{', num2str(modality), '} size: ', mat2str(size(a_history{trial}{modality}))]);
        end
    end

    % Calculate KL divergence over time for each context and resource
    kl_over_time = calculate_kl_over_time(A, a_history, true_states);

    % Plot the KL divergence over time for each context
    figure;
    for context = 1:4
        subplot(2, 2, context);
        hold on;
        plot(kl_over_time{context}(1, :), 'r', 'LineWidth', 2, 'DisplayName', 'No Resource');
        plot(kl_over_time{context}(2, :), 'g', 'LineWidth', 2, 'DisplayName', 'Food');
        plot(kl_over_time{context}(3, :), 'b', 'LineWidth', 2, 'DisplayName', 'Water');
        plot(kl_over_time{context}(4, :), 'm', 'LineWidth', 2, 'DisplayName', 'Sleep');
        title(['KL Divergence for Context ', num2str(context)]);
        xlabel('Time Steps');
        ylabel('KL Divergence');
        legend;
        grid on;
        hold off;
    end

    % Save KL divergence over time plot
    saveas(gcf, strcat(save_dir, '/KL_Divergence_Time_Seed_', num2str(seed), '.png'));

    % % Calculate final KL divergence at the last time step for each trial
    % final_kl = calculate_final_kl(A, a_history, true_states);
    % % Print final kl divergence for each context
    % disp('Final KL Divergence:');
    % for context = 1:4
    %     disp(['Context ', num2str(context), ':']);
    %     % Shape
    %     disp(['Shape of final_kl{', num2str(context), '}: ', mat2str(size(final_kl{context}))]);
    % end

    % Plot the final KL divergence for each context
    % figure;
    % for context = 1:4
    %     subplot(2, 2, context);
    %     hold on;
    %     plot(1:size(final_kl{context}, 2), final_kl{context}(1, :), 'r', 'LineWidth', 2, 'DisplayName', 'No Resource');
    %     plot(1:size(final_kl{context}, 2), final_kl{context}(2, :), 'g', 'LineWidth', 2, 'DisplayName', 'Food');
    %     plot(1:size(final_kl{context}, 2), final_kl{context}(3, :), 'b', 'LineWidth', 2, 'DisplayName', 'Water');
    %     plot(1:size(final_kl{context}, 2), final_kl{context}(4, :), 'm', 'LineWidth', 2, 'DisplayName', 'Sleep');
    %     title(['Final KL Divergence for Context ', num2str(context)]);
    %     xlabel('Trial Number');
    %     ylabel('Final KL Divergence');
    %     legend;
    %     grid on;
    %     hold off;
    % end

    % % Save final KL divergence plot
    % saveas(gcf, strcat(save_dir, '/Final_KL_Divergence_Seed_', num2str(seed), '.png'));
end

function kl_over_time = calculate_kl_over_time(A, a_history, true_states)
    num_trials = length(true_states);  % Total number of trials to process
    
    % Determine the number of contexts and resources dynamically
    num_contexts = size(A{2}, 3);  % Number of contexts
    num_resources = size(A{2}, 1);  % Number of resources
    
    % Display the determined number of contexts and resources
    disp(['Number of contexts: ', num2str(num_contexts)]);
    disp(['Number of resources: ', num2str(num_resources)]);
    disp(['Size of A{2}: ', mat2str(size(A{2}))]);

    % Initialize cell array to store KL divergences over time for each context
    kl_over_time = cell(1, num_contexts);
    for context = 1:num_contexts
        kl_over_time{context} = zeros(num_resources, 0);
    end

    time_counter = 1;  % Global time step counter across all trials

    for trial = 1:num_trials
        current_trial_a = a_history{trial};  % Get the belief at this trial
        num_time_steps = length(true_states{trial});  % Use true_states to determine the number of time steps

        disp(['Trial ', num2str(trial), ': num_time_steps = ', num2str(num_time_steps)]);

        for t = 1:num_time_steps
            disp(['  Time step: ', num2str(t)]);
            a_timestep = extract_belief_at_timestep(current_trial_a, t);

            % Display sizes of a_timestep for debugging
            disp(['  Size of a_timestep{2}: ', mat2str(size(a_timestep{2}))]);

            % Calculate KL divergences for each context
            kl_divergences = calculate_kl_per_resource(A, a_timestep);

            % Store KL divergences for each context
            for context_idx = 1:num_contexts
                kl_over_time{context_idx}(:, time_counter) = kl_divergences(:, context_idx);
            end

            time_counter = time_counter + 1;  % Increment global time counter
        end
    end
end

function final_kl = calculate_final_kl(A, a_history, true_states)
    num_trials = length(true_states);  % Total number of trials to process
    num_contexts = 4;  % Number of contexts
    num_resources = 4;  % Number of resources (No Resource, Food, Water, Sleep)
    
    % Initialize cell array to store final KL divergences for each context
    final_kl = cell(1, num_contexts);
    for context = 1:num_contexts
        final_kl{context} = zeros(num_resources, num_trials);
    end

    for trial = 1:num_trials
        current_trial_a = a_history{trial};  % Get the belief at this trial
        num_time_steps = length(true_states{trial});  % Use true_states to determine the number of time steps
        a_timestep = extract_belief_at_timestep(current_trial_a, num_time_steps);  % Extract final timestep

        % Calculate KL divergences for each context
        kl_divergences = calculate_kl_per_resource(A, a_timestep);

        % Store final KL divergences for each context
        for context = 1:num_contexts
            final_kl{context}(:, trial) = kl_divergences(:, context);
        end
    end
end

function kl_divergences = calculate_kl_per_resource(A, a_timestep)
    num_resources = size(A{2}, 1);  % Number of resources
    num_contexts = size(A{2}, 3);   % Number of contexts
    kl_divergences = zeros(num_resources, num_contexts);

    % Display sizes of A{2} and a_timestep{2}
    disp(['Number of resources (A{2}): ', num2str(num_resources)]);
    disp(['Size of A{2}: ', mat2str(size(A{2}))]);
    disp(['Size of a_timestep{2}: ', mat2str(size(a_timestep{2}))]);

    % Calculate KL divergence for each resource separately for each context
    for context = 1:num_contexts
        for resource = 1:num_resources
            disp(['      Processing context: ', num2str(context), ', resource: ', num2str(resource)]);
            p = A{2}(resource, :, context); % True distribution for the resource
            q = a_timestep{2}(resource, :, context); % Belief distribution for the resource

            % Display sizes before reshaping
            disp(['      Size of p: ', mat2str(size(p)), ', Number of elements in p: ', num2str(numel(p))]);
            disp(['      Size of q: ', mat2str(size(q)), ', Number of elements in q: ', num2str(numel(q))]);

            % Check if the number of elements match
            if numel(p) == numel(q)
                q = reshape(q, size(p));  % Reshape q to match p's dimensionality
            else
                error('Mismatch in the number of elements between p and q. Reshaping not possible.');
            end

            % Normalize to ensure they sum to 1
            p_sum = sum(p(:));
            q_sum = sum(q(:));
            p = p / p_sum;
            q = q / q_sum;

            % Display sum checks
            disp(['      Sum of p after normalization: ', num2str(sum(p(:)))]);
            disp(['      Sum of q after normalization: ', num2str(sum(q(:)))]);

            % Calculate KL divergence for this resource and context
            p(p == 0) = eps; % Avoid log of zero
            q(q == 0) = eps; % Avoid division by zero
            kl_divergences(resource, context) = sum(p .* log(p ./ q), 'all'); % Compute KL divergence

            % Display intermediate KL divergence value
            disp(['      KL divergence for resource ', num2str(resource), ', context ', num2str(context), ': ', num2str(kl_divergences(resource, context))]);
        end
    end
end

function a_timestep = extract_belief_at_timestep(current_trial_a, t)
    % Extract the belief distribution at a specific time step
    a_timestep = cell(size(current_trial_a));

    for modality = 1:length(current_trial_a)
        if ndims(current_trial_a{modality}) == 3
            % Since the third dimension isn't time steps, do not index it based on `t`
            % Instead, consider each full slice of the third dimension for processing
            a_timestep{modality} = current_trial_a{modality}(:, :, :); % Take the entire 3D slice
        else
            % Handle cases where the modality does not have a third dimension
            a_timestep{modality} = current_trial_a{modality}(:, t); % Handle 2D case, if applicable
        end
        disp(['    Extracted a_timestep{', num2str(modality), '} size: ', mat2str(size(a_timestep{modality}))]);
    end
end
