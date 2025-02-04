function main
    % Initialize environment parameters
    num_states = 100; % Total number of states
    grid_size = 10; % Size of the grid (10x10)
    start_position = 51; 
    hill_pos = 55; % Example hill position
    food_sources = [71, 43, 57, 78]; % Example food source positions
    water_sources = [73, 33, 48, 67]; % Example water source positions
    sleep_sources = [64, 44, 49, 59]; % Example sleep source positions

    % Initialize the environment matrices
    [A, ~, ~, ~, ~, ~, ~] = initialiseEnvironment(num_states, start_position, grid_size, hill_pos, food_sources, water_sources, sleep_sources);

    % Load state file
    seed = 120;
    directory_path = '/Users/stjohngrimbly/Documents/Sophisticated-Learning/src/MATLAB';
    % stateFile = strcat(directory_path, '/SI_Seed_120.mat');
    stateFile = strcat(directory_path, '/SI_Seed_171_GridID_3982848cccdb02f9f4f43953268e076d.mat');
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

    % Calculate final KL divergence at the last time step for each trial
    final_kl = calculate_final_kl(A, a_history, true_states);
    % Print final kl divergence for each context
    disp('Final KL Divergence:');
    for context = 1:4
        disp(['Context ', num2str(context), ':']);
        % disp(final_kl{context});
        % Shape
        disp(class(final_kl{context}));
        disp(['Shape of final_kl{', num2str(context), '}: ', mat2str(size(final_kl{context}))]);
    end
    % Shape of final kl
    disp(['Shape of final_kl: ', mat2str(size(final_kl))]);
    % Plot the final KL divergence for each context
    % Plot the final KL divergence for each context
    % Plot the final KL divergence for each context
    % figure;
    % % Plot the final KL divergence for each context
    % figure;
    % for context = 1:4
    %     subplot(2, 2, context);
    %     hold on;
        
    %     % Extract the matrix from the cell array
    %     kl_matrix = final_kl{context};  % This should be a 4x200 matrix
        
    %     % Check if the matrix is non-empty and valid
    %     if ~isempty(kl_matrix) && isnumeric(kl_matrix) && all(size(kl_matrix) == [4, 200])
    %         num_trials = size(kl_matrix, 2);  % Get the number of trials (should be 200)
            
    %         % Plot for each resource
    %         plot(1:num_trials, kl_matrix(1, :), 'r', 'DisplayName', 'No Resource');
    %         plot(1:num_trials, kl_matrix(2, :), 'g', 'LineWidth', 2, 'DisplayName', 'Food');
    %         plot(1:num_trials, kl_matrix(3, :), 'b', 'LineWidth', 2, 'DisplayName', 'Water');
    %         plot(1:num_trials, kl_matrix(4, :), 'm', 'LineWidth', 2, 'DisplayName', 'Sleep');
            
    %         title(['Final KL Divergence for Context ', num2str(context)]);
    %         xlabel('Trial Number');
    %         ylabel('Final KL Divergence');
    %         legend;
    %         grid on;
    %     else
    %         disp(['Context ', num2str(context), ' contains invalid or empty data. Skipping plot.']);
    %     end
    %     hold off;
    % end
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
            % disp(['  Time step: ', num2str(t)]);
            a_timestep = extract_belief_at_timestep(current_trial_a, t);

            % Calculate KL divergences for each context
            kl_divergences = calculate_kl_per_resource(A, a_timestep);

            % Display the size of kl_divergences
            disp(['kl_divergences size: ', mat2str(size(kl_divergences))]);

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

            % Only attempt to reshape if the number of elements match
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
        % disp(['Extracted a_timestep{', num2str(modality), '} size: ', mat2str(size(a_timestep{modality}))]);
    end
end

function render_pA(a_history, fps)
    % Number of frames to process
    num_frames = 55;
    disp(['Number of frames: ', num2str(num_frames)]);

    % Prepare output filenames for each context
    for context = 1:4 % Assuming 4 contexts
        gif_filename = sprintf('dynamic_experiment_visualization_context_%d.gif', context);
        disp(['Creating GIF for context: ', num2str(context)]);

        % Initialize GIF creation
        for time_step = 1:num_frames
            img_array = create_dynamic_frame(a_history{time_step}, context);

            % Convert RGB image to indexed image
            [indexedImg, map] = rgb2ind(img_array, 256);

            % Write frame to GIF
            if time_step == 1
                imwrite(indexedImg, map, gif_filename, 'gif', 'LoopCount', Inf, 'DelayTime', 1 / fps);
            else
                imwrite(indexedImg, map, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 1 / fps);
            end
        end
    end

    disp('Outputs for all contexts created successfully.');
end

function img_array = create_dynamic_frame(a, context_idx)
    fig = figure('Visible', 'off');
    set(gcf, 'Position', [100, 100, 1200, 800]); % Adjust size to fit visualization needs

    % Loop over each 'a' in the timestep
    for i = 1:length(a)
        subplot(1, length(a), i);
        data = squeeze(a{i}(:, :, context_idx));
        disp(['Data shape for subplot ', num2str(i), ': ', mat2str(size(data))]);

        imagesc(data);
        colormap('hot');
        colorbar;
        title(sprintf('Type %d, Context %d', i, context_idx));
        axis off;
    end

    % Save figure to image
    frame = getframe(gcf);
    img_array = frame2im(frame);
    close(fig);
end
