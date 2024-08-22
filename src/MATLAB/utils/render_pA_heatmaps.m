function render_pA(a_history, fps)
    % Number of frames to process
    % num_frames = length(a_history);
    num_frames = 55;

    % Prepare output filenames for each context
    for context = 1:4 % Assuming 4 contexts
        gif_filename = sprintf('dynamic_experiment_visualization_context_%d.gif', context);

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

% Test this rendering
seed = 120
directory_path = '/Users/stjohngrimbly/Documents/Sophisticated-Learning/src/MATLAB';
stateFile = strcat(directory_path, '/SI_Seed_', num2str(seed), '.mat')
[loadedState, isNew] = load_state(stateFile);
a_history = loadedState{3}; % 1x120 cell array, each element is the 'a' likelihood (1x3) for each timestep

% Set parameters for the test
fps = 10;
render_pA(a_history, fps);
