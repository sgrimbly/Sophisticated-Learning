function [A, a, B, b, D, T, num_modalities] = initialiseEnvironment(num_states, grid_size, hill_pos, food_sources, water_sources, sleep_sources)
    % Initialise Environment Variables
    A{1}(:, :, :) = zeros(num_states, num_states, 4);
    a{1}(:, :, :) = zeros(num_states, num_states, 4);

    for i = 1:num_states
        A{1}(i, i, :) = 1;
        a{1}(i, i, :) = 1;
    end

    A{2}(:, :, :) = zeros(4, num_states, 4);
    for i = 1:4
        A{2}(1, :, i) = 1;
        A{2}(2, food_sources(i), i) = 1;
        A{2}(1, food_sources(i), i) = 0;
        A{2}(3, water_sources(i), i) = 1;
        A{2}(1, water_sources(i), i) = 0;
        A{2}(4, sleep_sources(i), i) = 1;
        A{2}(1, sleep_sources(i), i) = 0;
    end

    A{3}(:, :, :) = zeros(5, num_states, 4);
    A{3}(5, :, :) = 1;
    for i = 1:4
        A{3}(i, hill_pos, i) = 1;
        A{3}(5, hill_pos, i) = 0;
    end
    a{3} = A{3};
    a{2}(:, :, :) = zeros(4, num_states, 4);
    a{2} = a{2} + 0.1;

    D{1} = zeros(1, num_states)'; 
    D{2} = [0.25, 0.25, 0.25, 0.25]';
    D{1}(51) = 1; 
    D{1} = normalise(D{1});
    T = 27;
    num_modalities = 3;

    short_term_memory(:, :, :, :) = zeros(35, 35, 35, 400);

    for action = 1:5
        B{1}(:, :, action) = eye(num_states);
        B{2}(:, :, action) = [0.95, 0, 0, 0.05;
                              0.05, 0.95, 0, 0;
                              0, 0.05, 0.95, 0;
                              0, 0, 0.05, 0.95];

        b{2}(:, :, action) = 0.25 * ones(4);
    end

    b = B;

    for i = 1:num_states
        if mod(i, grid_size) ~= 1
            B{1}(:, i, 2) = circshift(B{1}(:, i, 2), -1); % move left
        end
    end

    for i = 1:num_states
        if mod(i, grid_size) ~= 0
            B{1}(:, i, 3) = circshift(B{1}(:, i, 3), 1); % move right
        end
    end

    for i = 1:num_states
        if i > grid_size
            B{1}(:, i, 4) = circshift(B{1}(:, i, 4), grid_size); % move up
        end
    end

    for i = 1:num_states
        if i <= num_states - grid_size
            B{1}(:, i, 5) = circshift(B{1}(:, i, 5), -grid_size); % move down
        end
    end

    b{1} = B{1};
end

function [P, Q, true_states] = updateEnvironmentStates(t, num_states, chosen_action, D, B, bb, food_sources)
    for factor = 1:2
        if t == 1
            P{t, factor} = D{factor}';
            Q{t, factor} = D{factor}';
            true_states{1, t} = 51;
            true_states{2, t} = find(cumsum(D{2}) >= rand, 1);
        else
            if factor == 1
                Q{t, factor} = (B{1}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')';
                true_states{factor, t} = find(cumsum(B{1}(:, true_states{factor, t - 1}, chosen_action(t - 1))) >= rand, 1);
            else
                Q{t, factor} = (bb{2}(:, :, chosen_action(t - 1)) * Q{t - 1, factor}')';
                true_states{factor, t} = find(cumsum(B{2}(:, true_states{factor, t - 1), 1)) >= rand, 1);
            end
        end
    end
end
