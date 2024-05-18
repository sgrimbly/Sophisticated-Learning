% Call the function to generate configurations
generate_grids();

function generate_grids()
    rng(1);

    % Define grid sizes and horizons
    grid_sizes = [3, 5, 7, 10];
    horizons = {[1, 2], [2, 4], [3, 5], [3, 5, 7]};
    num_configs_per_setup = [1, 2, 3, 4];
    
    outputFile = fopen('src/MATLAB/grid_configs.txt', 'w');

    for i = 1:length(grid_sizes)
        grid_size = grid_sizes(i);
        center = floor((grid_size + 1) / 2);
        hill_position = sub2ind([grid_size, grid_size], center, center);

        for j = 1:length(horizons{i})
            horizon = horizons{i}(j);
            valid_positions = getValidPositions(grid_size, hill_position, horizon);

            generated_configs = containers.Map('KeyType', 'char', 'ValueType', 'logical');
            config_count = 0;

            while config_count < num_configs_per_setup(i)
                seasons = cell(1, 4);
                unique = true;
                
                % Generate four seasons
                for k = 1:4
                    seasons{k} = generateConfig(hill_position, valid_positions);
                end
                
                % Check uniqueness
                config_strs = cellfun(@getConfigString, seasons, 'UniformOutput', false);
                if any(isKey(generated_configs, config_strs))
                    unique = false;
                end
                
                if unique
                    for k = 1:4
                        generated_configs(config_strs{k}) = true;
                    end
                    config_count = config_count + 1;

                    % Write to file
                    fprintf(outputFile, 'Grid Size: %d, Horizon: %d, Hill: %d\n', grid_size, horizon, hill_position);
                    for k = 1:4
                        fprintf(outputFile, 'Season %d: Food(%d), Water(%d), Sleep(%d)\n', k, seasons{k}(2), seasons{k}(3), seasons{k}(4));
                    end
                    fprintf(outputFile, '\n');
                end
            end
        end
    end

    fclose(outputFile);
end

function positions = getValidPositions(grid_size, hill_position, horizon)
    [hill_row, hill_col] = ind2sub([grid_size, grid_size], hill_position);
    positions = [];
    for row = 1:grid_size
        for col = 1:grid_size
            if norm([row - hill_row, col - hill_col], 'inf') <= horizon
                positions = [positions, sub2ind([grid_size, grid_size], row, col)];
            end
        end
    end
end

function config = generateConfig(hill_position, valid_positions)
    valid_positions(valid_positions == hill_position) = [];
    selected_positions = randperm(length(valid_positions), 3);
    config = [hill_position, valid_positions(selected_positions)];
end

function config_str = getConfigString(config)
    config_str = sprintf('%d,', config);
    config_str = config_str(1:end-1);  % Remove the trailing comma
end
