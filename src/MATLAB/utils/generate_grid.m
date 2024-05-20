% Call the function to generate configurations
generate_grids();

function generate_grids()
    rng(1);  % Seed for reproducibility

    % Define grid sizes and horizons
    grid_sizes = [3, 5, 7, 10];
    horizons = {[1, 2], [2, 4], [3, 5], [4, 6]};
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
                resources = {'Food', 'Water', 'Sleep'};
                season_data = cell(1, 3);  % To store resource data across all seasons

                % Generate four seasons and a start position
                start_position = [];  % Initialize an empty array for start position
                for k = 1:4
                    seasons{k} = generateConfig(hill_position, valid_positions);
                    for r = 1:3
                        season_data{r}(k) = seasons{k}(r+1);
                    end
                end
                
                % Ensure start position is neither hill nor resources
                all_positions = [hill_position, [seasons{:}]];
                remaining_positions = setdiff(valid_positions, all_positions);
                if isempty(remaining_positions)
                    continue;  % Skip if no valid start positions are available
                end
                start_position = remaining_positions(randperm(numel(remaining_positions), 1));  % Select one random valid start position

                % Check uniqueness
                config_strs = cellfun(@getConfigString, seasons, 'UniformOutput', false);
                if any(isKey(generated_configs, config_strs))
                    continue;  % Skip if config is not unique
                end

                for k = 1:4
                    generated_configs(config_strs{k}) = true;
                end
                config_count = config_count + 1;

                % Construct and write formatted output to file
                line_to_write = sprintf('Grid Size: %d, Horizon: %d, Hill: %d, Start Position: %d, ', grid_size, horizon, hill_position, start_position);
                for r = 1:3
                    resource_string = sprintf('%d,', season_data{r});
                    resource_string = resource_string(1:end-1);  % Remove the trailing comma
                    line_to_write = [line_to_write sprintf('%s(%s), ', resources{r}, resource_string)];
                end
                line_to_write = line_to_write(1:end-2);  % Remove the last comma and space
                fprintf(outputFile, '%s\n', line_to_write);
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
