% Call the function to generate configurations
% Define the file path and grid size
CONFIGS_FILE = '/Users/stjohngrimbly/Documents/Sophisticated-Learning/src/MATLAB/grid_configs.txt';
GRID_SIZE = 10;  % Set this to your grid size
defaultConfigStr = 'Grid Size: 10, Horizon: 5, Hill: 55, Start Position: 51, Food(71,43,57,78), Water(73,33,48,67), Sleep(64,44,49,59)';

% TODO: Ensure the following config is always first, which is the config Rowan used (and therefore I did too) for extensive experimentation
% Grid Size: 10, Horizon: 5, Hill: 55, Start Position: 51, Food(71,43,57,78), Water(73,33,48,67), Sleep(64,44,49,59)

% Call the function to generate configurations
generate_grids(CONFIGS_FILE);
displayGridConfigsFromFile(CONFIGS_FILE, GRID_SIZE);
generateMD5(defaultConfigStr) 

function hash = generateMD5(inputString)
    md = java.security.MessageDigest.getInstance('MD5');
    hashBytes = md.digest(uint8(inputString));
    hash = sprintf('%02x', typecast(hashBytes, 'uint8'));
end

function generate_grids(output_file)
    rng(1);  % Seed for reproducibility

    % Define the default configuration
    defaultConfigStr = 'Grid Size: 10, Horizon: 5, Hill: 55, Start Position: 51, Food(71,43,57,78), Water(73,33,48,67), Sleep(64,44,49,59)';

    % Compute hash for the default configuration
    defaultConfigHash = generateMD5(defaultConfigStr);

    % Open the output file
    outputFile = fopen(output_file, 'w');

    % Write the default configuration as the first line
    fprintf(outputFile, 'Grid ID: %s, %s\n', defaultConfigHash, defaultConfigStr);

    grid_sizes = [10];
    horizons = {[3,5]};
    num_configs_per_setup = [5];

    for i = 1:length(grid_sizes)
        grid_size = grid_sizes(i);
        center = floor((grid_size + 1) / 2);
        hill_position = sub2ind([grid_size, grid_size], center, (center+1));

        for j = 1:length(horizons{i})
            horizon = horizons{i}(j);
            valid_positions = getValidPositions(grid_size, hill_position, horizon);

            generated_configs = containers.Map('KeyType', 'char', 'ValueType', 'logical');
            config_count = 0;

            while config_count < num_configs_per_setup(i)
                seasons = cell(1, 4);
                resources = {'Food', 'Water', 'Sleep'};
                season_data = cell(1, 3);  

                start_position = [];
                for k = 1:4
                    seasons{k} = generateConfig(hill_position, valid_positions);
                    for r = 1:3
                        season_data{r}(k) = seasons{k}(r+1);
                    end
                end
                
                all_positions = [hill_position, [seasons{:}]];
                remaining_positions = setdiff(valid_positions, all_positions);
                if isempty(remaining_positions)
                    continue;
                end
                start_position = remaining_positions(randperm(numel(remaining_positions), 1));

                config_strs = cellfun(@getConfigString, seasons, 'UniformOutput', false);
                if any(isKey(generated_configs, config_strs))
                    continue;
                end

                for k = 1:4
                    generated_configs(config_strs{k}) = true;
                end
                config_count = config_count + 1;

                % Generate unique hash ID
                config_str_full = [config_strs{:}];
                hash_id = generateMD5(config_str_full);

                % Construct and write formatted output to file
                line_to_write = sprintf('Grid ID: %s, Grid Size: %d, Horizon: %d, Hill: %d, Start Position: %d, ', hash_id, grid_size, horizon, hill_position, start_position);
                for r = 1:3
                    resource_string = sprintf('%d,', season_data{r});
                    resource_string = resource_string(1:end-1);
                    line_to_write = [line_to_write sprintf('%s(%s), ', resources{r}, resource_string)];
                end
                line_to_write = line_to_write(1:end-2);
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

function displayGridConfigsFromFile(filename, gridSize)
    % Open and read the file
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('File could not be opened, check the name or path.');
    end
    
    % Read all configurations into a cell array
    configs = textscan(fileID, '%s', 'Delimiter', '\n');
    configs = configs{1};
    fclose(fileID);
    
    % Create a larger figure to hold all subplots
    figure('Position', [100, 100, 1600, 900]);  % Larger figure size
    
    % Determine the number of configurations and sets per config
    numConfigs = length(configs);
    setsPerConfig = 4;  % As there are 4 sets in each configuration
    
    % Iterate over each configuration
    for i = 1:numConfigs
        configLine = configs{i};
        gridID = regexp(configLine, 'Grid ID: ([\w]+),', 'tokens');
        gridID = gridID{1}{1};  % Extract the Grid ID
        
        for j = 1:setsPerConfig
            % Calculate subplot index
            idx = (i-1) * setsPerConfig + j;
            
            % Subplot adjustment for spacing
            ax = subplot(numConfigs, setsPerConfig, idx);
            set(ax, 'Position', get(ax, 'Position') + [0.01 0.01 -0.02 -0.02]);  % Slightly increase each plot size
            
            displaySingleSet(configLine, j, gridSize, gridID);
            title(sprintf('Grid ID: %s, Set %d', gridID, j), 'FontSize', 10);  % Display Grid ID in the title
        end
    end
end

function displaySingleSet(config, setIndex, gridSize, gridID)
    % Extract positions from the config string for a specific set
    [startPos, hillPos, foodPositions, waterPositions, sleepPositions] = parseConfig(config, setIndex);

    % Prepare the subplot for display
    hold on;
    axis equal;
    xlim([0, gridSize + 1]);  % Dynamic adjustment based on grid size
    ylim([0, gridSize + 1]);  % Dynamic adjustment based on grid size
    set(gca, 'XTick', [], 'YTick', [], 'FontSize', 10);  % Clean plot presentation

    % Plot positions
    plotPosition(startPos, 'A', 'r', gridSize);
    plotPosition(hillPos, 'H', 'k', gridSize);
    plotPositions(foodPositions, 'F', 'g', gridSize);
    plotPositions(waterPositions, 'W', 'b', gridSize);
    plotPositions(sleepPositions, 'S', 'm', gridSize);

    hold off;
end

function [startPos, hillPos, foodPos, waterPos, sleepPos] = parseConfig(config, setIndex)
    % Extract numerical values from the config string for a specific set
    startPosMatch = regexp(config, 'Start Position: (\d+),', 'tokens');
    hillPosMatch = regexp(config, 'Hill: (\d+),', 'tokens');
    
    if ~isempty(startPosMatch)
        startPos = str2double(startPosMatch{1});
    else
        startPos = NaN; % or some error handling
    end
    
    if ~isempty(hillPosMatch)
        hillPos = str2double(hillPosMatch{1});
    else
        hillPos = NaN; % or some error handling
    end

    foodStr = regexp(config, 'Food\(([^)]+)\)', 'tokens');
    waterStr = regexp(config, 'Water\(([^)]+)\)', 'tokens');
    sleepStr = regexp(config, 'Sleep\(([^)]+)\)', 'tokens');
    
    foodArr = str2num(foodStr{1}{1});
    waterArr = str2num(waterStr{1}{1});
    sleepArr = str2num(sleepStr{1}{1});
    
    foodPos = foodArr(setIndex);
    waterPos = waterArr(setIndex);
    sleepPos = sleepArr(setIndex);
    
    % disp(['Parsed Start Position: ', num2str(startPos)]);
    % disp(['Parsed Hill Position: ', num2str(hillPos)]);
end

function plotPosition(pos, symbol, color, gridSize)
    % Converts linear index to subscript based on gridSize
    [row, col] = ind2sub([gridSize, gridSize], pos);
    % Ensure plotting occurs within the adjusted grid scale
    adjustedRow = gridSize + 1 - row;  % Adjust row for MATLAB's Y-axis inversion
    % Plot the position using text with visual adjustments
    text(col, adjustedRow, symbol, 'Color', color, 'FontSize', max(14 - (gridSize - 7)*2, 8), 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

function plotPositions(positions, symbol, color, gridSize)
    % Plot multiple positions with dynamic grid size
    for i = 1:length(positions)
        plotPosition(positions(i), symbol, color, gridSize);
    end
end