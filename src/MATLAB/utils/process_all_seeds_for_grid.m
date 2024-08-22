function process_all_seeds_for_grid(num_states, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, seeds, algorithm, grid_id, directory_path)
    % Print out the values and types of all inputs
    fprintf('Input types and values received:\n');
    
    fprintf('grid_size (type: %s):\n', class(grid_size));
    disp(grid_size);
        
    fprintf('start_position (type: %s):\n', class(start_position));
    disp(start_position);
    
    fprintf('hill_pos (type: %s):\n', class(hill_pos));
    disp(hill_pos);
    
    fprintf('food_sources (type: %s):\n', class(food_sources));
    disp(food_sources);
    
    fprintf('water_sources (type: %s):\n', class(water_sources));
    disp(water_sources);
    
    fprintf('sleep_sources (type: %s):\n', class(sleep_sources));
    disp(sleep_sources);
    
    fprintf('seeds (type: %s):\n', class(seeds));
    disp(seeds);
    
    fprintf('algorithm (type: %s):\n', class(algorithm));
    disp(algorithm);
    
    fprintf('grid_id (type: %s):\n', class(grid_id));
    disp(grid_id);
    
    fprintf('directory_path (type: %s):\n', class(directory_path));
    disp(directory_path);

    % Now process each seed
    for i = 1:length(seeds)
        seed = seeds(i);
        fprintf('Processing seed %d for grid_id %s with algorithm %s\n', seed, grid_id, algorithm);
        
        % Call the likelihood_divergence_analysis function
        likelihood_divergence_analysis(num_states, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, seed, algorithm, grid_id, directory_path);
    end
end