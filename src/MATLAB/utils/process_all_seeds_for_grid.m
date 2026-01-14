function process_all_seeds_for_grid(num_states, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, seeds, algorithm, grid_id, directory_path, config_id_filter)
    if nargin < 12
        config_id_filter = '';
    end

    algorithm_safe = sanitize_file_component(algorithm);
    grid_id_safe = sanitize_file_component(grid_id);
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

    seeds = unique(seeds);

    state_pattern = sprintf('%s_Seed_*_GridID_%s_Cfg_*.mat', algorithm_safe, grid_id_safe);
    all_state_files = dir(fullfile(directory_path, '**', state_pattern));
    if isempty(all_state_files)
        error('No state files found matching %s under %s', state_pattern, directory_path);
    end

    records = struct('seed', {}, 'config_id', {}, 'fullpath', {}, 'datenum', {});
    rx = sprintf('^%s_Seed_(\\d+)_GridID_%s_Cfg_(.+)\\.mat$', regexptranslate('escape', algorithm_safe), regexptranslate('escape', grid_id_safe));
    for i = 1:numel(all_state_files)
        f = all_state_files(i);
        tok = regexp(f.name, rx, 'tokens', 'once');
        if isempty(tok)
            continue
        end
        seed_i = str2double(tok{1});
        cfg_i = tok{2};
        records(end + 1) = struct(...
            'seed', seed_i, ...
            'config_id', cfg_i, ...
            'fullpath', fullfile(f.folder, f.name), ...
            'datenum', f.datenum ...
        ); %#ok<AGROW>
    end

    if isempty(records)
        error('State file parsing failed for pattern %s under %s', state_pattern, directory_path);
    end

    % Determine which config_id to analyse (avoid mixing configs across seeds).
    if ~isempty(config_id_filter)
        selected_cfg = config_id_filter;
    else
        cfg_ids = unique({records.config_id});
        if numel(cfg_ids) == 1
            selected_cfg = cfg_ids{1};
        else
            cfg_counts = zeros(size(cfg_ids));
            cfg_latest = zeros(size(cfg_ids));
            for c = 1:numel(cfg_ids)
                cfg = cfg_ids{c};
                cfg_records = records(strcmp({records.config_id}, cfg));
                cfg_seeds = unique([cfg_records.seed]);
                cfg_counts(c) = sum(ismember(seeds, cfg_seeds));
                cfg_latest(c) = max([cfg_records.datenum]);
            end
            max_count = max(cfg_counts);
            best_candidates = find(cfg_counts == max_count);
            if numel(best_candidates) == 1
                best_idx_final = best_candidates;
            else
                [~, latest_rel] = max(cfg_latest(best_candidates));
                best_idx_final = best_candidates(latest_rel);
            end
            selected_cfg = cfg_ids{best_idx_final};
            warning('Multiple config_id values found for %s GridID=%s. Auto-selecting config_id=%s (pass config_id_filter to override).', algorithm_safe, grid_id_safe, selected_cfg);
        end
    end

    % Now process each seed for the selected config_id.
    for i = 1:length(seeds)
        seed = seeds(i);
        fprintf('Processing seed %d for grid_id %s with algorithm %s (Cfg=%s)\n', seed, grid_id, algorithm, selected_cfg);

        seed_cfg_records = records([records.seed] == seed & strcmp({records.config_id}, selected_cfg));
        if isempty(seed_cfg_records)
            warning('No state file found for seed=%d algorithm=%s grid_id=%s config_id=%s. Skipping.', seed, algorithm_safe, grid_id_safe, selected_cfg);
            continue
        end

        [~, newest_idx] = max([seed_cfg_records.datenum]);
        state_file_path = seed_cfg_records(newest_idx).fullpath;

        % Call the likelihood_divergence_analysis function
        likelihood_divergence_analysis(num_states, grid_size, start_position, hill_pos, food_sources, water_sources, sleep_sources, seed, algorithm, grid_id, directory_path, state_file_path);
    end
end
