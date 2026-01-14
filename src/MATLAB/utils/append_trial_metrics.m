function append_trial_metrics(metrics_file, run_meta, trial_idx, survival_steps, param_update_kl, a2, food_sources, water_sources, sleep_sources)
    if nargin < 9
        error('append_trial_metrics requires 9 inputs.');
    end

    if ~isstruct(run_meta) || ~isfield(run_meta, 'config_id')
        error('run_meta must be a struct with field config_id.');
    end

    if isempty(metrics_file)
        return
    end

    [metrics_dir, ~, ~] = fileparts(metrics_file);
    if ~isempty(metrics_dir) && ~exist(metrics_dir, 'dir')
        [ok, msg] = mkdir(metrics_dir);
        if ~ok
            error('Unable to create metrics directory: %s', msg);
        end
    end

    write_header = ~exist(metrics_file, 'file');
    fid = fopen(metrics_file, 'a');
    if fid == -1
        error('Unable to open metrics file for writing: %s', metrics_file);
    end

    if write_header
        fprintf(fid, 'config_id,trial,survival,param_update_kl');
        for c = 1:4
            fprintf(fid, ',p_food_c%d,p_water_c%d,p_sleep_c%d', c, c, c);
        end
        fprintf(fid, '\n');
    end

    y2 = normalise_matrix(a2);
    probs = zeros(1, 12);
    for c = 1:4
        probs((c - 1) * 3 + 1) = y2(2, food_sources(c), c);
        probs((c - 1) * 3 + 2) = y2(3, water_sources(c), c);
        probs((c - 1) * 3 + 3) = y2(4, sleep_sources(c), c);
    end

    fprintf(fid, '%s,%d,%d,%.15g', run_meta.config_id, trial_idx, survival_steps, param_update_kl);
    for i = 1:numel(probs)
        fprintf(fid, ',%.15g', probs(i));
    end
    fprintf(fid, '\n');

    fclose(fid);
end

