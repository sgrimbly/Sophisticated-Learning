function append_step_metrics(step_metrics_file, run_meta, algorithm, seed, trial_idx, real_step, chosen_action, efe_components, param_update_kl_step, search_depth_so_far)
    if nargin < 10
        error('append_step_metrics requires at least 10 inputs.');
    end
    if nargin < 11
        search_depth_so_far = NaN;
    end

    if isempty(step_metrics_file)
        return
    end

    if ~isstruct(run_meta) || ~isfield(run_meta, 'config_id')
        error('run_meta must be a struct with field config_id.');
    end

    [metrics_dir, ~, ~] = fileparts(step_metrics_file);
    if ~isempty(metrics_dir) && ~exist(metrics_dir, 'dir')
        [ok, msg] = mkdir(metrics_dir);
        if ~ok
            error('Unable to create step metrics directory: %s', msg);
        end
    end

    novelty_sum = get_component_field(efe_components, 'novelty_term_sum');
    epistemic_sum = get_component_field(efe_components, 'epistemic_term_sum');
    extrinsic_sum = get_component_field(efe_components, 'extrinsic_term_sum');
    future_sum = get_component_field(efe_components, 'future_term_sum');
    node_count = get_component_field(efe_components, 'node_count');

    if ~isnan(node_count) && node_count > 0
        novelty_mean = novelty_sum / node_count;
        epistemic_mean = epistemic_sum / node_count;
        extrinsic_mean = extrinsic_sum / node_count;
    else
        novelty_mean = NaN;
        epistemic_mean = NaN;
        extrinsic_mean = NaN;
    end

    write_header = ~exist(step_metrics_file, 'file');
    fid = fopen(step_metrics_file, 'a');
    if fid == -1
        error('Unable to open step metrics file for writing: %s', step_metrics_file);
    end

    if write_header
        fprintf(fid, 'config_id,algorithm,seed,trial,real_step,chosen_action,planning_node_count,planning_novelty_term_sum,planning_epistemic_term_sum,planning_extrinsic_term_sum,planning_future_term_sum,planning_novelty_term_mean,planning_epistemic_term_mean,planning_extrinsic_term_mean,param_update_kl_step,search_depth_so_far\n');
    end

    fprintf(fid, '%s,%s,%d,%d,%d,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n', ...
        run_meta.config_id, char(string(algorithm)), seed, trial_idx, real_step, chosen_action, node_count, ...
        novelty_sum, epistemic_sum, extrinsic_sum, future_sum, novelty_mean, epistemic_mean, extrinsic_mean, ...
        param_update_kl_step, search_depth_so_far);

    fclose(fid);
end

function value = get_component_field(efe_components, field_name)
    value = NaN;
    if ~isstruct(efe_components) || ~isfield(efe_components, field_name)
        return
    end

    raw_value = efe_components.(field_name);
    if isempty(raw_value)
        return
    end

    value = double(raw_value);
    if ~isscalar(value)
        value = value(1);
    end
end
