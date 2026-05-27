function dashboard_run_one(algorithm, cfg, progress_queue, results_dir, grid_id, algorithm_label_override)
%DASHBOARD_RUN_ONE Run a single algorithm/seed and emit per-trial progress via DataQueue.

    % Each parfeval worker is intended to use one CPU; without this MKL inside
    % MATLAB will spawn multiple BLAS threads per worker and oversubscribe the
    % node. Single-thread BLAS keeps timing comparisons clean and matches the
    % implicit assumption in local_parallel_dashboard's worker count.
    maxNumCompThreads(1);

    if nargin < 5
        error('dashboard_run_one requires algorithm, cfg, progress_queue, results_dir, grid_id.');
    end
    if nargin < 6
        algorithm_label_override = '';
    end

    thisFile = mfilename('fullpath');
    matlabDir = fileparts(thisFile); % .../src/MATLAB
    addpath(genpath(matlabDir));

    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    weights = cfg.weights;
    if ~isfield(weights, 'learning_prune_threshold')
        weights.learning_prune_threshold = 0.2;
    end
    if ~isfield(weights, 'rng_algorithm')
        weights.rng_algorithm = 'twister';
    end

    algorithm_label_for_run = algorithm;
    if ~isempty(algorithm_label_override)
        algorithm_label_for_run = algorithm_label_override;
    end

    algorithm_spec = resolve_algorithm_spec(algorithm_label_for_run);

    novelty_for_run = weights.novelty;
    if ~algorithm_spec.novelty_on
        novelty_for_run = 0;
    end

    real_smoothing_for_run = logical(weights.real_smoothing);
    adaptive_for_run = logical(weights.adaptive_likelihood_in_plan);
    if algorithm_spec.is_unknown_model
        real_smoothing_for_run = algorithm_spec.smoothing_on;
        adaptive_for_run = algorithm_spec.adaptive_plan_on;
    end

    % Preference parameterisation:
    %   - preference_param='weight' : larger values strengthen the extrinsic term
    %   - preference_param='inverse_precision' : larger values weaken the extrinsic term
    if strcmp(weights.preference_param, 'inverse_precision')
        if isfield(weights, 'preference_inverse_precision')
            preference_value = weights.preference_inverse_precision;
        else
            preference_value = weights.preference;
        end
    else
        preference_value = weights.preference;
    end

    weight_vector = [novelty_for_run, weights.learning, weights.epistemic, preference_value];

    run_options = struct(...
        'state_selection', weights.state_selection, ...
        'preference_param', weights.preference_param, ...
        'rng_algorithm', weights.rng_algorithm, ...
        'baucb_variant', weights.baucb_variant, ...
        'real_smoothing', logical(real_smoothing_for_run), ...
        'adaptive_likelihood_in_plan', adaptive_for_run, ...
        'learning_prune_threshold', weights.learning_prune_threshold, ...
        'algorithm_label', algorithm_spec.label, ...
        'collect_efe_components', true, ...
        'compute_policy_sensitivity', true, ...
        'progress_queue', progress_queue ...
    );

    algorithm_label_safe = sanitize_file_component(algorithm_spec.label);
    results_file_override = fullfile(results_dir, sprintf('%s_Seed%d.txt', algorithm_label_safe, cfg.seed));

    switch algorithm_spec.implementation
        case 'SI'
            SI_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'SI_smooth'
            SI_smooth_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'SL'
            SL_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'SL_noSmooth'
            SL_noSmooth_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'BA'
            BA_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'BAUCB'
            BAUCB_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, weights.ucb_scale, results_file_override, cfg.max_horizon, run_options);
        otherwise
            error('Unsupported algorithm "%s" for local dashboard.', algorithm);
    end
end
