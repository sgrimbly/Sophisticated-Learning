function [survived, step_metrics_path] = perf_run_canonical(algorithm, num_trials, seed, max_horizon, workdir)
%PERF_RUN_CANONICAL Run a modular algorithm with the canonical local-dashboard
% config in single-process mode, with SL_LOG_METRICS=1 so the per-step CSV
% is emitted. Returns the survived array and the path to the step-metrics CSV.
%
% Used by perf_capture_golden_trace, perf_check_golden_trace, perf_time_step.
%
%   algorithm   : 'SI' | 'SL' | 'BA' | 'BAUCB' | 'SI_smooth' | 'SL_noSmooth'
%   num_trials  : positive integer
%   seed        : positive integer
%   max_horizon : positive integer (typically 9)
%   workdir     : directory for state files and per-step CSV (will be wiped)

    if nargin < 5
        error('perf_run_canonical requires algorithm, num_trials, seed, max_horizon, workdir.');
    end

    if exist(workdir, 'dir')
        rmdir(workdir, 's');
    end
    mkdir(workdir);

    cfg = struct();
    cfg.grid_size = 10;
    cfg.start_position = 51;
    cfg.hill_pos = 55;
    cfg.food_sources = [71, 43, 57, 78];
    cfg.water_sources = [73, 33, 48, 67];
    cfg.sleep_sources = [64, 44, 49, 59];
    cfg.num_states = cfg.grid_size ^ 2;
    cfg.num_trials = num_trials;
    cfg.max_horizon = max_horizon;
    cfg.seed = seed;

    weights = struct(...
        'novelty', 10, ...
        'learning', 40, ...
        'epistemic', 1, ...
        'preference', 10, ...
        'ucb_scale', 5, ...
        'state_selection', 'sample', ...
        'preference_param', 'weight', ...
        'baucb_variant', 'legacy', ...
        'real_smoothing', true, ...
        'adaptive_likelihood_in_plan', false, ...
        'learning_prune_threshold', 0.2, ...
        'rng_algorithm', 'twister');

    algorithm_spec = resolve_algorithm_spec(algorithm);

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

    weight_vector = [novelty_for_run, weights.learning, weights.epistemic, weights.preference];

    run_options = struct(...
        'state_selection', weights.state_selection, ...
        'preference_param', weights.preference_param, ...
        'rng_algorithm', weights.rng_algorithm, ...
        'baucb_variant', weights.baucb_variant, ...
        'real_smoothing', real_smoothing_for_run, ...
        'adaptive_likelihood_in_plan', adaptive_for_run, ...
        'learning_prune_threshold', weights.learning_prune_threshold, ...
        'algorithm_label', algorithm_spec.label, ...
        'collect_efe_components', true, ...
        'compute_policy_sensitivity', false);

    grid_id = 'perf';
    algorithm_label_safe = sanitize_file_component(algorithm_spec.label);
    results_file_override = fullfile(workdir, sprintf('%s_Seed%d.txt', algorithm_label_safe, seed));
    step_metrics_path = fullfile(workdir, sprintf('%s_Seed%d_step_metrics.csv', algorithm_label_safe, seed));

    prev_log = getenv('SL_LOG_METRICS');
    setenv('SL_LOG_METRICS', '1');
    cleanup = onCleanup(@() setenv('SL_LOG_METRICS', prev_log));

    switch algorithm_spec.implementation
        case 'SI'
            survived = SI_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'SI_smooth'
            survived = SI_smooth_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'SL'
            survived = SL_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'SL_noSmooth'
            survived = SL_noSmooth_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'BA'
            survived = BA_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, results_file_override, cfg.max_horizon, run_options);
        case 'BAUCB'
            survived = BAUCB_modular(cfg.seed, cfg.grid_size, cfg.start_position, cfg.hill_pos, cfg.food_sources, cfg.water_sources, cfg.sleep_sources, ...
                weight_vector, cfg.num_states, cfg.num_trials, grid_id, weights.ucb_scale, results_file_override, cfg.max_horizon, run_options);
        otherwise
            error('Unsupported algorithm: %s', algorithm);
    end

    if ~exist(step_metrics_path, 'file')
        warning('perf_run_canonical:NoStepMetrics', ...
            'Step metrics CSV not found at %s. SL_LOG_METRICS may have been ignored.', step_metrics_path);
    end
end
