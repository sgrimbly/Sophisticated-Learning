function local_parallel_dashboard(varargin)
%LOCAL_PARALLEL_DASHBOARD Run one or more seeds per algorithm in parallel and plot mean progress.
%
% Requirements:
%   - Parallel Computing Toolbox (parpool, parfeval, DataQueue)
%
% Usage:
%   local_parallel_dashboard
%
% Optional name-value overrides:
%   'Algorithms'  : cellstr of algorithm names
%   'Weights'     : struct overriding default weights/flags
%   'Seed'        : scalar integer seed
%   'Seeds'       : vector of integer seeds (overrides 'Seed')
%   'NumTrials'   : number of trials per algorithm
%   'MaxHorizon'  : planning horizon cap
%   'DashboardTitle' : figure title

    defaults = struct();
    defaults.Algorithms = { ...
        'SI', 'SI_novelty', 'SI_novelty_smooth', ...
        'SL_noSmooth', 'SL', ...
        'BA', 'BAUCB' ...
    };
	    defaults.Seed = 1;
	    defaults.NumTrials = 20;
	    defaults.MaxHorizon = 9;
	    defaults.RealSmoothing = true;
	    defaults.AdaptiveLikelihoodInPlan = false;
	    defaults.LearningPruneThreshold = 0.2;
	    defaults.Weights = [];
	    defaults.DashboardTitle = 'Local Parallel Dashboard';
	    defaults.Seeds = [];

    parser = inputParser();
    parser.addParameter('Algorithms', defaults.Algorithms, @(x) iscellstr(x) || (iscell(x) && all(cellfun(@ischar, x))));
	    parser.addParameter('Weights', defaults.Weights, @(x) isempty(x) || isstruct(x));
	    parser.addParameter('Seed', defaults.Seed, @(x) isnumeric(x) && isscalar(x) && x == floor(x));
	    parser.addParameter('Seeds', defaults.Seeds, @(x) isempty(x) || (isnumeric(x) && isvector(x) && all(x == floor(x))));
	    parser.addParameter('NumTrials', defaults.NumTrials, @(x) isnumeric(x) && isscalar(x) && x == floor(x) && x > 0);
	    parser.addParameter('MaxHorizon', defaults.MaxHorizon, @(x) isnumeric(x) && isscalar(x) && x == floor(x) && x > 0);
	    parser.addParameter('RealSmoothing', defaults.RealSmoothing, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
	    parser.addParameter('AdaptiveLikelihoodInPlan', defaults.AdaptiveLikelihoodInPlan, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
	    parser.addParameter('LearningPruneThreshold', defaults.LearningPruneThreshold, @(x) isnumeric(x) && isscalar(x) && x >= 0);
	    parser.addParameter('DashboardTitle', defaults.DashboardTitle, @(x) ischar(x) || isstring(x));
	    parser.parse(varargin{:});
	    algorithms = parser.Results.Algorithms;
	    dashboard_title = char(parser.Results.DashboardTitle);
	    seed = parser.Results.Seed;
	    seeds = parser.Results.Seeds;
	    num_trials = parser.Results.NumTrials;
	    max_horizon = parser.Results.MaxHorizon;
	    real_smoothing = logical(parser.Results.RealSmoothing);
	    adaptive_likelihood_in_plan = logical(parser.Results.AdaptiveLikelihoodInPlan);
	    learning_prune_threshold = parser.Results.LearningPruneThreshold;
	    user_weights = parser.Results.Weights;
	    using_defaults = parser.UsingDefaults;
	    if isempty(seeds)
	        seeds = seed;
	    end
	    seeds = unique(seeds(:)');
	    n_seeds = numel(seeds);

    thisFile = mfilename('fullpath');
    matlabDir = fileparts(thisFile);                 % .../src/MATLAB
    projectRoot = fullfile(matlabDir, '..', '..');   % repo root
    addpath(genpath(matlabDir));

    run_id = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    grid_id = sprintf('localDashboard_%s', run_id);
    results_dir = fullfile(projectRoot, 'results', 'local-dashboard', run_id);
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

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
		    weights = struct(...
		        'novelty', 10, ...
		        'learning', 40, ...
		        'epistemic', 1, ...
		        'preference', 10, ...
		        'ucb_scale', 5, ...
		        'state_selection', 'sample', ...
		        'preference_param', 'weight', ...
		        'baucb_variant', 'legacy', ...
		        'real_smoothing', real_smoothing, ...
		        'adaptive_likelihood_in_plan', adaptive_likelihood_in_plan, ...
		        'learning_prune_threshold', learning_prune_threshold ...
		    );
		    if ~isempty(user_weights)
		        provided_fields = fieldnames(user_weights);
		        for fi = 1:numel(provided_fields)
		            k = provided_fields{fi};
		            weights.(k) = user_weights.(k);
		        end
		    end
		    if ~ismember('RealSmoothing', using_defaults)
		        weights.real_smoothing = real_smoothing;
		    end
		    if ~ismember('AdaptiveLikelihoodInPlan', using_defaults)
		        weights.adaptive_likelihood_in_plan = adaptive_likelihood_in_plan;
		    end
		    if ~ismember('LearningPruneThreshold', using_defaults)
		        weights.learning_prune_threshold = learning_prune_threshold;
		    end
		    cfg.weights = weights;

    cluster = parcluster('local');
    requested_workers = min(numel(algorithms) * n_seeds, cluster.NumWorkers);
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool(cluster, requested_workers);
    end

    dq = parallel.pool.DataQueue();

    algoIndex = containers.Map(algorithms, 1:numel(algorithms));
    seedIndex = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for si = 1:n_seeds
        seedIndex(seeds(si)) = si;
    end
    n_alg = numel(algorithms);

    survival_raw = nan(n_alg, n_seeds, num_trials);
    param_update_kl_raw = nan(n_alg, n_seeds, num_trials);
    search_depth_raw = nan(n_alg, n_seeds, num_trials);
    policy_sensitivity_raw = nan(n_alg, n_seeds, num_trials);

    efe_novelty_sum_raw = nan(n_alg, n_seeds, num_trials);
    efe_epistemic_sum_raw = nan(n_alg, n_seeds, num_trials);
    efe_extrinsic_sum_raw = nan(n_alg, n_seeds, num_trials);
    efe_node_count_raw = nan(n_alg, n_seeds, num_trials);

    survival = nan(n_alg, num_trials);
    param_update_kl = nan(n_alg, num_trials);
    novelty_term_mean = nan(n_alg, num_trials);
    epistemic_term_mean = nan(n_alg, num_trials);
	    extrinsic_term_mean = nan(n_alg, num_trials);
	    search_depth = nan(n_alg, num_trials);
	    policy_sensitivity = nan(n_alg, num_trials);

	    fig = figure('Name', dashboard_title, 'NumberTitle', 'off');
	    layout = tiledlayout(fig, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
	    title(layout, dashboard_title, 'Interpreter', 'none');

    ax_survival = nexttile; hold(ax_survival, 'on'); title(ax_survival, 'Survival'); xlabel(ax_survival, 'Trial'); ylabel(ax_survival, 'Steps');
    ax_kl = nexttile; hold(ax_kl, 'on'); title(ax_kl, 'Param Update KL'); xlabel(ax_kl, 'Trial'); ylabel(ax_kl, 'Sum KL');
    ax_novelty = nexttile; hold(ax_novelty, 'on'); title(ax_novelty, 'Novelty Term (Mean/Planning Node)'); xlabel(ax_novelty, 'Trial'); ylabel(ax_novelty, 'Novelty');
	    ax_epi = nexttile; hold(ax_epi, 'on'); title(ax_epi, 'Epistemic Term (Mean/Planning Node)'); xlabel(ax_epi, 'Trial'); ylabel(ax_epi, 'Epistemic');
	    ax_ext = nexttile; hold(ax_ext, 'on'); title(ax_ext, 'Extrinsic Term (Mean/Planning Node)'); xlabel(ax_ext, 'Trial'); ylabel(ax_ext, 'Extrinsic');
	    ax_depth = nexttile; hold(ax_depth, 'on'); title(ax_depth, 'Search Depth'); xlabel(ax_depth, 'Trial'); ylabel(ax_depth, 'Depth');
	    ax_sens = nexttile; hold(ax_sens, 'on'); title(ax_sens, 'Novelty Policy Sensitivity'); xlabel(ax_sens, 'Trial'); ylabel(ax_sens, 'Frac. actions changed');

    colors = lines(n_alg);
    x = 1:num_trials;
    line_survival = gobjects(n_alg, 1);
    line_kl = gobjects(n_alg, 1);
    line_novelty = gobjects(n_alg, 1);
	    line_epi = gobjects(n_alg, 1);
	    line_ext = gobjects(n_alg, 1);
	    line_depth = gobjects(n_alg, 1);
	    line_sens = gobjects(n_alg, 1);

	    display_names = cell(1, n_alg);
	    for i = 1:n_alg
	        display_names{i} = pretty_algorithm_label(algorithms{i}, weights);
	    end

	    for i = 1:n_alg
	        style = line_style_for_algorithm(algorithms{i});
	        line_survival(i) = plot(ax_survival, x, survival(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	        line_kl(i) = plot(ax_kl, x, param_update_kl(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	        line_novelty(i) = plot(ax_novelty, x, novelty_term_mean(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	        line_epi(i) = plot(ax_epi, x, epistemic_term_mean(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	        line_ext(i) = plot(ax_ext, x, extrinsic_term_mean(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	        line_depth(i) = plot(ax_depth, x, search_depth(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	        line_sens(i) = plot(ax_sens, x, policy_sensitivity(i, :), style, 'Color', colors(i, :), 'DisplayName', display_names{i});
	    end

	    legend(ax_survival, 'Location', 'eastoutside');

	    ax_info = nexttile;
	    axis(ax_info, 'off');
	    seed_label = sprintf('%d', seeds(1));
	    if n_seeds > 1
	        seed_label = strjoin(arrayfun(@num2str, seeds, 'UniformOutput', false), ', ');
	    end
	    info_lines = { ...
	        sprintf('Seeds (n=%d): %s', n_seeds, seed_label), ...
	        sprintf('Plot: mean over seeds'), ...
	        sprintf('Trials: %d', num_trials), ...
	        sprintf('Max horizon: %d', max_horizon), ...
	        sprintf('state_selection: %s', string(weights.state_selection)), ...
	        sprintf('preference_param: %s', string(weights.preference_param)), ...
	        sprintf('real_smoothing: %d', double(logical(weights.real_smoothing))), ...
	        sprintf('adaptive_likelihood_in_plan: %d', double(logical(weights.adaptive_likelihood_in_plan))), ...
	        sprintf('learning_prune_threshold: %g', weights.learning_prune_threshold), ...
	        sprintf('baucb_variant: %s', string(weights.baucb_variant)), ...
	        sprintf('ucb_scale: %g', weights.ucb_scale), ...
	        '', ...
	        'EFE term panels show mean per planning node:', ...
	        '  sum(term over visited nodes) / node_count', ...
	        '', ...
	        'Algorithms:', ...
	    };
	    for i = 1:n_alg
	        info_lines{end+1} = sprintf('  - %s', display_names{i}); %#ok<AGROW>
	    end
	    text(ax_info, 0, 1, info_lines, 'VerticalAlignment', 'top', 'Interpreter', 'none', 'FontName', 'FixedWidth');

    afterEach(dq, @onProgress);

    futures = parallel.FevalFuture.empty(0, n_alg * n_seeds);
    fi = 0;
    for i = 1:n_alg
        for si = 1:n_seeds
            cfg_run = cfg;
            cfg_run.seed = seeds(si);
            fi = fi + 1;
            futures(fi) = parfeval(pool, @dashboard_run_one, 0, algorithms{i}, cfg_run, dq, results_dir, grid_id);
        end
    end

    wait(futures);

    function onProgress(msg)
        if ~isstruct(msg) || ~isfield(msg, 'algorithm') || ~isfield(msg, 'trial')
            return
        end
        if ~isKey(algoIndex, msg.algorithm)
            return
        end
        i = algoIndex(msg.algorithm);
        t = msg.trial;
        if t < 1 || t > num_trials
            return
        end

        s = 1;
        if isfield(msg, 'seed')
            seed_value = double(msg.seed);
            if isKey(seedIndex, seed_value)
                s = seedIndex(seed_value);
            elseif n_seeds > 1
                return
            end
        elseif n_seeds > 1
            return
        end

        survival_raw(i, s, t) = msg.survival;
        param_update_kl_raw(i, s, t) = msg.param_update_kl;
        search_depth_raw(i, s, t) = msg.search_depth;

        node_count = NaN;
        if isfield(msg, 'efe_node_count')
            node_count = msg.efe_node_count;
        elseif isfield(msg, 'efe_steps')
            node_count = msg.efe_steps;
        end
        if ~isnan(node_count) && node_count > 0
            efe_node_count_raw(i, s, t) = node_count;
            efe_novelty_sum_raw(i, s, t) = msg.efe_novelty_term_sum;
            efe_epistemic_sum_raw(i, s, t) = msg.efe_epistemic_term_sum;
            efe_extrinsic_sum_raw(i, s, t) = msg.efe_extrinsic_term_sum;
        end

        survival(i, t) = mean_over_seeds(survival_raw(i, :, t));
        param_update_kl(i, t) = mean_over_seeds(param_update_kl_raw(i, :, t));
        search_depth(i, t) = mean_over_seeds(search_depth_raw(i, :, t));

        novelty_term_mean(i, t) = weighted_mean_over_seeds(efe_novelty_sum_raw(i, :, t), efe_node_count_raw(i, :, t));
        epistemic_term_mean(i, t) = weighted_mean_over_seeds(efe_epistemic_sum_raw(i, :, t), efe_node_count_raw(i, :, t));
        extrinsic_term_mean(i, t) = weighted_mean_over_seeds(efe_extrinsic_sum_raw(i, :, t), efe_node_count_raw(i, :, t));

        line_survival(i).YData = survival(i, :);
        line_kl(i).YData = param_update_kl(i, :);
        line_novelty(i).YData = novelty_term_mean(i, :);
	        line_epi(i).YData = epistemic_term_mean(i, :);
	        line_ext(i).YData = extrinsic_term_mean(i, :);
	        line_depth(i).YData = search_depth(i, :);
	        if isfield(msg, 'policy_sensitivity')
	            policy_sensitivity_raw(i, s, t) = msg.policy_sensitivity;
	        else
	            policy_sensitivity_raw(i, s, t) = NaN;
	        end
	        policy_sensitivity(i, t) = mean_over_seeds(policy_sensitivity_raw(i, :, t));
	        line_sens(i).YData = policy_sensitivity(i, :);

	        update_axis_limits(ax_survival, survival, true);
	        update_axis_limits(ax_kl, param_update_kl, true);
	        update_axis_limits(ax_novelty, novelty_term_mean, false);
	        update_axis_limits(ax_epi, epistemic_term_mean, false);
	        update_axis_limits(ax_ext, extrinsic_term_mean, false);
	        update_axis_limits(ax_depth, search_depth, true);
	        update_axis_limits(ax_sens, policy_sensitivity, true);

	        drawnow limitrate
	    end

	    function m = mean_over_seeds(values)
	        values = values(isfinite(values));
	        if isempty(values)
	            m = NaN;
	        else
	            m = mean(values);
	        end
	    end

	    function m = weighted_mean_over_seeds(term_sums, node_counts)
	        node_counts = double(node_counts);
	        term_sums = double(term_sums);
	        valid = isfinite(term_sums) & isfinite(node_counts) & node_counts > 0;
	        if any(valid)
	            m = sum(term_sums(valid)) / sum(node_counts(valid));
	        else
	            m = NaN;
	        end
	    end

	    function label = pretty_algorithm_label(alg, weights_for_label)
	        switch alg
	            case 'SI'
	                label = 'SI (no novelty)';
	            case 'SI_novelty'
	                label = 'SI + novelty';
	            case 'SI_smooth'
	                label = 'SI (no novelty, smooth)';
	            case 'SI_novelty_smooth'
	                label = 'SI + novelty + smooth';
	            case 'SL'
	                label = 'SL';
	            case 'SL_adaptivePlan'
	                label = 'SL (adaptive plan)';
	            case 'SL_noAdaptivePlan'
	                label = 'SL (no adaptive plan)';
	            case 'SL_noSmooth'
	                label = 'SL (no smooth plan)';
	            case 'SL_noSmooth_adaptivePlan'
	                label = 'SL (no smooth plan, adaptive)';
	            case 'SL_noSmooth_noAdaptivePlan'
	                label = 'SL (no smooth plan, no adaptive)';
	            case 'SL_noNovelty'
	                label = 'SL (no novelty)';
	            case 'SL_noNovelty_noSmooth'
	                label = 'SL (no novelty, no smooth)';
	            case 'BA'
	                label = 'BA';
	            case 'BAUCB'
	                label = sprintf('BAUCB (%s)', string(weights_for_label.baucb_variant));
	            otherwise
	                label = alg;
	        end
	    end

	    function style = line_style_for_algorithm(alg)
	        if startsWith(alg, 'SI')
	            style = '-';
	        elseif startsWith(alg, 'SL')
	            style = '--';
	        elseif startsWith(alg, 'BA')
	            style = ':';
	        else
	            style = '-';
	        end
	    end

	    function update_axis_limits(ax, data_matrix, clamp_zero)
	        vals = data_matrix(:);
	        vals = vals(isfinite(vals));
	        if isempty(vals)
	            return
	        end
	        mn = min(vals);
	        mx = max(vals);
	        if mn == mx
	            pad = max(1e-6, abs(mn) * 0.1);
	            mn = mn - pad;
	            mx = mx + pad;
	        else
	            pad = 0.05 * (mx - mn);
	            mn = mn - pad;
	            mx = mx + pad;
	        end
	        if clamp_zero
	            mn = min(mn, 0);
	        end
	        ylim(ax, [mn mx]);
	    end
end
