function local_parallel_dashboard(varargin)
%LOCAL_PARALLEL_DASHBOARD Run one seed per algorithm in parallel and plot progress.
%
% Requirements:
%   - Parallel Computing Toolbox (parpool, parfeval, DataQueue)
%
% Usage:
%   local_parallel_dashboard
%
% Optional name-value overrides:
%   'Algorithms'  : cellstr of algorithm names
%   'Seed'        : scalar integer seed
%   'NumTrials'   : number of trials per algorithm
%   'MaxHorizon'  : planning horizon cap

    defaults = struct();
    defaults.Algorithms = { ...
        'SI', 'SI_novelty', 'SI_smooth', 'SI_novelty_smooth', ...
        'SL', 'SL_noSmooth', 'SL_noNovelty', 'SL_noNovelty_noSmooth', ...
        'BA', 'BAUCB' ...
    };
	    defaults.Seed = 1;
	    defaults.NumTrials = 20;
	    defaults.MaxHorizon = 9;
	    defaults.RealSmoothing = true;
	    defaults.AdaptiveLikelihoodInPlan = false;
	    defaults.LearningPruneThreshold = 0.2;

    parser = inputParser();
    parser.addParameter('Algorithms', defaults.Algorithms, @(x) iscellstr(x) || (iscell(x) && all(cellfun(@ischar, x))));
	    parser.addParameter('Seed', defaults.Seed, @(x) isnumeric(x) && isscalar(x) && x == floor(x));
	    parser.addParameter('NumTrials', defaults.NumTrials, @(x) isnumeric(x) && isscalar(x) && x == floor(x) && x > 0);
	    parser.addParameter('MaxHorizon', defaults.MaxHorizon, @(x) isnumeric(x) && isscalar(x) && x == floor(x) && x > 0);
	    parser.addParameter('RealSmoothing', defaults.RealSmoothing, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
	    parser.addParameter('AdaptiveLikelihoodInPlan', defaults.AdaptiveLikelihoodInPlan, @(x) islogical(x) || (isnumeric(x) && isscalar(x)));
	    parser.addParameter('LearningPruneThreshold', defaults.LearningPruneThreshold, @(x) isnumeric(x) && isscalar(x) && x >= 0);
	    parser.parse(varargin{:});
	    algorithms = parser.Results.Algorithms;
	    seed = parser.Results.Seed;
	    num_trials = parser.Results.NumTrials;
	    max_horizon = parser.Results.MaxHorizon;
	    real_smoothing = logical(parser.Results.RealSmoothing);
	    adaptive_likelihood_in_plan = logical(parser.Results.AdaptiveLikelihoodInPlan);
	    learning_prune_threshold = parser.Results.LearningPruneThreshold;

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
    cfg.seed = seed;
    cfg.grid_size = 10;
    cfg.start_position = 51;
    cfg.hill_pos = 55;
    cfg.food_sources = [71, 43, 57, 78];
    cfg.water_sources = [73, 33, 48, 67];
    cfg.sleep_sources = [64, 44, 49, 59];
    cfg.num_states = cfg.grid_size ^ 2;
    cfg.num_trials = num_trials;
    cfg.max_horizon = max_horizon;
		    cfg.weights = struct(...
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

    cluster = parcluster('local');
    requested_workers = min(numel(algorithms), cluster.NumWorkers);
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool(cluster, requested_workers);
    end

    dq = parallel.pool.DataQueue();

    algoIndex = containers.Map(algorithms, 1:numel(algorithms));
    n_alg = numel(algorithms);

    survival = nan(n_alg, num_trials);
    param_update_kl = nan(n_alg, num_trials);
    novelty_term_mean = nan(n_alg, num_trials);
    epistemic_term_mean = nan(n_alg, num_trials);
	    extrinsic_term_mean = nan(n_alg, num_trials);
	    search_depth = nan(n_alg, num_trials);
	    policy_sensitivity = nan(n_alg, num_trials);

	    fig = figure('Name', 'Local Parallel Dashboard', 'NumberTitle', 'off');
	    tiledlayout(fig, 4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    ax_survival = nexttile; hold(ax_survival, 'on'); title(ax_survival, 'Survival'); xlabel(ax_survival, 'Trial'); ylabel(ax_survival, 'Steps');
    ax_kl = nexttile; hold(ax_kl, 'on'); title(ax_kl, 'Param Update KL'); xlabel(ax_kl, 'Trial'); ylabel(ax_kl, 'Sum KL');
    ax_novelty = nexttile; hold(ax_novelty, 'on'); title(ax_novelty, 'Novelty Term (Mean/Step)'); xlabel(ax_novelty, 'Trial'); ylabel(ax_novelty, 'Novelty');
	    ax_epi = nexttile; hold(ax_epi, 'on'); title(ax_epi, 'Epistemic Term (Mean/Step)'); xlabel(ax_epi, 'Trial'); ylabel(ax_epi, 'Epistemic');
	    ax_ext = nexttile; hold(ax_ext, 'on'); title(ax_ext, 'Extrinsic Term (Mean/Step)'); xlabel(ax_ext, 'Trial'); ylabel(ax_ext, 'Extrinsic');
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

    for i = 1:n_alg
        line_survival(i) = plot(ax_survival, x, survival(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
        line_kl(i) = plot(ax_kl, x, param_update_kl(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
        line_novelty(i) = plot(ax_novelty, x, novelty_term_mean(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
	        line_epi(i) = plot(ax_epi, x, epistemic_term_mean(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
	        line_ext(i) = plot(ax_ext, x, extrinsic_term_mean(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
	        line_depth(i) = plot(ax_depth, x, search_depth(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
	        line_sens(i) = plot(ax_sens, x, policy_sensitivity(i, :), '-', 'Color', colors(i, :), 'DisplayName', algorithms{i});
	    end

    legend(ax_survival, 'Location', 'eastoutside');

    afterEach(dq, @onProgress);

    futures = parallel.FevalFuture.empty(0, n_alg);
    for i = 1:n_alg
        futures(i) = parfeval(pool, @dashboard_run_one, 0, algorithms{i}, cfg, dq, results_dir, grid_id);
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

        survival(i, t) = msg.survival;
        param_update_kl(i, t) = msg.param_update_kl;
        search_depth(i, t) = msg.search_depth;

        if isfield(msg, 'efe_steps') && msg.efe_steps > 0
            novelty_term_mean(i, t) = msg.efe_novelty_term_sum / msg.efe_steps;
            epistemic_term_mean(i, t) = msg.efe_epistemic_term_sum / msg.efe_steps;
            extrinsic_term_mean(i, t) = msg.efe_extrinsic_term_sum / msg.efe_steps;
        else
            novelty_term_mean(i, t) = NaN;
            epistemic_term_mean(i, t) = NaN;
            extrinsic_term_mean(i, t) = NaN;
        end

        line_survival(i).YData = survival(i, :);
        line_kl(i).YData = param_update_kl(i, :);
        line_novelty(i).YData = novelty_term_mean(i, :);
	        line_epi(i).YData = epistemic_term_mean(i, :);
	        line_ext(i).YData = extrinsic_term_mean(i, :);
	        line_depth(i).YData = search_depth(i, :);
	        if isfield(msg, 'policy_sensitivity')
	            policy_sensitivity(i, t) = msg.policy_sensitivity;
	        else
	            policy_sensitivity(i, t) = NaN;
	        end
	        line_sens(i).YData = policy_sensitivity(i, :);

	        drawnow limitrate
	    end
end
