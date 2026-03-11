function local_parallel_dashboard_sweep(varargin)
%LOCAL_PARALLEL_DASHBOARD_SWEEP Run a small hyperparameter sweep locally and pick best configs.
%
% This dashboard runs multiple (algorithm, novelty_weight) configurations
% across paired seeds and reports the best novelty setting per algorithm.
%
% The sweep is intended for "rough" local tuning before scaling up on HPC.
%
% Name-value overrides:
%   'Algorithms'              : cellstr of algorithm variants to tune
%   'Seeds'                   : vector of integer seeds
%   'NumTrials'               : number of trials per run
%   'MaxHorizon'              : planning horizon cap
%   'Weights'                 : base weights struct (novelty overwritten by sweep)
%   'NoveltyWeightsNonSmooth' : novelty weights for non-smooth variants
%   'NoveltyWeightsSmooth'    : novelty weights for smooth variants
%   'ScoreWindow'             : number of final trials to score (default=min(10, NumTrials))
%   'DashboardTitle'          : figure title
%
% Example:
%   local_parallel_dashboard_sweep('Seeds', 1:5, 'NumTrials', 50)

    defaults = struct();
    defaults.Algorithms = { ...
        'SI', ...
        'SI_novelty', 'SI_novelty_smooth', ...
        'SL_noSmooth_noAdaptivePlan', 'SL_noAdaptivePlan', ...
        'SL_noSmooth_adaptivePlan', 'SL_adaptivePlan' ...
    };
    defaults.Seeds = 1:5;
    defaults.NumTrials = 50;
    defaults.MaxHorizon = 9;
    defaults.ScoreWindow = [];
    defaults.DashboardTitle = 'Local Sweep Dashboard';
    defaults.NumWorkers = [];
    defaults.NoveltyWeightsNonSmooth = [2, 5, 10];
    defaults.NoveltyWeightsSmooth = [0.25, 0.5, 1, 2];
    defaults.Weights = struct(...
        'novelty', 10, ...
        'learning', 40, ...
        'epistemic', 1, ...
        'preference', 10, ...
        'ucb_scale', 5, ...
        'state_selection', 'sample', ...
        'preference_param', 'weight', ...
        'baucb_variant', 'fixed_joint_counts', ...
        'real_smoothing', true, ...
        'adaptive_likelihood_in_plan', false, ...
        'learning_prune_threshold', 0.2 ...
    );

    parser = inputParser();
    parser.addParameter('Algorithms', defaults.Algorithms, @(x) iscellstr(x) || (iscell(x) && all(cellfun(@ischar, x))));
    parser.addParameter('Seeds', defaults.Seeds, @(x) isnumeric(x) && isvector(x) && all(x == floor(x)));
    parser.addParameter('NumTrials', defaults.NumTrials, @(x) isnumeric(x) && isscalar(x) && x == floor(x) && x > 0);
    parser.addParameter('MaxHorizon', defaults.MaxHorizon, @(x) isnumeric(x) && isscalar(x) && x == floor(x) && x > 0);
    parser.addParameter('Weights', defaults.Weights, @(x) isstruct(x));
    parser.addParameter('NoveltyWeightsNonSmooth', defaults.NoveltyWeightsNonSmooth, @(x) isnumeric(x) && isvector(x));
    parser.addParameter('NoveltyWeightsSmooth', defaults.NoveltyWeightsSmooth, @(x) isnumeric(x) && isvector(x));
    parser.addParameter('ScoreWindow', defaults.ScoreWindow, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x == floor(x) && x > 0));
    parser.addParameter('NumWorkers', defaults.NumWorkers, @(x) isempty(x) || (isnumeric(x) && isscalar(x) && x == floor(x) && x > 0));
    parser.addParameter('DashboardTitle', defaults.DashboardTitle, @(x) ischar(x) || isstring(x));
    parser.parse(varargin{:});

    algorithms = parser.Results.Algorithms;
    seeds = unique(parser.Results.Seeds(:)');
    n_seeds = numel(seeds);
    num_trials = parser.Results.NumTrials;
    max_horizon = parser.Results.MaxHorizon;
    base_weights = parser.Results.Weights;
    novelty_non_smooth = parser.Results.NoveltyWeightsNonSmooth(:)';
    novelty_smooth = parser.Results.NoveltyWeightsSmooth(:)';
    dashboard_title = char(parser.Results.DashboardTitle);
    score_window = parser.Results.ScoreWindow;
    requested_workers_override = parser.Results.NumWorkers;
    if isempty(score_window)
        score_window = min(10, num_trials);
    end

    thisFile = mfilename('fullpath');
    matlabDir = fileparts(thisFile);                 % .../src/MATLAB
    projectRoot = fullfile(matlabDir, '..', '..');   % repo root
    addpath(genpath(matlabDir));

    run_id = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    grid_id = sprintf('localSweep_%s', run_id);
    results_root = fullfile(projectRoot, 'results', 'local-sweep', run_id);
    if ~exist(results_root, 'dir')
        mkdir(results_root);
    end

    cfg_base = struct();
    cfg_base.grid_size = 10;
    cfg_base.start_position = 51;
    cfg_base.hill_pos = 55;
    cfg_base.food_sources = [71, 43, 57, 78];
    cfg_base.water_sources = [73, 33, 48, 67];
    cfg_base.sleep_sources = [64, 44, 49, 59];
    cfg_base.num_states = cfg_base.grid_size ^ 2;
    cfg_base.num_trials = num_trials;
    cfg_base.max_horizon = max_horizon;

    configs = build_sweep_configs(algorithms, base_weights, novelty_non_smooth, novelty_smooth);
    n_cfg = numel(configs);
    if n_cfg == 0
        error('No sweep configs generated.');
    end

    seedIndex = containers.Map('KeyType', 'double', 'ValueType', 'double');
    for si = 1:n_seeds
        seedIndex(seeds(si)) = si;
    end
    cfgIndex = containers.Map('KeyType', 'char', 'ValueType', 'double');
    for ci = 1:n_cfg
        cfgIndex(configs(ci).label) = ci;
    end

    cluster = parcluster('local');
    requested_workers = min(n_cfg * n_seeds, cluster.NumWorkers);
    if ~isempty(requested_workers_override)
        requested_workers = min(requested_workers, requested_workers_override);
    end
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool(cluster, requested_workers);
    end

    dq = parallel.pool.DataQueue();
    afterEach(dq, @onProgress);

    survival_raw = nan(n_cfg, n_seeds, num_trials);
    kl_raw = nan(n_cfg, n_seeds, num_trials);
    depth_raw = nan(n_cfg, n_seeds, num_trials);
    sens_raw = nan(n_cfg, n_seeds, num_trials);
    node_count_raw = nan(n_cfg, n_seeds, num_trials);
    novelty_sum_raw = nan(n_cfg, n_seeds, num_trials);
    epistemic_sum_raw = nan(n_cfg, n_seeds, num_trials);
    extrinsic_sum_raw = nan(n_cfg, n_seeds, num_trials);

    seed_done = false(n_cfg, n_seeds);
    cfg_done = false(n_cfg, 1);
    cfg_score = nan(n_cfg, 1);

    families = unique({configs.family}, 'stable');
    n_fam = numel(families);
    best_cfg_for_family = nan(n_fam, 1);
    best_score_for_family = nan(n_fam, 1);

    fig = figure('Name', dashboard_title, 'NumberTitle', 'off');
    layout = tiledlayout(fig, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    title(layout, dashboard_title, 'Interpreter', 'none');

    ax_survival = nexttile; hold(ax_survival, 'on'); title(ax_survival, 'Survival (Best Per Family)'); xlabel(ax_survival, 'Trial'); ylabel(ax_survival, 'Steps');
    ax_score = nexttile; hold(ax_score, 'on'); title(ax_score, sprintf('Score (Mean Last %d Trials)', score_window)); xlabel(ax_score, 'Algorithm'); ylabel(ax_score, 'Score');
    ax_info = nexttile; axis(ax_info, 'off');
    ax_unused = nexttile; axis(ax_unused, 'off');

    x = 1:num_trials;
    colors = lines(n_fam);
    line_best = gobjects(n_fam, 1);
    for fi = 1:n_fam
        line_best(fi) = plot(ax_survival, x, nan(1, num_trials), '-', 'Color', colors(fi, :), 'DisplayName', families{fi});
    end
    legend(ax_survival, 'Location', 'eastoutside');

    bar_handle = bar(ax_score, zeros(1, n_fam));
    ax_score.XTick = 1:n_fam;
    ax_score.XTickLabel = families;
    ax_score.XTickLabelRotation = 30;

    render_info_text();

    futures = parallel.FevalFuture.empty(0, n_cfg * n_seeds);
    fi = 0;
    for ci = 1:n_cfg
        cfg_dir = fullfile(results_root, sanitize_file_component(configs(ci).label));
        if ~exist(cfg_dir, 'dir')
            mkdir(cfg_dir);
        end
        for si = 1:n_seeds
            cfg_run = cfg_base;
            cfg_run.seed = seeds(si);
            cfg_run.weights = configs(ci).weights;
            fi = fi + 1;
            futures(fi) = parfeval(pool, @dashboard_run_one, 0, configs(ci).algorithm, cfg_run, dq, cfg_dir, grid_id, configs(ci).label);
        end
    end

    wait(futures);

    % Final report.
    summary = build_summary_table();
    summary_file = fullfile(results_root, 'sweep_summary.csv');
    writetable(summary, summary_file);
    disp('Sweep complete. Best settings per algorithm family:');
    disp(summary(summary.is_best, :));
    fprintf('Wrote %s\n', summary_file);

    function onProgress(msg)
        if ~isstruct(msg) || ~isfield(msg, 'algorithm') || ~isfield(msg, 'trial')
            return
        end
        label = char(msg.algorithm);
        if ~isKey(cfgIndex, label)
            return
        end
        ci = cfgIndex(label);
        t = msg.trial;
        if t < 1 || t > num_trials
            return
        end

        s = 1;
        if isfield(msg, 'seed')
            seed_value = double(msg.seed);
            if isKey(seedIndex, seed_value)
                s = seedIndex(seed_value);
            else
                return
            end
        else
            return
        end

        survival_raw(ci, s, t) = msg.survival;
        kl_raw(ci, s, t) = msg.param_update_kl;
        depth_raw(ci, s, t) = msg.search_depth;
        if isfield(msg, 'policy_sensitivity')
            sens_raw(ci, s, t) = msg.policy_sensitivity;
        end

        if isfield(msg, 'efe_node_count') && msg.efe_node_count > 0
            node_count_raw(ci, s, t) = msg.efe_node_count;
            novelty_sum_raw(ci, s, t) = msg.efe_novelty_term_sum;
            epistemic_sum_raw(ci, s, t) = msg.efe_epistemic_term_sum;
            extrinsic_sum_raw(ci, s, t) = msg.efe_extrinsic_term_sum;
        end

        if t == num_trials
            seed_done(ci, s) = true;
        end
        if all(seed_done(ci, :)) && ~cfg_done(ci)
            cfg_done(ci) = true;
            cfg_score(ci) = score_config(ci);
            update_best_for_family(configs(ci).family, ci, cfg_score(ci));
            render_info_text();
        end

        update_plots();
    end

    function update_plots()
        % Update score bars.
        for fam_i = 1:n_fam
            bar_handle.YData(fam_i) = best_score_for_family(fam_i);
        end

        % Update survival curves for each family's best config.
        all_y = nan(n_fam, num_trials);
        for fam_i = 1:n_fam
            ci = best_cfg_for_family(fam_i);
            if isnan(ci)
                continue
            end
            line_best(fam_i).YData = mean_over_seeds(survival_raw(ci, :, :));
            all_y(fam_i, :) = line_best(fam_i).YData;
        end

        update_axis_limits(ax_survival, all_y, true);
        drawnow limitrate
    end

    function update_best_for_family(family, ci, score)
        fam_i = find(strcmp(families, family), 1);
        if isempty(fam_i)
            return
        end
        if isnan(best_score_for_family(fam_i)) || score > best_score_for_family(fam_i)
            best_score_for_family(fam_i) = score;
            best_cfg_for_family(fam_i) = ci;
        end
    end

    function s = score_config(ci)
        curve = mean_over_seeds(survival_raw(ci, :, :));
        if all(~isfinite(curve))
            s = NaN;
            return
        end
        start_idx = max(1, num_trials - score_window + 1);
        window_vals = curve(start_idx:num_trials);
        window_vals = window_vals(isfinite(window_vals));
        if isempty(window_vals)
            s = NaN;
        else
            s = mean(window_vals);
        end
    end

    function curve = mean_over_seeds(values)
        % values: 1 x n_seeds x num_trials
        v = squeeze(values);
        if size(v, 1) ~= n_seeds
            v = reshape(v, n_seeds, num_trials);
        end
        curve = nan(1, num_trials);
        for tt = 1:num_trials
            x = v(:, tt);
            x = x(isfinite(x));
            if ~isempty(x)
                curve(tt) = mean(x);
            end
        end
    end

    function render_info_text()
        best_lines = {};
        for fam_i = 1:n_fam
            ci = best_cfg_for_family(fam_i);
            if isnan(ci)
                best_lines{end+1} = sprintf('  - %s: (pending)', families{fam_i}); %#ok<AGROW>
            else
                best_lines{end+1} = sprintf('  - %s: %s (score=%.2f)', families{fam_i}, configs(ci).label, best_score_for_family(fam_i)); %#ok<AGROW>
            end
        end
        info_lines = { ...
            sprintf('Seeds (n=%d): %s', n_seeds, strjoin(arrayfun(@num2str, seeds, 'UniformOutput', false), ', ')), ...
            sprintf('Trials: %d', num_trials), ...
            sprintf('Max horizon: %d', max_horizon), ...
            sprintf('Score window: last %d trials', score_window), ...
            '', ...
            sprintf('Non-smooth novelty grid: %s', mat2str(novelty_non_smooth)), ...
            sprintf('Smooth novelty grid: %s', mat2str(novelty_smooth)), ...
            '', ...
            'Best per family:', ...
        };
        info_lines = [info_lines, best_lines]; %#ok<AGROW>
        cla(ax_info);
        text(ax_info, 0, 1, info_lines, 'VerticalAlignment', 'top', 'Interpreter', 'none', 'FontName', 'FixedWidth');
    end

    function summary = build_summary_table()
        family_col = strings(n_cfg, 1);
        alg_col = strings(n_cfg, 1);
        label_col = strings(n_cfg, 1);
        novelty_col = nan(n_cfg, 1);
        score_col = nan(n_cfg, 1);
        mean_survival_col = nan(n_cfg, 1);
        std_survival_col = nan(n_cfg, 1);
        p_survival_ge50_col = nan(n_cfg, 1);
        p_survival_ge100_col = nan(n_cfg, 1);
        mean_depth_per_step_col = nan(n_cfg, 1);
        done_col = cfg_done;
        is_best = false(n_cfg, 1);

        for ci = 1:n_cfg
            family_col(ci) = string(configs(ci).family);
            alg_col(ci) = string(configs(ci).algorithm);
            label_col(ci) = string(configs(ci).label);
            novelty_col(ci) = configs(ci).novelty;
            score_col(ci) = cfg_score(ci);

            surv_vals = squeeze(survival_raw(ci, :, :));
            surv_vals = surv_vals(isfinite(surv_vals));
            if ~isempty(surv_vals)
                mean_survival_col(ci) = mean(surv_vals);
                std_survival_col(ci) = std(surv_vals);
                p_survival_ge50_col(ci) = mean(surv_vals >= 50);
                p_survival_ge100_col(ci) = mean(surv_vals >= 100);
            end

            depth_vals = squeeze(depth_raw(ci, :, :));
            surv_matrix = squeeze(survival_raw(ci, :, :));
            depth_per_step = depth_vals ./ max(surv_matrix, 1);
            depth_per_step = depth_per_step(isfinite(depth_per_step));
            if ~isempty(depth_per_step)
                mean_depth_per_step_col(ci) = mean(depth_per_step);
            end
        end

        for fam_i = 1:n_fam
            ci = best_cfg_for_family(fam_i);
            if ~isnan(ci)
                is_best(ci) = true;
            end
        end

        summary = table( ...
            family_col, alg_col, label_col, novelty_col, done_col, score_col, ...
            mean_survival_col, std_survival_col, p_survival_ge50_col, p_survival_ge100_col, mean_depth_per_step_col, ...
            is_best, ...
            'VariableNames', { ...
                'family', 'algorithm', 'label', 'novelty', 'done', 'score', ...
                'mean_survival', 'std_survival', 'p_survival_ge50', 'p_survival_ge100', 'mean_depth_per_step', ...
                'is_best' ...
            } ...
        );
        summary = sortrows(summary, {'family', 'score'}, {'ascend', 'descend'});
    end

    function configs = build_sweep_configs(algorithm_variants, base_weights_in, novelty_grid_non_smooth, novelty_grid_smooth)
        configs = struct('family', {}, 'algorithm', {}, 'label', {}, 'weights', {}, 'novelty', {});
        for ai = 1:numel(algorithm_variants)
            alg = algorithm_variants{ai};
            family = alg;
            if strcmp(alg, 'SI')
                weights = base_weights_in;
                weights.novelty = 0;
                configs(end+1) = struct(...
                    'family', family, ...
                    'algorithm', alg, ...
                    'label', alg, ...
                    'weights', weights, ...
                    'novelty', 0 ...
                ); %#ok<AGROW>
                continue
            end

            is_smooth = contains(alg, 'smooth') || (startsWith(alg, 'SL') && ~contains(alg, 'noSmooth'));
            novelty_grid = novelty_grid_non_smooth;
            if is_smooth
                novelty_grid = novelty_grid_smooth;
            end
            for nv = novelty_grid(:)'
                weights = base_weights_in;
                weights.novelty = nv;
                label = sprintf('%s_nv%s', alg, sanitize_file_component(num2str(nv)));
                configs(end+1) = struct(...
                    'family', family, ...
                    'algorithm', alg, ...
                    'label', label, ...
                    'weights', weights, ...
                    'novelty', nv ...
                ); %#ok<AGROW>
            end
        end
    end

    function update_axis_limits(ax, values, clamp_zero)
        vals = values(:);
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
