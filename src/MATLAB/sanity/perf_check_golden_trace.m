function ok = perf_check_golden_trace(mode, algorithms, num_trials, seed, max_horizon)
%PERF_CHECK_GOLDEN_TRACE Re-run the canonical config and assert the per-step
% CSV and survived array match the saved golden trace.
%
% Modes:
%   'exact' : require survived to match exactly AND CSV bytes to match exactly
%             (only valid for changes that should be bit-exact, e.g. Phase 1).
%   'tol'   : require survived to match exactly AND numeric CSV columns to match
%             within max-abs-diff < 1e-12 (use for Phases 2-3 where FP order
%             may differ).
%
%   algorithms / num_trials / seed / max_horizon: see perf_capture_golden_trace.
%
% Returns true on success; throws on failure (so it works inside sanity hooks).

    if nargin < 1 || isempty(mode); mode = 'exact'; end
    if nargin < 2 || isempty(algorithms); algorithms = {'SI', 'SL'}; end
    if nargin < 3 || isempty(num_trials); num_trials = 3; end
    if nargin < 4 || isempty(seed); seed = 1; end
    if nargin < 5 || isempty(max_horizon); max_horizon = 9; end

    if ~ismember(mode, {'exact', 'tol'})
        error('perf_check_golden_trace: mode must be ''exact'' or ''tol''.');
    end

    sanity_dir = fileparts(mfilename('fullpath'));
    golden_dir = fullfile(sanity_dir, 'golden');
    workdir = fullfile(sanity_dir, '_perf_workdir');

    ok = true;

    for k = 1:numel(algorithms)
        alg = algorithms{k};
        csv_golden = fullfile(golden_dir, sprintf('golden_%s_seed%d_n%d.csv', alg, seed, num_trials));
        mat_golden = fullfile(golden_dir, sprintf('golden_%s_seed%d_n%d.mat', alg, seed, num_trials));

        if ~exist(mat_golden, 'file')
            error('perf_check_golden_trace: missing golden file %s. Run perf_capture_golden_trace first.', mat_golden);
        end

        loaded = load(mat_golden, 'survived');
        survived_golden = loaded.survived;

        [survived, csv_current] = perf_run_canonical(alg, num_trials, seed, max_horizon, workdir);

        if ~isequal(survived, survived_golden)
            error('perf_check_golden_trace: [%s] survived array differs.\n  golden: %s\n  current: %s', ...
                alg, mat2str(survived_golden), mat2str(survived));
        end

        if exist(csv_golden, 'file')
            switch mode
                case 'exact'
                    if ~files_byte_equal(csv_golden, csv_current)
                        error('perf_check_golden_trace: [%s, exact] CSV bytes differ.\n  golden:  %s\n  current: %s', ...
                            alg, csv_golden, csv_current);
                    end
                case 'tol'
                    diff_summary = compare_step_metrics_with_tol(csv_golden, csv_current);
                    if diff_summary.max_abs_diff > 1e-12 || diff_summary.chosen_action_mismatches > 0
                        error('perf_check_golden_trace: [%s, tol] CSV diverges.\n  max_abs_diff = %.3e\n  chosen_action mismatches = %d', ...
                            alg, diff_summary.max_abs_diff, diff_summary.chosen_action_mismatches);
                    end
            end
        end

        fprintf('  [%s] OK (mode=%s, survived=%s)\n', alg, mode, mat2str(survived));
    end

    if exist(workdir, 'dir'); rmdir(workdir, 's'); end
end

function eq = files_byte_equal(p1, p2)
    f1 = fopen(p1, 'rb'); b1 = fread(f1); fclose(f1);
    f2 = fopen(p2, 'rb'); b2 = fread(f2); fclose(f2);
    eq = isequal(b1, b2);
end

function s = compare_step_metrics_with_tol(p_golden, p_current)
    Tg = readtable(p_golden);
    Tc = readtable(p_current);
    if ~isequal(size(Tg), size(Tc))
        error('Step-metrics CSVs differ in shape: golden=%s current=%s', mat2str(size(Tg)), mat2str(size(Tc)));
    end

    s.chosen_action_mismatches = sum(Tg.chosen_action ~= Tc.chosen_action);

    numeric_cols = {'planning_node_count', 'planning_novelty_term_sum', ...
        'planning_epistemic_term_sum', 'planning_extrinsic_term_sum', 'planning_future_term_sum', ...
        'planning_novelty_term_mean', 'planning_epistemic_term_mean', 'planning_extrinsic_term_mean', ...
        'param_update_kl_step', 'search_depth_so_far'};
    s.max_abs_diff = 0;
    for k = 1:numel(numeric_cols)
        c = numeric_cols{k};
        if ~ismember(c, Tg.Properties.VariableNames); continue; end
        a = Tg.(c); b = Tc.(c);
        valid = ~(isnan(a) & isnan(b));
        d = abs(a(valid) - b(valid));
        if ~isempty(d)
            s.max_abs_diff = max(s.max_abs_diff, max(d));
        end
    end
end
