function perf_capture_golden_trace(algorithms, num_trials, seed, max_horizon)
%PERF_CAPTURE_GOLDEN_TRACE Capture the reference per-step CSV and survived
% array for one or more algorithms. Saved under sanity/golden/ for use by
% perf_check_golden_trace.
%
% Run this ONCE on the baseline commit. Commit the resulting golden_*.csv
% and golden_*.mat files alongside the code.
%
%   algorithms  : cellstr (default {'SI','SL'})
%   num_trials  : positive integer (default 3)
%   seed        : positive integer (default 1)
%   max_horizon : positive integer (default 9)

    if nargin < 1 || isempty(algorithms); algorithms = {'SI', 'SL'}; end
    if nargin < 2 || isempty(num_trials); num_trials = 3; end
    if nargin < 3 || isempty(seed); seed = 1; end
    if nargin < 4 || isempty(max_horizon); max_horizon = 9; end

    sanity_dir = fileparts(mfilename('fullpath'));
    golden_dir = fullfile(sanity_dir, 'golden');
    if ~exist(golden_dir, 'dir'); mkdir(golden_dir); end

    workdir = fullfile(sanity_dir, '_perf_workdir');

    fprintf('Capturing golden traces (num_trials=%d, seed=%d, max_horizon=%d)\n', ...
        num_trials, seed, max_horizon);

    for k = 1:numel(algorithms)
        alg = algorithms{k};
        fprintf('  [%s] running...\n', alg);
        t0 = tic;
        [survived, csv_src] = perf_run_canonical(alg, num_trials, seed, max_horizon, workdir);
        elapsed = toc(t0);

        csv_dst = fullfile(golden_dir, sprintf('golden_%s_seed%d_n%d.csv', alg, seed, num_trials));
        mat_dst = fullfile(golden_dir, sprintf('golden_%s_seed%d_n%d.mat', alg, seed, num_trials));

        if exist(csv_src, 'file')
            copyfile(csv_src, csv_dst);
        else
            warning('No step-metrics CSV produced for %s; only survived will be saved.', alg);
        end
        save(mat_dst, 'survived', 'num_trials', 'seed', 'max_horizon');

        fprintf('  [%s] done in %.1fs. survived = %s\n', alg, elapsed, mat2str(survived));
        fprintf('         CSV  -> %s\n', csv_dst);
        fprintf('         MAT  -> %s\n', mat_dst);
    end

    if exist(workdir, 'dir')
        rmdir(workdir, 's');
    end

    fprintf('Done.\n');
end
