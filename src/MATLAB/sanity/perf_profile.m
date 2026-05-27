function perf_profile(algorithm, num_trials, seed, max_horizon, top_n)
%PERF_PROFILE Run an algorithm under MATLAB's profiler and dump the top
% time-consuming functions to stdout. Used to identify real hot-paths before
% targeting them — replaces the audit-by-inspection that overestimated some
% function-overhead wins in the 2026-05-01 rollout.
%
%   algorithm   : 'SI' | 'SL' | ... (passed to perf_run_canonical)
%   num_trials  : default 5
%   seed        : default 1
%   max_horizon : default 9
%   top_n       : how many top functions to show (default 30)

    if nargin < 2 || isempty(num_trials); num_trials = 5; end
    if nargin < 3 || isempty(seed); seed = 1; end
    if nargin < 4 || isempty(max_horizon); max_horizon = 9; end
    if nargin < 5 || isempty(top_n); top_n = 30; end

    sanity_dir = fileparts(mfilename('fullpath'));
    workdir = fullfile(sanity_dir, '_perf_workdir_profile');
    out_dir = fullfile(sanity_dir, sprintf('profile_%s_seed%d', algorithm, seed));

    fprintf('=== perf_profile %s (num_trials=%d, seed=%d, horizon=%d) ===\n', ...
        algorithm, num_trials, seed, max_horizon);

    % single-thread BLAS so per-function times reflect single-CPU work
    prev_threads = maxNumCompThreads(1);
    cleanup_threads = onCleanup(@() maxNumCompThreads(prev_threads));

    % warm-up to absorb first-run JIT cost
    perf_run_canonical(algorithm, 1, seed, max_horizon, workdir);

    profile('off');
    profile('clear');
    profile('on', '-timer', 'real');
    survived = perf_run_canonical(algorithm, num_trials, seed, max_horizon, workdir);
    profile('off');

    p = profile('info');
    if exist(out_dir, 'dir'); rmdir(out_dir, 's'); end
    profsave(p, out_dir);
    save(fullfile(out_dir, 'profile_info.mat'), 'p');
    fprintf('saved profile -> %s\n', out_dir);
    fprintf('survived = %s\n', mat2str(survived));

    if isempty(p.FunctionTable)
        fprintf('No FunctionTable rows — profiler was off?\n');
        return;
    end

    % Build a sortable summary
    n = numel(p.FunctionTable);
    names = cell(n, 1);
    total_times = zeros(n, 1);
    self_times = zeros(n, 1);
    num_calls = zeros(n, 1);
    for k = 1:n
        f = p.FunctionTable(k);
        names{k} = f.FunctionName;
        total_times(k) = f.TotalTime;
        if isfield(f, 'TotalRecursiveTime')
            self_times(k) = f.TotalTime - f.TotalRecursiveTime;
        else
            self_times(k) = f.TotalTime;
        end
        num_calls(k) = f.NumCalls;
    end

    total_wall = sum(self_times);  % approx — sum of self-times
    fprintf('\nApprox. total self-time across all functions: %.2fs\n\n', total_wall);

    [~, idx] = sort(total_times, 'descend');
    fprintf('--- TOP %d by TOTAL time (incl. callees) ---\n', top_n);
    fprintf('%-60s %10s %10s %10s\n', 'function', 'total(s)', 'self(s)', 'calls');
    fprintf('%s\n', repmat('-', 1, 95));
    for k = 1:min(top_n, n)
        i = idx(k);
        fprintf('%-60s %10.3f %10.3f %10d\n', truncate_name(names{i}, 60), total_times(i), self_times(i), num_calls(i));
    end

    [~, idx2] = sort(self_times, 'descend');
    fprintf('\n--- TOP %d by SELF time (excl. callees) ---\n', top_n);
    fprintf('%-60s %10s %10s %10s\n', 'function', 'self(s)', 'total(s)', 'calls');
    fprintf('%s\n', repmat('-', 1, 95));
    for k = 1:min(top_n, n)
        i = idx2(k);
        fprintf('%-60s %10.3f %10.3f %10d\n', truncate_name(names{i}, 60), self_times(i), total_times(i), num_calls(i));
    end

    if exist(workdir, 'dir'); rmdir(workdir, 's'); end
end

function s = truncate_name(s, n)
    if numel(s) > n
        s = ['...' s(end-n+4:end)];
    end
end
