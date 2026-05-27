function results = perf_time_step(algorithms, num_trials, seed, max_horizon)
%PERF_TIME_STEP Measure mean ms/step for one or more algorithms with the
% canonical config, single-thread BLAS (so worker oversubscription doesn't
% pollute the signal). Returns a struct array with fields algorithm,
% total_steps, total_seconds, ms_per_step.
%
%   algorithms  : cellstr (default {'SI','SL'})
%   num_trials  : positive integer (default 2). Higher = more stable signal,
%                 but expensive — for SL each trial can take ~30s.
%   seed        : positive integer (default 1)
%   max_horizon : positive integer (default 9)

    if nargin < 1 || isempty(algorithms); algorithms = {'SI', 'SL'}; end
    if nargin < 2 || isempty(num_trials); num_trials = 2; end
    if nargin < 3 || isempty(seed); seed = 1; end
    if nargin < 4 || isempty(max_horizon); max_horizon = 9; end

    prev_threads = maxNumCompThreads(1);
    cleanup = onCleanup(@() maxNumCompThreads(prev_threads));

    sanity_dir = fileparts(mfilename('fullpath'));
    workdir = fullfile(sanity_dir, '_perf_workdir');

    results = struct('algorithm', {}, 'total_steps', {}, 'total_seconds', {}, 'ms_per_step', {});
    fprintf('Timing (single-thread BLAS, num_trials=%d, max_horizon=%d):\n', num_trials, max_horizon);

    for k = 1:numel(algorithms)
        alg = algorithms{k};
        % warm-up: first call pays JIT cost; we don't want that in the reading.
        perf_run_canonical(alg, 1, seed, max_horizon, workdir);

        t0 = tic;
        survived = perf_run_canonical(alg, num_trials, seed, max_horizon, workdir);
        elapsed = toc(t0);

        total_steps = sum(survived);
        ms_per_step = (elapsed / total_steps) * 1000;

        results(end+1) = struct('algorithm', alg, 'total_steps', total_steps, ...
            'total_seconds', elapsed, 'ms_per_step', ms_per_step); %#ok<AGROW>

        fprintf('  [%s] total_steps=%d  total_seconds=%.2f  ms_per_step=%.1f\n', ...
            alg, total_steps, elapsed, ms_per_step);
    end

    if exist(workdir, 'dir'); rmdir(workdir, 's'); end
end
