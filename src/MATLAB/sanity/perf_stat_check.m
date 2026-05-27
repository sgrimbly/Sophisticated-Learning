function result = perf_stat_check(algorithm, seeds, num_trials, max_horizon, baseline_mat)
%PERF_STAT_CHECK Run an algorithm across multiple seeds and compare survival
% statistics to a stored baseline. Used for Phases 2-3 where bit-equality is
% not expected, only statistical equivalence.
%
%   algorithm     : 'SI' | 'SL' | 'BA' | 'BAUCB' | ...
%   seeds         : vector of seeds (e.g., 1:10)
%   num_trials    : positive integer
%   max_horizon   : positive integer (typically 9)
%   baseline_mat  : path to a saved baseline (or '' to capture fresh and save)
%
% On capture (empty baseline_mat): runs the panel, saves to
% sanity/golden/stat_baseline_<algo>.mat, returns result.
% On check: re-runs and asserts mean within 1 SE of baseline.

    if nargin < 5; baseline_mat = ''; end

    sanity_dir = fileparts(mfilename('fullpath'));
    workdir = fullfile(sanity_dir, '_perf_workdir_stat');

    n_seeds = numel(seeds);
    survival_per_seed = nan(n_seeds, num_trials);
    fprintf('perf_stat_check %s: %d seeds x %d trials @ horizon=%d\n', algorithm, n_seeds, num_trials, max_horizon);
    for i = 1:n_seeds
        s = seeds(i);
        survived = perf_run_canonical(algorithm, num_trials, s, max_horizon, workdir);
        survival_per_seed(i, :) = survived(:)';
        fprintf('  seed %d: %s\n', s, mat2str(survived));
    end

    mean_survival = mean(survival_per_seed(:));
    std_survival = std(survival_per_seed(:));
    se_survival = std_survival / sqrt(numel(survival_per_seed));

    result = struct('algorithm', algorithm, 'seeds', seeds, 'num_trials', num_trials, ...
        'max_horizon', max_horizon, 'survival_per_seed', survival_per_seed, ...
        'mean', mean_survival, 'sd', std_survival, 'se', se_survival);

    fprintf('  mean=%.2f, sd=%.2f, se=%.3f\n', mean_survival, std_survival, se_survival);

    if isempty(baseline_mat)
        baseline_mat = fullfile(sanity_dir, 'golden', sprintf('stat_baseline_%s.mat', algorithm));
        save(baseline_mat, '-struct', 'result');
        fprintf('  saved baseline -> %s\n', baseline_mat);
    else
        if ~exist(baseline_mat, 'file')
            error('Baseline file not found: %s', baseline_mat);
        end
        baseline = load(baseline_mat);
        delta = abs(mean_survival - baseline.mean);
        threshold = baseline.se;  % 1 SE
        fprintf('  baseline mean=%.2f, |delta|=%.3f, 1-SE threshold=%.3f\n', baseline.mean, delta, threshold);
        if delta > threshold
            error('perf_stat_check FAILED: |mean_survival_delta| (%.3f) > 1 SE (%.3f)', delta, threshold);
        end
        fprintf('  PASS (within 1 SE).\n');
    end

    if exist(workdir, 'dir'); rmdir(workdir, 's'); end
end
