function local_parallel_dashboard_bugfixed_sweep(varargin)
%LOCAL_PARALLEL_DASHBOARD_BUGFIXED_SWEEP Convenience wrapper for local sweep tuning.
%
% Runs the novelty-weight sweep dashboard under bugfixed defaults.
%
% Example:
%   local_parallel_dashboard_bugfixed_sweep('Seeds', 1:5, 'NumTrials', 50)

    has_title = false;
    if mod(numel(varargin), 2) == 0
        keys = varargin(1:2:end);
        if iscell(keys) && all(cellfun(@(x) ischar(x) || isstring(x), keys))
            has_title = any(strcmpi(string(keys), "DashboardTitle"));
        end
    end

    if has_title
        local_parallel_dashboard_sweep(varargin{:});
    else
        local_parallel_dashboard_sweep('DashboardTitle', 'Local Sweep Dashboard - Bugfixed', varargin{:});
    end
end
