function idx = select_from_posterior(p, mode)
    %SELECT_FROM_POSTERIOR Select an index from a categorical distribution.
    %   idx = select_from_posterior(p, mode)
    %     p    : vector (row or column)
    %     mode : 'sample' or 'map'
    %     idx  : integer in 1..numel(p)

    if nargin < 2 || isempty(mode)
        mode = 'sample';
    end

    if ~isvector(p) || isempty(p)
        error('select_from_posterior expects a non-empty vector.');
    end

    p = double(p(:));
    p(~isfinite(p)) = 0;
    p(p < 0) = 0;

    total = sum(p);
    if total <= 0
        p = ones(size(p)) / numel(p);
    else
        tol = 1e-12;
        if abs(total - 1) > tol
            p = p / total;
        end
    end

    switch lower(mode)
        case 'map'
            [~, idx] = max(p);
        case 'sample'
            r = rand;
            idx = find(cumsum(p) >= r, 1, 'first');
            if isempty(idx)
                idx = numel(p);
            end
        otherwise
            error('Unknown mode "%s". Expected ''sample'' or ''map''.', mode);
    end
end

