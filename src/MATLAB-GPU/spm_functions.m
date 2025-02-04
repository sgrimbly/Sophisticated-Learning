% -------------------------------------------------------------------------
function [L] = spm_backwards(O, Q, A, B, u, t, T)
% Evaluate posterior over initial states by backwards smoothing (GPU-based)

    L = Q{t, 2};
    p = gpuArray(1);

    for timestep = (t + 1):T
        p = B{2}(:, :, 1) * p;

        for state = 1:numel(L)
            for g = 3:3
                obs = find(cumsum(O{g, timestep}) >= rand, 1);
                temp = A{g}(obs, :, :);
                temp = permute(temp, [3,2,1]);
                temp = temp * Q{timestep, 1}';
                aaa  = temp' * p(:, state);
                L(state) = L(state) .* aaa;
            end
        end
    end

    L = spm_norm(L(:));
end

% -------------------------------------------------------------------------
function Y = spm_cross(X, x, varargin)
% Multidimensional outer product on GPU
%
% Recursively handles cell inputs if present.

    if nargin < 2
        if isnumeric(X)
            Y = X;
        else
            Y = spm_cross(X{:});
        end
        return;
    end

    if iscell(X), X = spm_cross(X{:}); end
    if iscell(x), x = spm_cross(x{:}); end

    A = reshape(X, [size(X), ones(1, ndims(x))]);
    B = reshape(x, [ones(1, ndims(X)), size(x)]);
    Y = squeeze(bsxfun(@times, A, B));

    for i = 1:numel(varargin)
        Y = spm_cross(Y, varargin{i});
    end
end

% -------------------------------------------------------------------------
function A = spm_norm(A)
% normalisation of a probability transition matrix (columns) on GPU
    A = bsxfun(@rdivide, A, sum(A, 1));
    A(isnan(A)) = 1 / size(A, 1);
end