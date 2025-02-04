function kl = kldir(a, b)
% Kullback-Leibler divergence on GPU
    if ~isequal(size(a), size(b))
        error('Input matrices must have the same dimensions.');
    end
    kl = sum(a .* log(a ./ b), 'all');
    if ~isfinite(kl)
        kl = realmax('double');
    end
end

% -------------------------------------------------------------------------
function y = nat_log(x)
% numeric log on GPU with small offset to prevent log(0)
    y = log(x + exp(-500));
end