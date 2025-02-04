function x = normalise(array)
% normalise a gpuArray vector or matrix
    s = sum(array(:));
    if s == 0
        x = array;
        return;
    end
    x = array / s;
    if any(isnan(x(:)))
        x = gpuArray.ones(size(x), 'like', x);
        x = x / numel(x);
    end
end