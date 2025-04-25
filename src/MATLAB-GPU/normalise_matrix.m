function m = normalise_matrix(m)
% normalise columns in a 2D gpuArray
    for i = 1:size(m,2)
        s = sum(m(:, i));
        if s == 0
            m(:, i) = 1/size(m,1);
        else
            m(:, i) = m(:, i) / s;
        end
    end
end