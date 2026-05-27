function m = normalise_matrix(m)
    % Bit-exact vectorised replacement for the per-column for-loop. sum(m, 1)
    % sums down each column in the same order as the original loop, so the
    % element-wise divisions are identical under IEEE 754. Behaviour for
    % zero-sum columns (NaN output) is preserved.
    m = bsxfun(@rdivide, m, sum(m, 1));
end
