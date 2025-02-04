function G = G_epistemic_value(A, s)
% Bayesian surprise or mutual information, adapted for GPU usage
%
% A   - cell array of likelihood arrays on GPU
% s   - probability density of causes on GPU

    qx = spm_cross(s);  % outer product across states
    G  = gpuArray(0);
    qo = gpuArray(0);

    idxNonNeg = find(qx > exp(-16));
    for i = 1:numel(idxNonNeg)
        ix = idxNonNeg(i);
        po = gpuArray(1);
        for g = 1:numel(A)
            po = spm_cross(po, A{g}(:, ix));
        end
        po = po(:);
        qo = qo + qx(ix) * po;
        G  = G  + qx(ix) * (po' * nat_log(po));
    end

    % subtract entropy of expectations
    G = G - qo' * nat_log(qo);
end