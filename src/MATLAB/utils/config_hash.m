function hash = config_hash(config, nChars)

    if nargin < 2
        nChars = 8;
    end

    try
        payload = jsonencode(config);
    catch
        payload = evalc('disp(config)');
    end

    try
        md = java.security.MessageDigest.getInstance('MD5');
        md.update(uint8(payload));
        digest = typecast(md.digest(), 'uint8');
        fullHex = lower(reshape(dec2hex(digest, 2).', 1, []));
    catch
        bytes = uint8(payload);
        digest = uint8(mod(cumsum(bytes), 256));
        fullHex = lower(reshape(dec2hex(digest(end - min(15, numel(digest) - 1):end), 2).', 1, []));
    end

    hash = fullHex(1:min(nChars, numel(fullHex)));

end
