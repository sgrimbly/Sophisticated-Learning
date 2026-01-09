function out = sanitize_file_component(in)

    if nargin < 1 || isempty(in)
        in = 'none';
    end

    in = char(in);
    out = regexprep(in, '[^a-zA-Z0-9_-]', '_');

    if isempty(out)
        out = 'none';
    end

end
