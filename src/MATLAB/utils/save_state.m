function save_state(filename, state)
    [dir_path, ~, ~] = fileparts(filename);
    if ~isempty(dir_path) && ~exist(dir_path, 'dir')
        [ok, msg] = mkdir(dir_path);
        if ~ok
            error('Unable to create state directory: %s', msg);
        end
    end
    save(filename, 'state');
end
