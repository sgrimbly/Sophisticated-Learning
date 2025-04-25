function [loadedState, isNew] = load_state(filename)
    if exist(filename, 'file')
        data = load(filename);
        loadedState = data.state; % Assume 'state' is the variable name used in save
        isNew = false; % Existing state found, not a new simulation
    else
        isNew = true; % No existing state found, start a new simulation
        loadedState = {}; % Initial empty cell array for a new state
    end
end