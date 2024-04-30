function main(algorithm, seed, horizon, k_factor, root_folder, mct, num_mct)
    % Get the directory of the current file
    currentDir = fileparts(mfilename('fullpath'));

    % Generate path for the 'src' folder and all its subfolders
    srcPath = fullfile(currentDir, '../MATLAB');
    fullSrcPath = genpath(srcPath);

    % Add the generated path to MATLAB's search path
    addpath(fullSrcPath);

    % Execute based on the selected algorithm
    switch algorithm
        case 'SL'
            SL(seed);
            disp('SL run complete');
        case 'SI'
            SI(seed);
            disp('SI run complete');
        case 'BA'
            BA(seed);
            disp('BA run complete');
        case 'BAUCB'
            BA_UCB(seed);
            disp('BA_UCB run complete');
        case 'known_large_MCT'
            known_large_MCT(seed, horizon, k_factor, root_folder, mct, num_mct);
            disp('Known large MCT run complete');
        otherwise
            error('Unknown algorithm specified');
    end
end
