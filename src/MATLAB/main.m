function main(algorithm, seed, horizon, k_factor, root_folder, mct, num_mct)
    % Check the number of arguments and set default values if necessary
    if nargin < 1
        algorithm = 'SL';  % Default value for algorithm
    end
    if nargin < 2
        seed = '1';  % Default seed should be an integer
    end
    if nargin < 3
        horizon = 1000;  % Default value for horizon
    end
    if nargin < 4
        k_factor = 1.5;  % Default value for k_factor
    end
    if nargin < 5
        root_folder = 'defaultFolder';  % Default value for root_folder
    end
    if nargin < 6
        mct = 500;  % Default value for mct
    end
    if nargin < 7
        num_mct = 10;  % Default value for num_mct
    end

    % Directory and path setup
    currentDir = fileparts(mfilename('fullpath'));
    srcPath = fullfile(currentDir, '../MATLAB');
    fullSrcPath = genpath(srcPath);
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
