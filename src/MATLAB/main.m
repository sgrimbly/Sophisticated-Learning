% Get the directory of the current file
currentDir = fileparts(mfilename('fullpath'));

% Generate path for the 'src' folder and all its subfolders
srcPath = fullfile(currentDir, '../MATLAB');
fullSrcPath = genpath(srcPath);

% Add the generated path to MATLAB's search path
addpath(fullSrcPath);

run_SL = 0;
run_SI = 0;
run_BA = 0;
run_BAUCB = 0;
run_known_large_MCT = 1;

if run_SL == 1
    SL("1")
elseif run_SI == 1
    SI("1")
elseif run_BA == 1
    BA("1")
elseif run_BAUCB == 1
    BA_UCB("1")
elseif run_known_large_MCT == 1

    if ispc
        root = 'L:';
        seed = '1'; % Subject ID
        horizon = '2'; % Where the subject file is located
        k_factor = '0.7';
        mct = '3';
        num_mct = '100';
    elseif isunix
        root = '/media/labs';
        seed = getenv('seed'); % Subject ID
        horizon = getenv('horizon'); % Where the subject file is located
        k_factor = getenv('k_factor');
        mct = getenv('mct');
        num_mct = getenv('num_mct');
    end

    seed = seed
    horizon = horizon
    k_factor = k_factor
    mct = mct
    num_mct = num_mct

    known_large_MCT(seed, horizon, k_factor, root, mct, num_mct);
    %known_large_MCT_v2(seed, horizon, k_factor,root, mct) ;

    disp('run complete')
end
