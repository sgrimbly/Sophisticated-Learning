if ispc
    root ='L:';
    seed = '1';   % Subject ID
    horizon = '2';  % Where the subject file is located
    k_factor = '0.7';
    mct = '3';
    num_mct = '100';
elseif isunix
    root='/media/labs';
    seed = getenv('seed');   % Subject ID
    horizon = getenv('horizon');  % Where the subject file is located
    k_factor = getenv('k_factor');
    mct = getenv('mct');
    num_mct = getenv('num_mct');
end
seed = seed
horizon = horizon
k_factor = k_factor
mct = mct
num_mct = num_mct

known_large_MCT(seed, horizon, k_factor,root, mct, num_mct) ;
%known_large_MCT_v2(seed, horizon, k_factor,root, mct) ;

disp('run complete')
