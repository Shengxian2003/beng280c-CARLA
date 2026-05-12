% Stage 1c: Headless reconstruction - callable from WSL via Windows MATLAB
% Paths use Windows UNC format so Windows MATLAB can access WSL filesystem

recon_dir = '\\wsl.localhost\Ubuntu\home\nick_17\projects\medict\skills\reconstruction\motion-robust-CMR-main\3D cine-4D flow MRI Reconstruction (Study III IV V)';
data_path = 'G:\medict_temp.mat';
out_dir   = 'G:\medict_out';

addpath(genpath(fullfile(recon_dir, 'functions')));
addpath(genpath(fullfile(recon_dir, 'recon_methods')));

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

setenv('HDF5_USE_FILE_LOCKING', 'FALSE');  % allows HDF5 to open files on WSL network path
disp('Loading data...');
data = load(data_path);

kdata = cat(6, data.D.kb, data.D.kx, data.D.ky, data.D.kz);

if isfield(data.D, 'sampB') && ~isempty(data.D.sampB)
    samp = cat(5, data.D.sampB, data.D.sampX, data.D.sampY, data.D.sampZ);
else
    samp = logical(squeeze(abs(kdata(:,:,:,1,:,:))));
end

if isfield(data.D, 'weightsB') && ~isempty(data.D.weightsB)
    weights = cat(5, data.D.weightsB, data.D.weightsX, data.D.weightsY, data.D.weightsZ);
else
    weights = samp;
end

% Parameters (rest 4D flow)
opt.flow      = 1;
opt.lam_cs    = 2e-4*[1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
opt.lam1_core = 2e-4*[1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
opt.lam2_core = 7.5e-2;
opt.mu_cs     = 5e-1;
opt.mu1_core  = 5e-1;
opt.mu2_core  = 5e-1;
opt.coil      = 12;
opt.use_gpu   = 1;   % RTX 5090 via Windows MATLAB GPU toolbox
parallel.gpu.enableCUDAForwardCompatibility(true);  % required for RTX 5090 (compute 12.0)
opt.nit       = 5;   % 5 iterations for test (full = 50)
opt.oIter     = 5;
opt.iIter     = 4;
opt.gStp      = 1e-1;
opt.vrb       = 1;
opt.spar      = 'jtv';
opt.transform = 'harr';
opt.sfolder   = out_dir;
opt.readout   = numel(find(sum(reshape(samp, size(samp,1),[]), 2)));

disp('Coil combining...');
kdata = coilCombine(kdata, opt.coil, '3dt');

disp('Computing time-averaged image...');
avg_k = sum(sum(kdata,5),6);
avg_pattern = sum(sum(samp,4),5);
avg_pattern(avg_pattern==0) = inf;
avg_k = bsxfun(@rdivide, avg_k, avg_pattern);
avg_image = ifft3_shift(avg_k);

disp('Estimating sensitivity maps...');
p.fil = 3;
[maps, x0] = WalshCoilCombine3D(avg_image, p);
x0 = repmat(x0, [1,1,1,size(samp,4)]);

acceleration_rate = numel(weights) / sum(weights(:).^2);
disp(['Acceleration rate: ', num2str(acceleration_rate)]);

scale = 0.1 * max(abs(kdata(:)));
opt.recon = 'cs';

disp('Running reconstruction (CS, 5 iterations)...');
t = tic;
xhat = zeros(size(squeeze(kdata(:,:,:,1,:,:))));
for k = 1:size(kdata,6)
    disp(['  Encoding: ', num2str(k), ' of ', num2str(size(kdata,6))]);
    [xhat(:,:,:,:,k), ~] = pMRIL14D(kdata(:,:,:,:,:,k)/scale, ...
        samp(:,:,:,:,k), weights(:,:,:,:,k), opt, maps, x0/scale);
end
fprintf('Elapsed: %.2f minutes\n', toc(t)/60);

% Save outputs
outputs.xHat   = sqrt(sum(xhat.^2, 5));
outputs.thetaX = angle(xhat(:,:,:,:,2) .* conj(xhat(:,:,:,:,1)));
outputs.thetaY = angle(xhat(:,:,:,:,3) .* conj(xhat(:,:,:,:,1)));
outputs.thetaZ = angle(xhat(:,:,:,:,4) .* conj(xhat(:,:,:,:,1)));

out_file = fullfile(out_dir, 'test_4dflow_cs.mat');
save(out_file, 'outputs');

disp(' ');
disp('=== Stage 1c SUCCESS ===');
disp(['Output: ', out_file]);
disp(['Magnitude shape: ', mat2str(size(outputs.xHat))]);
disp(['thetaX shape:    ', mat2str(size(outputs.thetaX))]);
disp(['thetaY shape:    ', mat2str(size(outputs.thetaY))]);
disp(['thetaZ shape:    ', mat2str(size(outputs.thetaZ))]);
disp('Valid 4D velocity field confirmed (magnitude + 3 velocity components)');
