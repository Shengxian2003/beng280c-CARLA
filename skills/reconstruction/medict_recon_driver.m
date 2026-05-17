function medict_recon_driver(config_path)
% Stage 2a MATLAB driver: parameterizable batch reconstruction.
%
% Reads a JSON config file describing the run (input/output paths, method,
% iterations, GPU flag, etc.), executes the CS or CORe reconstruction from
% motion-robust-CMR-main, and saves results to a deterministic output path.
%
% Designed to be called from the Python wrapper in subprocess/batch mode:
%   matlab -batch "medict_recon_driver('G:\medict_tmp\config.json')"
%
% Config JSON schema (all paths in Windows format for Windows MATLAB):
%   {
%     "input_mat":      "G:\\...\\input.mat",
%     "output_mat":     "G:\\...\\output.mat",
%     "recon_dir":      "\\\\wsl.localhost\\...\\3D cine-4D flow MRI Reconstruction (Study III IV V)",
%     "method":         "cs" | "core",
%     "is_flow":        1 | 0,
%     "is_rest":        1 | 0,
%     "n_iterations":   5 | 50 | ...,
%     "n_coils":        12,
%     "use_gpu":        1 | 0,
%     "data_field":     "D"        (top-level struct field in input_mat that holds kb/kx/ky/kz),
%     "venc_m_per_s":   1.5         (recorded in output for downstream tools; not used by recon itself)
%   }
%
% Output .mat contains an 'outputs' struct with:
%   xHat   [Z Y X T]  magnitude (sum-of-squares for 4D flow)
%   thetaX [Z Y X T]  background-corrected phase, x velocity encoding
%   thetaY [Z Y X T]  background-corrected phase, y velocity encoding
%   thetaZ [Z Y X T]  background-corrected phase, z velocity encoding
%   meta             struct with elapsed_minutes, method, n_iterations, shape, etc.

cfg = jsondecode(fileread(config_path));

% ---- Add reconstruction code paths ---------------------------------------
addpath(genpath(fullfile(cfg.recon_dir, 'functions')));
addpath(genpath(fullfile(cfg.recon_dir, 'recon_methods')));

% Some HDF5 v7.3 .mat files on WSL network paths need this
setenv('HDF5_USE_FILE_LOCKING', 'FALSE');

% ---- Load and unpack k-space --------------------------------------------
fprintf('Loading %s ...\n', cfg.input_mat);
raw = load(cfg.input_mat);
if ~isfield(raw, cfg.data_field)
    error('Input .mat has no top-level field "%s"', cfg.data_field);
end
D = raw.(cfg.data_field);

if cfg.is_flow
    kdata = cat(6, D.kb, D.kx, D.ky, D.kz);
else
    kdata = D.kb;
end

if isfield(D, 'sampB') && ~isempty(D.sampB)
    samp = cat(5, D.sampB, D.sampX, D.sampY, D.sampZ);
else
    samp = logical(squeeze(abs(kdata(:,:,:,1,:,:))));
end

if isfield(D, 'weightsB') && ~isempty(D.weightsB)
    weights = cat(5, D.weightsB, D.weightsX, D.weightsY, D.weightsZ);
else
    weights = samp;
end

% ---- Regularization parameters (from main_recon.m, rest/exercise * 4D/3D) -
opt.flow = cfg.is_flow;
if cfg.is_flow
    if cfg.is_rest
        opt.lam_cs    = 2e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
        opt.lam1_core = 2e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
    else
        opt.lam_cs    = 4e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
        opt.lam1_core = 4e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
    end
else
    if cfg.is_rest
        opt.lam_cs    = 5e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
        opt.lam1_core = 5e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
    else
        opt.lam_cs    = 7e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
        opt.lam1_core = 7e-4 * [1e-2, 1,1,1,1,1,1,1, 5,5,5,5,5,5,5,5];
    end
end
opt.lam2_core = 7.5e-2;
opt.mu_cs     = 5e-1;
opt.mu1_core  = 5e-1;
opt.mu2_core  = 5e-1;

opt.coil      = cfg.n_coils;
opt.use_gpu   = cfg.use_gpu;
opt.nit       = cfg.n_iterations;
opt.oIter     = cfg.n_iterations;
opt.iIter     = 4;
opt.gStp      = 1e-1;
opt.vrb       = 1;
opt.spar      = 'jtv';
opt.transform = 'harr';
opt.sfolder   = fileparts(cfg.output_mat);
opt.readout   = numel(find(sum(reshape(samp, size(samp,1), []), 2)));
opt.recon     = cfg.method;

% RTX 5090 forward-compatibility (compute capability 12.0)
if opt.use_gpu
    try
        parallel.gpu.enableCUDAForwardCompatibility(true);
    catch
        % older MATLAB versions don't have this; ignore
    end
end

% ---- Pre-processing ------------------------------------------------------
fprintf('Coil combining to %d coils...\n', opt.coil);
kdata = coilCombine(kdata, opt.coil, '3dt');

fprintf('Computing time-averaged image...\n');
avg_k = sum(sum(kdata, 5), 6);
avg_pattern = sum(sum(samp, 4), 5);
avg_pattern(avg_pattern == 0) = inf;
avg_k = bsxfun(@rdivide, avg_k, avg_pattern);
avg_image = ifft3_shift(avg_k);

fprintf('Estimating Walsh sensitivity maps...\n');
p.fil = 3;
[maps, x0] = WalshCoilCombine3D(avg_image, p);
x0 = repmat(x0, [1, 1, 1, size(samp, 4)]);

acceleration_rate = numel(weights) / sum(weights(:).^2);
fprintf('Acceleration rate: %.2fx\n', acceleration_rate);

scale = 0.1 * max(abs(kdata(:)));

% ---- Reconstruction ------------------------------------------------------
fprintf('Running %s reconstruction, %d iterations...\n', opt.recon, opt.nit);
t_start = tic;
xhat = zeros(size(squeeze(kdata(:,:,:,1,:,:))));
for k = 1:size(kdata, 6)
    fprintf('  Encoding %d of %d\n', k, size(kdata, 6));
    if opt.use_gpu
        gpuDevice(1);
    end
    [xhat(:,:,:,:,k), ~] = pMRIL14D( ...
        kdata(:,:,:,:,:,k) / scale, ...
        samp(:,:,:,:,k), ...
        weights(:,:,:,:,k), ...
        opt, maps, x0 / scale);
end
elapsed_min = toc(t_start) / 60;
fprintf('Reconstruction elapsed: %.2f minutes\n', elapsed_min);

% ---- Magnitude (sum-of-squares for 4D flow) -----------------------------
if cfg.is_flow
    outputs.xHat = zeros(size(xhat(:,:,:,:,1)));
    for i = 1:size(kdata, 6)
        outputs.xHat = outputs.xHat + xhat(:,:,:,:,i).^2;
    end
    outputs.xHat = sqrt(outputs.xHat);
else
    outputs.xHat = xhat(:,:,:,:,1);
end

% ---- Background phase correction (4D flow only) -------------------------
if cfg.is_flow
    fprintf('Background phase correction...\n');
    cmapx = backgroundCorrection3D(xhat(:,:,:,:,1), xhat(:,:,:,:,2));
    cmapy = backgroundCorrection3D(xhat(:,:,:,:,1), xhat(:,:,:,:,3));
    cmapz = backgroundCorrection3D(xhat(:,:,:,:,1), xhat(:,:,:,:,4));

    thetaXo = xhat(:,:,:,:,2) .* conj(xhat(:,:,:,:,1));
    thetaYo = xhat(:,:,:,:,3) .* conj(xhat(:,:,:,:,1));
    thetaZo = xhat(:,:,:,:,4) .* conj(xhat(:,:,:,:,1));

    outputs.thetaX = angle(bsxfun(@times, thetaXo, exp(-1i * cmapx)));
    outputs.thetaY = angle(bsxfun(@times, thetaYo, exp(-1i * cmapy)));
    outputs.thetaZ = angle(bsxfun(@times, thetaZo, exp(-1i * cmapz)));
end

% ---- Metadata for the Python wrapper ------------------------------------
outputs.meta.elapsed_minutes = elapsed_min;
outputs.meta.method          = char(opt.recon);
outputs.meta.n_iterations    = opt.nit;
outputs.meta.n_coils         = opt.coil;
outputs.meta.use_gpu         = opt.use_gpu;
outputs.meta.acceleration    = acceleration_rate;
outputs.meta.is_flow         = cfg.is_flow;
outputs.meta.is_rest         = cfg.is_rest;
outputs.meta.venc_m_per_s    = cfg.venc_m_per_s;
outputs.meta.shape_ZYXT      = size(outputs.xHat);

% ---- Save ----------------------------------------------------------------
out_dir = fileparts(cfg.output_mat);
if ~exist(out_dir, 'dir'); mkdir(out_dir); end
save(cfg.output_mat, 'outputs', '-v7');
fprintf('Saved: %s\n', cfg.output_mat);
fprintf('=== RECON COMPLETE ===\n');
end
