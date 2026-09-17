import os
import glob
import argparse
import numpy as np
import torch
from astropy.io import fits
from tqdm import tqdm
import torchmfbd
import yaml

def read_hifi_dataset(fits_path, n_frames=100, crop_size=None):
    """
    Reads HiFI+ level1 dataset from a .fts file.
    Camera 1 (Ext 1, 3, 5...) -> Narrow-band (656.3 nm Lyot)
    Camera 2 (Ext 2, 4, 6...) -> Broad-band (656.7 nm Wideband)

    Frame selection uses MFGS (Median Filter-Gradient Similarity), the seeing
    metric the instrument pipeline already stores in every extension header,
    rather than an ad-hoc variance/contrast score. Note that level 1 has
    normally already selected the best frames upstream (NSETFRMS frames
    reduced to NEXTEN, with non-contiguous FRAMEIDs), so this only matters
    when fewer frames than are available are requested.

    The retained frames are returned in *temporal* order: frame 0 is the
    destretch reference and the tip-tilt anchor of the deconvolution, so
    keeping it the earliest retained frame makes that anchor reproducible
    from burst to burst instead of an arbitrary quality-sorted pick.
    """
    f = fits.open(fits_path)
    total_ext = len(f) - 1  # total ImageHDUs
    n_avail = total_ext // 2

    # Header-only pass: MFGS per frame, averaged over the two cameras.
    mfgs = np.full(n_avail, np.nan)
    for i in range(n_avail):
        vals = [f[e].header.get('MFGSMED') for e in (1 + 2*i, 2 + 2*i)]
        vals = [v for v in vals if v is not None]
        if vals:
            mfgs[i] = float(np.mean(vals))

    n_keep = min(n_avail, n_frames)
    if np.all(np.isnan(mfgs)):
        keep = np.arange(n_keep)
    else:
        best = np.argsort(np.nan_to_num(mfgs, nan=-np.inf))[::-1][:n_keep]
        keep = np.sort(best)

    img_h, img_w = f[1].data.shape
    if crop_size is not None:
        ch, cw = min(img_h, crop_size), min(img_w, crop_size)
    else:
        ch, cw = img_h, img_w

    nb_frames = np.zeros((1, n_keep, ch, cw), dtype=np.float32)
    wb_frames = np.zeros((1, n_keep, ch, cw), dtype=np.float32)

    for k, i in enumerate(keep):
        # Ext 1 + 2*i: Camera 1 (Narrowband)
        # Ext 2 + 2*i: Camera 2 (Broadband)
        nb_frames[0, k, :, :] = f[1 + 2*i].data[:ch, :cw]
        wb_frames[0, k, :, :] = f[2 + 2*i].data[:ch, :cw]

    header = f[0].header
    f.close()
    return nb_frames, wb_frames, header, mfgs[keep]


def dark_fraction(wb_frames, dark_level=0.35):
    """
    Fraction of pixels well below the median intensity, i.e. off-limb sky.
    Used to decide per burst whether the field is off-limb: a single global
    flag is not enough, since a dataset can cross the limb during a run.
    """
    ref = wb_frames[0]
    med = np.median(ref)
    if med <= 0:
        return 0.0
    return float(np.mean(ref < dark_level * med))

def process_single_file(fits_path, output_path, config_path, args, device):
    """
    Processes a single HiFI+ FITS observation file through GPU-accelerated MOMFBD.
    """
    nb_frames, wb_frames, header, mfgs = read_hifi_dataset(fits_path, n_frames=args.n_frames, crop_size=args.crop_size)

    with open(config_path, 'r') as fh:
        cfg = yaml.safe_load(fh)
    # The deconvolution tapers this many pixels at each patch edge, so the same
    # number is discarded when mosaicking - otherwise apodized, poorly
    # constrained pixels get blended into the final image.
    apod = int(cfg['images'].get('apodization_border', 0))

    # Off-limb / on-disk decision, per burst rather than per run: a dataset can
    # cross the limb during an observing sequence.
    dfrac = dark_fraction(wb_frames)
    if args.limb_mode == 'auto':
        off_limb = dfrac > args.limb_threshold
    else:
        off_limb = (args.limb_mode == 'off_limb')

    # Intensity scaling. The 95th percentile is a robust high-signal reference
    # and is measurably more stable than the frame mean in *both* regimes
    # (residual disk-level scatter 0.07% vs 0.51% off-limb, 0.02% vs 0.06%
    # on-disk), so it is used unconditionally.
    p95_wb = np.percentile(wb_frames, 95, axis=(-2, -1), keepdims=True)
    p95_nb = np.percentile(nb_frames, 95, axis=(-2, -1), keepdims=True)

    # Convert to PyTorch tensors and move to target device (CUDA / CPU)
    wb_tensor = torch.tensor(wb_frames, dtype=torch.float32, device=device)
    nb_tensor = torch.tensor(nb_frames, dtype=torch.float32, device=device)

    wb_tensor /= torch.tensor(np.maximum(p95_wb, 1e-5), dtype=torch.float32, device=device)
    nb_tensor /= torch.tensor(np.maximum(p95_nb, 1e-5), dtype=torch.float32, device=device)

    # Destretching / alignment if enabled
    if not args.no_destretch:
        lambda_tt = 0.08 if off_limb else 0.01
        # Destretch WB sequence
        wb_warped, _ = torchmfbd.destretch(
            wb_tensor[:, None, :, :, :],
            ngrid=64, lr=0.50, reference_frame=0, border=6, n_iterations=40, lambda_tt=lambda_tt
        )
        wb_tensor = wb_warped[:, 0, :, :, :]

        # Destretch NB sequence
        nb_warped, _ = torchmfbd.destretch(
            nb_tensor[:, None, :, :, :],
            ngrid=64, lr=0.50, reference_frame=0, border=6, n_iterations=40, lambda_tt=lambda_tt
        )
        nb_tensor = nb_warped[:, 0, :, :, :]

    # Create deconvolution object
    decSI = torchmfbd.Deconvolution(config_path)

    # Patchify
    patchify = torchmfbd.Patchify4D()
    wb_patches = patchify.patchify(wb_tensor, patch_size=args.patch_size, stride_size=args.stride_size, flatten_sequences=True)
    nb_patches = patchify.patchify(nb_tensor, patch_size=args.patch_size, stride_size=args.stride_size, flatten_sequences=True)

    # Estimate noise
    wb_noise = torchmfbd.compute_noise(wb_patches[0:1, 0:1, ...])
    nb_noise = torchmfbd.compute_noise(nb_patches[0:1, 0:1, ...])

    # Add frames to MOMFBD engine
    decSI.add_frames(wb_patches, id_object=0, id_diversity=0, diversity=0.0, sigma=wb_noise)
    decSI.add_frames(nb_patches, id_object=1, id_diversity=0, diversity=0.0, sigma=nb_noise)

    # Deconvolve on GPU
    decSI.deconvolve(
        infer_object=False,
        optimizer='adam',
        simultaneous_sequences=args.simultaneous_seq,
        n_iterations=args.n_iterations
    )

    # Unpatchify results. `apodization` crops this many pixels from every patch
    # edge before blending, so the mosaic origin sits at pixel (apod, apod) of
    # the raw frame - recorded below as APODCROP so downstream plotting can
    # co-align the raw and reconstructed images.
    obj_wb = patchify.unpatchify(decSI.obj[0][:, None, ...], apodization=apod, weight_type='cosine', weight_params=30).cpu().numpy()
    obj_nb = patchify.unpatchify(decSI.obj[1][:, None, ...], apodization=apod, weight_type='cosine', weight_params=30).cpu().numpy()

    # Save reconstructed objects to FITS
    hdu0 = fits.PrimaryHDU(
    data=obj_wb[0, :, :],
    header=header)
    
    h = hdu0.header
    h['PIPELINE'] = ('hifi_momfbd_batch_gpu', 'Reconstruction script')
    h['TMFBDVER'] = (getattr(torchmfbd, '__version__', 'unknown'), 'torchmfbd version')
    h['CFGFILE'] = (os.path.basename(args.config), 'MOMFBD configuration file')
    h['NFRAMES'] = (nb_frames.shape[1], 'Frames used per camera')
    h['PATCHSZ'] = (args.patch_size, 'Patch size [px]')
    h['STRIDESZ'] = (args.stride_size, 'Patch stride [px]')
    h['APODCROP'] = (apod, 'Px cropped per patch edge; mosaic origin offset')
    h['CROPSIZE'] = (str(args.crop_size), 'Input crop size (None = full FOV)')
    h['NITER'] = (args.n_iterations, 'Optimization iterations')
    h['LIMBMODE'] = ('off_limb' if off_limb else 'on_disk', 'Regime used for this burst')
    h['DARKFRAC'] = (round(dfrac, 5), 'Dark-sky pixel fraction (limb indicator)')
    h['NORMTYPE'] = ('p95', 'Per-frame intensity normalization')
    h['DESTRTCH'] = (not args.no_destretch, 'Destretching applied')
    if np.isfinite(mfgs).any():
        h['MFGSMEAN'] = (round(float(np.nanmean(mfgs)), 5), 'Mean MFGS of frames used')

    hdu1 = fits.ImageHDU(data=obj_wb[0, :, :], name="WIDEBAND_RECONSTRUCTED")
    hdu2 = fits.ImageHDU(data=obj_nb[0, :, :], name="NARROWBAND_RECONSTRUCTED")
    hdu3 = fits.ImageHDU(data=decSI.rho[0].cpu().numpy(), name="PHASE_MODES_WB")
    hdu4 = fits.ImageHDU(data=decSI.rho[1].cpu().numpy(), name="PHASE_MODES_NB")
    hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])

    hdul.writeto(output_path, overwrite=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GPU-Optimized Batch MOMFBD for HiFI+ Datasets.")
    parser.add_argument("--input_dir", type=str, 
                        default="/dat/andreuva/data/hifiplus/level1/20260714",
                        help="Input directory containing HiFI+ FITS files")
    parser.add_argument("--pattern", type=str, default="*_sd.fts", help="File matching pattern")
    parser.add_argument("--output_dir", type=str, default="results_momfbd", help="Output directory for deconvolved FITS")
    parser.add_argument("--config", type=str, default="hifi_momfbd_gpu.yaml", help="Path to config YAML file")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU, 0+ for GPU)")
    parser.add_argument("--n_frames", type=int, default=100, help="Number of frames per camera per burst")
    parser.add_argument("--patch_size", type=int, default=96, help="Patch size")
    parser.add_argument("--stride_size", type=int, default=50, help="Stride size")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop region size (None for full FOV)")
    parser.add_argument("--start_time", type=str, default=None, help="Start time (HHMMSS)")
    parser.add_argument("--end_time", type=str, default=None, help="End time (HHMMSS)")
    parser.add_argument("--no_destretch", action="store_true", help="Disable destretching")
    parser.add_argument("--limb_mode", choices=['auto', 'off_limb', 'on_disk'], default='auto',
                        help="Off-limb handling. 'auto' (default) classifies each burst from its "
                             "dark-sky fraction, which matters because a dataset can cross the limb "
                             "during a run; 'off_limb'/'on_disk' force one regime for all bursts.")
    parser.add_argument("--limb_threshold", type=float, default=0.02,
                        help="Dark-sky pixel fraction above which a burst counts as off-limb (--limb_mode auto)")
    parser.add_argument("--n_iterations", type=int, default=250, help="Optimization iterations")
    parser.add_argument("--simultaneous_seq", type=int, default=1000, help="Simultaneous patch sequences on GPU")
    parser.add_argument("--no_resume", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of files to process")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of parallel shards splitting the file list (for multi-GPU runs)")
    parser.add_argument("--shard_id", type=int, default=0, help="This shard's index in [0, num_shards), selecting every num_shards-th file starting at shard_id")
    args = parser.parse_args()

    # Determine device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Running GPU-accelerated batch MOMFBD on {torch.cuda.get_device_name(device)} (cuda:{args.gpu})")
    else:
        device = torch.device("cpu")
        print("CUDA GPU not requested or not available. Running on CPU.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Search for matching FITS files
    search_path = os.path.join(args.input_dir, args.pattern)
    fits_files = sorted(glob.glob(search_path))

    if args.start_time is not None or args.end_time is not None:
        start_time = args.start_time or "000000"
        end_time = args.end_time or "235959"
        fits_files = [
            path for path in fits_files
            if start_time <= os.path.basename(path).split("_")[2] <= end_time
        ]

    if not fits_files:
        print(f"No FITS files found matching pattern '{search_path}'.")
        exit(1)

    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            print(f"--shard_id must be in [0, {args.num_shards}).")
            exit(1)
        fits_files = fits_files[args.shard_id::args.num_shards]

    if args.limit is not None:
        fits_files = fits_files[:args.limit]

    print(f"Found {len(fits_files)} observation files to process in {args.input_dir}"
          f"{f' (shard {args.shard_id}/{args.num_shards})' if args.num_shards > 1 else ''}.")

    # Sync GPU option into config YAML if needed
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['optimization']['gpu'] = args.gpu
    temp_config = f"tmp_gpu_config_gpu{args.gpu}_shard{args.shard_id}.yaml"
    with open(temp_config, 'w') as f:
        yaml.dump(cfg, f)

    # Batch execution loop
    success_count = 0
    for idx, fits_path in enumerate(tqdm(fits_files, desc="Batch MOMFBD")):
        base_name = os.path.basename(fits_path).replace('.fts', '_momfbd.fits')
        output_path = os.path.join(args.output_dir, base_name)

        # Skip if already processed and resume enabled
        if not args.no_resume and os.path.exists(output_path):
            tqdm.write(f"[{idx+1}/{len(fits_files)}] Skipping existing output: {base_name}")
            continue

        tqdm.write(f"[{idx+1}/{len(fits_files)}] Processing {os.path.basename(fits_path)} -> {base_name}")
        try:
            process_single_file(fits_path, output_path, temp_config, args, device)
            success_count += 1
        except Exception as e:
            tqdm.write(f"Error processing {fits_path}: {e}")
        finally:
            # Must run on the failure path too: a mid-file exception otherwise
            # leaves the allocation in place and the next file OOMs.
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"Batch MOMFBD processing completed: {success_count}/{len(fits_files)} files processed successfully.")
