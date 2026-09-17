"""
MOMFBD reconstruction of a whole GREGOR / HiFI+ dataset, one GPU per shard.

    python hifi_momfbd_batch_gpu.py --input_dir <dir> --output_dir <dir> --gpu 0

Shares its pipeline with hifi_momfbd.py through hifi_common.py. See
hifi_momfbd.yaml for why the cutoffs and the number of modes are what they are.
"""

import os
import glob
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm
import hifi_common as hifi


def process_single_file(fits_path, output_path, config_dict, args, device):
    nb_frames, wb_frames, header, mfgs = hifi.read_hifi_dataset(
        fits_path, n_frames=args.n_frames, crop_size=args.crop_size)

    obj_wb, obj_nb, dec, meta = hifi.reconstruct_burst(
        nb_frames, wb_frames, config_dict, device,
        patch_size=args.patch_size, stride_size=args.stride_size,
        n_iterations=args.n_iterations, simultaneous_sequences=args.simultaneous_seq,
        destretch=not args.no_destretch, regime=args.regime,
        disk_modes=args.disk_modes, limb_modes=args.limb_modes,
        logger=tqdm.write)

    extra = {}
    if np.isfinite(mfgs).any():
        extra['MFGSMEAN'] = (round(float(np.nanmean(mfgs)), 5), 'Mean MFGS of frames used')

    hifi.write_output(output_path, header, obj_wb, obj_nb, dec, meta,
                      config_name=os.path.basename(args.config), extra=extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch MOMFBD for HiFI+ datasets.")
    parser.add_argument("--input_dir", type=str,
                        default="/dat/andreuva/data/hifiplus/level1/20260714",
                        help="Directory holding the HiFI+ FITS bursts")
    parser.add_argument("--pattern", type=str, default="*_sd.fts", help="File pattern")
    parser.add_argument("--output_dir", type=str, default="results_momfbd", help="Output directory")
    parser.add_argument("--config", type=str, default="hifi_momfbd.yaml", help="Configuration YAML")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    parser.add_argument("--n_frames", type=int, default=100, help="Frames per camera per burst")
    parser.add_argument("--patch_size", type=int, default=96, help="Patch size")
    parser.add_argument("--start_time", type=str, default=None, help="Start time (HHMMSS)")
    parser.add_argument("--end_time", type=str, default=None, help="End time (HHMMSS)")
    parser.add_argument("--stride_size", type=int, default=32,
                        help="Stride between patches. 32 with apodization_border 20 leaves 24 px of "
                             "overlap, which is what keeps the limb artifact from surviving the mosaic")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop the field (for quick tests)")
    parser.add_argument("--no_destretch", action="store_true", help="Skip destretching")
    parser.add_argument("--regime", choices=['auto', 'on_disk', 'off_limb'], default='auto',
                        help="'auto' decides on-disk vs off-limb per patch, per burst, which matters "
                             "because a dataset can cross the limb during a run")
    parser.add_argument("--disk_modes", type=int, default=44, help="Wavefront modes, on-disk patches")
    parser.add_argument("--limb_modes", type=int, default=20, help="Wavefront modes, off-limb patches")
    parser.add_argument("--n_iterations", type=int, default=250, help="Optimization iterations")
    parser.add_argument("--simultaneous_seq", type=int, default=200, help="Patches optimized together")
    parser.add_argument("--no_resume", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many files")
    parser.add_argument("--num_shards", type=int, default=1, help="Number of parallel shards")
    parser.add_argument("--shard_id", type=int, default=0, help="This shard's index in [0, num_shards)")
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Running on {torch.cuda.get_device_name(device)} (cuda:{args.gpu})")
    else:
        device = torch.device("cpu")
        print("CUDA not requested or not available. Running on CPU.")

    os.makedirs(args.output_dir, exist_ok=True)

    fits_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if args.start_time is not None or args.end_time is not None:
        start_time = args.start_time or "000000"
        end_time = args.end_time or "235959"
        fits_files = [
            path for path in fits_files
            if start_time <= os.path.basename(path).split("_")[2] <= end_time
        ]
    if not fits_files:
        print(f"No files matching '{args.pattern}' in {args.input_dir}.")
        raise SystemExit(1)

    if args.num_shards > 1:
        if not (0 <= args.shard_id < args.num_shards):
            print(f"--shard_id must be in [0, {args.num_shards}).")
            raise SystemExit(1)
        fits_files = fits_files[args.shard_id::args.num_shards]

    if args.limit is not None:
        fits_files = fits_files[:args.limit]

    print(f"Found {len(fits_files)} bursts in {args.input_dir}"
          f"{f' (shard {args.shard_id}/{args.num_shards})' if args.num_shards > 1 else ''}.")

    # Read the configuration once and pass it as a dict, so that parallel shards
    # do not need a temporary file each.
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict['optimization']['gpu'] = args.gpu

    success = 0
    for idx, fits_path in enumerate(tqdm(fits_files, desc="Batch MOMFBD")):
        base_name = os.path.basename(fits_path).replace('.fts', '_momfbd.fits')
        output_path = os.path.join(args.output_dir, base_name)

        if not args.no_resume and os.path.exists(output_path):
            tqdm.write(f"[{idx+1}/{len(fits_files)}] Skipping existing {base_name}")
            continue

        tqdm.write(f"[{idx+1}/{len(fits_files)}] {os.path.basename(fits_path)} -> {base_name}")
        try:
            process_single_file(fits_path, output_path, config_dict, args, device)
            success += 1
        except Exception as e:
            tqdm.write(f"Error processing {fits_path}: {e}")
        finally:
            # Must run on the failure path too: a mid-file exception otherwise
            # leaves the allocation in place and the next burst runs out of memory.
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"Completed: {success}/{len(fits_files)} bursts reconstructed.")
