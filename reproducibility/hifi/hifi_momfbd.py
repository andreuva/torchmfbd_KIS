"""
MOMFBD reconstruction of a single GREGOR / HiFI+ H-alpha burst.

    python hifi_momfbd.py --fits <burst.fts> --gpu 0 --output out.fits

The pipeline lives in hifi_common.py, which hifi_momfbd_batch_gpu.py shares.
See hifi_momfbd.yaml for why the cutoffs and the number of modes are what they
are.
"""

import argparse
import numpy as np
import torch
import yaml
import hifi_common as hifi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MOMFBD on one HiFI+ level 1 burst.")
    parser.add_argument("--fits", type=str,
                        default="/dat/andreuva/data/hifiplus/level1/20260714/hifiplus2_20260714_080658_sd.fts",
                        help="Path to the HiFI+ FITS burst")
    parser.add_argument("--config", type=str, default="hifi_momfbd.yaml", help="Configuration YAML")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index, -1 for CPU")
    parser.add_argument("--n_frames", type=int, default=100, help="Frames per camera")
    parser.add_argument("--patch_size", type=int, default=96, help="Patch size for the sub-field deconvolution")
    parser.add_argument("--stride_size", type=int, default=32,
                        help="Stride between patches. 32 with apodization_border 20 leaves 24 px of "
                             "overlap, which is what keeps the limb artifact from surviving the mosaic")
    parser.add_argument("--crop_size", type=int, default=None, help="Crop the field to this size (for quick tests)")
    parser.add_argument("--no_destretch", action="store_true", help="Skip destretching")
    parser.add_argument("--regime", choices=['auto', 'on_disk', 'off_limb'], default='auto',
                        help="'auto' decides on-disk vs off-limb per patch from the wideband brightness. "
                             "The forced modes exist for fields that do not cross the limb, where the "
                             "automatic classification has nothing to separate.")
    parser.add_argument("--disk_modes", type=int, default=44,
                        help="Wavefront modes for on-disk patches; must not exceed psf/nmax_modes")
    parser.add_argument("--limb_modes", type=int, default=20,
                        help="Wavefront modes for off-limb patches. Fewer modes is what stops the "
                             "wavefront from being fitted to noise where there is no signal")
    parser.add_argument("--n_iterations", type=int, default=250, help="Optimization iterations")
    parser.add_argument("--simultaneous_seq", type=int, default=200, help="Patches optimized together")
    parser.add_argument("--output", type=str, default="hifi_momfbd_result.fits", help="Output FITS path")
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Running on {torch.cuda.get_device_name(device)} (cuda:{args.gpu})")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    print(f"Reading {args.fits}...")
    nb_frames, wb_frames, header, mfgs = hifi.read_hifi_dataset(
        args.fits, n_frames=args.n_frames, crop_size=args.crop_size)

    with open(args.config, 'r') as fh:
        config_dict = yaml.safe_load(fh)
    config_dict['optimization']['gpu'] = args.gpu

    obj_wb, obj_nb, dec, meta = hifi.reconstruct_burst(
        nb_frames, wb_frames, config_dict, device,
        patch_size=args.patch_size, stride_size=args.stride_size,
        n_iterations=args.n_iterations, simultaneous_sequences=args.simultaneous_seq,
        destretch=not args.no_destretch, regime=args.regime,
        disk_modes=args.disk_modes, limb_modes=args.limb_modes)

    extra = {}
    if np.isfinite(mfgs).any():
        extra['MFGSMEAN'] = (round(float(np.nanmean(mfgs)), 5), 'Mean MFGS of frames used')

    hifi.write_output(args.output, header, obj_wb, obj_nb, dec, meta,
                      config_name=args.config, extra=extra)
    print(f"Done. Output written to {args.output}")
