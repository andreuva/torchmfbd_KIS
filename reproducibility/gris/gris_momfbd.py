"""
MOMFBD reconstruction of one CRISP 630 nm scan (SAC "gris" dataset), one wavelength step at a time.

    python gris_momfbd.py --scan science1 --gpu 0

For every wavelength step the 32 NB exposures and the 32 simultaneous BB exposures are
reconstructed together: two objects, one wavefront per frame. The pipeline is the validated
HiFi one, imported unchanged from ../hifi/hifi_common.py. This script only pairs the files,
reads the crops (gris_common.py), restores the ADU scale and names the outputs by scan step:

    <output_dir>/<scan>/step039_6301.5010A_momfbd.fits

Existing outputs are skipped, so a run can be resumed by starting it again.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml
from astropy.io import fits
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'hifi'))
import hifi_common as hifi      # noqa: E402
import gris_common as gris      # noqa: E402


def parse_steps(spec):
    """'1,39,100' or '30-50' or a mix of both -> set of SCANSTEP numbers; None -> all."""
    if spec is None:
        return None
    steps = set()
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            steps.update(range(int(a), int(b) + 1))
        else:
            steps.add(int(part))
    return steps


def output_name(header):
    return f"step{int(header['SCANSTEP']):03d}_{float(header['WAVE']):.4f}A_momfbd.fits"


def process_step(nb_path, bb_path, out_path, config_dict, args, device, offset):
    nb, bb, header, scales = gris.read_step(nb_path, bb_path, crop_size=args.crop_size,
                                            offset=offset, n_frames=args.n_frames)

    obj_bb, obj_nb, dec, meta = hifi.reconstruct_burst(
        nb, bb, config_dict, device,
        patch_size=args.patch_size, stride_size=args.stride_size,
        n_iterations=args.n_iterations, simultaneous_sequences=args.simultaneous_seq,
        destretch=not args.no_destretch, regime='on_disk',
        disk_modes=args.modes, limb_modes=args.modes, logger=tqdm.write)

    # The blending weights of unpatchify vanish on the outermost pixel of every patch, so the
    # outer pixel ring of the mosaic, covered by a single patch, comes out exactly zero.
    # Drop it; APODCROP (written from meta['apod']) then still gives the mosaic origin.
    obj_bb, obj_nb = obj_bb[1:-1, 1:-1], obj_nb[1:-1, 1:-1]
    meta['apod'] += 1

    # reconstruct_burst() normalises every frame by its 95th percentile, so its result is
    # near 1 at every step. Scaling it back keeps the line profile across the scan.
    obj_bb = obj_bb * scales['bb']
    obj_nb = obj_nb * scales['nb']

    y0, x0 = gris.crop_window(args.crop_size)
    extra = {
        'PIPELINE': ('gris_momfbd', 'Reconstruction script'),
        'SCAN': (args.scan, 'Scan (series) name'),
        'NBFILE': (os.path.basename(nb_path), 'Narrow-band input file'),
        'BBFILE': (os.path.basename(bb_path), 'Broad-band input file'),
        'CROPSIZE': (args.crop_size, 'Side of the square crop [px]'),
        'CROPY0': (y0, 'Row of the NB crop origin in the 2048 px frame'),
        'CROPX0': (x0, 'Column of the NB crop origin in the 2048 px frame'),
        'BBFLIPY': (True, 'BB flipped in y to the NB orientation'),
        'BBOFFY': (int(offset[0]), 'Row offset of the flipped BB window vs NB [px]'),
        'BBOFFX': (int(offset[1]), 'Column offset of the flipped BB window vs NB [px]'),
        'NBSCALE': (round(scales['nb'], 3), 'NB normalisation (mean per-frame p95) [ADU]'),
        'BBSCALE': (round(scales['bb'], 3), 'BB normalisation (mean per-frame p95) [ADU]'),
        'BUNIT': ('ADU', 'Reconstructions rescaled to the input units'),
        'PIXSCALE': (gris.PIX_SCALE, 'Pixel scale [arcsec/px]'),
        'APODCROP': (meta['apod'], 'Mosaic origin in the crop [px]: apodization+1'),
    }
    hifi.write_output(out_path, header, obj_bb, obj_nb, dec, meta,
                      config_name=os.path.basename(args.config), extra=extra)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MOMFBD of one CRISP 630 nm scan, step by step.")
    parser.add_argument("--scan", required=True, choices=['science1', 'science2'])
    parser.add_argument("--data_root", default=gris.DATA_ROOT)
    parser.add_argument("--output_dir", default="results_momfbd", help="Results go to <output_dir>/<scan>/")
    parser.add_argument("--config", default=os.path.join(HERE, "gris_momfbd.yaml"))
    parser.add_argument("--gpu", type=int, default=0, help="GPU index, -1 for CPU")
    parser.add_argument("--crop_size", type=int, default=1024,
                        help="Centred square crop; the field is a circle, anything up to ~1400 is fully lit")
    parser.add_argument("--offset", type=int, nargs=2, metavar=('DY', 'DX'), default=None,
                        help="Offset of the flipped BB window vs NB; measured on the first step if omitted")
    parser.add_argument("--n_frames", type=int, default=None, help="Frames per step (default: all 32)")
    parser.add_argument("--patch_size", type=int, default=128, help="Must equal images/n_pixel of the config")
    parser.add_argument("--stride_size", type=int, default=64)
    parser.add_argument("--n_iterations", type=int, default=250)
    parser.add_argument("--simultaneous_seq", type=int, default=225,
                        help="Patches optimised together; 225 is the whole 1024 px field in one batch")
    parser.add_argument("--modes", type=int, default=None,
                        help="Wavefront modes per patch; default psf/nmax_modes of the config")
    parser.add_argument("--no_destretch", action="store_true")
    parser.add_argument("--steps", type=str, default=None, help="Scan steps to do, e.g. '1,39,100' or '30-50'")
    parser.add_argument("--limit", type=int, default=None, help="Do at most this many steps")
    parser.add_argument("--no_resume", action="store_true", help="Redo steps whose output exists")
    args = parser.parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Running on {torch.cuda.get_device_name(device)} (cuda:{args.gpu})")
    else:
        device = torch.device("cpu")
        print("Running on CPU")

    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    config_dict['optimization']['gpu'] = args.gpu
    # The KL basis has to cover whole radial orders: nmax_modes + 1 must be a triangular number
    # (2, 5, 9, 14, 20, 27, 35, 44, 54, ...), otherwise torchmfbd stops on every step. To use
    # fewer modes per patch keep nmax_modes at a valid value and pass --modes.
    n_max = int(config_dict['psf']['nmax_modes'])
    order = (np.sqrt(8 * (n_max + 1) + 1) - 1) / 2
    if order != int(order):
        valid = [n * (n + 1) // 2 - 1 for n in range(2, 12)]
        raise SystemExit(f"psf/nmax_modes {n_max} is not a whole number of radial orders; use one of {valid}")
    if args.modes is None:
        args.modes = n_max
    if args.modes > config_dict['psf']['nmax_modes']:
        raise SystemExit(f"--modes {args.modes} exceeds psf/nmax_modes {config_dict['psf']['nmax_modes']}")
    if config_dict['images']['n_pixel'] != args.patch_size:
        raise SystemExit(f"--patch_size {args.patch_size} must equal images/n_pixel "
                         f"{config_dict['images']['n_pixel']} in {args.config}")

    out_dir = os.path.join(args.output_dir, args.scan)
    os.makedirs(out_dir, exist_ok=True)

    pairs = gris.list_steps(args.scan, args.data_root)
    if not pairs:
        raise SystemExit(f"No files for scan {args.scan} under {args.data_root}")

    wanted = parse_steps(args.steps)
    jobs = []
    for nb_path, bb_path in pairs:
        header = fits.getheader(nb_path)
        step = int(header['SCANSTEP'])
        if wanted is None or step in wanted:
            jobs.append((step, nb_path, bb_path, os.path.join(out_dir, output_name(header))))
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print(f"{args.scan}: {len(pairs)} steps on disk, {len(jobs)} selected, output in {out_dir}")

    # The camera offset is a property of the setup, not of the step: measure it once on the
    # first step of the scan (a continuum step) and use it everywhere.
    if args.offset is not None:
        offset = tuple(args.offset)
        print(f"Using the given BB offset (dy, dx) = {offset}")
    else:
        offset, residual, corr = gris.measure_offset(pairs[0][0], pairs[0][1])
        print(f"BB offset (dy, dx) = {offset} px measured on {os.path.basename(pairs[0][0])}: "
              f"residual {np.round(residual, 2)} px, correlation after alignment {corr:.3f}")
        if corr < 0.5:
            print("WARNING: the channels do not correlate after alignment; check the flip/offset")

    success = 0
    for idx, (step, nb_path, bb_path, out_path) in enumerate(tqdm(jobs, desc=f"MOMFBD {args.scan}")):
        name = os.path.basename(out_path)
        if not args.no_resume and os.path.exists(out_path):
            tqdm.write(f"[{idx + 1}/{len(jobs)}] step {step}: {name} exists, skipped")
            continue
        tqdm.write(f"[{idx + 1}/{len(jobs)}] step {step}: {os.path.basename(nb_path)} -> {name}")
        t0 = time.time()
        try:
            process_step(nb_path, bb_path, out_path, config_dict, args, device, offset)
            success += 1
            tqdm.write(f"    done in {time.time() - t0:.0f} s")
        except Exception as e:
            tqdm.write(f"    ERROR on step {step}: {e}")
        finally:
            # Also on failure, otherwise the next step runs out of GPU memory.
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    print(f"Completed: {success}/{len(jobs)} steps reconstructed.")
