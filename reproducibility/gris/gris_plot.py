"""
Quick-look figures of the per-step results: raw frame 0 next to the reconstruction, both channels.

    python gris_plot.py --results_dir results_momfbd/science1 [--output_dir plots/science1]

One PNG per step, with one display range per channel for the whole scan so that the line
darkening is visible from step to step. The movie through the scan then uses the generic HiFi
script unchanged:

    python ../hifi/make_hifi_movie.py --images_dir plots/science1 --pattern "step*_momfbd.png" \
        --fps 8 --output plots/science1_scan.mp4
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
from astropy.io import fits         # noqa: E402
from tqdm import tqdm               # noqa: E402

import gris_common as gris          # noqa: E402


def load_result(path):
    with fits.open(path) as f:
        return (f[0].header,
                f['WIDEBAND_RECONSTRUCTED'].data.astype(np.float32),
                f['NARROWBAND_RECONSTRUCTED'].data.astype(np.float32))


def raw_frames(header, shape, data_root):
    """Frame 0 of each channel, read exactly as the reconstruction read it, cut to the mosaic."""
    nb_path = os.path.join(data_root, 'NB', header['SCAN'], header['NBFILE'])
    bb_path = os.path.join(data_root, 'BB', header['SCAN'], header['BBFILE'])
    nb, bb, _, _ = gris.read_step(nb_path, bb_path, crop_size=header['CROPSIZE'],
                                  offset=(header['BBOFFY'], header['BBOFFX']), n_frames=1)
    # The mosaic starts at pixel (APODCROP, APODCROP) of the crop (unpatchify drops the
    # apodized border of every patch), so the raw frame is cut the same way to stay aligned.
    a = int(header['APODCROP'])
    ny, nx = shape
    return nb[0, 0, a:a + ny, a:a + nx], bb[0, 0, a:a + ny, a:a + nx]


def display_range(files, low=0.5, high=99.5):
    """One (vmin, vmax) per channel from the reconstructions of all steps."""
    bb_lo, bb_hi, nb_lo, nb_hi = [], [], [], []
    for path in tqdm(files, desc="Display range"):
        _, bb, nb = load_result(path)
        lo, hi = np.percentile(bb, [low, high]); bb_lo.append(lo); bb_hi.append(hi)
        lo, hi = np.percentile(nb, [low, high]); nb_lo.append(lo); nb_hi.append(hi)
    return (min(bb_lo), max(bb_hi)), (min(nb_lo), max(nb_hi))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick-look PNGs of per-step MOMFBD results.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_dir", default=None, help="Default: plots/<scan>")
    parser.add_argument("--data_root", default=gris.DATA_ROOT, help="Raw data, for the raw-frame panels")
    parser.add_argument("--no_raw", action="store_true", help="Reconstructions only")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, 'step*_momfbd.fits')))
    if not files:
        raise SystemExit(f"No step*_momfbd.fits in {args.results_dir}")
    if args.limit is not None:
        files = files[:args.limit]

    scan = fits.getheader(files[0]).get('SCAN', 'scan')
    output_dir = args.output_dir or os.path.join('plots', scan)
    os.makedirs(output_dir, exist_ok=True)

    bb_range, nb_range = display_range(files)
    print(f"Display range BB {np.round(bb_range)} ADU, NB {np.round(nb_range)} ADU")

    for path in tqdm(files, desc="Plotting"):
        png = os.path.join(output_dir, os.path.basename(path).replace('.fits', '.png'))
        if os.path.exists(png) and not args.overwrite:
            continue
        header, bb, nb = load_result(path)
        ny, nx = nb.shape
        pix = float(header.get('PIXSCALE', gris.PIX_SCALE))
        extent = (0, nx * pix, 0, ny * pix)
        panels = [('MOMFBD broad band', bb, bb_range), ('MOMFBD narrow band', nb, nb_range)]
        if not args.no_raw:
            raw_nb, raw_bb = raw_frames(header, nb.shape, args.data_root)
            panels = [('Raw broad band, frame 0', raw_bb, bb_range), panels[0],
                      ('Raw narrow band, frame 0', raw_nb, nb_range), panels[1]]

        n_rows = 2 if len(panels) == 4 else 1
        fig, axes = plt.subplots(n_rows, 2, figsize=(13, 6.3 * n_rows), squeeze=False)
        for ax, (title, image, vrange) in zip(axes.ravel(), panels):
            im = ax.imshow(image, cmap='gray', origin='lower', extent=extent, vmin=vrange[0], vmax=vrange[1])
            ax.set_title(title)
            ax.set_xlabel('x [arcsec]')
            ax.set_ylabel('y [arcsec]')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='ADU')
        fig.suptitle(f"{header['SCAN']}  step {int(header['SCANSTEP'])}  "
                     f"λ = {float(header['WAVE']):.4f} Å  ({header.get('DATE-OBS', '')})", fontsize=14)
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)

    print(f"PNGs in {output_dir}")
