"""
Figures of a reconstructed scan, one PNG per wavelength step, and optionally a movie of them.

    python gris_plot.py --results_dir results_momfbd_v2/science1 --output_dir plots_steps/science1
    python gris_plot.py --cube results_momfbd_v2/science1_cube.fits --bb_mean 5 \
        --output_dir plots_cube_v2/science1 --movie plots_cube_v2/science1_scan.mp4

From the per-step files (--results_dir) every figure shows raw frame 0 next to the
reconstruction, for both channels. Every step keeps the geometry of its own burst, so a movie
made from these figures wobbles from step to step.

From the cube (--cube, written by gris_make_cube.py) the figures show the two reconstructions
only, in the common geometry of the scan, which is what a movie through the scan should use.
The raw frames are left out because they are not in that geometry.

Full resolution: every image pixel is drawn as exactly one screen pixel (no resampling), so the
figures are as large as the images (3104 x 1488 px for the 1326 px v2 cube). The PNGs are 8-bit
grey. The movie (--movie) is lossless by default; --crf 1 makes a near-lossless copy that
standard players can open (see write_movie).

--bb_mean N (cube only) shows BB as the mean of the N planes centred on each step. BB sees the
same continuum scene at every step (the whole scan lasts 45 s), so its step-to-step changes are
reconstruction noise that differs from burst to burst; in the aligned movie that noise is no
longer hidden by image motion. NB is always shown per step (averaging it would mix wavelengths).

One display range per channel is used for the whole scan so that the line darkening is visible
from step to step.
"""

import argparse
import glob
import itertools
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402
from astropy.io import fits         # noqa: E402
from PIL import Image               # noqa: E402
from tqdm import tqdm               # noqa: E402

import gris_common as gris          # noqa: E402

# Frame layout in screen pixels (100 dpi, so 1 pt = 100/72 px)
DPI = 100
LEFT = 90        # y tick labels and label, left of each panel
RIGHT = 130      # colour bar and its labels, right of each panel
BELOW = 70       # x tick labels and label, below each panel
ABOVE = 40       # panel title
TOP = 50         # figure title


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


def steps_from_files(files, data_root=None):
    """(info, bb, nb, raw) per step from the per-step result files; raw only with data_root."""
    for path in files:
        header, bb, nb = load_result(path)
        info = dict(scan=header['SCAN'], step=int(header['SCANSTEP']), wave=float(header['WAVE']),
                    date=header.get('DATE-OBS', ''), pix=float(header.get('PIXSCALE', gris.PIX_SCALE)),
                    png=os.path.basename(path).replace('.fits', '.png'))
        raw = raw_frames(header, nb.shape, data_root) if data_root else None
        yield info, bb, nb, raw


def steps_from_cube(path, bb_mean=1):
    """(info, bb, nb, None) per plane of a cube written by gris_make_cube.py; NaN planes skipped.

    With bb_mean = N, BB is the mean of the valid planes among the N centred on the step (fewer
    at the ends of the scan)."""
    half = bb_mean // 2
    with fits.open(path) as f:
        h = f[0].header
        nb_cube, bb_cube, wave = f['NB'].data, f['BB'].data, f['WAVE'].data
        n = nb_cube.shape[0]
        valid = [bool(np.isfinite(bb_cube[k]).all()) for k in range(n)]   # a missing step is NaN
        for k in range(n):
            if not valid[k]:
                continue
            window = [j for j in range(max(0, k - half), min(n, k + half + 1)) if valid[j]]
            bb = np.mean([bb_cube[j].astype(np.float32) for j in window], axis=0)
            nb = nb_cube[k].astype(np.float32)
            step0 = int(h['STEP0'])
            step = step0 + k
            bb_title = 'MOMFBD broad band'
            if bb_mean > 1:
                bb_title += f', mean of {len(window)} steps ({step0 + window[0]}-{step0 + window[-1]})'
            info = dict(scan=h.get('SCAN', ''), step=step, wave=float(wave[k]), date='',
                        pix=float(h.get('PIXSCALE', gris.PIX_SCALE)), bb_title=bb_title,
                        png=f"step{step:03d}_{float(wave[k]):.4f}A_cube.png")
            yield info, bb, nb, None


def display_range(frames, low=0.5, high=99.5, border=8):
    """One (vmin, vmax) per channel from the reconstructions of all steps, away from the edges."""
    bb_lo, bb_hi, nb_lo, nb_hi = [], [], [], []
    for _, bb, nb, _ in tqdm(frames, desc="Display range"):
        bb, nb = bb[border:-border, border:-border], nb[border:-border, border:-border]
        lo, hi = np.percentile(bb, [low, high]); bb_lo.append(lo); bb_hi.append(hi)
        lo, hi = np.percentile(nb, [low, high]); nb_lo.append(lo); nb_hi.append(hi)
    return (min(bb_lo), max(bb_hi)), (min(nb_lo), max(nb_hi))


def plot_step(info, bb, nb, raw, bb_range, nb_range, png):
    ny, nx = nb.shape
    pix = info['pix']
    panels = [(info.get('bb_title', 'MOMFBD broad band'), bb, bb_range),
              ('MOMFBD narrow band', nb, nb_range)]
    if raw is not None:
        raw_nb, raw_bb = raw
        panels = [('Raw broad band, frame 0', raw_bb, bb_range), panels[0],
                  ('Raw narrow band, frame 0', raw_nb, nb_range), panels[1]]
    n_rows = len(panels) // 2

    # The figure is laid out in pixels. Its size is rounded up to multiples of 16 so that the
    # movie encoder (macro blocks of 16 px) never rescales it.
    width = 2 * (LEFT + nx + RIGHT)
    height = TOP + n_rows * (ABOVE + ny + BELOW)
    width, height = -(-width // 16) * 16, -(-height // 16) * 16
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    for i, (title, image, vrange) in enumerate(panels):
        row, col = divmod(i, 2)
        x0 = LEFT + col * (LEFT + nx + RIGHT)
        y0 = height - TOP - (row + 1) * (ABOVE + ny + BELOW) + BELOW
        # 8-bit grey levels computed here: matplotlib's 'gray' colour map rounds down and merges
        # 24 of the 256 levels. figimage then puts them on the canvas pixel for pixel, with no
        # resampling.
        grey = np.clip(np.round((image - vrange[0]) / (vrange[1] - vrange[0]) * 255), 0, 255).astype(np.uint8)
        fig.figimage(np.dstack([grey] * 3), xo=x0, yo=y0, origin='lower')
        # Axes only for the frame, ticks and labels, 2 px outside the image so that the frame
        # line does not cover image pixels
        ax = fig.add_axes([(x0 - 2) / width, (y0 - 2) / height, (nx + 4) / width, (ny + 4) / height])
        ax.patch.set_alpha(0)
        ax.set_xlim(-2 * pix, (nx + 2) * pix)
        ax.set_ylim(-2 * pix, (ny + 2) * pix)
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('x [arcsec]', fontsize=13)
        ax.set_ylabel('y [arcsec]', fontsize=13)
        ax.tick_params(labelsize=12)
        cax = fig.add_axes([(x0 + nx + 14) / width, y0 / height, 24 / width, ny / height])
        mappable = plt.cm.ScalarMappable(plt.Normalize(*vrange), cmap='gray')
        fig.colorbar(mappable, cax=cax).set_label('ADU', fontsize=13)
        cax.tick_params(labelsize=12)
    date = f"  ({info['date']})" if info['date'] else ''
    fig.suptitle(f"{info['scan']}  step {info['step']}  λ = {info['wave']:.4f} Å{date}", fontsize=18,
                 y=1 - 12 / height, va='top')
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    if rgba.shape[:2] != (height, width):
        raise RuntimeError(f"Canvas is {rgba.shape[1]} x {rgba.shape[0]} px, expected {width} x {height}")
    Image.fromarray(rgba).convert('L').save(png)    # everything is grey, so 8-bit grey loses nothing


def write_movie(pngs, output, fps, crf):
    """H.264 movie of the PNGs, unscaled, in full-range grey levels.

    In an aligned, still movie the step-to-step changes are only ~1 grey level, so ordinary
    settings (the HiFi movie script uses crf 18) change the fine texture from frame to frame as
    much as the data do. crf 0 is lossless: the movie has exactly the pixels of the PNGs, but
    uses the High 4:4:4 Predictive profile, which VLC, mpv and ffplay play and browsers do not.
    crf 1 gives a standard (High profile) file within ~0.5 grey level rms of the PNGs, which any
    player opens for the cube figures (3104 x 1488 px, level 5.0); the two-row per-step figures
    with raw panels (~3200 x 3000 px) need level 6, which many players lack. Full range
    (yuvj420p) avoids the +-1 rounding of converting 0-255 grey to video levels 16-235."""
    import imageio
    params = ['-crf', str(crf), '-preset', 'slow', '-tune', 'grain', '-bf', '0', '-g', str(2 * fps),
              '-color_range', 'pc']
    with imageio.get_writer(output, fps=fps, codec='libx264', pixelformat='yuvj420p',
                            macro_block_size=16, output_params=params) as writer:
        for png in tqdm(pngs, desc="Movie"):
            writer.append_data(np.asarray(Image.open(png).convert('L')))
    print(f"Movie {output}: {len(pngs)} frames at {fps} fps")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Quick-look PNGs of a reconstructed scan.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results_dir", help="Per-step results (step*_momfbd.fits): raw vs reconstruction")
    source.add_argument("--cube", help="Cube from gris_make_cube.py: reconstructions in the scan geometry")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_root", default=gris.DATA_ROOT, help="Raw data, for the raw-frame panels")
    parser.add_argument("--no_raw", action="store_true", help="Per-step mode: reconstructions only")
    parser.add_argument("--bb_mean", type=int, default=1,
                        help="Cube only: show BB as the mean of this many steps centred on each step (odd)")
    parser.add_argument("--movie", default=None, help="Also write an H.264 movie of the figures to this file")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--crf", type=int, default=0,
                        help="Movie quality, 0-51: 0 = lossless (default), 1 = near-lossless, standard profile")
    parser.add_argument("--limit", type=int, default=None, help="Plot only the first N steps (N >= 1)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.bb_mean < 1 or args.bb_mean % 2 == 0:
        parser.error("--bb_mean must be odd and at least 1")
    if args.bb_mean > 1 and not args.cube:
        parser.error("--bb_mean needs --cube: the per-step files are not in a common geometry")
    if not 0 <= args.crf <= 51:
        parser.error("--crf must be between 0 and 51")
    if args.movie and os.path.exists(args.movie) and not args.overwrite:
        parser.error(f"{args.movie} exists (use --overwrite)")
    if args.movie:
        os.makedirs(os.path.dirname(os.path.abspath(args.movie)), exist_ok=True)

    if args.cube:
        def frames(with_raw=False):
            return itertools.islice(steps_from_cube(args.cube, args.bb_mean), args.limit)
    else:
        files = sorted(glob.glob(os.path.join(args.results_dir, 'step*_momfbd.fits')))[:args.limit]
        if not files:
            raise SystemExit(f"No step*_momfbd.fits in {args.results_dir}")

        def frames(with_raw=False):
            return steps_from_files(files, args.data_root if with_raw and not args.no_raw else None)

    os.makedirs(args.output_dir, exist_ok=True)
    bb_range, nb_range = display_range(frames())
    print(f"Display range BB {np.round(bb_range)} ADU, NB {np.round(nb_range)} ADU")

    pngs = []
    for info, bb, nb, raw in tqdm(frames(with_raw=True), desc="Plotting"):
        png = os.path.join(args.output_dir, info['png'])
        pngs.append(png)
        if os.path.exists(png) and not args.overwrite:
            continue
        plot_step(info, bb, nb, raw, bb_range, nb_range, png)
    print(f"PNGs in {args.output_dir}")

    if args.movie:
        write_movie(pngs, args.movie, args.fps, args.crf)
