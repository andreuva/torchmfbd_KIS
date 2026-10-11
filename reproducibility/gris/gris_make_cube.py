"""
Stack the per-step MOMFBD results of one scan into a wavelength cube.

    python gris_make_cube.py --results_dir results_momfbd/science1 [--gpu 0]

Writes results_momfbd/science1_cube.fits (or --output) with the extensions
    NB      (n_steps, ny, nx) narrow-band reconstructions in ADU, wavelength along axis 3
    BB      (n_steps, ny, nx) broad-band reconstructions
    WAVE    wavelength of every plane [Angstrom]
    SHIFTS  table with the mean and the largest displacement applied to every step
A step missing from results_dir becomes a NaN plane, so that axis 3 stays linear in wavelength.

Every step is a different burst, reconstructed in the geometry its own seeing left behind:
between two steps the images differ by a global shift of about a pixel and by a local
distortion of several pixels, which would smear the line profile of every small feature.
The BB channel sees the same continuum at every wavelength, so the cube is destretched on BB
to the average geometry of the scan (torchmfbd.destretch, the routine the reconstruction
itself uses within a burst) and the identical displacement field is applied to NB, which
keeps the two channels co-registered. --no_register switches this off.
"""

import argparse
import glob
import os

import numpy as np
import torch
from astropy.io import fits
from einops import rearrange
from tqdm import tqdm

import torchmfbd
from torchmfbd.destretch import apply_destretch


def register_cubes(nb_cube, bb_cube, device, ngrid=64, n_iterations=40):
    """
    Destretch every plane to the average geometry of the scan. The displacement field is
    measured on BB and applied to both channels. Returns the two warped cubes, the mean
    displacement (dy, dx) of every plane and its largest local displacement, in pixels.
    """
    valid = np.isfinite(bb_cube).all(axis=(1, 2))
    ref = int(np.where(valid)[0][0])
    # Missing planes borrow the reference image so that the flow is defined everywhere;
    # they are set back to NaN afterwards.
    bb_in = np.where(valid[:, None, None], bb_cube, bb_cube[ref])
    nb_in = np.where(valid[:, None, None], nb_cube, nb_cube[ref])

    bb_t = torch.tensor(bb_in, dtype=torch.float32, device=device)[None, None]   # (1, 1, n, ny, nx)
    warped_bb, tt = torchmfbd.destretch(bb_t, ngrid=ngrid, lr=0.50, reference_frame=ref, border=6,
                                        n_iterations=n_iterations, lambda_tt=0.08, average_geometry=True)
    flow = rearrange(tt, 'b f d x y -> (b f) d x y')
    nb_t = torch.tensor(nb_in, dtype=torch.float32, device=device)[None, None]
    warped_nb = apply_destretch(nb_t, flow)

    bb_out = warped_bb[0, 0].cpu().numpy().astype(np.float32)
    nb_out = warped_nb[0, 0].cpu().numpy().astype(np.float32)
    bb_out[~valid] = np.nan
    nb_out[~valid] = np.nan

    # Statistics of the displacement field, excluding the border where the flow is only
    # weakly constrained by the data.
    f = flow.cpu().numpy()[:, :, 32:-32, 32:-32]
    mean_shift = f.mean(axis=(2, 3))
    max_shift = np.hypot(f[:, 0], f[:, 1]).max(axis=(1, 2))
    return nb_out, bb_out, mean_shift, max_shift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assemble per-step MOMFBD results into a wavelength cube.")
    parser.add_argument("--results_dir", required=True, help="Directory with the step*_momfbd.fits files")
    parser.add_argument("--output", default=None, help="Default: <results_dir>/../<scan>_cube.fits")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU for the destretch (-1: CPU, a few minutes)")
    parser.add_argument("--no_register", action="store_true", help="Keep every step in its own geometry")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, 'step*_momfbd.fits')))
    if not files:
        raise SystemExit(f"No step*_momfbd.fits in {args.results_dir}")
    headers = [fits.getheader(f) for f in files]
    steps = np.array([int(h['SCANSTEP']) for h in headers])
    waves = np.array([float(h['WAVE']) for h in headers])
    order = np.argsort(steps)
    files = [files[i] for i in order]
    headers = [headers[i] for i in order]
    steps, waves = steps[order], waves[order]

    first, last = int(steps[0]), int(steps[-1])
    n_planes = last - first + 1
    missing = sorted(set(range(first, last + 1)) - set(steps.tolist()))
    if missing:
        print(f"WARNING: {len(missing)} missing steps become NaN planes: {missing}")
    # Linear wavelength axis fitted through the steps present (3.15 pm per step nominally).
    if len(steps) > 1:
        cdelt = float(np.polyfit(steps, waves, 1)[0])
    else:
        cdelt = float(headers[0].get('STEPPM', 3.15)) / 100.0
    wave_axis = waves[0] + cdelt * np.arange(n_planes)
    print(f"{len(files)} steps {first}..{last}, {wave_axis[0]:.4f}..{wave_axis[-1]:.4f} A, {cdelt * 1e3:.3f} mA/step")

    with fits.open(files[0]) as f:
        ny, nx = f['NARROWBAND_RECONSTRUCTED'].data.shape
    nb_cube = np.full((n_planes, ny, nx), np.nan, dtype=np.float32)
    bb_cube = np.full((n_planes, ny, nx), np.nan, dtype=np.float32)
    for path, step in tqdm(list(zip(files, steps)), desc="Reading"):
        with fits.open(path) as f:
            nb_cube[int(step) - first] = f['NARROWBAND_RECONSTRUCTED'].data
            bb_cube[int(step) - first] = f['WIDEBAND_RECONSTRUCTED'].data

    mean_shift = np.zeros((n_planes, 2), dtype=np.float32)
    max_shift = np.zeros(n_planes, dtype=np.float32)
    if not args.no_register:
        device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
        print(f"Destretching the {n_planes} planes to the average geometry of the scan on {device}...")
        nb_cube, bb_cube, mean_shift, max_shift = register_cubes(nb_cube, bb_cube, device)
        print(f"Mean displacement up to {np.abs(mean_shift).max():.2f} px, local displacement up to {max_shift.max():.2f} px")

    # Primary: the provenance of the first step, minus what is specific to one step.
    primary = fits.PrimaryHDU(header=headers[0])
    for key in ('WAVE', 'SCANSTEP', 'DATE-OBS', 'NBFILE', 'BBFILE'):
        primary.header.remove(key, ignore_missing=True)
    primary.header['NSTEPS'] = (n_planes, 'Planes along axis 3')
    primary.header['STEP0'] = (first, 'Scan step of plane 1')
    primary.header['REGISTER'] = (not args.no_register, 'Destretched to scan-average geometry on BB')
    primary.header['MAXDISP'] = (float(max_shift.max()), 'Largest displacement, inner region [px]')

    def image_hdu(cube, name):
        hdu = fits.ImageHDU(data=cube, name=name)
        h = hdu.header
        pix = float(headers[0].get('PIXSCALE', 0.032))
        h['BUNIT'] = 'ADU'
        h['CTYPE1'], h['CUNIT1'], h['CRPIX1'], h['CRVAL1'], h['CDELT1'] = 'X', 'arcsec', 1, 0.0, pix
        h['CTYPE2'], h['CUNIT2'], h['CRPIX2'], h['CRVAL2'], h['CDELT2'] = 'Y', 'arcsec', 1, 0.0, pix
        h['CTYPE3'], h['CUNIT3'], h['CRPIX3'] = 'WAVE', 'Angstrom', 1
        h['CRVAL3'], h['CDELT3'] = float(wave_axis[0]), float(cdelt)
        return hdu

    shift_table = fits.BinTableHDU.from_columns([
        fits.Column(name='STEP', format='J', array=np.arange(first, last + 1)),
        fits.Column(name='WAVE', format='D', unit='Angstrom', array=wave_axis),
        fits.Column(name='DY', format='E', unit='px', array=mean_shift[:, 0]),
        fits.Column(name='DX', format='E', unit='px', array=mean_shift[:, 1]),
        fits.Column(name='MAXDISP', format='E', unit='px', array=max_shift),
    ], name='SHIFTS')

    scan = headers[0].get('SCAN', os.path.basename(os.path.normpath(args.results_dir)))
    output = args.output or os.path.join(os.path.dirname(os.path.normpath(args.results_dir)), f"{scan}_cube.fits")
    fits.HDUList([primary, image_hdu(nb_cube, 'NB'), image_hdu(bb_cube, 'BB'),
                  fits.ImageHDU(data=wave_axis, name='WAVE'), shift_table]).writeto(output, overwrite=True)
    print(f"Cube written to {output}: NB/BB {nb_cube.shape}")
