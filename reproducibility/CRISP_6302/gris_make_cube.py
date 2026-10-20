"""
Stack the per-step MOMFBD results of one scan into a wavelength cube.

    python gris_make_cube.py --results_dir results_momfbd_v2/science1 [--gpu 0]

Writes results_momfbd_v2/science1_cube.fits (or --output; an existing file is only replaced with
--overwrite) with the extensions
    NB      (n_steps, ny, nx) narrow-band reconstructions in ADU, wavelength along axis 3
    BB      (n_steps, ny, nx) broad-band reconstructions
    WAVE    wavelength of every plane [Angstrom]
    SHIFTS  table per step: mean displacement applied (DY, DX: the plane is resampled at
            (y + DY, x + DX)), largest local displacement, and the per-step NBSCALE, BBSCALE, PFENV
A step missing from results_dir becomes a NaN plane, so that axis 3 stays linear in wavelength.
The cube is --border pixels (default 16) smaller on every side than the per-step mosaics: next to
the edge the warp has to invent the pixels it pulls in from outside the field (they are reflected
copies), so that border is dropped. Pixel (0, 0) of the cube is pixel (APODCROP + CUBEBORD,
APODCROP + CUBEBORD) of the crop that was reconstructed, up to the +-0.5 px rescale that
torchmfbd's warp applies (see README, known issues).

Every step is a different burst, reconstructed in the geometry its own seeing left behind:
between two steps the images differ by a global shift of about a pixel and by a local
distortion of several pixels, which would smear the line profile of every small feature.
The BB channel sees the same continuum at every wavelength, so the cube is destretched on BB
to the average geometry of the scan (torchmfbd.destretch, the routine the reconstruction
itself uses within a burst) and the identical displacement field is applied to NB, so the
registration does not change the relative alignment of the two channels. (That alignment is
the one of the cameras: a static ~0.37 deg rotation and ~0.19 % scale difference remain, see
README.) --no_register switches this off.

Wavelengths are air wavelengths from the step headers (WAVE), i.e. the data providers' nominal
3.15 pm per step; the line positions in these data indicate ~3.22 pm per step (README).
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

import gris_common as gris


def register_cubes(nb_cube, bb_cube, device, n_iterations=60, n_refine=3):
    """
    Destretch every present plane to the average geometry of the present planes. The
    displacement field is measured on BB and applied to both channels. Returns the two warped
    cubes, the mean displacement (dy, dx) of every plane and its largest local displacement, in
    pixels; NaN for missing planes.

    Three passes (n_refine=3: the second and third use the mean of the warped planes as the
    reference) with 60 iterations each. On the science2 v2 planes this roughly halves the
    residual of the worst-seeing planes against the scan mean compared with one 40-iteration
    pass (to ~0.3-0.9 px rms in 256 px tiles, depending on the estimator), while the median over
    all steps stays ~0.1 px.
    """
    # ngrid is a number of control points, not a spacing: keep the ~15 px spacing of the first
    # run (64 points over 984 px) whatever the size of the field.
    ngrid = max(16, int(round(bb_cube.shape[-1] / 15.4)))
    valid = np.isfinite(bb_cube).all(axis=(1, 2))

    bb_t = torch.tensor(bb_cube[valid], dtype=torch.float32, device=device)[None, None]   # (1, 1, n, ny, nx)
    warped_bb, tt = torchmfbd.destretch(bb_t, ngrid=ngrid, lr=0.50, reference_frame=0, border=6,
                                        n_iterations=n_iterations, lambda_tt=0.08, average_geometry=True,
                                        n_refine=n_refine)
    flow = rearrange(tt, 'b f d x y -> (b f) d x y')
    nb_t = torch.tensor(nb_cube[valid], dtype=torch.float32, device=device)[None, None]
    warped_nb = apply_destretch(nb_t, flow)

    nb_out = np.full(nb_cube.shape, np.nan, dtype=np.float32)
    bb_out = np.full(bb_cube.shape, np.nan, dtype=np.float32)
    bb_out[valid] = warped_bb[0, 0].cpu().numpy()
    nb_out[valid] = warped_nb[0, 0].cpu().numpy()

    # Statistics of the displacement field, excluding the border where the flow is only
    # weakly constrained by the data. Flow channel 0 is the x (column) displacement, 1 the y.
    f = flow.cpu().numpy()[:, :, 32:-32, 32:-32]
    mean_shift = np.full((len(valid), 2), np.nan, dtype=np.float32)
    max_shift = np.full(len(valid), np.nan, dtype=np.float32)
    mean_shift[valid] = f.mean(axis=(2, 3))[:, ::-1]          # (dy, dx)
    max_shift[valid] = np.hypot(f[:, 0], f[:, 1]).max(axis=(1, 2))
    return nb_out, bb_out, mean_shift, max_shift


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assemble per-step MOMFBD results into a wavelength cube.")
    parser.add_argument("--results_dir", required=True, help="Directory with the step*_momfbd.fits files")
    parser.add_argument("--output", default=None, help="Default: <results_dir>/../<scan>_cube.fits")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU for the destretch (-1: CPU, a few minutes)")
    parser.add_argument("--no_register", action="store_true", help="Keep every step in its own geometry")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file")
    parser.add_argument("--border", type=int, default=16,
                        help="Pixels dropped on every side after the destretch (warp edge effects)")
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
        print(f"Mean displacement up to {np.nanmax(np.abs(mean_shift)):.2f} px, local displacement up to {np.nanmax(max_shift):.2f} px")
        if np.nanmax(max_shift) > args.border - 4:
            print(f"WARNING: displacements reach {np.nanmax(max_shift):.1f} px; consider --border {int(np.nanmax(max_shift)) + 6}")
    b = args.border
    if b > 0:
        nb_cube, bb_cube = nb_cube[:, b:-b, b:-b], bb_cube[:, b:-b, b:-b]

    # Primary: the provenance of the first step, minus what is specific to one step.
    primary = fits.PrimaryHDU(header=headers[0])
    for key in ('WAVE', 'SCANSTEP', 'DATE-OBS', 'NBFILE', 'BBFILE', 'NBSCALE', 'BBSCALE', 'PFENV', 'FLATFILE'):
        primary.header.remove(key, ignore_missing=True)    # per-step values: in the SHIFTS table
    # Inherited from the raw NB header but wrong or NB-only for a cube that holds both channels:
    # WAVELNTH = 430 and TARGET/COMMENT 'tbd' are placeholders of the data providers.
    for key in ('WAVELNTH', 'TARGET', 'CAMERA', 'COMMENT'):
        while key in primary.header:
            primary.header.remove(key)
    if 'EXPTIME' in primary.header:
        primary.header.rename_keyword('EXPTIME', 'NBEXPT')
        primary.header['NBEXPT'] = (primary.header['NBEXPT'], 'NB exposure time [s]')
    try:
        bb_raw = os.path.join(gris.DATA_ROOT, 'BB', headers[0]['SCAN'], headers[0]['BBFILE'])
        primary.header['BBEXPT'] = (fits.getheader(bb_raw)['EXPTIME'], 'BB exposure time [s]')
    except Exception:
        pass
    primary.header['COMMENT'] = ("WAVE / axis 3: air wavelengths with the providers' nominal 3.15 pm/step; "
                                 "the Fe I and telluric O2 line positions indicate ~3.22 pm/step (README).")
    primary.header['NSTEPS'] = (n_planes, 'Planes along axis 3')
    primary.header['STEP0'] = (first, 'Scan step of plane 1')
    primary.header['REGISTER'] = (not args.no_register, 'Destretched to scan-average geometry on BB')
    primary.header['MAXDISP'] = (float(np.nanmax(max_shift)), 'Largest displacement, inner region [px]')
    primary.header['CUBEBORD'] = (b, 'Pixels dropped on every side of the mosaics')
    if not args.no_register:
        primary.header['COMMENT'] = ("torchmfbd's warp rescales the registered planes by W/(W-1) about "
                                     "the centre (+-0.5 px at the edges); see README, known issues.")

    def image_hdu(cube, name):
        hdu = fits.ImageHDU(data=cube, name=name)
        h = hdu.header
        pix = float(headers[0].get('PIXSCALE', 0.032))
        h['BUNIT'] = 'ADU'
        h['CTYPE1'], h['CUNIT1'], h['CRPIX1'], h['CRVAL1'], h['CDELT1'] = 'X', 'arcsec', 1, 0.0, pix
        h['CTYPE2'], h['CUNIT2'], h['CRPIX2'], h['CRVAL2'], h['CDELT2'] = 'Y', 'arcsec', 1, 0.0, pix
        h['CTYPE3'], h['CUNIT3'], h['CRPIX3'] = 'AWAV', 'Angstrom', 1     # air wavelengths
        h['CRVAL3'], h['CDELT3'] = float(wave_axis[0]), float(cdelt)
        return hdu

    per_step = {key: np.full(n_planes, np.nan) for key in ('NBSCALE', 'BBSCALE', 'PFENV')}
    for h, step in zip(headers, steps):
        for key in per_step:
            per_step[key][int(step) - first] = h.get(key, np.nan)
    shift_table = fits.BinTableHDU.from_columns([
        fits.Column(name='STEP', format='J', array=np.arange(first, last + 1)),
        fits.Column(name='WAVE', format='D', unit='Angstrom', array=wave_axis),
        fits.Column(name='DY', format='E', unit='px', array=mean_shift[:, 0]),   # plane resampled at y + DY
        fits.Column(name='DX', format='E', unit='px', array=mean_shift[:, 1]),   # and x + DX
        fits.Column(name='MAXDISP', format='E', unit='px', array=max_shift),
        fits.Column(name='NBSCALE', format='E', unit='ADU', array=per_step['NBSCALE']),
        fits.Column(name='BBSCALE', format='E', unit='ADU', array=per_step['BBSCALE']),
        fits.Column(name='PFENV', format='E', array=per_step['PFENV']),
    ], name='SHIFTS')

    scan = headers[0].get('SCAN', os.path.basename(os.path.normpath(args.results_dir)))
    output = args.output or os.path.join(os.path.dirname(os.path.normpath(args.results_dir)), f"{scan}_cube.fits")
    wave_hdu = fits.ImageHDU(data=wave_axis, name='WAVE')
    wave_hdu.header['BUNIT'] = ('Angstrom', 'air wavelength')
    if os.path.exists(output) and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite to replace it or choose another --output")
    fits.HDUList([primary, image_hdu(nb_cube, 'NB'), image_hdu(bb_cube, 'BB'),
                  wave_hdu, shift_table]).writeto(output, overwrite=args.overwrite)
    print(f"Cube written to {output}: NB/BB {nb_cube.shape}")
