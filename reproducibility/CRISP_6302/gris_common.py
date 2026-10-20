"""
Reading of the CRISP 630 nm scans of 2026-10-06 (the SAC "gris" dataset) for the HiFi MOMFBD pipeline.

The reconstruction is not here: gris_momfbd.py imports reconstruct_burst() and write_output()
from ../hifi/hifi_common.py unchanged. This module only knows the data layout:

  * NB/<scan>/corr_sd*.fits and BB/<scan>/corr_sd*.fits: one file per wavelength step, the same
    file name in both directories, each a (32, 2048, 2048) float32 cube of 32 simultaneous
    exposures (NB 9.5 ms, BB 0.95 ms, 80 Hz), dark/flat corrected, in ADU.
  * The two cameras are mirrored in y with respect to each other, and after the flip a constant
    offset of a few pixels remains (about 12 px in x). Measured on the data: the correlation of
    the frame averages is 0.98 with the flip and 0.12 without, and the per-frame image motion of
    the two cameras agrees frame by frame once flipped, so the exposures are simultaneous.
  * The field is a circle of radius ~1093 px centred within 5 px of the image centre, so any
    centred square of up to ~1400 px is fully illuminated. The corners are exactly zero.

BB is flipped to the NB orientation, not the other way round, so that the reconstructed NB
cube keeps the orientation of the level-1 NB products (NB/science*_mean.fits).
"""

import glob
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation

DATA_ROOT = '/dat/andreuva/data/SAC_gris_data'
IMAGE_SIZE = 2048
PIX_SCALE = 0.032      # arcsec per pixel, both cameras


def list_steps(scan, data_root=DATA_ROOT):
    """Sorted (nb_path, bb_path) pairs of one scan; sorted file name order is wavelength order."""
    pairs = []
    for nb in sorted(glob.glob(os.path.join(data_root, 'NB', scan, 'corr_sd*.fits'))):
        bb = os.path.join(data_root, 'BB', scan, os.path.basename(nb))
        if os.path.exists(bb):
            pairs.append((nb, bb))
        else:
            print(f"Warning: {os.path.basename(nb)} has no BB counterpart and is skipped")
    return pairs


def crop_window(crop_size, offset=(0, 0)):
    """Origin (y0, x0) of a centred square window, displaced by offset=(dy, dx)."""
    y0 = IMAGE_SIZE // 2 - crop_size // 2 + int(offset[0])
    x0 = IMAGE_SIZE // 2 - crop_size // 2 + int(offset[1])
    if not (0 <= y0 <= IMAGE_SIZE - crop_size and 0 <= x0 <= IMAGE_SIZE - crop_size):
        raise ValueError(f"A {crop_size} px crop with offset {offset} leaves the {IMAGE_SIZE} px frame")
    return y0, x0


def read_crop(path, crop_size, offset=(0, 0), n_frames=None, flip_y=False):
    """
    Centred crop of one cube as a native, C-contiguous float32 array (n_frames, crop, crop).

    The copy is not optional: astropy returns the data big-endian and a flipped view has
    negative strides, and torch.tensor() refuses both.

    With flip_y the window is taken from the vertically flipped image img[::-1]: its rows
    [y0, y0 + crop) are the stored rows [N - y0 - crop, N - y0), read backwards.
    """
    y0, x0 = crop_window(crop_size, offset)
    with fits.open(path, memmap=True) as f:
        n_avail = f[0].header['NAXIS3']
        n = n_avail if n_frames is None else min(n_frames, n_avail)
        if flip_y:
            rows = slice(IMAGE_SIZE - y0 - crop_size, IMAGE_SIZE - y0)
        else:
            rows = slice(y0, y0 + crop_size)
        data = f[0].data[:n, rows, x0:x0 + crop_size]
        if flip_y:
            data = data[:, ::-1, :]
        return np.ascontiguousarray(data, dtype=np.float32)


def _highpass(image, sigma=12.0):
    """Remove the large-scale intensity pattern so the cross-correlation sees the granulation."""
    return image - gaussian_filter(image, sigma)


def measure_offset(nb_path, bb_path, crop_size=768, n_frames=8):
    """
    Integer offset (dy, dx) of the flipped BB window relative to the NB window, such that
    read_crop(bb, offset=(dy, dx), flip_y=True) shows the same field as read_crop(nb).

    Phase cross-correlation of the high-pass filtered frame averages of the central crop_size
    pixels, checked afterwards by re-reading BB with the offset applied. Use a continuum step:
    at the line core the NB and BB granulation patterns correlate much less.

    Returns the offset, the residual sub-pixel shift left after the integer offset, and the
    correlation of the two filtered images after alignment (about 0.95 when the flip is right,
    about 0.1 when it is not).
    """
    nb = _highpass(read_crop(nb_path, crop_size, n_frames=n_frames).mean(0))
    bb = _highpass(read_crop(bb_path, crop_size, n_frames=n_frames, flip_y=True).mean(0))
    # phase_cross_correlation returns the shift s that registers the moving image on the
    # reference, moving[y - s_y, x - s_x] = reference[y, x]; the BB window has to move the
    # other way.
    shift, _, _ = phase_cross_correlation(nb, bb, upsample_factor=10, normalization=None)
    offset = (-int(np.round(shift[0])), -int(np.round(shift[1])))

    bb = _highpass(read_crop(bb_path, crop_size, offset=offset, n_frames=n_frames, flip_y=True).mean(0))
    residual, _, _ = phase_cross_correlation(nb, bb, upsample_factor=10, normalization=None)
    m = 32
    corr = np.corrcoef(nb[m:-m, m:-m].ravel(), bb[m:-m, m:-m].ravel())[0, 1]
    return offset, residual, float(corr)


def read_step(nb_path, bb_path, crop_size=1024, offset=(0, 0), n_frames=None):
    """
    One wavelength step in the form the HiFi pipeline takes.

    Returns
    -------
    nb_frames, bb_frames : float32 (1, n_frames, crop_size, crop_size)
        NB in its native orientation; BB flipped in y and displaced by `offset`, so that the two
        channels are co-registered to the pixel.
    header : the NB primary header (SCANSTEP, WAVE, DATE-OBS, EXPTIME, ...).
    scales : dict, mean over frames of the per-frame 95th percentile of each channel, in ADU.
        reconstruct_burst() divides every frame by this quantity, so multiplying its result by
        it puts the reconstruction back into ADU, which keeps the line profile across the scan.
    """
    nb = read_crop(nb_path, crop_size, n_frames=n_frames)
    bb = read_crop(bb_path, crop_size, offset=offset, n_frames=n_frames, flip_y=True)
    header = fits.getheader(nb_path)
    scales = {'nb': float(np.percentile(nb.reshape(nb.shape[0], -1), 95, axis=1).mean()),
              'bb': float(np.percentile(bb.reshape(bb.shape[0], -1), 95, axis=1).mean())}
    return nb[None], bb[None], header, scales
