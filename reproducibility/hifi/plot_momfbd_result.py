import argparse
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import gaussian_filter
import numpy as np

def plot_momfbd_results(fits_path, output_png=None, raw_fits_path=None,
                         wb_shift=(0.0, 0.0), nb_shift=(0.0, 0.0),
                         wb_scale=1.0, nb_scale=1.0,
                         wb_vrange=None, nb_vrange=None, smooth=0.0):
    """
    wb_shift / nb_shift : (dy, dx) sub-pixel translation applied to the
        reconstructed frame before display, to compensate for burst-to-burst
        pointing jitter (e.g. from cross-correlation registration across a
        movie sequence).
    wb_scale / nb_scale : multiplicative brightness correction applied to
        the reconstructed frame before display, to compensate for
        burst-to-burst normalization jumps.
    wb_vrange / nb_vrange : optional (vmin, vmax) fixed display range; when
        None, matplotlib auto-scales to this frame's own data (the original
        single-frame behavior).
    smooth : Gaussian sigma in pixels applied to the reconstructed frames before
        display, to take the edge off whatever noise the reconstruction still
        carries. This is cosmetic: it is applied to the displayed copy only, not
        to the FITS, and the raw panels are left alone so the comparison still
        shows what the raw data looked like. At 0.0462 arcsec/pixel a sigma of 2
        is a FWHM of 0.22 arcsec, about 2.3 times the diffraction limit at
        656 nm, so it does cost real resolution.
    """
    f = fits.open(fits_path)

    wb_data = f['WIDEBAND_RECONSTRUCTED'].data
    nb_data = f['NARROWBAND_RECONSTRUCTED'].data

    if wb_data.ndim == 3:
        wb_data = wb_data[0]
    if nb_data.ndim == 3:
        nb_data = nb_data[0]

    if wb_shift != (0.0, 0.0):
        wb_data = ndi_shift(wb_data, wb_shift, order=3, mode='nearest')
    if nb_shift != (0.0, 0.0):
        nb_data = ndi_shift(nb_data, nb_shift, order=3, mode='nearest')
    if wb_scale != 1.0:
        wb_data = wb_data * wb_scale
    if nb_scale != 1.0:
        nb_data = nb_data * nb_scale

    if smooth > 0:
        wb_data = gaussian_filter(wb_data, smooth)
        nb_data = gaussian_filter(nb_data, smooth)

    smooth_label = f'\nGaussian smoothed, sigma = {smooth:g} px' if smooth > 0 else ''

    wb_vmin, wb_vmax = wb_vrange if wb_vrange is not None else (None, None)
    nb_vmin, nb_vmax = nb_vrange if nb_vrange is not None else (None, None)

    # Create figure comparing raw vs deconvolved
    if raw_fits_path is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        f_raw = fits.open(raw_fits_path)
        # The mosaic starts at pixel (off, off) of the raw frame, because
        # unpatchify crops that many pixels from every patch edge. Cropping the
        # raw frame from (0, 0) instead would leave the two panels misaligned.
        off = f[0].header.get('APODCROP', 6)
        raw_nb = f_raw[1].data[off:off + nb_data.shape[0], off:off + nb_data.shape[1]]
        raw_wb = f_raw[2].data[off:off + wb_data.shape[0], off:off + wb_data.shape[1]]
        f_raw.close()

        im0 = axes[0, 0].imshow(raw_wb, cmap='gray', origin='lower')
        axes[0, 0].set_title('Raw Broad-band Frame (Camera 2)')
        axes[0, 0].axis('off')
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(wb_data, cmap='gray', origin='lower', vmin=wb_vmin, vmax=wb_vmax)
        axes[0, 1].set_title('MOMFBD Reconstructed Broad-band (656.7 nm)' + smooth_label)
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(raw_nb, cmap='gray', origin='lower')
        axes[1, 0].set_title('Raw Narrow-band Frame (Camera 1 / H-alpha Lyot)')
        axes[1, 0].axis('off')
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(nb_data, cmap='gray', origin='lower', vmin=nb_vmin, vmax=nb_vmax)
        axes[1, 1].set_title('MOMFBD Reconstructed Narrow-band H-alpha Lyot (656.3 nm)' + smooth_label)
        axes[1, 1].axis('off')
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        im0 = axes[0].imshow(wb_data, cmap='gray', origin='lower', vmin=wb_vmin, vmax=wb_vmax)
        axes[0].set_title('MOMFBD Reconstructed Broad-band (656.7 nm)' + smooth_label)
        axes[0].axis('off')
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(nb_data, cmap='gray', origin='lower', vmin=nb_vmin, vmax=nb_vmax)
        axes[1].set_title('MOMFBD Reconstructed Narrow-band H-alpha Lyot (656.3 nm)' + smooth_label)
        axes[1].axis('off')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if output_png is not None:
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Saved visualization figure to {output_png}")
    
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot MOMFBD reconstructed results.")
    parser.add_argument("--fits", type=str, default="hifi_momfbd_result.fits", help="Path to MOMFBD result FITS file")
    parser.add_argument("--raw_fits", type=str, default="/dat/andreuva/data/hifiplus/level1/20260714/hifiplus2_20260714_080658_sd.fts", help="Path to raw FITS dataset file")
    parser.add_argument("--output_png", type=str, default="momfbd_reconstructed.png", help="PNG output path")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian sigma in pixels applied to the reconstructed panels before "
                             "display (0 = off). Cosmetic only: the FITS is untouched and the raw "
                             "panels are left alone.")
    args = parser.parse_args()

    plot_momfbd_results(args.fits, output_png=args.output_png, raw_fits_path=args.raw_fits,
                        smooth=args.smooth)
