

import argparse
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import gaussian_filter
import numpy as np


# Manual instrument calibration assumptions. These values are not read from
# the FITS data: update them here if the instrument calibration changes.
INSTRUMENT_PIXEL_SCALES_ARCSEC = {
    1: 0.027,
    2: 0.0498,
    3: 0.0498,
}


def _intensity_colormap(color=None, cmap=None):
    """Return a black-to-color map while preserving data-driven intensity."""
    if color is not None and cmap is not None:
        raise ValueError("Use either color or cmap, not both")
    if cmap is not None:
        try:
            return mpl.colormaps[cmap]
        except KeyError as exc:
            raise ValueError(f"Unknown Matplotlib colormap: {cmap}") from exc
    if color is None:
        return 'gray'
    return LinearSegmentedColormap.from_list(
        'momfbd_intensity',
        [(0.0, (0.0, 0.0, 0.0, 1.0)), (1.0, to_rgba(color))],
    )


def _instrument_pixel_scale_arcsec(header):
    """Return the manually supplied scale for HiFI No. 1, 2, or 3."""
    instrument = str(header.get('INSTRUME', ''))
    match = re.search(r'hifi\+?\s*no\.?\s*([123])\b', instrument, re.IGNORECASE)
    if match is None:
        return None, instrument
    instrument_number = int(match.group(1))
    return INSTRUMENT_PIXEL_SCALES_ARCSEC[instrument_number], instrument


def _observation_time_from_filename(fits_path):
    """Return HH:MM:SS from the six-digit observation time in the filename."""
    match = re.search(r'(?:^|_)(\d{6})(?:_|\.|$)', os.path.basename(fits_path))
    if match is None:
        return None
    value = match.group(1)
    return f'{value[:2]}:{value[2:4]}:{value[4:]}'


def _display_extent(image_shape, scale_arcsec):
    """Return an imshow extent in arcsec, or native pixels when disabled."""
    if scale_arcsec is None:
        return None
    height, width = image_shape[-2:]
    return (0.0, width * scale_arcsec, 0.0, height * scale_arcsec)


def _configure_angular_axis(axis, show_scale):
    if show_scale:
        axis.set_xlabel('Solar X [arcsec]')
        axis.set_ylabel('Solar Y [arcsec]')
        axis.tick_params(axis='both', which='both', direction='out')
        axis.grid(False)
    else:
        axis.axis('off')


def _apply_unsharp_mask(image, sigma, amount):
    if sigma <= 0 or amount == 0:
        return image
    return image + amount * (image - gaussian_filter(image, sigma))


def plot_momfbd_results(fits_path, output_png=None, raw_fits_path=None,
                         wb_shift=(0.0, 0.0), nb_shift=(0.0, 0.0),
                         wb_scale=1.0, nb_scale=1.0,
                         wb_vrange=None, nb_vrange=None, smooth=0.0,
                         unsharp_sigma=0.0, unsharp_amount=1.0,
                         color=None, cmap=None, show_scale=True,
                         narrowband_only=False, narrowband_only_with_scale=False):
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
        shows what the raw data looked like. The angular resolution impact
        depends on the instrument pixel scale and observing wavelength.
    unsharp_sigma : Gaussian sigma in pixels used for Unsharp Masking; zero disables it.
    unsharp_amount : Strength of the high-frequency component added by Unsharp Masking.
    color : optional matplotlib color specification, for example ``#e63946``.
        Reconstructed panels use a black-to-color intensity map; FITS values
        still determine intensity and vmin/vmax remains effective.
    cmap : optional Matplotlib colormap name, for example ``Reds``. This is
        useful for sequential maps where low intensities are light and high
        intensities become progressively red. Use either color or cmap.
    The observation time is read from the six-digit HHMMSS component of the
        FITS filename and shown in the title outside the image.
    show_scale : show angular axes in arcseconds around the reconstructed panels.
    narrowband_only : plot only the reconstructed narrow-band image, without
        axes, title, colorbar, or other annotations.
    narrowband_only_with_scale : plot only the reconstructed narrow-band image
        with angular axes and a title, but without an intensity colorbar.
    The angular scale is selected from the ``INSTRUME`` header using the
        manually maintained INSTRUMENT_PIXEL_SCALES_ARCSEC table above:
        HiFI 1 = 0.027 arcsec/pixel, HiFI 2/3 = 0.0497 arcsec/pixel.
        Unknown instruments are plotted without angular axes and produce a
        terminal warning rather than receiving an inaccurate scale.
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
    wb_data = _apply_unsharp_mask(wb_data, unsharp_sigma, unsharp_amount)
    nb_data = _apply_unsharp_mask(nb_data, unsharp_sigma, unsharp_amount)


    #smooth_label = f'\nGaussian smoothed, \u03C3 = {smooth:g} px' if smooth > 0 else ''
    #unsharp_label =  f'\nUnsharp Mask, \u03C3 = {unsharp_sigma:g}, amount = {unsharp_amount:g}' if unsharp_sigma > 0 and unsharp_amount != 0 else ''

    wb_vmin, wb_vmax = wb_vrange if wb_vrange is not None else (None, None)
    nb_vmin, nb_vmax = nb_vrange if nb_vrange is not None else (None, None)
    reconstructed_cmap = _intensity_colormap(color, cmap)
    header = f[0].header
    observation_time = _observation_time_from_filename(fits_path)
    scale_arcsec = None
    if show_scale:
        scale_arcsec, instrument = _instrument_pixel_scale_arcsec(header)
        if scale_arcsec is None:
            print(
                f"Warning: unknown instrument {instrument!r} in {os.path.basename(fits_path)}; "
                "angular scale omitted because it would not be accurate."
            )
    wb_extent = _display_extent(wb_data.shape, scale_arcsec)
    nb_extent = _display_extent(nb_data.shape, scale_arcsec)
    time_label = f'\nt = {observation_time}' if observation_time is not None else ''

    if narrowband_only or narrowband_only_with_scale:
        height, width = nb_data.shape[-2:]
        if narrowband_only:
            fig, axis = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        else:
            fig, axis = plt.subplots(figsize=(8, 7))
        axis.imshow(nb_data, cmap=reconstructed_cmap, origin='lower',
                    extent=nb_extent, vmin=nb_vmin, vmax=nb_vmax)
        if narrowband_only:
            axis.axis('off')
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        else:
            axis.set_title('MOMFBD Reconstructed Narrow-band H-alpha Lyot (656.3 nm)' + time_label)  # + smooth_label + unsharp_label''' + time_label)
            _configure_angular_axis(axis, show_scale)
            fig.tight_layout()
        if output_png is not None:
            save_kwargs = {'dpi': 300}
            if narrowband_only:
                save_kwargs['pad_inches'] = 0
            else:
                save_kwargs['bbox_inches'] = 'tight'
            fig.savefig(output_png, **save_kwargs)
            print(f"Saved visualization figure to {output_png}")
        plt.close(fig)
        f.close()
        return

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

        im0 = axes[0, 0].imshow(raw_wb, cmap='gray', origin='lower', extent=wb_extent)
        axes[0, 0].set_title('Raw Broad-band Frame (Camera 2)' + time_label)
        _configure_angular_axis(axes[0, 0], show_scale)
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(wb_data, cmap=reconstructed_cmap, origin='lower', extent=wb_extent, vmin=wb_vmin, vmax=wb_vmax)
        axes[0, 1].set_title('MOMFBD Reconstructed Broad-band (656.7 nm)' '''+ smooth_label + unsharp_label''' + time_label)
        _configure_angular_axis(axes[0, 1], show_scale)
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(raw_nb, cmap='gray', origin='lower', extent=nb_extent)
        axes[1, 0].set_title('Raw Narrow-band Frame (Camera 1 / H-alpha Lyot)' + time_label)
        _configure_angular_axis(axes[1, 0], show_scale)
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        im3 = axes[1, 1].imshow(nb_data, cmap=reconstructed_cmap, origin='lower', extent=nb_extent, vmin=nb_vmin, vmax=nb_vmax)
        axes[1, 1].set_title('MOMFBD Reconstructed Narrow-band H-alpha Lyot (656.3 nm)' '''+ smooth_label + unsharp_label''' + time_label)
        _configure_angular_axis(axes[1, 1], show_scale)
        fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        im0 = axes[0].imshow(wb_data, cmap=reconstructed_cmap, origin='lower', extent=wb_extent, vmin=wb_vmin, vmax=wb_vmax)
        axes[0].set_title('MOMFBD Reconstructed Broad-band (656.7 nm)' '''+ smooth_label + unsharp_label''' + time_label)
        _configure_angular_axis(axes[0], show_scale)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(nb_data, cmap=reconstructed_cmap, origin='lower', extent=nb_extent, vmin=nb_vmin, vmax=nb_vmax)
        axes[1].set_title('MOMFBD Reconstructed Narrow-band H-alpha Lyot (656.3 nm)' '''+ smooth_label + unsharp_label''' + time_label)
        _configure_angular_axis(axes[1], show_scale)
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
    parser.add_argument("--unsharp_sigma", type=float, default=0.0,
                        help="Gaussian sigma in pixels for Unsharp Masking (0 = off; default: off).")
    parser.add_argument("--unsharp_amount", type=float, default=1.0,
                        help="Strength of Unsharp Masking high-frequency enhancement.")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--color", type=str, default=None,
                             help="Color for reconstructed panels, e.g. '#e63946'. "
                                  "Intensity remains data-driven.")
    color_group.add_argument("--cmap", type=str, default=None,
                             help="Matplotlib colormap for reconstructed panels, "
                                  "e.g. 'Reds' or 'Reds_r'. Default is grayscale.")
    parser.add_argument("--no_scale", action="store_true",
                        help="Disable the default angular scale bar.")
    parser.add_argument("--narrowband_only", action="store_true",
                        help="Plot only reconstructed narrow-band as an annotation-free PNG.")
    parser.add_argument("--narrowband_only_with_scale", action="store_true",
                        help="Plot only reconstructed narrow-band with angular axes and title, without an intensity colorbar.")
    args = parser.parse_args()

    plot_momfbd_results(args.fits, output_png=args.output_png, raw_fits_path=args.raw_fits,
                        smooth=args.smooth, unsharp_sigma=args.unsharp_sigma,
                        unsharp_amount=args.unsharp_amount, color=args.color, cmap=args.cmap,
                        show_scale=not args.no_scale, narrowband_only=args.narrowband_only,
                        narrowband_only_with_scale=args.narrowband_only_with_scale)
