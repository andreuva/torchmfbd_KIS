import os
import glob
import argparse
import re
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

from plot_momfbd_result import plot_momfbd_results


DEFAULT_CONTRAST_FACTOR = 1.5 # 1.0 means use the measured reference range; <1.0 gives stronger contrast, >1.0 gives softer contrast.


def _load_channel(path, ext):
    with fits.open(path) as f:
        data = f[ext].data
    if data.ndim == 3:
        data = data[0]
    return data


def _filename_time_seconds(path):
    """Return seconds since midnight from a FITS filename timestamp, or None."""
    match = re.search(r'(?:^|_)(\d{6})(?:_|\.|$)', os.path.basename(path))
    if match is None:
        return None
    value = match.group(1)
    hh, mm, ss = int(value[:2]), int(value[2:4]), int(value[4:])
    return hh * 3600 + mm * 60 + ss


def _select_reference_files(result_files, reference_window=None):
    """Return a subset of files representing the reference-range frames.

    `reference_window` is a comma-separated range like "08:45:00,08:55:00".
    Files with timestamps inside the interval are used to set the shared
    display range; if no files match, the full sequence is used as a fallback.
    """
    if reference_window is None:
        return list(result_files)

    parts = [p.strip() for p in str(reference_window).split(',')]
    if len(parts) != 2:
        raise ValueError(
            "--reference_window must be formatted like '08:45:00,08:55:00'"
        )

    start = parts[0]
    end = parts[1]
    lower = _parse_hhmmss(start)
    upper = _parse_hhmmss(end)
    refs = []
    for path in result_files:
        t = _filename_time_seconds(path)
        if t is not None and lower <= t <= upper:
            refs.append(path)
    if refs:
        return refs
    return list(result_files)


def _parse_hhmmss(value):
    value = str(value).strip()
    if re.fullmatch(r'\d{6}', value):
        return int(value[:2]) * 3600 + int(value[2:4]) * 60 + int(value[4:])
    if re.fullmatch(r'\d{2}:\d{2}:\d{2}', value):
        h, m, s = [int(x) for x in value.split(':')]
        return h * 3600 + m * 60 + s
    raise ValueError(f"Invalid time value: {value!r}; expected HHMMSS or HH:MM:SS")


def measure_drift_and_brightness(result_files, upsample_factor=20, max_step_shift=15.0,
                                 reference_files=None):
    """
    Pass 1 over the sequence: for the WIDEBAND and NARROWBAND reconstructed
    channels independently, measure

      - a per-frame (dy, dx) shift, via phase cross-correlation against the
        *previous* frame and accumulated over the sequence, to counteract
        burst-to-burst pointing jitter ("movement" flicker); and
      - a per-frame median brightness level, later turned into a scale
        factor that matches every frame to the sequence's median level, to
        counteract independent per-burst reconstruction normalization
        jumps ("brightness" flicker).

    A registered shift step larger than `max_step_shift` pixels is treated
    as a registration failure (e.g. a genuine solar-evolution mismatch
    rather than instrumental jitter) and clamped to zero for that step, so
    a single bad frame can't throw off the whole cumulative chain.

    Returns dicts keyed by result file path.
    """
    wb_shifts, nb_shifts = {}, {}
    wb_levels, nb_levels = {}, {}

    prev_wb = prev_nb = None
    cum_wb = np.zeros(2)
    cum_nb = np.zeros(2)

    for path in tqdm(result_files, desc="Pass 1/2: measuring drift & brightness"):
        wb = _load_channel(path, 'WIDEBAND_RECONSTRUCTED')
        nb = _load_channel(path, 'NARROWBAND_RECONSTRUCTED')

        wb_levels[path] = float(np.median(wb))
        nb_levels[path] = float(np.median(nb))

        if prev_wb is not None:
            step_wb, _, _ = phase_cross_correlation(prev_wb, wb, upsample_factor=upsample_factor, normalization=None)
            step_nb, _, _ = phase_cross_correlation(prev_nb, nb, upsample_factor=upsample_factor, normalization=None)
            # The two cameras look through the same telescope at the same
            # instant, so the pointing jitter is common. Estimating and applying
            # it separately would slowly de-register the two channels from each
            # other, so the per-channel estimates are averaged and one common
            # shift is applied to both. Estimates that fail the sanity clamp are
            # dropped from the average rather than pulling it toward zero.
            valid = [s for s in (step_wb, step_nb) if np.hypot(*s) <= max_step_shift]
            step = np.mean(valid, axis=0) if valid else np.zeros(2)
            cum_wb = cum_wb + step
            cum_nb = cum_nb + step

        wb_shifts[path] = tuple(cum_wb)
        nb_shifts[path] = tuple(cum_nb)
        prev_wb, prev_nb = wb, nb

    brightness_reference = result_files if reference_files is None else reference_files
    ref_wb_level = float(np.median([wb_levels[p] for p in brightness_reference]))
    ref_nb_level = float(np.median([nb_levels[p] for p in brightness_reference]))
    wb_scales = {p: ref_wb_level / max(lvl, 1e-6) for p, lvl in wb_levels.items()}
    nb_scales = {p: ref_nb_level / max(lvl, 1e-6) for p, lvl in nb_levels.items()}

    return wb_shifts, nb_shifts, wb_scales, nb_scales


def measure_display_range(result_files, wb_scales, nb_scales, sample_size=40, low_pct=0.4, high_pct=99.9,
                          smooth=0.0, contrast_factor=DEFAULT_CONTRAST_FACTOR, reference_files=None):
    """
    Pick one fixed (vmin, vmax) per channel for the whole movie from either
    a representative reference subset or an evenly spaced sample of the full
    batch. The resulting range is then optionally scaled by `contrast_factor`:

      contrast_factor < 1.0  -> stronger contrast (narrower range)
      contrast_factor = 1.0 -> reference range as measured
      contrast_factor > 1.0  -> softer contrast (wider range)

    `smooth` has to match what the frames are plotted with: smoothing pulls in
    the tails of the histogram, so a range measured on unsmoothed data would
    stretch the movie slightly flatter than it should be.
    """
    if reference_files is None:
        idx = np.linspace(0, len(result_files) - 1, num=min(sample_size, len(result_files)), dtype=int)
        sample = [result_files[i] for i in np.unique(idx)]
    else:
        sample = list(reference_files)

    wb_values, nb_values = [], []
    for path in tqdm(sample, desc="Pass 1/2: measuring display range"):
        wb = _load_channel(path, 'WIDEBAND_RECONSTRUCTED') * wb_scales[path]
        nb = _load_channel(path, 'NARROWBAND_RECONSTRUCTED') * nb_scales[path]
        if smooth > 0:
            wb = gaussian_filter(wb, smooth)
            nb = gaussian_filter(nb, smooth)
        wb_values.append(wb.ravel())
        nb_values.append(nb.ravel())

    if contrast_factor <= 0:
        raise ValueError(f"contrast_factor must be positive, got {contrast_factor!r}")

    wb_lo, wb_hi = np.percentile(np.concatenate(wb_values), [low_pct, high_pct])
    nb_lo, nb_hi = np.percentile(np.concatenate(nb_values), [low_pct, high_pct])
    wb_center = (wb_lo + wb_hi) / 2.0
    wb_half_range = (wb_hi - wb_lo) * contrast_factor / 2.0
    nb_center = (nb_lo + nb_hi) / 2.0
    nb_half_range = (nb_hi - nb_lo) * contrast_factor / 2.0
    wb_vrange = (float(wb_center - wb_half_range), float(wb_center + wb_half_range))
    nb_vrange = (float(nb_center - nb_half_range), float(nb_center + nb_half_range))
    return wb_vrange, nb_vrange


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot MOMFBD reconstructed results for a batch of files.")
    parser.add_argument("--results_dir", type=str,
                        default="results_momfbd",
                        help="Directory containing batch MOMFBD reconstructed FITS files (output of hifi_momfbd_batch_gpu.py)")
    parser.add_argument("--pattern", type=str, default="*_momfbd.fits", help="File matching pattern for reconstructed FITS files")
    parser.add_argument("--raw_dir", type=str,
                        default="/dat/andreuva/data/hifiplus/level1/",
                        help="Directory containing the raw HiFI+ FITS observation files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for PNG figures (default: same as --results_dir)")
    parser.add_argument("--no_raw", action="store_true", help="Do not overlay raw frames, even if found")
    parser.add_argument("--limit", type=int, default=None, help="Limit total number of files to plot")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files (default: skip existing)")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian sigma in pixels applied to the reconstructed panels before "
                             "display (0 = off). Cosmetic only: the FITS is untouched and the raw "
                             "panels are left alone.")
    parser.add_argument("--unsharp_sigma", type=float, default=0.0,
                        help="Gaussian sigma in pixels for Unsharp Masking (0 = off; default: off).")
    parser.add_argument("--unsharp_amount", type=float, default=1.0,
                        help="Strength of Unsharp Masking high-frequency enhancement.")
    parser.add_argument("--reference_window", type=str, default=None,
                        help="Time window for reference frames used to define a shared display range, "
                             "formatted as 'HH:MM:SS,HH:MM:SS' (e.g. '08:45:00,08:55:00').")
    parser.add_argument("--contrast_factor", "--range_factor", dest="contrast_factor",
                        type=float, default=DEFAULT_CONTRAST_FACTOR,
                        help="Scale the reference display range around its center: <1 gives stronger contrast, "
                             ">1 gives softer contrast. This is applied after selecting the reference frames.")
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
    parser.add_argument("--no_stabilize", action="store_true",
                        help="Disable cross-correlation shift stabilization and brightness normalization across the sequence (original per-frame behavior)")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    search_path = os.path.join(args.results_dir, args.pattern)
    result_files = sorted(glob.glob(search_path))

    if not result_files:
        print(f"No reconstructed FITS files found matching pattern '{search_path}'.")
        exit(1)

    if args.limit is not None:
        result_files = result_files[:args.limit]

    print(f"Found {len(result_files)} reconstructed files to plot in {args.results_dir}.")

    reference_files = _select_reference_files(result_files, args.reference_window)
    if args.no_stabilize:
        wb_shifts = nb_shifts = {p: (0.0, 0.0) for p in result_files}
        wb_scales = nb_scales = {p: 1.0 for p in result_files}
    else:
        wb_shifts, nb_shifts, wb_scales, nb_scales = measure_drift_and_brightness(
            result_files, reference_files=reference_files
        )
    if args.reference_window is not None:
        print(f"Using {len(reference_files)} reference frames for display-range calibration between {args.reference_window}")
    wb_vrange, nb_vrange = measure_display_range(
        result_files,
        wb_scales,
        nb_scales,
        smooth=args.smooth,
        contrast_factor=args.contrast_factor,
        reference_files=reference_files,
    )
    print(f"Sequence display range: WB {wb_vrange}, NB {nb_vrange} (contrast_factor={args.contrast_factor})")

    success_count = 0
    if args.cmap is not None:
        color_suffix = f"_cmap-{args.cmap}"
    elif args.color is not None:
        safe_color = re.sub(r"[^A-Za-z0-9]+", "", args.color)
        color_suffix = f"_color-{safe_color}"
    else:
        color_suffix = ""

    for idx, result_path in enumerate(tqdm(result_files, desc="Pass 2/2: batch plotting")):
        base_name = os.path.basename(result_path)
        stem = base_name.replace('_momfbd.fits', '')
        output_png = os.path.join(output_dir, f"{stem}{color_suffix}_momfbd.png")

        if not args.overwrite and os.path.exists(output_png):
            tqdm.write(f"[{idx+1}/{len(result_files)}] Skipping existing figure: {os.path.basename(output_png)}")
            continue

        raw_fits_path = None
        if not args.no_raw:
            candidate = os.path.join(args.raw_dir, f"{stem}.fts")
            if os.path.exists(candidate):
                raw_fits_path = candidate

        tqdm.write(f"[{idx+1}/{len(result_files)}] Plotting {base_name} -> {os.path.basename(output_png)}")
        try:
            plot_momfbd_results(
                result_path, output_png=output_png, raw_fits_path=raw_fits_path,
                wb_shift=wb_shifts[result_path], nb_shift=nb_shifts[result_path],
                wb_scale=wb_scales[result_path], nb_scale=nb_scales[result_path],
                wb_vrange=wb_vrange, nb_vrange=nb_vrange, smooth=args.smooth,
                unsharp_sigma=args.unsharp_sigma, unsharp_amount=args.unsharp_amount,
                color=args.color, cmap=args.cmap, show_scale=not args.no_scale,
                narrowband_only=args.narrowband_only,
                narrowband_only_with_scale=args.narrowband_only_with_scale,
            )
            success_count += 1
        except Exception as e:
            tqdm.write(f"Error plotting {result_path}: {e}")

    print(f"Batch plotting completed: {success_count}/{len(result_files)} files plotted successfully.")
