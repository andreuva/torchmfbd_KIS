#!/usr/bin/env python3
"""Apply Gaussian smoothing to a PNG or FITS image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


FITS_SUFFIXES = {".fit", ".fits", ".fts"}


def default_output_path(input_path: Path, sigma: float) -> Path:
    """Return an output filename next to the input image."""
    sigma_label = f"{sigma:g}".replace(".", "p")
    return input_path.with_name(f"{input_path.stem}_gaussian_sigma{sigma_label}{input_path.suffix}")


def build_output_path(input_path: Path, output_arg: Path | None, sigma: float) -> Path:
    """Resolve output path: either an explicit file or a directory containing a _smooth filename."""
    if output_arg is None:
        return default_output_path(input_path, sigma)

    if output_arg.suffix:
        return output_arg

    output_dir = output_arg
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_label = f"{sigma:g}".replace(".", "p")
    return output_dir / f"{input_path.stem}_smooth{input_path.suffix}"


def smooth_fits(input_path: Path, output_path: Path, sigma: float, hdu: int, overwrite: bool) -> None:
    from astropy.io import fits

    with fits.open(input_path) as hdul:
        data = hdul[hdu].data
        if data is None:
            raise ValueError(f"FITS HDU {hdu} enthält keine Bilddaten.")

        array = np.asarray(data, dtype=np.float32)
        spatial_sigma = (0.0,) * max(array.ndim - 2, 0) + (sigma, sigma)
        smoothed = gaussian_filter(array, sigma=spatial_sigma, mode="nearest")

        header = hdul[hdu].header.copy()
        fits.PrimaryHDU(data=smoothed, header=header).writeto(output_path, overwrite=overwrite)


def smooth_png(input_path: Path, output_path: Path, sigma: float, overwrite: bool) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("PNG-Dateien benötigen Pillow. Installiere es mit: pip install pillow") from exc

    with Image.open(input_path) as image:
        source_mode = image.mode
        array = np.asarray(image)
        channel_axis = -1 if array.ndim == 3 else None
        smoothed = gaussian_filter(array.astype(np.float32), sigma=(sigma, sigma, 0) if channel_axis else sigma, mode="nearest")

        if np.issubdtype(array.dtype, np.integer):
            info = np.iinfo(array.dtype)
            smoothed = np.clip(np.rint(smoothed), info.min, info.max).astype(array.dtype)
        else:
            smoothed = smoothed.astype(array.dtype)

        result = Image.fromarray(smoothed, mode=source_mode)
        result.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Glättet ein PNG- oder FITS-Bild mit einem Gaußfilter.")
    parser.add_argument("input", type=Path, help="Eingabedatei (.png, .fits, .fit oder .fts)")
    parser.add_argument("-o", "--output", type=Path, help="Ausgabedatei; standardmäßig neben der Eingabe")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gauß-Sigma in Pixeln (Standard: 2)")
    parser.add_argument("--hdu", type=int, default=0, help="FITS-HDU mit den Bilddaten (Standard: 0)")
    parser.add_argument("--overwrite", action="store_true", help="Eine vorhandene Ausgabedatei überschreiben")
    args = parser.parse_args()

    if args.sigma <= 0:
        parser.error("--sigma muss größer als 0 sein.")
    if not args.input.is_file():
        parser.error(f"Eingabedatei nicht gefunden: {args.input}")

    output_path = build_output_path(args.input, args.output, args.sigma)
    if output_path.exists() and not args.overwrite:
        parser.error(f"Ausgabedatei existiert bereits: {output_path} (verwende --overwrite)")

    if output_path.parent != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.input.suffix.lower() in FITS_SUFFIXES:
        smooth_fits(args.input, output_path, args.sigma, args.hdu, args.overwrite)
    elif args.input.suffix.lower() == ".png":
        smooth_png(args.input, output_path, args.sigma, args.overwrite)
    else:
        parser.error("Nur PNG und FITS werden unterstützt.")

    print(f"Gespeichert: {output_path}")


if __name__ == "__main__":
    main()