"""
Split-burst test to choose the Fourier cutoffs, the number of wavefront modes and the noise model.

    python gris_tune.py --gpu 0 --shard 0/2 --configs m44_c85_white,m44_c85_auto
    python gris_tune.py --summary

Every (scan, step) is reconstructed three times per configuration: with all 32 frames, with the
16 even frames and with the 16 odd frames. The two halves see the same Sun through independent
noise, so comparing their reconstructions ring by ring in Fourier space separates what the
reconstruction puts in the image into a reproducible part (signal) and the rest (noise). This is
the test the HiFi settings were chosen with (see ../hifi/hifi_momfbd.yaml).

A configuration name is  m<modes>_<cutoffs>_<noise model>[_f<NN>][_nods], e.g. m30_all75_auto_f85:
    modes      number of wavefront modes per patch
    cutoffs    c85   BB and NB [0.70, 0.85]  (the first full run)
               nb75  NB [0.60, 0.75], BB [0.70, 0.85];   nb65  NB [0.50, 0.65], BB [0.70, 0.85]
               all75 BB and NB [0.60, 0.75];              all65 BB and NB [0.50, 0.65]
    noise      white or auto: filter_noise_shape of both objects
    f<NN>      filter_noise_flat = NN/100 (auto only): the measured noise shape is held flat below
               this frequency (default 0.25)
    nods       no destretching inside the burst (diagnostic)

Outputs in --output_dir, per (scan, step, configuration):
    <scan>_step<NNN>_<config>.fits   full, even and odd reconstructions of BB and NB (ADU)
    <scan>_step<NNN>_<config>.json   ring spectra and band summaries of the split-burst test
--summary prints the comparison table and writes figures to <output_dir>/summary/.
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from astropy.io import fits
from scipy.ndimage import fourier_shift, gaussian_filter
from skimage.registration import phase_cross_correlation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'hifi'))
import hifi_common as hifi      # noqa: E402
import gris_common as gris      # noqa: E402

CUTS = {   # (BB, NB) cutoff pairs in units of the diffraction cutoff
    'c85': ((0.70, 0.85), (0.70, 0.85)),
    'nb75': ((0.70, 0.85), (0.60, 0.75)),
    'nb65': ((0.70, 0.85), (0.50, 0.65)),
    'all75': ((0.60, 0.75), (0.60, 0.75)),
    'all65': ((0.50, 0.65), (0.50, 0.65)),
}
DEFAULT_STEPS = 'science2:36,science2:40,science2:71,science2:5,science1:40'
STAGE1 = ('m44_c85_white,m44_c85_auto,m44_c85_white_nods,m44_nb65_white,'
          'm44_all65_white,m44_all75_white,m30_c85_white,m20_c85_white')

# Diffraction cutoff D / lambda in cycles per pixel (0.354; Nyquist is 0.5, i.e. 1.41 of it)
CUTOFF_CYC_PX = 1.44 / 6302e-10 / 206265.0 * gris.PIX_SCALE
RHO_EDGES = np.arange(0.0, 1.425, 0.025)
BANDS = {'low': (0.10, 0.30), 'mid': (0.30, 0.50), 'high': (0.50, 0.70), 'top': (0.70, 0.85)}


def zoom(image):
    """8" x 8" of plage and granulation, placed relative to the mosaic centre."""
    cy, cx = np.array(image.shape) // 2
    return image[cy + 68:cy + 324, cx + 96:cx + 352]


def highpass(image):
    return image - gaussian_filter(image, 12)


def parse_config(name):
    parts = name.split('_')
    bb_cut, nb_cut = CUTS[parts[1]]
    flat = [p for p in parts[3:] if p.startswith('f')]
    if (parts[2] not in ('white', 'auto') or not parts[0].startswith('m')
            or any(p != 'nods' and not p.startswith('f') for p in parts[3:])):
        raise ValueError(f"Bad configuration name {name}")
    return dict(name=name, modes=int(parts[0][1:]), bb_cut=bb_cut, nb_cut=nb_cut, noise=parts[2],
                noise_flat=int(flat[0][1:]) / 100 if flat else None, destretch='nods' not in parts[3:])


def reconstruct(nb, bb, cfg, base_config, device, args):
    c = copy.deepcopy(base_config)
    c['object1']['cutoff'] = list(cfg['bb_cut'])
    c['object2']['cutoff'] = list(cfg['nb_cut'])
    for obj in ('object1', 'object2'):
        c[obj]['filter_noise_shape'] = cfg['noise']
        # Always explicit, so that the name means the same whatever the base --config sets
        c[obj]['filter_noise_flat'] = cfg['noise_flat'] if cfg['noise_flat'] is not None else 0.25
    t0 = time.time()
    obj_bb, obj_nb, dec, _ = hifi.reconstruct_burst(
        np.ascontiguousarray(nb), np.ascontiguousarray(bb), c, device,
        patch_size=args.patch_size, stride_size=args.stride_size, n_iterations=args.n_iterations,
        simultaneous_sequences=args.simultaneous_seq, destretch=cfg['destretch'], regime='on_disk',
        disk_modes=cfg['modes'], limb_modes=cfg['modes'], logger=print)
    del dec
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return obj_bb, obj_nb, time.time() - t0


def ring_spectra(a, b):
    """
    Ring sums of the cross power and of the two auto powers of the contrast images a/<a> - 1 and
    b/<b> - 1, Hann windowed, in units of contrast variance (summing a band gives the variance
    of the image in that band). Rings are in units of the diffraction cutoff.
    """
    ny, nx = a.shape
    win = np.outer(np.hanning(ny), np.hanning(nx))
    fa = np.fft.fft2((a / a.mean() - 1.0) * win)
    fb = np.fft.fft2((b / b.mean() - 1.0) * win)
    ky, kx = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx), indexing='ij')
    idx = np.digitize(np.hypot(ky, kx).ravel() / CUTOFF_CYC_PX, RHO_EDGES) - 1
    ok = (idx >= 0) & (idx < len(RHO_EDGES) - 1)
    norm = a.size * (win ** 2).sum()

    def ring(x):
        return np.bincount(idx[ok], x.ravel()[ok], minlength=len(RHO_EDGES) - 1) / norm

    return ring((fa * fb.conj()).real), ring(np.abs(fa) ** 2), ring(np.abs(fb) ** 2)


def split_burst_metrics(even, odd):
    """FRC curve, ring powers and band summaries of one channel from its two half reconstructions."""
    cross, pa, pb = ring_spectra(even, odd)
    rho = 0.5 * (RHO_EDGES[1:] + RHO_EDGES[:-1])
    frc = cross / np.sqrt(pa * pb + 1e-30)
    out = dict(frc=frc.tolist(), p_signal=cross.tolist(), p_total=(0.5 * (pa + pb)).tolist())
    smooth = np.convolve(frc, np.ones(3) / 3, mode='same')
    for thr, key in ((0.5, 'rho_frc05'), (0.143, 'rho_frc0143')):
        below = np.where((rho > 0.05) & (smooth < thr))[0]
        out[key] = float(rho[below[0]]) if len(below) else float(rho[-1])
    for band, (lo, hi) in BANDS.items():
        sel = (rho >= lo) & (rho < hi)
        s, tot = cross[sel].sum(), 0.5 * (pa[sel] + pb[sel]).sum()
        out[band] = dict(frc=float(s / np.sqrt(pa[sel].sum() * pb[sel].sum() + 1e-30)),
                         signal_rms=float(np.sqrt(max(s, 0.0))),
                         noise_rms=float(np.sqrt(max(tot - s, 0.0))))
    return out


def seeing_contrast(bb):
    """Mean small-scale rms contrast of the raw BB frames: a seeing indicator for the burst."""
    return float(np.mean([np.std(f - gaussian_filter(f, 6)) / f.mean() for f in bb[0, ::4]]))


def find_step(scan, step):
    for nb_path, bb_path in gris.list_steps(scan):
        if int(fits.getheader(nb_path)['SCANSTEP']) == step:
            return nb_path, bb_path
    raise SystemExit(f"Step {step} not found in {scan}")


def run(args):
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    base_config['optimization']['gpu'] = args.gpu
    os.makedirs(args.output_dir, exist_ok=True)
    n_max = base_config['psf']['nmax_modes']
    for name in args.configs.split(','):
        if parse_config(name)['modes'] > n_max:
            raise SystemExit(f"{name} asks for more modes than psf/nmax_modes = {n_max} in {args.config}")

    steps = [(s.split(':')[0], int(s.split(':')[1])) for s in args.steps.split(',')]
    combos = [(scan, step, name) for scan, step in steps for name in args.configs.split(',')]
    shard, n_shards = (int(x) for x in args.shard.split('/'))
    combos = combos[shard::n_shards]
    print(f"{len(combos)} (step, configuration) combinations on {device}")

    offsets, cache = {}, {}
    for i, (scan, step, name) in enumerate(combos):
        stem = os.path.join(args.output_dir, f"{scan}_step{step:03d}_{name}")
        if os.path.exists(stem + '.json') and not args.no_resume:
            done = json.load(open(stem + '.json'))
            if (done['crop_size'], done['n_iterations']) != (args.crop_size, args.n_iterations):
                raise SystemExit(f"{stem}.json was made with another crop size or iteration count; "
                                 f"use another --output_dir or --no_resume")
            print(f"[{i + 1}/{len(combos)}] {os.path.basename(stem)} exists, skipped")
            continue
        cfg = parse_config(name)
        if scan not in offsets:
            offsets[scan] = gris.measure_offset(*gris.list_steps(scan)[0])[0]
        if (scan, step) not in cache:
            cache = {(scan, step): gris.read_step(*find_step(scan, step), crop_size=args.crop_size,
                                                  offset=offsets[scan])}
        nb, bb, header, scales = cache[(scan, step)]
        print(f"[{i + 1}/{len(combos)}] {scan} step {step} {name}")

        full_bb, full_nb, t_full = reconstruct(nb, bb, cfg, base_config, device, args)
        ev_bb, ev_nb, t_ev = reconstruct(nb[:, 0::2], bb[:, 0::2], cfg, base_config, device, args)
        od_bb, od_nb, t_od = reconstruct(nb[:, 1::2], bb[:, 1::2], cfg, base_config, device, args)

        # The two halves are destretched to the average geometry of interleaved frames, so
        # they differ by a small global shift at most; measure it per channel and apply it to
        # that channel. Measured on the interior: the outer pixel ring of every mosaic is exactly
        # zero and that edge would pin the result to zero.
        inner = (slice(8, -8), slice(8, -8))
        shifts = {}
        for ch, ev, od in (('bb', ev_bb, od_bb), ('nb', ev_nb, od_nb)):
            shifts[ch] = phase_cross_correlation(highpass(ev[inner]), highpass(od[inner]),
                                                 upsample_factor=20, normalization=None)[0]
        od_bb = np.fft.ifft2(fourier_shift(np.fft.fft2(od_bb), shifts['bb'])).real.astype(np.float32)
        od_nb = np.fft.ifft2(fourier_shift(np.fft.fft2(od_nb), shifts['nb'])).real.astype(np.float32)

        result = dict(scan=scan, step=step, wave=float(header['WAVE']), config=cfg,
                      crop_size=args.crop_size, n_iterations=args.n_iterations,
                      base_config=os.path.basename(args.config), seeing_bb_contrast=seeing_contrast(bb),
                      shift_odd_vs_even={ch: [float(x) for x in s] for ch, s in shifts.items()},
                      seconds=dict(full=t_full, even=t_ev, odd=t_od),
                      rho=(0.5 * (RHO_EDGES[1:] + RHO_EDGES[:-1])).tolist(),
                      bb=split_burst_metrics(ev_bb, od_bb), nb=split_burst_metrics(ev_nb, od_nb))

        h = fits.Header()
        for key in ('SCANSTEP', 'WAVE', 'DATE-OBS'):
            h[key] = header[key]
        h['SCAN'], h['CONFIG'], h['CROPSIZE'] = scan, name, args.crop_size
        fits.HDUList([fits.PrimaryHDU(header=h)] + [
            fits.ImageHDU(data=(img * scales[ch]).astype(np.float32), name=nm) for nm, img, ch in (
                ('BB', full_bb, 'bb'), ('NB', full_nb, 'nb'), ('BB_EVEN', ev_bb, 'bb'),
                ('NB_EVEN', ev_nb, 'nb'), ('BB_ODD', od_bb, 'bb'), ('NB_ODD', od_nb, 'nb'))
        ]).writeto(stem + '.fits', overwrite=True)
        # The JSON marks the combination as done, so it is written last and in one step.
        with open(stem + '.json.tmp', 'w') as f:
            json.dump(result, f)
        os.replace(stem + '.json.tmp', stem + '.json')
        print(f"    {t_full + t_ev + t_od:.0f} s; NB FRC=0.5 at {result['nb']['rho_frc05']:.2f}, "
              f"BB at {result['bb']['rho_frc05']:.2f} of the diffraction cutoff")


def summary(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.output_dir, '*.json')))]
    if not results:
        raise SystemExit(f"No results in {args.output_dir}")
    out_dir = os.path.join(args.output_dir, 'summary')
    os.makedirs(out_dir, exist_ok=True)
    lines = ["Split-burst test. FRC05: frequency where the half-burst FRC drops below 0.5 (units of the",
             "diffraction cutoff). S/N per band: rms contrast of the reproducible and of the",
             "non-reproducible part of a half-burst reconstruction, in percent.", ""]
    for key in sorted({(r['scan'], r['step'], r['crop_size']) for r in results}):
        group = sorted([r for r in results if (r['scan'], r['step'], r['crop_size']) == key],
                       key=lambda r: r['config']['name'])
        r0 = group[0]
        name = f"{key[0]}_step{key[1]:03d}_crop{key[2]}"
        lines.append(f"{key[0]} step {key[1]} ({r0['wave']:.3f} A), crop {key[2]} px, raw BB small-scale "
                     f"contrast {100 * r0['seeing_bb_contrast']:.2f} %")
        lines.append(f"  {'config':22s} | {'ch':2s} {'FRC05':>5s} | " +
                     " | ".join(f"{b:>4s} S/N" for b in BANDS) + " | seconds")
        for r in group:
            for ch in ('bb', 'nb'):
                m = r[ch]
                bands = " | ".join(f"{100 * m[b]['signal_rms']:4.2f}/{100 * m[b]['noise_rms']:4.2f}" for b in BANDS)
                sec = f"{sum(r['seconds'].values()):.0f}" if ch == 'bb' else ''
                lines.append(f"  {r['config']['name'] if ch == 'bb' else '':22s} | {ch} {m['rho_frc05']:5.2f} | {bands} | {sec}")
        lines.append("")

        # FRC curves and zoomed reconstructions of every configuration for this step
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        for r in group:
            for ax, ch in zip(axes, ('bb', 'nb')):
                ax.plot(r['rho'], r[ch]['frc'], label=r['config']['name'])
        for ax, ch in zip(axes, ('BB', 'NB')):
            ax.axhline(0.5, color='k', lw=0.5)
            ax.set(xlim=(0, 1.0), ylim=(-0.1, 1.0), xlabel='frequency / diffraction cutoff',
                   ylabel='half-burst FRC', title=f"{key[0]} step {key[1]} {ch}")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_frc.png"), dpi=90)
        plt.close(fig)

        fits_paths = [os.path.join(args.output_dir, f"{key[0]}_step{key[1]:03d}_{r['config']['name']}.fits")
                      for r in group]
        if not all(os.path.exists(p) for p in fits_paths):
            print(f"{name}: reconstructions not on disk (removed after the tuning), zoom figure skipped")
            continue
        fig, axes = plt.subplots(2, len(group), figsize=(2.6 * len(group), 5.6), squeeze=False)
        for col, (r, path) in enumerate(zip(group, fits_paths)):
            with fits.open(path) as f:
                for row, ch in enumerate(('BB', 'NB')):
                    z = zoom(f[ch].data)
                    lo, hi = np.percentile(z, [0.5, 99.5])
                    axes[row, col].imshow(z, cmap='gray', origin='lower', vmin=lo, vmax=hi)
                    axes[row, col].set_title(f"{r['config']['name']} {ch}", fontsize=7)
                    axes[row, col].axis('off')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{name}_zoom.png"), dpi=90)
        plt.close(fig)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(text + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split-burst tuning of cutoffs, modes and noise model.")
    parser.add_argument("--summary", action="store_true", help="Only summarise existing results")
    parser.add_argument("--output_dir", default="tuning")
    parser.add_argument("--config", default=os.path.join(HERE, "gris_momfbd.yaml"))
    parser.add_argument("--steps", default=DEFAULT_STEPS, help="scan:step list")
    parser.add_argument("--configs", default=STAGE1, help="Comma-separated configuration names")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard", default="0/1", help="i/n: run every n-th combination starting at i")
    parser.add_argument("--crop_size", type=int, default=768)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride_size", type=int, default=64)
    parser.add_argument("--n_iterations", type=int, default=250)
    parser.add_argument("--simultaneous_seq", type=int, default=225)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()
    summary(args) if args.summary else run(args)
