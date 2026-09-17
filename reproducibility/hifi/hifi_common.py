"""
Shared reconstruction pipeline for GREGOR / HiFI+ H-alpha bursts.

Both hifi_momfbd.py (one burst) and hifi_momfbd_batch_gpu.py (a whole dataset)
call reconstruct_burst() so that the two cannot drift apart.

What this does differently from a plain MOMFBD run, and why:

  * The noise is estimated for every patch instead of once for the whole field.
    A field crossing the limb spans a factor ~5 in noise amplitude, so a single
    number is wrong nearly everywhere, and it was wrong by a different amount in
    each patch - which is what produced the rectangular seams in the mosaics.

  * Patches are split into on-disk and off-limb, and the two get different
    Fourier cutoffs and a different number of wavefront modes. The wavefront is
    fitted to each patch independently, so a patch with no signal has nothing
    but noise to fit, and a large mode basis fits it: the result is spurious
    power spread over the whole band the loss covers. Off-limb patches are the
    extreme case and get a small basis; the disk keeps the modes it can
    actually constrain.

  * The mosaic discards exactly the pixels the deconvolution apodized. They are
    tapered and poorly constrained, and blending them into the neighbours is
    another source of seams. The apodization border and the stride go together:
    the border has to be wide enough that the window error at the limb stays
    small, and the stride short enough that what is left still overlaps.
"""

import numpy as np
import torch
from astropy.io import fits
import torchmfbd


def read_hifi_dataset(fits_path, n_frames=100, crop_size=None):
    """
    Read a HiFI+ level 1 burst.

    Extensions alternate between the two cameras:
      ext 1, 3, 5, ... -> camera 1, 656.3 nm H-alpha Lyot (narrow band)
      ext 2, 4, 6, ... -> camera 2, 656.7 nm wideband

    Frames are ranked by MFGS, the seeing metric the instrument pipeline already
    stores in every extension header. Level 1 has normally already kept the best
    frames of the original set (NSETFRMS reduced to NEXTEN), so the MFGS spread
    over what is left is tiny and this only matters when fewer frames than are
    available are requested.

    The frames that are kept are returned in temporal order. Frame 0 is the
    destretch reference, so keeping it the earliest retained frame makes that
    reference reproducible from burst to burst rather than an arbitrary pick.

    Returns
    -------
    nb_frames, wb_frames : ndarray of shape (1, n_frames, ny, nx)
    header : the primary header
    mfgs : the MFGS of the frames that were kept
    """
    f = fits.open(fits_path)
    n_avail = (len(f) - 1) // 2

    mfgs = np.full(n_avail, np.nan)
    for i in range(n_avail):
        vals = [f[e].header.get('MFGSMED') for e in (1 + 2 * i, 2 + 2 * i)]
        vals = [v for v in vals if v is not None]
        if vals:
            mfgs[i] = float(np.mean(vals))

    n_keep = min(n_avail, n_frames)
    if np.all(np.isnan(mfgs)):
        keep = np.arange(n_keep)
    else:
        keep = np.sort(np.argsort(np.nan_to_num(mfgs, nan=-np.inf))[::-1][:n_keep])

    img_h, img_w = f[1].data.shape
    if crop_size is not None:
        ch, cw = min(img_h, crop_size), min(img_w, crop_size)
    else:
        ch, cw = img_h, img_w

    nb_frames = np.zeros((1, n_keep, ch, cw), dtype=np.float32)
    wb_frames = np.zeros((1, n_keep, ch, cw), dtype=np.float32)

    for k, i in enumerate(keep):
        nb_frames[0, k] = f[1 + 2 * i].data[:ch, :cw]
        wb_frames[0, k] = f[2 + 2 * i].data[:ch, :cw]

    header = f[0].header
    f.close()
    return nb_frames, wb_frames, header, mfgs[keep]


def _percentile(cube, q):
    """
    Per-frame percentile of (1, n_frames, ny, nx). torch.quantile refuses tensors
    beyond ~16M elements, and a full HiFI+ burst is an order of magnitude larger,
    so go frame by frame.
    """
    flat = cube.reshape(cube.shape[0], cube.shape[1], -1)
    k = max(1, int(round(q * (flat.shape[-1] - 1))))
    return flat.kthvalue(k, dim=-1).values


def classify_patches(wb_patches, scale=1.0, dark_fraction=0.25, min_contrast=8.0,
                     disk_level=0.15):
    """
    Split the patches into on-disk (0) and off-limb (1) from the wideband
    brightness. The wideband carries no signal at all off the limb, so its
    brightness separates the two regimes by about two orders of magnitude and
    needs no tuning.

    A patch that straddles the limb counts as on-disk: it does contain
    photospheric signal and should keep the bandwidth that goes with it.

    Parameters
    ----------
    wb_patches : torch.Tensor
        Wideband patches, of shape (n_patches, n_frames, nx, ny).
    scale : float
        Factor that converts the patch means back to the intensity of the raw
        frames, i.e. the normalization that was divided out. Only used when the
        field does not cross the limb.
    dark_fraction : float
        A patch is off-limb below this fraction of the disk brightness.
    min_contrast : float
        Below this ratio between the brightest and the faintest patch the field
        does not cross the limb, and the brightness of one patch no longer says
        anything relative to the others. The absolute level decides instead.
    disk_level : float
        Raw wideband intensity above which a uniform field is on the disk. The
        level 1 frames are stored in normalized intensity, where the disk sits
        near 0.5 and the sky near 0.005, so this is not a delicate threshold.

    Returns
    -------
    index : torch.LongTensor of shape (n_patches,)
    crosses_limb : bool
    """
    mean = wb_patches.mean(dim=(1, 2, 3)).float().cpu()
    hi = torch.quantile(mean, 0.95)
    lo = torch.quantile(mean, 0.05).clamp(min=1e-8)

    if hi / lo > min_contrast:
        return (mean < dark_fraction * hi).long(), True

    off_limb = float(hi) * scale < disk_level
    return torch.full((mean.shape[0],), int(off_limb), dtype=torch.long), False


def reconstruct_burst(nb_frames, wb_frames, config, device,
                      patch_size=96, stride_size=32, n_iterations=250,
                      simultaneous_sequences=200, destretch=True,
                      disk_cutoff=((0.70, 0.85), (0.50, 0.65)),
                      limb_cutoff=((0.35, 0.50), (0.40, 0.55)),
                      disk_modes=44, limb_modes=20,
                      regime='auto', n_sigma_frames=8, logger=print):
    """
    Run the full MOMFBD reconstruction of one burst.

    Parameters
    ----------
    nb_frames, wb_frames : ndarray of shape (1, n_frames, ny, nx)
        Narrow-band and wideband frames as returned by read_hifi_dataset().
    config : str or dict
        torchmfbd configuration.
    device : torch.device
    disk_cutoff, limb_cutoff : pair of (lower, upper)
        Fourier cutoffs, in units of the diffraction cutoff, for the wideband
        (object 1) and the narrow band (object 2), for each of the two regimes.
    disk_modes, limb_modes : int
        Number of wavefront modes each regime is allowed to use. `disk_modes`
        must not exceed psf/nmax_modes in the configuration.
    regime : 'auto' | 'on_disk' | 'off_limb'
        'auto' classifies every patch; the other two force one regime.

    Returns
    -------
    obj_wb, obj_nb : ndarray
        The reconstructed wideband and narrow-band images.
    meta : dict
        Provenance, to be written into the output header.
    """
    n_frames = nb_frames.shape[1]

    nb = torch.tensor(nb_frames, dtype=torch.float32, device=device)
    wb = torch.tensor(wb_frames, dtype=torch.float32, device=device)

    # The 95th percentile is a robust high-signal reference and is much steadier
    # than the frame mean when most of the field is dark sky.
    nb_scale = _percentile(nb, 0.95).clamp(min=1e-5)
    wb_scale = _percentile(wb, 0.95).clamp(min=1e-5)
    nb /= nb_scale[..., None, None]
    wb /= wb_scale[..., None, None]

    if destretch:
        logger("Destretching the burst...")
        for name, cube in (('wb', wb), ('nb', nb)):
            warped, _ = torchmfbd.destretch(cube[:, None], ngrid=64, lr=0.50,
                                            reference_frame=0, border=6,
                                            n_iterations=40, lambda_tt=0.08)
            if name == 'wb':
                wb = warped[:, 0]
            else:
                nb = warped[:, 0]

    dec = torchmfbd.Deconvolution(config)
    apod = int(dec.config['images']['apodization_border'])

    patchify = torchmfbd.Patchify4D()
    wb_patches = patchify.patchify(wb, patch_size=patch_size, stride_size=stride_size, flatten_sequences=True)
    nb_patches = patchify.patchify(nb, patch_size=patch_size, stride_size=stride_size, flatten_sequences=True)

    # One noise estimate per patch. A handful of frames is enough - the noise
    # varies over the field, not from frame to frame - and estimating it for all
    # of them would dominate the run time.
    k = min(n_sigma_frames, n_frames)
    wb_sigma = torchmfbd.compute_noise(wb_patches[:, :k]).median(dim=1, keepdim=True).values
    nb_sigma = torchmfbd.compute_noise(nb_patches[:, :k]).median(dim=1, keepdim=True).values

    if regime == 'auto':
        index, crosses_limb = classify_patches(wb_patches, scale=float(wb_scale.mean()))
        if not crosses_limb:
            where = 'off the limb' if int(index[0]) == 1 else 'on the disk'
            logger(f"The field does not cross the limb; treating every patch as {where}.")
    elif regime == 'off_limb':
        index = torch.ones(wb_patches.shape[0], dtype=torch.long)
    else:
        index = torch.zeros(wb_patches.shape[0], dtype=torch.long)

    n_limb = int(index.sum())
    logger(f"Patches: {len(index) - n_limb} on disk, {n_limb} off limb")

    dec.add_frames(wb_patches, id_object=0, id_diversity=0, diversity=0.0, sigma=wb_sigma)
    dec.add_frames(nb_patches, id_object=1, id_diversity=0, diversity=0.0, sigma=nb_sigma)

    dec.set_patch_cutoffs([[list(disk_cutoff[0]), list(limb_cutoff[0])],
                           [list(disk_cutoff[1]), list(limb_cutoff[1])]],
                          [index, index])
    dec.set_patch_modes(torch.where(index == 1, limb_modes, disk_modes))

    dec.deconvolve(infer_object=False, optimizer='adam',
                   simultaneous_sequences=simultaneous_sequences,
                   n_iterations=n_iterations)

    # Crop exactly what the deconvolution apodized, so that tapered and poorly
    # constrained pixels are not blended into the mosaic. The mosaic origin then
    # sits at pixel (apod, apod) of the input field.
    obj_wb = patchify.unpatchify(dec.obj[0][:, None], apodization=apod,
                                 weight_type='cosine', weight_params=30).cpu().numpy()
    obj_nb = patchify.unpatchify(dec.obj[1][:, None], apodization=apod,
                                 weight_type='cosine', weight_params=30).cpu().numpy()

    meta = dict(apod=apod, n_frames=n_frames, patch_size=patch_size, stride_size=stride_size,
                n_iterations=n_iterations, n_patches=len(index), n_limb=n_limb,
                disk_modes=disk_modes, limb_modes=limb_modes,
                disk_cutoff=disk_cutoff, limb_cutoff=limb_cutoff,
                destretch=destretch, regime=regime)

    return np.squeeze(obj_wb), np.squeeze(obj_nb), dec, meta


def write_output(path, header, obj_wb, obj_nb, dec, meta, config_name, extra=None):
    """Write the reconstruction plus enough provenance to reproduce it."""
    hdu0 = fits.PrimaryHDU(header=header)
    h = hdu0.header
    h['PIPELINE'] = ('hifi_momfbd', 'Reconstruction script')
    h['TMFBDVER'] = (getattr(torchmfbd, '__version__', 'unknown'), 'torchmfbd version')
    h['CFGFILE'] = (config_name, 'MOMFBD configuration file')
    h['NFRAMES'] = (meta['n_frames'], 'Frames used per camera')
    h['PATCHSZ'] = (meta['patch_size'], 'Patch size [px]')
    h['STRIDESZ'] = (meta['stride_size'], 'Patch stride [px]')
    h['APODCROP'] = (meta['apod'], 'Px cropped per patch edge; mosaic origin offset')
    h['NITER'] = (meta['n_iterations'], 'Optimization iterations')
    h['NPATCH'] = (meta['n_patches'], 'Number of patches')
    h['NLIMB'] = (meta['n_limb'], 'Patches treated as off-limb')
    h['REGIME'] = (meta['regime'], 'Patch classification mode')
    h['MODESDSK'] = (meta['disk_modes'], 'Wavefront modes, on-disk patches')
    h['MODESLMB'] = (meta['limb_modes'], 'Wavefront modes, off-limb patches')
    h['CUTWBDSK'] = (str(list(meta['disk_cutoff'][0])), 'WB Fourier cutoff, on-disk')
    h['CUTNBDSK'] = (str(list(meta['disk_cutoff'][1])), 'NB Fourier cutoff, on-disk')
    h['CUTWBLMB'] = (str(list(meta['limb_cutoff'][0])), 'WB Fourier cutoff, off-limb')
    h['CUTNBLMB'] = (str(list(meta['limb_cutoff'][1])), 'NB Fourier cutoff, off-limb')
    h['NORMTYPE'] = ('p95', 'Per-frame intensity normalization')
    h['SIGMATYP'] = ('per_patch', 'Noise estimated per patch')
    h['DESTRTCH'] = (meta['destretch'], 'Destretching applied')
    for k, v in (extra or {}).items():
        h[k] = v

    hdul = fits.HDUList([
        hdu0,
        fits.ImageHDU(data=obj_wb, name="WIDEBAND_RECONSTRUCTED"),
        fits.ImageHDU(data=obj_nb, name="NARROWBAND_RECONSTRUCTED"),
        fits.ImageHDU(data=dec.rho[0].cpu().numpy(), name="PHASE_MODES_WB"),
        fits.ImageHDU(data=dec.rho[1].cpu().numpy(), name="PHASE_MODES_NB"),
    ])
    hdul.writeto(path, overwrite=True)
