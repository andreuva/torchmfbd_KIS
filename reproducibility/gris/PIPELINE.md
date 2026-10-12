# What the gris MOMFBD pipeline does to the data, step by step

This document follows one CRISP 630 nm scan from the level-1 files to the final movie. For every
operation it gives what is done to the pixels, the parameters, the reason, and a link to the code
that does it. [README.md](README.md) covers how to run the pipeline, the products and the known
issues; this document is the detailed account of the processing. The last section explains how
the first movie (`plots/science1_scan.mp4`) differs from the final one
(`plots_cube_v2/science1_scan.mp4`).

Line links refer to the code of the `gris-momfbd` branch as of 2026-10-12. Three code bases are
involved:

* `reproducibility/gris/`: the scripts written for this dataset (reading, driver, cube, figures);
* `reproducibility/hifi/hifi_common.py`: the reconstruction recipe validated on HiFI+ data, used
  unchanged ([`reconstruct_burst`](../hifi/hifi_common.py#L150), [`write_output`](../hifi/hifi_common.py#L275));
* `src/torchmfbd/`: the library (destretch, patches, noise estimate, MOMFBD, Wiener filter), used
  unchanged.

Naming: BB = broad band (the configuration's `object1`, index 0 in the code,
`WIDEBAND_RECONSTRUCTED` in the files); NB = narrow band (`object2`, index 1,
`NARROWBAND_RECONSTRUCTED`). The numbers below are for the recommended run, v2
([gris_momfbd_v2.yaml](gris_momfbd_v2.yaml), 1408 px crop). The first run, v1, differs only where
stated.

## 0. The chain at a glance

```
level-1 files: per scan 100 wavelength steps x 32 frames x 2048 x 2048, NB and BB
 |
 |  gris_momfbd.py, once per step (GPU, ~100 s per step)
 |   1  pair NB/BB files; measure the camera offset once per scan
 |   2  read a 1408 px crop of both cameras; flip BB in y and shift it onto NB
 |   3  divide every frame by its 95th percentile
 |   4  destretch the 32 frames of each camera to the average geometry of the burst
 |   5  cut into 441 overlapping 128 px patches; estimate the noise of every patch
 |   6  MOMFBD: fit 44 wavefront modes per frame and patch, shared by NB and BB
 |   7  object of every patch: multi-frame inverse with a data-driven Wiener filter
 |   8  stitch the patches into a 1366 px mosaic; scale back to ADU; write a FITS file
 |
 |  gris_make_cube.py, once per scan (GPU, minutes)
 |   9  stack the 100 steps; destretch them on BB to the average geometry of the scan,
 |      apply the same displacement to NB; drop a 20 px border -> 100 x 1326 x 1326 cube
 |
 |  gris_plot.py, once per scan (CPU, ~2 min)
 |  10  BB as a 5-step running mean (display only), fixed grey scale, one image pixel per
 |      data pixel, lossless movie
```

What the data look like after each stage:

| after | array (per channel) | where on the detector | geometry | units |
|---|---|---|---|---|
| level-1 file | 32 x 2048 x 2048 per step | full frame, round field of radius ~1093 px | every frame has its own seeing distortion; BB mirrored in y and shifted by (+3, -12) px relative to NB | ADU |
| reading (2) | 32 x 1408 x 1408 | NB rows/columns 320-1727 | BB flipped and shifted onto NB with whole pixels (no resampling) | ADU |
| normalisation (3) | same | same | same | every frame divided by its 95th percentile (~1) |
| destretch (4) | same | same | every frame resampled to the average geometry of its 32-frame burst | normalised |
| patches (5-7) | 441 x 32 x 128 x 128, then 441 x 128 x 128 objects | stride 64 px | per patch, the object sits where it is in the destretched frame 0 | normalised |
| mosaic (8) | 1366 x 1366 | NB detector 341-1706 | geometry of that step's burst | ADU |
| cube (9) | 100 x 1326 x 1326 | NB detector 361-1686 | every step resampled to the average geometry of the scan | ADU |
| movie frame (10) | 3104 x 1488 px, 8-bit grey, BB and NB side by side | one cube pixel = one image pixel | as the cube | grey level = ADU through a fixed range |

## 1. The input

`/dat/andreuva/data/SAC_gris_data/{NB,BB}/{science1,science2}/corr_sd*.fits`, as delivered by
the data providers:

* One file per wavelength step and camera, each holding 32 exposures of 2048 x 2048 px
  (float32, big-endian, ADU).
* Frames from both cameras are taken at 80 Hz, so a burst lasts 0.4 s. The 100 steps of a scan
  take 44.6 s (`DATE-OBS` of steps 1 and 100).
* Exposure: NB 9.5 ms, BB 0.95 ms.
* Dark and flat corrections are already applied, plus the prefilter correction on NB. The
  pipeline applies no further calibration.
* Pixel scale 0.032", for both cameras. The diffraction limit lambda/D at 630.2 nm with the 1.44 m
  aperture is 0.090", so the diffraction cutoff is 0.3545 cycles/px. That is 0.71 of the
  Nyquist frequency, i.e. Nyquist is at 1.41 of the cutoff.
* NB and BB files of the same step have the same name, and sorting the names gives wavelength
  order.

The reader's docstring summarises what was measured on these files:
[gris_common.py:1-19](gris_common.py#L1-L19).

## 2. How the pipeline is run

* [run_gris.sh](run_gris.sh) starts one `gris_momfbd.py` per scan in the background: science1 on
  GPU 0, science2 on GPU 2 ([line 19](run_gris.sh#L19), [line 30](run_gris.sh#L30)). Each writes
  `logs/<scan>_gpu<N>[_<TAG>].log`, and the script refuses to overwrite an existing log
  ([lines 20-23](run_gris.sh#L20-L23)). For v2 the arguments were
  `--config gris_momfbd_v2.yaml --crop_size 1408 --simultaneous_seq 441 --output_dir results_momfbd_v2`.
* `gris_momfbd.py` processes the steps one after the other. It skips a step whose output file
  exists ([gris_momfbd.py:177-193](gris_momfbd.py#L177-L193)), so a run is resumed by starting it
  again (with a new `TAG`).
* Then, per scan: `gris_make_cube.py --results_dir results_momfbd_v2/<scan> --gpu 0 --border 20`
  and `gris_plot.py --cube ... --bb_mean 5 --movie ...` (commands in
  [README.md](README.md#how-to-run-v2)).

The configuration is read once ([gris_momfbd.py:127-144](gris_momfbd.py#L127-L144)). Two checks
happen there:

* `nmax_modes` must cover whole radial orders: 44 + 1 is a triangular number.
* The patch size must equal `images/n_pixel`.

## 3. Reconstruction of one wavelength step

### 3.1 Pairing the files and registering the cameras (once per scan)

* **Pairing.** [`list_steps`](gris_common.py#L34) pairs each NB file with the BB file of the same
  name.
* **The flip.** The two cameras are mirrored in y with respect to each other. The flip is applied
  to BB, so the NB products keep the orientation of the level-1 NB files.
* **The offset.** After the flip, the BB field is shifted relative to NB. The shift is measured
  once per scan, on the first step (continuum), by [`measure_offset`](gris_common.py#L84),
  called at [gris_momfbd.py:166-174](gris_momfbd.py#L166-L174):
  1. Take the central 768 px of the mean of the first 8 frames of each camera, with BB read
     flipped.
  2. Subtract a Gaussian blur (sigma 12 px) to keep only the granulation
     ([`_highpass`](gris_common.py#L79)).
  3. Phase cross-correlation with 1/10 px precision
     ([line 102](gris_common.py#L102)), rounded to whole pixels ([line 103](gris_common.py#L103)).
  4. Re-read BB with the offset and check: correlation of the two filtered images and the
     residual shift ([lines 105-108](gris_common.py#L105-L108)).
* **Result.** (dy, dx) = (+3, -12) px in both scans. The residual is at most 0.2 px and the
  correlation 0.970 (science1) and 0.961 (science2), from the log of the v2 run.

The offset is applied as an integer shift of the BB read window, so no resampling is involved.
Two differences between the cameras are not corrected anywhere in the pipeline: a static rotation
of ~0.36 deg and a scale difference of 0.17-0.19 % (README).

### 3.2 Reading a step

[`read_step`](gris_common.py#L112), called at [gris_momfbd.py:52-53](gris_momfbd.py#L52-L53):

* **NB.** The centred 1408 x 1408 window, origin (320, 320) ([`crop_window`](gris_common.py#L46)).
  The whole crop lies inside the illuminated circle.
* **BB.** The same window displaced by the offset and taken from the y-flipped image. It is read
  as the stored rows [2048 - y0 - 1408, 2048 - y0), reversed
  ([`read_crop`](gris_common.py#L55), [lines 69-75](gris_common.py#L69-L75)).
* **Format.** Both are copied to native little-endian, C-contiguous float32
  ([line 76](gris_common.py#L76)), because torch refuses big-endian and negative-stride arrays.
* **ADU scale.** For each channel, the mean over the 32 frames of the per-frame 95th percentile
  ([lines 129-130](gris_common.py#L129-L130)). It is used in step 3.9 to put the result back into
  ADU.

### 3.3 Per-frame normalisation

[hifi_common.py:190-193](../hifi/hifi_common.py#L190-L193): every frame is divided by its own 95th
percentile ([`_percentile`](../hifi/hifi_common.py#L93)). This removes frame-to-frame intensity
changes within the burst (transparency, exposure jitter). At the end the reconstruction is
multiplied by the mean of these percentiles, so the absolute level of each step, and with it the
spectral line profile, is preserved (section 3.9).

### 3.4 Destretch within the burst

[hifi_common.py:195-204](../hifi/hifi_common.py#L195-L204) calls
[`torchmfbd.destretch`](../../src/torchmfbd/destretch.py#L11) once for BB and once for NB. Each
camera is aligned on its own images.

* **Model.** One displacement vector (dx, dy) per control point and frame, on a 64 x 64 grid of
  control points (~22 px apart over 1408 px), interpolated to every pixel.
* **Fit.** 40 Adam iterations (learning rate 0.5) maximise the normalised correlation between
  each warped frame and frame 0, ignoring a 6 px border
  ([destretch.py:104-132](../../src/torchmfbd/destretch.py#L104-L132)). A penalty of 0.08 times
  the mean squared difference between neighbouring control points keeps the field smooth
  ([lines 126-128](../../src/torchmfbd/destretch.py#L126-L128)).
* **Average geometry.** The fitted fields of the 32 frames are made zero-mean
  ([lines 149-150](../../src/torchmfbd/destretch.py#L149-L150)). The frames therefore end up in
  the **average geometry of the burst**, not in the geometry of frame 0.
* **Resampling.** Each original frame is resampled once, with bicubic interpolation and reflected
  edges ([lines 152-155](../../src/torchmfbd/destretch.py#L152-L155),
  [warp.py](../../src/torchmfbd/warp.py#L3)). This is the first time the data are interpolated.

Purpose: MOMFBD models each 128 px patch with one PSF per frame, which assumes the patch moves
rigidly between frames. Removing the differential seeing motion first makes that assumption
hold.

Side effects (README, known issues):

* The interpolation smooths the noise above ~0.85 of the cutoff. This is why v2 measures the
  shape of the noise spectrum instead of assuming it flat (section 3.8).
* Because of the `align_corners` issue in [warp.py:34-42](../../src/torchmfbd/warp.py#L34-L42),
  even a zero displacement samples at offsets from -0.5 px (one edge) to +0.5 px (the other).
  The smoothing therefore varies over the field, and the frames are rescaled by W/(W-1).

### 3.5 Patches and noise

* **Patches.** [hifi_common.py:220-222](../hifi/hifi_common.py#L220-L222) cuts every frame into
  128 x 128 patches every 64 px ([`Patchify4D.patchify`](../../src/torchmfbd/patchify4D.py#L15)).
  That gives (1408 - 128)/64 + 1 = 21 per axis, 441 patches (v1: 15 x 15 = 225).
* **Patch size.** 128 px = 4.1", the same angular size as the patches validated on HiFI+.
* **Noise.** [hifi_common.py:227-230](../hifi/hifi_common.py#L227-L230) estimates one noise sigma
  per patch and channel. It uses the SVD-based estimator of
  [`compute_noise`](../../src/torchmfbd/noise.py#L4) (median singular value, Gavish & Donoho;
  [noise_svd.py:83](../../src/torchmfbd/noise_svd.py#L83)) on the first 8 frames of the patch,
  and takes the median. This sigma enters the merit function (section 3.7).
* **Regime.** The driver forces every patch to be "on disk" (`regime='on_disk'`). All 441
  patches therefore get the same Fourier cutoff [0.70, 0.85] and 44 modes, for both channels
  ([hifi_common.py:233-252](../hifi/hifi_common.py#L233-L252)). The off-limb branch of the HiFI+
  recipe is unused.

### 3.6 MOMFBD set-up

[`Deconvolution`](../../src/torchmfbd/deconvolution.py#L40) is built from the yaml
([hifi_common.py:206](../hifi/hifi_common.py#L206)) and receives the BB patches as object 0 and
the NB patches as object 1 ([lines 246-247](../hifi/hifi_common.py#L246-L247)).
[`deconvolve`](../../src/torchmfbd/deconvolution.py#L2227) then prepares the following.

* **Gradient removal and edge taper.** In every patch, a plane fitted to the mean of the 32
  frames is subtracted, then the frames are multiplied by a 2D Hanning taper over the outer
  20 px ([deconvolution.py:122-137](../../src/torchmfbd/deconvolution.py#L122-L137),
  [`combine_frames`](../../src/torchmfbd/deconvolution.py#L2007),
  [`util.apodize`](../../src/torchmfbd/util.py#L77)). The plane is added back to the final object
  ([deconvolution.py:1603-1604](../../src/torchmfbd/deconvolution.py#L1603-L1604)). This avoids
  the edge discontinuities that a periodic FFT would otherwise see. The frames are then Fourier
  transformed ([lines 2280-2282](../../src/torchmfbd/deconvolution.py#L2280-L2282)).
* **Pupil and frequency grid** ([`_define_basis`](../../src/torchmfbd/deconvolution.py#L204)).
  * The 128 px pupil array is padded by the factor lambda/(D x pixel) = 2.82
    ([util.psf_scale](../../src/torchmfbd/util.py#L44)). The pupil is therefore ~45 px across,
    with the 40/144 central obscuration ([util.aperture](../../src/torchmfbd/util.py#L10)).
  * Spatial frequencies are expressed in units of the diffraction cutoff
    ([deconvolution.py:452-457](../../src/torchmfbd/deconvolution.py#L452-L457)).
  * Both objects get the same wavelength (6302 A), hence the same pupil and basis, and so
    **exactly the same PSF** for the simultaneous NB and BB frames. This is the shared wavefront
    of the strategy agreed with the observers. The model ignores that NB integrates 10 times
    longer than BB.
* **Wavefront basis.** 44 Karhunen-Loeve modes of Kolmogorov turbulence
  ([`KL.precalculate`](../../src/torchmfbd/kl_modes.py#L72)), loaded from `basis/` or computed
  once.
  * Modes 1-2 are tip and tilt.
  * Piston is left out ([kl_modes.py:145](../../src/torchmfbd/kl_modes.py#L145)). As a
    consequence the 44th array is zero, so 43 modes are effective.
  * The mode count must close a radial order
    ([deconvolution.py:211-246](../../src/torchmfbd/deconvolution.py#L211-L246)).
* **Fourier cutoff.** A mask that is 1 up to 0.70 of the cutoff and falls as a cosine to 0 at
  0.85 ([`_diffraction_mask_from_cutoff`](../../src/torchmfbd/deconvolution.py#L764)). It weights
  the merit function and multiplies the final object.

### 3.7 The wavefront fit

The unknowns are 44 KL coefficients for each of the 32 frames of each of the 441 patches, the same
coefficients for NB and BB. They start at zero
([deconvolution.py:2457-2460](../../src/torchmfbd/deconvolution.py#L2457-L2460)). All 441 patches
are fitted together in one batch (`--simultaneous_seq 441`; 42.6 GB of GPU memory).

* **PSF of frame j** ([`compute_psfs`](../../src/torchmfbd/deconvolution.py#L1020)).
  * The wavefront phi_j is the sum of a_jk K_k over the 44 modes
    ([line 1100](../../src/torchmfbd/deconvolution.py#L1100)).
  * PSF_j = |FFT(P exp(i phi_j))|^2, normalised to unit sum
    ([lines 1111-1118](../../src/torchmfbd/deconvolution.py#L1111-L1118)).
  * The OTF S_j is its FFT ([line 1121](../../src/torchmfbd/deconvolution.py#L1121)).
* **Merit function.** The "joint" MOMFBD metric, with the object eliminated analytically
  ([`compute_loss`](../../src/torchmfbd/deconvolution.py#L1655),
  [lines 1694-1721](../../src/torchmfbd/deconvolution.py#L1694-L1721)). For each channel and
  patch, with D_j the Fourier transforms of the frames:

  $$L=\sum_\nu M(\nu)\,\frac{1}{2\sigma^2}\Big[\sum_j|D_j|^2-\frac{s_u\,|\sum_j D_j S_j^*|^2}{\sigma^2+s_u\sum_j|S_j|^2}\Big]$$

  * sigma^2 is the noise variance of the patch (section 3.5).
  * s_u = 100 (`s_u_joint`).
  * M is the cutoff mask.
  * The two channels and all patches are summed or averaged into one number. The zero-frequency
    row and column are left out ([line 1831](../../src/torchmfbd/deconvolution.py#L1831)).
  * Since the PSFs are shared, the NB data also constrain the wavefront that restores BB, and
    vice versa.
* **Tip-tilt reference.** Inside the fit, the tip and tilt of every frame are taken relative to
  those of frame 0 of the same patch
  ([lines 2710-2711](../../src/torchmfbd/deconvolution.py#L2710-L2711)). The image position is
  degenerate in MOMFBD; this pins each patch of the object to its position in the destretched
  frame 0.
* **Optimiser.** Adam, learning rate 0.05, 250 iterations
  ([line 2660](../../src/torchmfbd/deconvolution.py#L2660)). A cosine learning-rate scheduler is
  created ([line 2680](../../src/torchmfbd/deconvolution.py#L2680)) but never stepped, so the rate
  stays constant.
* **Mode annealing** ([`compute_annealing`](../../src/torchmfbd/deconvolution.py#L671),
  [lines 2319-2331](../../src/torchmfbd/deconvolution.py#L2319-L2331),
  [line 2871](../../src/torchmfbd/deconvolution.py#L2871)). The number of free modes grows by
  whole radial orders, 2, 5, 9, 14, 20, 27, 35, 44. It increases linearly over the first 60 % of
  the iterations (one step about every 21 iterations), and all 44 modes are free for the last
  100. Low orders settle before high orders can fit noise.

### 3.8 The object and the Wiener filter

After the fit, the PSFs are recomputed with all modes
([deconvolution.py:2882-2885](../../src/torchmfbd/deconvolution.py#L2882-L2885)). The object of
every patch and channel is the regularised multi-frame inverse
([`compute_object`](../../src/torchmfbd/deconvolution.py#L1490),
[lines 1518-1521](../../src/torchmfbd/deconvolution.py#L1518-L1521),
[1577-1583](../../src/torchmfbd/deconvolution.py#L1577-L1583)):

$$\hat O(\nu)=\frac{\sum_j S_j^*(\nu)\,D_j(\nu)}{\sum_j|S_j(\nu)|^2+R(\nu)}\;w_{\rm support}(\nu)\,M(\nu)$$

The result is transformed back to the image, and the plane removed in section 3.6 is added again
([lines 1595-1604](../../src/torchmfbd/deconvolution.py#L1595-L1604)).

The regularisation R is measured from the data of each patch, separately for NB and BB
([`wiener_filter`](../../src/torchmfbd/deconvolution.py#L908)):

1. **Power ratio.** Q(nu) = |sum_j S_j* D_j|^2 / sum_j |S_j|^2, averaged over 64 rings of
   constant |nu| ([lines 949-952](../../src/torchmfbd/deconvolution.py#L949-L952),
   [`_setup_radial_bins`](../../src/torchmfbd/deconvolution.py#L872)). Q is noise plus signal,
   and pure noise beyond the diffraction cutoff.
2. **Noise level.** Per patch, the median of Q over the rings between 0.9 and 1.45 of the cutoff
   ([lines 955-958](../../src/torchmfbd/deconvolution.py#L955-L958),
   [982-983](../../src/torchmfbd/deconvolution.py#L982-L983)). The telescope passes no signal
   there, and the data are sampled up to 1.41.
3. **Noise shape.** This is the only difference between v1 and v2
   ([lines 960-979](../../src/torchmfbd/deconvolution.py#L960-L979)):
   * **v1, `white`:** flat. The level measured beyond the cutoff is assumed at all frequencies.
     Because the destretch interpolation smoothed the noise (section 3.4), this underestimates the
     noise where the signal is, by x1.45 in power.
   * **v2, `auto` with `filter_noise_flat: 0.70`:** the shape is measured, as the 25 % quantile
     over the 441 patches of each patch's Q profile divided by its own level. It is forced not to
     increase with frequency and held flat below 0.70 of the cutoff, where the signal
     contaminates the measurement. One shape is estimated per burst and channel. Section
     "The noise model" of the README gives the split-burst test that chose it.
4. **Wiener weight per ring.** w = 1 - N/Q, clipped to [0, 1]
   ([line 985](../../src/torchmfbd/deconvolution.py#L985)). It is then smoothed with a 3-ring
   median ([991-992](../../src/torchmfbd/deconvolution.py#L991-L992)), made non-increasing
   outwards ([1000](../../src/torchmfbd/deconvolution.py#L1000)), and soft-thresholded as
   (w - 0.3)/0.7 ([1006-1007](../../src/torchmfbd/deconvolution.py#L1006-L1007)).
5. **Regularisation.** R = <sum_j |S_j|^2>_ring x (1 - w)/w. Rings with w = 0 are removed
   entirely (`w_support`) ([lines 1013-1016](../../src/torchmfbd/deconvolution.py#L1013-L1016)).
   The cutoff mask M (cosine from 0.70 to 0.85) is applied on top
   ([line 1542](../../src/torchmfbd/deconvolution.py#L1542)).

The effective resolution of each patch is set by its own signal-to-noise ratio. It is lower in
poor seeing and in the dark line-core steps, and it differs from patch to patch and from burst to
burst.

### 3.9 Mosaic, ADU and output file

* **Mosaic.** [`Patchify4D.unpatchify`](../../src/torchmfbd/patchify4D.py#L81), called at
  [hifi_common.py:261-264](../hifi/hifi_common.py#L261-L264):
  1. Drop the 20 px that were tapered on each patch edge, keeping 88 x 88
     ([patchify4D.py:108](../../src/torchmfbd/patchify4D.py#L108)).
  2. Weight the patch with a 2D cosine taper over its outer 30 px
     ([line 122](../../src/torchmfbd/patchify4D.py#L122)).
  3. Add the overlapping patches and divide by the summed weights
     ([lines 146-149](../../src/torchmfbd/patchify4D.py#L146-L149)).

  The mosaic is (21 - 1) x 64 + 88 = 1368 px and starts at pixel 20 of the crop. Its outermost
  ring is covered by a single patch whose weight is zero there, so it is exactly 0.
  `gris_momfbd.py` drops it, giving 1366 px starting at crop pixel 21 (`APODCROP` = 21;
  [gris_momfbd.py:65-66](gris_momfbd.py#L65-L66)). v1 kept the ring: 984 px, `APODCROP` = 20.
* **ADU.** The mosaics are multiplied by the ADU scale of section 3.2
  ([gris_momfbd.py:70-71](gris_momfbd.py#L70-L71)).
* **Output.** [`write_output`](../hifi/hifi_common.py#L275), with extra keys at
  [gris_momfbd.py:74-92](gris_momfbd.py#L74-L92).
  * Name: `results_momfbd_v2/<scan>/stepNNN_<wavelength>A_momfbd.fits`.
  * Contents: the raw NB primary header plus provenance (crop, flip and offset, ADU scales,
    MOMFBD settings), then `WIDEBAND_RECONSTRUCTED` and `NARROWBAND_RECONSTRUCTED`. The
    `PHASE_MODES_*` extensions hold the frequency grid, not wavefronts (README).

Each per-step image is in the geometry of its own burst: the average seeing distortion of those
0.4 s, with each patch pinned to the destretched frame 0. Consecutive steps therefore differ by
this distortion.

## 4. The cube ([gris_make_cube.py](gris_make_cube.py))

1. **Stack.** The 100 step files are stacked by `SCANSTEP`. A missing step would become a NaN
   plane so that the wavelength axis stays linear
   ([lines 102-133](gris_make_cube.py#L102-L133)).
2. **Register the steps, measured on BB.** Registering means resampling every step so that a
   given pixel shows the same point on the Sun in all 100 steps. Without it, the step-to-step
   distortion of section 3.9 would smear the line profile of every small feature.
   * The distortion is measured only on BB, because BB shows the same continuum scene at every
     step, while NB changes with wavelength. An alignment on NB would mistake those spectral
     changes for distortions.
   * The correction measured on BB is then applied to the NB plane of the same step (item 3).
   * Code: [`register_cubes`](gris_make_cube.py#L50),
     [destretch call at lines 69-71](gris_make_cube.py#L69-L71).
   * Same `torchmfbd.destretch` as in section 3.4, with 89 x 89 control points (~15 px apart;
     [line 65](gris_make_cube.py#L65)) and 3 passes of 60 iterations.
   * Pass 1 aligns every plane to step 1. Passes 2 and 3 align to the mean of the BB planes warped
     so far ([destretch.py:85-97](../../src/torchmfbd/destretch.py#L85-L97)). This mean image is
     only an alignment target; it has nothing to do with the 5-step BB average of the movie
     (section 5).
   * The displacements are made zero-mean over the 100 steps. Every step ends up in the
     **average geometry of the scan**, and each plane is resampled once (bicubic).
3. **Apply the same displacement to NB** ([line 74](gris_make_cube.py#L74),
   [`apply_destretch`](../../src/torchmfbd/destretch.py#L166)).
   * This is valid because the NB and BB frames of a step were exposed simultaneously through the
     same atmosphere, and were restored with the same wavefronts. So the NB image of a step
     carries the same distortion as its BB image.
   * NB and BB therefore keep the relative alignment they had: the cameras' alignment, with the
     uncorrected 0.36 deg rotation and 0.17-0.19 % scale.
4. **Record the displacements** in the `SHIFTS` table: mean shift per step and largest local
   displacement ([lines 83-87](gris_make_cube.py#L83-L87),
   [191-200](gris_make_cube.py#L191-L200)). For science1 the mean shift of a step reaches 3.9 px,
   and the largest local displacement is 7.5 px for the median step and up to 13.6 px.
5. **Crop.** Drop 20 px on every side, where the warp only has reflected pixels
   ([lines 144-146](gris_make_cube.py#L144-L146)). The cube is 100 x 1326 x 1326 = 42.4" square,
   and cube pixel (0, 0) is NB detector pixel (361, 361).
6. **Header and axes** ([lines 148-185](gris_make_cube.py#L148-L185)).
   * Per-step keys are removed from the primary header; NB and BB exposure times are added.
   * Linear axes: x and y in arcsec, axis 3 `AWAV` with 31.5 mA per step. This is a linear fit
     of the providers' `WAVE`; the line positions suggest ~32.2 mA (README caveat).

The registration leaves a median residual of ~0.05-0.1 px against the scan mean. For science2's
worst-seeing planes it is ~0.3-0.9 px (README).

## 5. Figures and movie ([gris_plot.py](gris_plot.py))

* **BB running mean, for display only** ([`steps_from_cube`](gris_plot.py#L85),
  [lines 99-100](gris_plot.py#L99-L100)).
  * Each BB frame is the mean of the planes of steps k-2..k+2: 3 or 4 steps at the ends of the
    scan, as the panel title says. NB is the single step k.
  * Why: BB shows the same scene at every step (45 s in total), so its step-to-step changes are
    reconstruction noise of independent bursts. The mean reduces that noise without changing the
    scene. NB is not averaged, because that would mix wavelengths.
* **Display range** ([`display_range`](gris_plot.py#L113)). One range per channel for the whole
  scan, so the line darkening stays visible. It runs from the lowest 0.5 percentile to the
  highest 99.5 percentile over the steps, measured 8 px inside the edges (BB after averaging).
  science1: BB 805-2952 ADU, NB 403-2521 ADU.
* **Full resolution** ([`plot_step`](gris_plot.py#L123)).
  * Each panel is converted to 8-bit grey here: round((x - vmin)/(vmax - vmin) x 255), clipped
    ([line 147](gris_plot.py#L147)).
  * It is pasted pixel for pixel with `figimage` ([line 148](gris_plot.py#L148)), so one cube
    pixel is one image pixel, with no resampling.
  * Axes and colour bars are drawn around the panels and do not cover image pixels.
  * The frame is 3104 x 1488 px, a multiple of 16 so that the encoder does not rescale it. It is
    saved as an 8-bit grey PNG ([line 171](gris_plot.py#L171)).
* **Movie** ([`write_movie`](gris_plot.py#L174)): H.264 at 8 frames/s, full-range grey.
  * `--crf 0` gives the lossless `<scan>_scan.mp4`: every frame decodes exactly to its PNG.
  * `--crf 1` gives `<scan>_scan_compatible.mp4`, a standard High-profile copy within ~0.5 grey
    level rms.

## 6. Where each image sits: the geometry chain

The difference between the two movies (section 8) comes down to this chain, so here it is in one
place:

1. **Raw frame:** the instantaneous seeing distortion of that 0.95 ms (BB) or 9.5 ms (NB)
   exposure.
2. **After the destretch within the burst:** the average distortion over the 0.4 s burst.
   Consecutive bursts have different averages.
3. **MOMFBD object:** in that burst-average geometry, each patch pinned to the destretched
   frame 0. These are the per-step files in `results_momfbd*/`.
4. **Cube plane:** destretched to the average geometry over the 100 bursts (45 s). Consecutive
   planes now agree to ~0.1 px.

Resampling happens exactly twice: once in step 2 (raw frames, bicubic) and once in step 4 (cube
planes, bicubic). The crop, the flip and the camera offset (sections 3.1 and 3.2) are
whole-pixel operations.

## 7. What the pipeline does not do

* No further calibration (dark, flat, prefilter come from the providers). NB detector defects and
  fringes are passed through (README, known issues).
* No rotation or scale correction between NB and BB (~0.36 deg, 0.17-0.19 %): pixel-by-pixel NB/BB
  comparisons need an affine correction first.
* No correction of the wavelength scale (`WAVE` as delivered).
* No temporal or spectral smoothing of the science products. The BB running mean exists only in
  the figures and movies.
* No frame selection: all 32 frames of every step are used.
* The per-step reconstructions are not bit-reproducible: a rerun differs by ~3 % of the image rms,
  mostly at small scales (README).

## 8. The first movie and the final movie

`plots/science1_scan.mp4` was the first look at the v1 run (11 Oct). It was made with the original
`gris_plot.py` (commit 2de3545; `git show 2de3545:reproducibility/gris/gris_plot.py`) and the HiFI+
movie script [make_hifi_movie.py](../hifi/make_hifi_movie.py). `plots_cube_v2/science1_scan.mp4`
is the final product, made with the current `gris_plot.py` from the v2 cube.

| | first: `plots/science1_scan.mp4` | final: `plots_cube_v2/science1_scan.mp4` |
|---|---|---|
| data | v1 per-step files, `results_momfbd/science1/step*.fits` | v2 cube, `results_momfbd_v2/science1_cube.fits` |
| noise model | `white` | `auto`, flat below 0.70 |
| field | 984 px = 31.5" (1024 px crop), including the 1 px zero ring | 1326 px = 42.4" (1408 px crop); the first movie's field is its central part, ~171 px in from each edge |
| geometry from step to step | each step in its own burst geometry: mean shifts up to ~4 px, local distortions of typically 7-8 px and up to ~14 px between a step and the scan average | all steps registered to the scan-average geometry, ~0.1 px residual |
| panels | 2 x 2: raw frame 0 and reconstruction, for BB and NB | 1 x 2: reconstructions only, BB and NB |
| BB | the single step | mean of 5 steps (3-4 at the ends) |
| NB | the single step | the single step |
| display scale | 984 px drawn in ~516 px (1.9 data px per screen px), resampled by matplotlib's anti-aliasing (Hanning) filter | 1 data px = 1 screen px, no resampling |
| grey levels | matplotlib's `gray` colour map (232 of 256 levels) | all 256 levels, computed directly |
| display range | BB 706-2945, NB 349-2504 ADU (0.5/99.5 percentiles of the whole mosaics) | BB 805-2952, NB 403-2521 ADU (interior; BB after averaging) |
| frame | 1430 x 1386 px RGBA PNG, stretched by the encoder to 1440 x 1392 (0.7 % in x, 0.4 % in y) | 3104 x 1488 px 8-bit grey PNG, encoded as is |
| encoding | H.264 High, CRF 18, keyframe every 16 frames, video levels 16-235; 7 MB, 4.5 Mb/s | lossless H.264 (High 4:4:4 Predictive), full range; 104 MB, 67 Mb/s; plus a CRF 1 standard copy |
| length | 100 frames at 8 frames/s, 12.5 s | same |

**What changes on screen.** The rms change between consecutive frames in the BB reconstruction
panel, in grey levels:

| | median | 90th percentile |
|---|---|---|
| first movie | 9.7 | 12.4 |
| final movie | 1.0 | 1.3 |
| final movie, NB panel (the line moving through the scan plus per-step noise) | 4.2 | 6.2 |

All three are measured on the PNG frames. They correspond to different steps of the pipeline:

1. **Alignment.** In the first movie almost all of the change is the image itself moving: every
   step is shown in the distortion its own burst left behind (section 6). That distortion is not
   a rigid shift. In the earlier analysis of the high-pass filtered frames, the change was 9.2
   grey levels, and removing the best global shift of each frame lowered it only to 8.4-8.6. The
   cube registration removes the distortion, which is why the final movie is still.
2. **Reconstruction noise, exposed by the alignment.** Every step is an independent burst and is
   restored a little differently (~0.7 % rms contrast at 0.3-0.9", independent from step to
   step). In the first movie this flicker was ~17 % of the variance of the change and was hidden
   by the motion, which the eye follows. Once the image stands still, it is the only thing that
   changes in BB. The 5-step BB mean reduces it to the 1.0 grey level above.
3. **Noise model.** Not the cause of the moving texture: on the same field and scale, v1 (white)
   changes more from step to step than v2 (README, "Moving texture in the aligned movies").
   v2 removes the fine grain that the white model let through, especially in bad seeing.
4. **Display scale.** The first movie showed each data pixel at about half size, and the
   resampling filter averaged neighbouring pixels. The final movie shows every pixel: the full
   detail, but also all the fine-scale noise. A larger field (42.4" instead of 31.5") includes
   outer patches, where the noise is somewhat higher.
5. **Encoding.** CRF 18 is fine for a moving scene, but in a still one it changes the fine
   texture about as much as the data do. The final movie is lossless.

In short: the first movie shows each step as it came out of its own burst, small and moving. The
final movie shows the registered cube at full resolution and full fidelity, with the BB averaged
over 5 steps so that its burst-to-burst noise does not flicker.

## 9. Code index

| operation | code |
|---|---|
| pair files | [gris_common.list_steps](gris_common.py#L34) |
| camera offset | [gris_common.measure_offset](gris_common.py#L84), called at [gris_momfbd.py:166](gris_momfbd.py#L166) |
| crop, flip, float32, ADU scale | [gris_common.read_step](gris_common.py#L112), [read_crop](gris_common.py#L55) |
| per-step driver | [gris_momfbd.process_step](gris_momfbd.py#L51) |
| burst recipe | [hifi_common.reconstruct_burst](../hifi/hifi_common.py#L150) |
| per-frame normalisation | [hifi_common.py:190](../hifi/hifi_common.py#L190) |
| destretch | [destretch.destretch](../../src/torchmfbd/destretch.py#L11), [warp.warp](../../src/torchmfbd/warp.py#L3) |
| patches, mosaic | [Patchify4D.patchify](../../src/torchmfbd/patchify4D.py#L15), [unpatchify](../../src/torchmfbd/patchify4D.py#L81) |
| noise per patch | [noise.compute_noise](../../src/torchmfbd/noise.py#L4), [noise_svd.noise_estimation](../../src/torchmfbd/noise_svd.py#L83) |
| gradient removal, taper | [Deconvolution.combine_frames](../../src/torchmfbd/deconvolution.py#L1955), [util.apodize](../../src/torchmfbd/util.py#L77) |
| pupil, KL basis, frequency grid | [Deconvolution._define_basis](../../src/torchmfbd/deconvolution.py#L204), [KL.precalculate](../../src/torchmfbd/kl_modes.py#L72) |
| cutoff mask | [_diffraction_mask_from_cutoff](../../src/torchmfbd/deconvolution.py#L764) |
| PSFs | [Deconvolution.compute_psfs](../../src/torchmfbd/deconvolution.py#L1020) |
| merit function | [Deconvolution.compute_loss](../../src/torchmfbd/deconvolution.py#L1655) |
| optimisation, annealing | [Deconvolution.deconvolve](../../src/torchmfbd/deconvolution.py#L2227), [compute_annealing](../../src/torchmfbd/deconvolution.py#L671) |
| object and Wiener filter | [Deconvolution.compute_object](../../src/torchmfbd/deconvolution.py#L1490), [wiener_filter](../../src/torchmfbd/deconvolution.py#L908) |
| output file | [hifi_common.write_output](../hifi/hifi_common.py#L275) |
| cube registration | [gris_make_cube.register_cubes](gris_make_cube.py#L50) |
| BB running mean | [gris_plot.steps_from_cube](gris_plot.py#L85) |
| full-resolution frame | [gris_plot.plot_step](gris_plot.py#L123) |
| movie | [gris_plot.write_movie](gris_plot.py#L174) |
| configuration (v2) | [gris_momfbd_v2.yaml](gris_momfbd_v2.yaml), defaults in [configuration.py](../../src/torchmfbd/configuration.py#L1) |
