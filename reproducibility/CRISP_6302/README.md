# MOMFBD reconstruction of the CRISP 630 nm scans (SAC dataset, 2026-10-06)

Reconstruction of the two spectral scans `science1` and `science2` taken with the CRISP
narrow-band (NB) and broad-band (BB) cameras at GREGOR, with the torchmfbd pipeline that was
validated on HiFI+ data (`../hifi/`). For every one of the 100 wavelength steps, the 32 NB
exposures and the 32 simultaneous BB exposures are reconstructed together: two objects,
one wavefront per frame. Nothing in `../hifi/` or in `src/torchmfbd/` is modified;
`reconstruct_burst()` and `write_output()` are imported from `../hifi/hifi_common.py`.

[PIPELINE.md](PIPELINE.md) describes every operation applied to the data, in order, with links to
the code that does it, and compares the first and the final movie.

Data: `/dat/andreuva/data/SAC_gris_data/{NB,BB}/{science1,science2}/corr_sd*.fits`, one
(32, 2048, 2048) float32 cube per step, same file name in NB and BB, dark/flat corrected
(NB also prefilter corrected), in ADU, 0.032"/px, exposures 9.5 ms (NB) and 0.95 ms (BB).

> **Wavelength caveat (data providers' calibration).** The step headers (`WAVE`) and therefore
> the cubes use the nominal 3.15 pm per step, with step 39 = 6301.501 A (`CORESTEP`). The line
> positions in these data disagree: the Fe I 6301.50/6302.49 minima are 30.9 steps apart and the
> telluric O2 6302.00/6302.76 minima (independent of solar velocities) 23.6 steps apart, in
> both scans, which implies ~3.22 pm per step (3.19-3.25 depending on the estimator), ~2 % more
> than the nominal "ca 3.15 pm" of the observers. The observed NB
> minima are at steps ~40 (39.9) and ~71 (70.8). This is the same in the raw data, so it is not
> a reconstruction effect; confirm the step with the data providers before deriving Doppler
> velocities from the `WAVE` axis.

## Two runs

| run | config | crop | output | status |
|---|---|---|---|---|
| v1 | `gris_momfbd.yaml` (white noise model) | 1024 px | `results_momfbd/` | first run, kept unchanged |
| v2 | `gris_momfbd_v2.yaml` (measured noise model, see below) | 1408 px | `results_momfbd_v2/` | recommended |

v2 differs from v1 in: the Wiener-filter noise model; the larger field, processed as one batch
of 21 x 21 = 441 patches per burst; and the dropped zero edge ring of the mosaic (`APODCROP` 21
instead of 20, see known issues). Cutoffs, 44 modes, patches and destretch are the same.

## What was found in the data (and how the scripts deal with it)

* The two cameras are **mirrored in y** with respect to each other. BB is flipped to the NB
  orientation at read time, so the NB products keep the orientation of the level-1 NB files.
* After the flip a **constant offset** remains (dy = +3, dx = -12 px, flipped BB vs NB),
  measured once per scan on the first step and applied as an integer shift of the BB window.
* A **static rotation and scale difference** remain between the cameras: ~0.36 deg and
  0.17-0.19 % (NB larger), plus ~0.4 px at the centre. Over the 1326 px v2 cube field NB and BB
  differ by ~3.4 px (median), 6-7 px at the corners, and by more than 1 px over almost the whole
  field. Not corrected: it does not affect the shared-wavefront fit, and each channel is fine
  on its own, but pixel-by-pixel NB/BB comparisons need this affine correction first.
* The 1408 px crop is **fully illuminated** in both channels (no unlit pixel; the field stop is
  at r ~ 1093 px). BB noise is uniform over it; NB has lower counts toward the top-right, so its
  noise per ADU is ~2.5x higher there than at the bottom-left.
* **Seeing**: science1 is good except for a dip at steps ~46-55; science2 is moderate to poor,
  worst at steps ~21-28, ~33-40 (the 6301.5 core), ~76-81 and ~85-88. In those steps the
  effective resolution is set by the seeing and no setting recovers it, so line profiles mix
  steps of different resolution. For quantitative spectral work prefer science1, or degrade
  the steps of a scan to a common resolution first.
* Different steps are different bursts: their reconstructions differ by ~1 px globally and by
  up to ~10-15 px of local seeing distortion. Per-step files therefore wobble when played as a
  movie; the cube assembly destretches all steps to a common geometry (see Outputs).

## The noise model (why v2)

v1 left a fine worm-like noise texture over both channels in the worst-seeing bursts. The cause:
the Wiener filter calibrates the noise level beyond the diffraction cutoff and, with
`filter_noise_shape: white`, assumes that level at all frequencies. The destretch applied to the
frames before MOMFBD interpolates them, which removes noise only above ~0.85 of the cutoff, so the
white model underestimates the noise power by x1.45 where the signal is. In bad seeing that lets
noise through. Narrower cutoffs and fewer modes do not remove it (they only move it).

`gris_tune.py` tested 13 settings on 5 steps (good, moderate and worst seeing; continuum and both
line cores) with a split-burst test: each burst is reconstructed from its even and its odd frames
separately, and what the two halves share is signal, the rest noise. Three independent judges
(error budget, visual inspection, spectral-analysis needs) and an adversarial check compared the
results. Chosen: `filter_noise_shape: auto` with `filter_noise_flat: 0.70` for both channels;
cutoffs and modes unchanged. Half-burst signal / noise in % rms contrast, from
`tuning/summary/summary.txt` (768 px crop, Hann window; bands in units of the diffraction cutoff):

| step | channel, band | v1 (white) | v2 (auto, flat below 0.70) |
|---|---|---|---|
| science1 40, good seeing | BB 0.3-0.5 | 1.53 / 0.82 | 1.41 / 0.75 |
| science1 40 | NB 0.3-0.5 | 0.93 / 0.64 | 0.79 / 0.48 |
| science1 40 | NB 0.5-0.7 | 0.23 / 0.42 | 0.11 / 0.12 |
| science2 40, worst seeing | NB 0.3-0.5 | 0.38 / 0.60 | 0.14 / 0.14 |
| science2 40 | NB 0.5-0.7 | 0.20 / 0.66 | 0.02 / 0.04 |
| science2 71, 6302.5 core | NB 0.3-0.5 | 0.33 / 0.55 | 0.15 / 0.20 |

Good steps keep ~90 % of their reproducible mid-band detail; in the worst steps the noise drops
3-15x. On the full 32-frame reconstructions the large-scale contrast and the mean level are the
same in v1 and v2. The `auto` default (flat below 0.25) over-smooths good seeing; flat below 0.85
still let noise through near the field edges, because of the warp issue listed below.

**What v2 does not fix.** In science2's worst steps (~22-27, ~33-41, ~76-78) the outer
~150-300 px of the 1408 px field, mostly the top-right and bottom-left corners, keep part of the
noise texture (the warp issue smooths the noise most at the corners, so the global noise shape
fits those patches least); the NB top-right is noisier throughout science2 and in science1's
seeing dip (steps ~50-52), where the NB counts are lowest. For quantitative work on those steps
use the central ~1000 px, as the observers also suggested. The split-burst tuning looked only at
a central 768 px crop. In the worst steps the filter strength also varies visibly from one 64 px
patch to the next.

**Destretch before MOMFBD.** Leaving out the destretch of the frames before MOMFBD (tested on
step 40 of both scans, 1408 px; the run without destretch used the white noise model, which fits
frames that were not interpolated) lowers the large-scale contrast (to 0.76-0.91 of the v2 result
in science2, 0.93-1.00 in science1) and reduces the noise only in two corners of the worst step,
so the destretch is kept. The test reconstructions were deleted; to repeat it:
`$PY gris_tune.py --crop_size 1408 --simultaneous_seq 441 --steps science2:40,science1:40
--configs m44_c85_auto_f70,m44_c85_white_nods --output_dir tuning_1408`.

**Moving texture in the aligned movies (not the noise model).** Played as a still, aligned movie,
the BB of every scan shows a texture that changes from step to step. BB sees the same scene at
every step (a scan lasts 45 s), so this is reconstruction noise: every step is an independent
32-frame burst, restored a little differently (~0.7 % rms contrast at 0.3-0.9", independent from
step to step). Movies of the per-step files hide it because the seeing distorts every burst
differently and the eye follows that motion; aligning the steps removes the motion and shows the
texture. The noise model does not cause it: on the same 896 px field at full resolution, v1 (white)
changes more from step to step than v2 (BB rms of consecutive-step differences 1.73 vs 1.55 % in
science1, 1.64 vs 1.23 % in science2). Below 0.3 of the cutoff (0.3-0.9"), which holds 56-75 % of
v2's change, the two are equal; above it v1 changes more. v2 only makes its finest detail come and
go a little more in bad seeing. The movies therefore show BB
as a running mean of 5 steps (see Outputs). For spectral work, the BB step-to-step scatter is a
direct estimate of the per-step reconstruction error at each position, also for NB.

**Texture in the umbrae.** At full resolution the umbrae (e.g. the large spot at x 5-8", y 30-38")
show a fine hair-like and grainy texture, most visible in single NB steps. Investigated on
2026-10-13:
* **Coarser than ~0.3": real.** It repeats from burst to burst; it is still there in the mean of
  all 100 BB steps and in science2, observed a minute later. It includes umbral dots, the light
  bridge and filament tips.
* **Finer grain: mostly noise.** It is burst-to-burst reconstruction error. In ADU it is as large
  as in the nearby quiet Sun, but it is ~2.5x larger relative to the umbral intensity, and no
  granulation hides it. About half of it at these scales is photon noise amplified by the
  deconvolution.
* **Not the stretching.** Step files and cube planes show the same texture, so the cube
  registration does not cause it. Switching off the destretch within each burst makes the noise
  28-32 % worse.
* **Configuration changes do not help.** Tested on GPU, on science1 steps 18-21 and 40:
  filter_threshold 0.5 gives -3 %, 27 modes -8 %, 96 px patches +16 %. Each acts on the whole
  field like a uniform blur, because the Wiener filter is set per 128 px patch.
* **The display made it worse.** The BB grey scale clipped ~10 % of the umbral pixels to black.
* **Optional smoothing (display only, off by default; the movies are not smoothed):**
  `gris_plot.py --smooth SIGMA` applies a Gaussian of that width to every pixel of both channels,
  BB after the 5-step mean. The display range is still computed from the unsmoothed frames.
  * With sigma = 1.5 px (FWHM 3.5 px = 0.11", a little coarser than the diffraction limit
    lambda/D = 0.09"), it keeps 71 % of the amplitude at 0.25 of the diffraction cutoff, 25 % at
    0.5 and 6.5 % at 0.7.
  * The grain disappears, and so does most of the real detail finer than ~0.2".

## Files

| file | job |
|---|---|
| `gris_common.py` | reads one step: centred crop, BB flip and offset, native float32 arrays, the ADU scale; measures the offset |
| `gris_momfbd.yaml` | v1 configuration: `hifi_momfbd.yaml` with the CRISP pixel scale, 128 px patches, 6302 A for both objects, the wideband cutoff [0.70, 0.85] for both channels (HiFi used [0.50, 0.65] for its NB) and no off-limb settings |
| `gris_momfbd_v2.yaml` | v2 configuration: v1 with the measured noise model |
| `gris_momfbd.py` | reconstructs one scan step by step (resumable), one FITS per step in ADU |
| `run_gris.sh` | runs science1 on GPU 0 and science2 on GPU 2 in the background, logs in `logs/` |
| `gris_make_cube.py` | stacks the steps of a scan into a wavelength cube, destretched to a common geometry |
| `gris_plot.py` | one PNG per step at full resolution (one image pixel per screen pixel): from the step files (raw vs reconstruction) or from the cube (reconstructions in the common geometry, optionally BB averaged over steps), and the movie |
| `gris_tune.py` | split-burst test of cutoffs, modes and noise model (`--summary` for the table and figures) |

## How to run (v2)

```bash
cd /dat/andreuva/gpu/torchmfbd/reproducibility/gris      # basis/ cache and logs/ live here
PY=/dat/andreuva/gpu/miniconda/envs/torchmfbd/bin/python

# 1. one step first, into a separate directory that can be deleted afterwards (this also builds
#    the KL basis if it is missing)
$PY gris_momfbd.py --scan science2 --gpu 0 --steps 36 --crop_size 1408 --simultaneous_seq 441 \
    --config gris_momfbd_v2.yaml --output_dir results_test

# 2. both scans, one per GPU (edit the scan:gpu pairs in the script). TAG names the logs and the
#    script refuses to overwrite existing logs, so to resume an interrupted run start it again
#    with a new TAG: finished steps are skipped.
TAG=v2 ./run_gris.sh --config gris_momfbd_v2.yaml --crop_size 1408 --simultaneous_seq 441 \
    --output_dir results_momfbd_v2

# 3. per scan: cube, full-resolution PNGs from the cube and the lossless movie through the scan,
#    then a standard-profile copy of the movie from the same PNGs (existing PNGs are reused)
$PY gris_make_cube.py --results_dir results_momfbd_v2/science1 --gpu 0 --border 20
$PY gris_plot.py --cube results_momfbd_v2/science1_cube.fits --bb_mean 5 --quiet_sun_box 35 40 35 40 \
    --output_dir plots_cube_v2/science1 --movie plots_cube_v2/science1_scan.mp4
$PY gris_plot.py --cube results_momfbd_v2/science1_cube.fits --bb_mean 5 --quiet_sun_box 35 40 35 40 \
    --output_dir plots_cube_v2/science1 --crf 1 --movie plots_cube_v2/science1_scan_compatible.mp4
```

PNGs that already exist in `--output_dir` are not redrawn (unless `--overwrite`), so use a new
directory (or `--overwrite`) when changing `--bb_mean`, `--quiet_sun_box` or `--smooth` or after
rebuilding the cube, or the movie would show the old frames.

`run_gris.sh` starts the reconstructions with `nohup`. If the shell that launches it may be killed
as a whole (a terminal manager cleaning up its jobs did this once, after step 72), start it in
its own session instead: `TAG=v2 setsid nohup ./run_gris.sh ... > logs/run.out 2>&1 &`.

`--simultaneous_seq 441` puts the whole 1408 px field in one batch, so every burst gets one
noise-shape estimate (42.6 GB of GPU memory, ~100 s per step on an H100, ~2.8 h per scan).
Mode counts: `psf/nmax_modes` must be a whole number of radial orders (2, 5, 9, 14, 20, 27, 35,
44, 54, ...); to use fewer modes keep it at 44 and pass `--modes N`. The cube script warns when
the destretch displacements approach `--border`; 20 is enough for the 1408 px field.

## Outputs

* `results_momfbd*/<scan>/stepNNN_<wavelength>A_momfbd.fits`: `WIDEBAND_RECONSTRUCTED` (BB) and
  `NARROWBAND_RECONSTRUCTED` (NB) in ADU. The mosaic starts at pixel (`APODCROP`, `APODCROP`) of
  the crop (`CROPY0`, `CROPX0`, `CROPSIZE`). v2 mosaics are 1366 px; v1 mosaics are 984 px and
  their outermost pixel ring is exactly zero. The header also carries the step (`SCANSTEP`,
  `WAVE`), the camera registration (`BBFLIPY`, `BBOFFY`, `BBOFFX`), the normalisation that was
  undone (`NBSCALE`, `BBSCALE`) and the MOMFBD settings. The rest of the primary header is the
  raw NB header, including the providers' placeholders (`WAVELNTH = 430`, `TARGET = 'tbd'`) and
  the NB-only `CAMERA`/`EXPTIME`.
* `results_momfbd_v2/<scan>_cube.fits` (1326 px, `CUBEBORD` 20): extensions `NB` and `BB`
  (100, ny, nx) with a linear air-wavelength
  WCS on axis 3 (`AWAV`, Angstrom), `WAVE`, and `SHIFTS` (per step: mean displacement applied,
  the plane being resampled at (y + DY, x + DX); largest local displacement; `NBSCALE`,
  `BBSCALE`, `PFENV`; NaN for a missing step). The planes are destretched to the average
  geometry of the scan on BB in three passes, and the same displacement field is applied to NB;
  the residual misregistration against the scan mean is ~0.05-0.1 px for most planes and
  ~0.3-0.9 px rms for the worst-seeing science2 planes (36-40; the exact value depends on the
  estimator, since tile shifts are noisy in those low-contrast planes). `CUBEBORD` pixels are dropped on every side,
  where the warp only has reflected pixels to work with. The primary header has `NBEXPT` and
  `BBEXPT` and no placeholders.
* `results_momfbd/<scan>_cube.fits` are the original v1 cubes, kept as they were. They have
  corrupted (partly negative) outer ~8 px, swapped `DY`/`DX` columns, and step 1's
  `NBSCALE`/`BBSCALE`/`PFENV`/`FLATFILE` in the primary header. To make a v1 cube with the
  current script (952 px): `mkdir -p cubes_v1 && $PY gris_make_cube.py --results_dir
  results_momfbd/science1 --gpu 0 --border 16 --output cubes_v1/science1_cube.fits`.
  `gris_make_cube.py` refuses to replace an existing cube unless `--overwrite` is given.
* `plots/<scan>/` and `plots/<scan>_scan.mp4`: v1 per-step figures with raw panels, as first made
  (downsampled; these movies wobble).
* `plots_cube_v2/<scan>/stepNNN_<wavelength>A_cube.png`: the v2 cube planes at full resolution,
  every cube pixel one image pixel (3104 x 1488 px, 8-bit grey), one display range per channel
  for the whole scan. BB is the mean of the 5 steps centred on each step (3 or 4 at the ends of
  the scan; the panel title gives the steps), NB is the single step. An inset in the top-right
  corner of the NB panel shows the mean NB spectrum of the quiet Sun at x, y = 35-40" (I/Ic,
  continuum = straight line through the first and last 3 steps), with an arrow at the wavelength
  on display, to make clear that the movie runs through wavelength, not time. The inset covers
  that box in NB, so the box is outlined on BB (`--quiet_sun_box`; PIPELINE.md section 5).
  The title gives the telescope, instrument and date (GREGOR / CRISP,
  2026-10-06) and the subtitle the step and wavelength. The colour bars are labelled in pixel
  counts, which are the ADU of the data.
* `plots_cube_v2/<scan>_scan.mp4`: those PNGs at 8 frames/s, H.264 lossless (every pixel equals
  the PNG; ~0.9-1 MB per frame, ~88-101 MB per movie). Lossless H.264 uses the High 4:4:4
  Predictive profile, which VLC, mpv and ffplay play but browsers, QuickTime and the VS Code
  preview do not; for those use `<scan>_scan_compatible.mp4` (standard High profile, level 5.0,
  within ~0.5 grey level rms of the PNGs, about the same size).
* `tuning/`: split-burst test results, one JSON per configuration and step (the half-burst
  reconstructions themselves were deleted after the tuning); `tuning/summary/summary.txt` and
  the `*_zoom.png` / `*_frc.png` figures. `gris_tune.py --summary` still rebuilds the table and
  the FRC figures from the JSON files.
* `logs/`: v1 run (`science1_gpu0.log`, `science2_gpu2.log`, `*_quality.npy`,
  `post_science*.log`); v2 run, which was killed externally after step 72 and resumed with the
  same settings, so its logs are `*_v2.log` (steps 1-72) and `*_v2b.log` (steps 73-100), plus
  `resume_v2.log`; the v2 cubes (`rebuild_v2_cube_*.log`) and the figures and movies
  (`plots_cube_v2*.log`). Steps 1-72 were written before the `APODCROP` header comment was
  corrected; the comment was then fixed in place (the value, 21, was right throughout).

## Known issues and possible follow-ups

* **torchmfbd warp** (`src/torchmfbd/warp.py`): the grid is normalised with `2v/(W-1) - 1` but
  sampled with `align_corners=False`, so even a zero displacement resamples every frame, at an
  offset running from -0.5 px at one edge to +0.5 px at the other (zero at the centre). It
  smooths the noise unevenly over the field (the cause of the leftover corner noise above) and
  rescales the destretched frames by W/(W-1) about the centre. Fix: `align_corners=True` (or
  normalise with `(2v+1)/W - 1`). It affects the HiFi pipeline too; after a fix the noise-model
  tuning should be repeated.
* **KL basis**: the last mode of every computed KL basis is zero (`kl_modes.py`), so "44 modes"
  are 43 effective modes. Same in both runs and in HiFi.
* **`PHASE_MODES_WB/NB`** extensions (written by `hifi_common.write_output`) contain the Fourier
  radius grid, not the fitted wavefronts.
* **Mosaic edge**: `unpatchify`'s blending weights vanish on the outermost pixel of every patch,
  so the outer pixel ring of a mosaic is zero. `gris_momfbd.py` drops it since v2; HiFi outputs
  have it too.
* **Not bit-reproducible**: two runs of the same step with the same settings differ by ~3 % of
  the image rms (BB 2.7 %, NB 3.7 %), mostly at small scales.
* **NB detector defects**: compact dots at fixed NB detector positions, amplified by the
  reconstruction and mostly visible in the dark line-core planes. As (row, col) in the 1408 px
  crop (detector = crop + 320): (660.5, 400.2) the strongest, with a companion at (663.9, 389.8);
  (633.7, 437.2); (413.8, 401.7); (486.2, 766.0); (730.7, 939.3). A calibration residual; they
  could be interpolated in the NB frames before the reconstruction.
* **NB fringes**: concentric rings ~30-40 px apart in the outer field, centred near the detector
  centre, present in the raw NB data (clear in science2, weak in science1, absent in BB) and
  passed through by MOMFBD; visible in the outer field of science2 line-core planes.
* **NB continuum slope**: the prefilter-corrected NB continuum is ~4 % brighter at the red end of
  the scan than at the blue end (in the cubes, with a straight continuum through both ends; 3.2 % from
  the raw `NBSCALE`) (BB is flat to < 1 %), the same in the raw data; normalise line
  profiles with a continuum fitted through both ends of the scan.
