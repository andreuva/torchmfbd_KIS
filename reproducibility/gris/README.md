# MOMFBD reconstruction of the CRISP 630 nm scans (SAC "gris" dataset, 2026-10-06)

Reconstruction of the two spectral scans `science1` and `science2` taken with the CRISP
narrow-band (NB) and broad-band (BB) cameras at GREGOR, with the torchmfbd pipeline that was
validated on HiFI+ data (`../hifi/`). For every one of the 100 wavelength steps, the 32 NB
exposures and the 32 simultaneous BB exposures are reconstructed together: two objects,
one wavefront per frame. Nothing in `../hifi/` is modified; `reconstruct_burst()` and
`write_output()` are imported from `../hifi/hifi_common.py`.

Data: `/dat/andreuva/data/SAC_gris_data/{NB,BB}/{science1,science2}/corr_sd*.fits`, one
(32, 2048, 2048) float32 cube per step, same file name in NB and BB, dark/flat corrected
(NB also prefilter corrected), in ADU. 0.032"/px, 3.15 pm/step, Fe I 6301.5 core at step 39.

## What was found in the data (and how the scripts deal with it)

* The two cameras are **mirrored in y** with respect to each other. BB is flipped to the NB
  orientation at read time, so the NB products keep the orientation of the level-1 NB files.
* After the flip a **constant offset** remains (about dy = +3, dx = -12 px, flipped BB vs NB).
  It is measured once per scan on the first (continuum) step by cross-correlation and applied
  as an integer displacement of the BB crop window; the value is written to every header.
* A **static rotation of about 0.3 deg and a scale difference of 0.15 %** remain between the
  cameras (about 4 px at the corners of the 1024 px crop, 0.5 px rms residual). It is not
  corrected: it does not affect the shared-wavefront fit and the NB cube is self-consistent.
  Only NB-vs-BB comparisons far from the field centre see it.
* The field is a circle of radius ~1093 px; the default centred crop of 1024 px (as suggested
  by Thomas Berkefeld) is fully illuminated, and so is anything up to ~1400 px.
* Different steps are different bursts: their reconstructions differ by a global shift of
  about a pixel and by a local seeing distortion of several pixels. The cube assembly
  destretches all steps to the average geometry of the scan on BB and applies the same
  displacement field to NB.

## Files

| file | job |
|---|---|
| `gris_common.py` | reads one step: centred crop, BB flip and offset, native float32 arrays, the ADU scale; measures the offset |
| `gris_momfbd.yaml` | torchmfbd configuration: `hifi_momfbd.yaml` with the CRISP pixel scale, 128 px patches and 6302 A |
| `gris_momfbd.py` | reconstructs one scan step by step (resumable), one FITS per step in ADU |
| `run_gris.sh` | runs science1 on GPU 0 and science2 on GPU 2 in the background, logs in `logs/` |
| `gris_make_cube.py` | stacks the steps of a scan into a registered wavelength cube |
| `gris_plot.py` | one PNG per step: raw frame 0 next to the reconstruction, both channels |

## How to run

```bash
cd /dat/andreuva/gpu/torchmfbd/reproducibility/gris      # basis/ cache and logs/ live here
PY=/dat/andreuva/gpu/miniconda/envs/torchmfbd/bin/python

# 1. smoke test: three steps (continuum, 6301.5 core, last) into a separate directory, ~1 min each.
#    This also builds the KL basis basis/kl_144cm_128px_6302A_44.npz before two runs could race for it.
$PY gris_momfbd.py --scan science1 --gpu 0 --steps 1,39,100 --output_dir results_test
$PY gris_plot.py --results_dir results_test/science1 --output_dir plots_test/science1

# 2. both scans, one per GPU (edit the scan:gpu pairs in the script); re-running resumes
./run_gris.sh
tail -f logs/science1_gpu0.log

# 3. per scan: cube, quick-look PNGs, movie through the scan
$PY gris_make_cube.py --results_dir results_momfbd/science1 --gpu 0
$PY gris_plot.py --results_dir results_momfbd/science1 --output_dir plots/science1
$PY ../hifi/make_hifi_movie.py --images_dir plots/science1 --pattern "step*_momfbd.png" --fps 8 --output plots/science1_scan.mp4
```

Useful driver options: `--crop_size 1408` (larger field), `--steps 30-50` or `--steps 39`,
`--limit N`, `--no_resume`, `--offset DY DX` (skip the measurement), `--n_iterations`,
`--modes`. `--patch_size` must equal `images/n_pixel` in the yaml.

## Outputs

* `results_momfbd/<scan>/stepNNN_<wavelength>A_momfbd.fits`: `WIDEBAND_RECONSTRUCTED` (BB) and
  `NARROWBAND_RECONSTRUCTED` (NB), 984 x 984 px for the 1024 crop (the apodized 20 px border of
  every patch is dropped, so the mosaic starts at pixel (20, 20) of the crop: `APODCROP`), in ADU.
  The header carries the step (`SCANSTEP`, `WAVE`), the crop (`CROPSIZE`, `CROPY0`, `CROPX0`),
  the camera registration (`BBFLIPY`, `BBOFFY`, `BBOFFX`), the normalisation that was undone
  (`NBSCALE`, `BBSCALE`) and the MOMFBD settings.
* `results_momfbd/<scan>_cube.fits`: extensions `NB` and `BB` of shape (100, 984, 984) with a
  linear wavelength WCS on axis 3, `WAVE`, and `SHIFTS` (mean and largest displacement per step).
* `plots/<scan>/stepNNN_*.png` and `plots/<scan>_scan.mp4`.

## Settings worth revisiting after a first look

The cutoffs (`object1/object2: cutoff`), `psf/nmax_modes` and `s_u_joint` in the yaml are the
HiFi on-disk values; `hifi_momfbd.yaml` documents how they were chosen and what the symptoms
of a wrong value look like (grain in low-signal regions, seams). Steps take about a minute on
an H100 (225 patches of 128 px, 32 frames, 2 objects).
