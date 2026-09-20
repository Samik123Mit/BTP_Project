# Chirp HDF5 Enhancement Run

This is the data and reproduction guide for the **primary project**. See the [main README](README.md) for visuals and results and [PROJECT_WRITEUP.md](PROJECT_WRITEUP.md) for the formal problem statement. Earlier optical–thermal experiments are preserved as [additional work](additional_experiments/optical_thermal_benchmark/README.md), outside the assigned task.

## Current gap-reconstruction benchmark

The latest experiment directly measures recovery inside missing regions and includes an actual raster-image input path. Its full protocol is in [GAP_RECONSTRUCTION.md](GAP_RECONSTRUCTION.md).

```powershell
python -m pip install -r requirements-gap.txt
python run_chirp_gap_benchmark.py --data-root "C:\Users\DELL\Desktop\hs lit theiory\Chirp data"
python render_chirp_gap_report.py
```

Results are written to `outputs/chirp/gap_reconstruction/` in the repository, including numerical predictions, CSV measurements, gap close-ups, all-method sheets and a local HTML gallery. The [review notebook](notebooks/chirp_gap_review.ipynb) can display committed evidence without access to the original HDF5 files.

The following sections document the earlier image-filter and short-context CNN experiments, which remain available for comparison.

This document describes the run against the local data placed at:

```text
C:\Users\DELL\Desktop\hs lit theiory\Chirp data
```

## Source files read

| File | Dataset read | Shape |
|---|---|---:|
| `ORB3H_p1350_tr60_AM_9dB_v1.h5` | `/data` | 19,600 × 2,048 signed samples |
| `ORB3H_p1350_tr60_Ch_9dB_v1.h5` | `/data` | 19,600 × 2,048 signed samples |
| `ORB3H_p1350_tr60_v3.h5` | `/data` | 19,600 × 2,048 signed samples |

The script reads these HDF5 files without modifying them. It renders source rows `500` and `5000` from each file into reference waveform images, then evaluates restoration on controlled degraded copies. The recordings are not certified noise-free, and their interpretation as synchronized optical–thermal biomedical pairs has not been established.

## Degradation model

```text
clean waveform image
  → 4× downsampling and bicubic expansion
  → Gaussian blur
  → contrast / brightness loss
  → additive noise
  → two missing waveform-image regions
  → JPEG compression
```

## Restoration methods

- Conservative NLM and bilateral denoising
- Gaussian, median, bilateral, and NLM with CLAHE
- NLM + CLAHE + unsharp mask
- Mask-assisted Telea inpainting
- Blind image-trace interpolation

## Results

Outputs are at:

```text
Chirp_enhancement_results/
  featured_before_after/  # 6 direct clean → degraded → best-restored panels
  all_methods/            # all restoration attempts on every source trace
  individual_outputs/     # each image separately
  metrics_per_waveform.csv
  metrics_summary.csv
```

Average SSIM on the two selected rows:

| Source file | Degraded input | Best method | Best SSIM |
|---|---:|---|---:|
| AM 9 dB | 0.649 | NLM + CLAHE + unsharp | 0.739 |
| Chirp 9 dB | 0.657 | NLM + CLAHE + unsharp | 0.736 |
| v3 | 0.611 | Mask-assisted Telea inpainting | 0.718 |

These improvements use the global SSIM implementation in `src/pipeline.py` and are measured against the original rendered waveform image. Missing-region repair is an estimate; whole-image scores include backgrounds and labels.

Published snapshots are in [outputs/chirp/raster_enhancement](outputs/chirp/raster_enhancement/), including all six before/after panels, all six all-method sheets, and the two metric CSVs.

## Re-run

```powershell
cd C:\Users\DELL\biomedical-waveform-restoration
python run_chirp_enhancement.py
```

Install dependencies first with `python -m pip install -r requirements.txt`. On another machine, change the `DATA_ROOT` constant near the top of each Chirp runner to the folder containing the HDF5 files.

## Learned numerical waveform reconstruction

```powershell
python run_chirp_learned_restoration.py --source chirp_9db
python run_chirp_learned_restoration.py --source am_9db
```

Each command trains a separate model on CPU and writes results next to the input data:

```text
Chirp_enhancement_results/learned_signal_restoration/<source>/
  row_16000_signal_restoration.png
  row_17200_signal_restoration.png
  row_18400_signal_restoration.png
  row_19500_signal_restoration.png
  heldout_metrics.csv
  <source>_restorer.pt
```

Training uses 6,000 rows sampled from `0–14999`, with four evaluation rows (`16000`, `17200`, `18400`, `19500`) in the same recording. The residual 1D CNN receives the degraded signal and a valid-sample mask. Training runs for 14 epochs with seed `29`; the loss gives missing samples four times the weight of observed samples.

Signal corruption reduces 2,048 samples to 512, expands to 2,048 using linear interpolation, applies seven-sample averaging, compresses amplitude, adds noise, and removes two spans. Original traces are median-centered and scaled by their standard deviation before corruption. A naturally degraded input will require a separately validated normalization and mask-detection procedure.

| Source | Degraded RMSE | Linear baseline RMSE | Learned RMSE | Degraded correlation | Learned correlation |
|---|---:|---:|---:|---:|---:|
| Chirp 9 dB | 0.991 | 0.992 | 0.507 | 0.131 | 0.852 |
| AM 9 dB | 0.971 | 0.972 | 0.691 | 0.238 | 0.720 |

These are means across all four evaluated rows, in normalized units. The saved plots use independently scaled vertical axes. The results show improved overall signal agreement; they do not establish accurate recovery of every erased interval.

[Published Chirp plots and metrics](outputs/chirp/learned_signal_restoration/chirp_9db/) · [Published AM plots and metrics](outputs/chirp/learned_signal_restoration/am_9db/)

The first Chirp run saved its files directly in `learned_signal_restoration/`; the current runner uses the source subfolder. Its original evidence was copied into the published `chirp_9db` folder without changing the plots or metrics. AM evidence was copied from the `am_9db` subfolder. Re-running does not automatically overwrite the GitHub snapshots.

The runner supports `--source v3`, but a learned V3 result is not part of the published evidence. Raw HDF5 recordings and trained checkpoints are local; GitHub contains the scripts, result images, and metric CSVs.
