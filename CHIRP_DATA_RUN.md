# Chirp HDF5 Enhancement Run

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

The script reads these HDF5 files without modifying them. It renders genuine source rows `500` and `5000` from each file into clean waveform images, then evaluates restoration on controlled degraded copies.

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

These improvements are measured against the original clean rendered waveform image. Missing-region repair is an estimate, not recovery of known source data.

## Re-run

```powershell
cd C:\Users\DELL\biomedical-waveform-restoration
python run_chirp_enhancement.py
```
