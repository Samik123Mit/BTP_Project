# BTP Phase 1 — Optical–Thermal Image Restoration Benchmark

> A reproducible Phase 1 investigation of how image degradation affects paired optical and thermal imagery, and how classical restoration, inpainting, and neural super-resolution can recover usable visual structure for downstream feature extraction.

![Real thermal moderate degradation comparison](outputs/real_image_enhancement/00_thermal_moderate_comparison.png)

## Why this project

Biomedical optical and thermal images may be affected by limited sensor resolution, defocus or motion blur, low contrast, sensor noise, compression, and incomplete image regions. These effects can make later tasks—region identification, boundary detection, temperature-pattern analysis, waveform tracing, and feature extraction—less reliable.

This Phase 1 work addresses the **image-quality layer** first:

```text
Clean paired image  ->  controlled realistic degradation  ->  restoration alternatives
                                                         ->  visual + objective comparison
                                                         ->  candidate input for later feature extraction
```

The goal is not to claim a clinical diagnostic or material-property estimate. It is to establish a careful, repeatable enhancement baseline and demonstrate which processing choices preserve image structure.

## What was implemented

### 1. Real paired optical–thermal data

The benchmark uses the public **ULB17-VT** dataset. Every selected example contains an aligned optical RGB image, a high-resolution thermal image (320×240), and a low-resolution thermal image (80×60). The repository includes a compact visible subset of six real aligned image triplets under `data/samples/ulb17_vt_test/`; the 512 MB original download is intentionally excluded from Git.

Dataset: [ULB17-VT on Zenodo](https://doi.org/10.5281/zenodo.2557535) · Related work: Almasri & Debeir, *Multimodal Sensor Fusion in Single Thermal Image Super-Resolution* (2018).

### 2. Controlled degradation suite

Each clean image is transformed into several reproducible failure cases.

| Degradation | Simulated failure |
|---|---|
| Resolution reduction | Low-resolution sensor acquisition / undersampling |
| Gaussian blur | Defocus and optical blur |
| Motion blur | Motion or camera shake |
| Contrast reduction | Weak thermal/optical separation |
| Additive noise | Sensor noise |
| JPEG compression | Transmission or storage artifacts |
| Missing patches | Occlusion, dropped pixels, or incomplete acquisition |

Three profiles are run on every real image: `moderate`, `severe`, and `motion`.

### 3. Broad restoration comparison

No single method is assumed to be best. All results are retained, including those that underperform.

| Family | Methods implemented | Purpose |
|---|---|---|
| Baseline | Bicubic interpolation | Reference for low-resolution upscaling |
| Denoising + local contrast | Gaussian+CLAHE, Median+CLAHE, NLM+CLAHE | Reduce noise and reveal local structure |
| Edge-aware processing | Bilateral+CLAHE, NLM+CLAHE+unsharp mask | Retain / emphasize boundaries |
| Missing-data repair | Automatic-mask Telea inpainting | Fill detected missing regions |
| Neural super-resolution | Pre-trained EDSR ×2 | Exploratory deep-learning super-resolution comparison |
| Waveform-specific recovery | Blind trace interpolation | Recover a missing plotted trace from neighbouring visible trace points |

`CLAHE` is Contrast Limited Adaptive Histogram Equalization. `NLM` is Non-Local Means denoising. `EDSR` is a real pre-trained residual neural super-resolution model; it is **not** a GAN and is labelled exploratory because it was trained on generic natural images, not biomedical images.

### 4. Quantitative evaluation

Because the clean image is retained before degradation, every restoration can be objectively scored.

| Metric | Meaning | Better result |
|---|---|---|
| PSNR | Pixel-level closeness to the clean reference | Higher |
| SSIM | Brightness, contrast, and structural similarity (0–1) | Higher |
| MAE | Mean absolute pixel error | Lower |
| RMS contrast | Visible intensity variation | Context-dependent; higher can mean noise |

Visual sharpness alone is never treated as proof of improvement. A method that looks sharper but decreases SSIM/PSNR is kept and reported as an informative failure case.

## Results and evidence

The current run uses six real RGB–thermal samples × two modalities × three degradation profiles.

- **36** screenshot-ready comparison sheets
- **288** standalone clean / degraded / enhanced image outputs
- Per-image and aggregate CSV metrics
- Six original visible image triplets committed under `data/samples/`

| Example evidence | Link |
|---|---|
| Real RGB + HR thermal + LR thermal source | [open](outputs/ulb17_vt/00_source_pair.png) |
| Real thermal image with severe degradation and every method | [open](outputs/real_image_enhancement/00_thermal_severe_comparison.png) |
| Real optical image with motion degradation and every method | [open](outputs/real_image_enhancement/00_optical_motion_comparison.png) |
| Per-image metrics | [open](outputs/real_image_enhancement/metrics_per_image.csv) |
| Aggregate metrics | [open](outputs/real_image_enhancement/metrics_summary.csv) |

### Best observed Phase 1 outcomes

The best method is chosen separately for the modality and degradation—not by visual preference alone.

| Case | Best evaluated method | Mean SSIM: degraded -> restored |
|---|---|---:|
| Optical, moderate degradation | NLM + CLAHE + unsharp | 0.841 -> 0.856 |
| Optical, motion degradation | NLM + CLAHE + unsharp | 0.792 -> 0.817 |
| Optical, severe degradation | NLM + CLAHE + unsharp | 0.636 -> 0.739 |
| Thermal, moderate degradation | Conservative NLM | 0.865 -> 0.868 |
| Thermal, motion degradation | Conservative NLM | 0.845 -> 0.848 |
| Thermal, severe degradation | NLM + CLAHE | 0.681 -> 0.766 |

The gains on moderate thermal inputs are intentionally reported as small. This is a useful finding: when an input is already structurally good, aggressive processing should not be expected to create a dramatic or trustworthy change.

## Reproducibility

### Requirements

```powershell
python -m pip install -r requirements.txt
```

### Run the real-image enhancement benchmark

The compact real samples are already present. This command regenerates all clean/degraded/restored results.

```powershell
python run_real_image_enhancement.py
```

Generated files:

```text
outputs/real_image_enhancement/
  00_thermal_severe_comparison.png
  00_optical_motion_comparison.png
  ...                                   # 36 visual comparison sheets
  individual_outputs/                   # 288 standalone images
  metrics_per_image.csv
  metrics_summary.csv
```

### Rebuild the visible data subset from the original archive

```powershell
curl.exe -L --output data\ULB17-VT.pkl "https://zenodo.org/records/2557535/files/ULB17-VT.pkl?download=1"
python export_real_samples.py
```

### Run the 4× thermal super-resolution comparison

```powershell
python run_ulb17_benchmark.py
```

### Run waveform-image recovery experiments

```powershell
python run_waveform_image_benchmark.py
```

This additional controlled experiment creates paired optical and thermal waveform-image renders, applies blur/noise/lower resolution/missing trace segments, then evaluates filters, inpainting, blind trace interpolation, and EDSR. It demonstrates how image restoration can support later waveform feature extraction. The waveform renders are synthetic demonstrations, explicitly separate from the real-image benchmark.

## Repository layout

```text
data/samples/ulb17_vt_test/        # visible real RGB/thermal source images
src/pipeline.py                    # degradation-independent restorers and metrics
export_real_samples.py             # exports small real-image subset from public archive
run_real_image_enhancement.py      # main Phase 1 benchmark
run_ulb17_benchmark.py             # original LR thermal -> HR thermal 4× benchmark
run_waveform_image_benchmark.py    # missing-trace and waveform-image experiment
outputs/                           # figures and numerical evidence
DATA_AND_ETHICS.md                 # source, privacy, and radiometric-data safeguards
PROJECT_GUIDE.md                   # plain-language explanation
```

## Technical observations from Phase 1

1. Restoration is problem-dependent: one filter cannot be assumed to solve blur, noise, contrast loss, and missing data simultaneously.
2. Contrast enhancement can make an image more readable yet reduce fidelity to the clean reference. Therefore visual and numerical scoring are both required.
3. Generic inpainting can fill a hole but cannot guarantee that the recovered pixels represent true anatomy or thermal values.
4. Neural super-resolution is promising as an experiment, but a generic pre-trained model must not be presented as medically validated.
5. Raw radiometric thermal values must be retained for quantitative temperature analysis; contrast-enhanced images are display/segmentation aids unless separately validated.

## Next phase

The next step is to run this same protocol on a licensed biomedical RGB–thermal dataset with synchronized physiological ground truth, use subject-level separation, and evaluate downstream feature extraction against independent references. The image-restoration framework, output organization, and metrics are already in place.

## Responsible use

This code is for research and educational benchmarking. It is not a diagnostic device and does not directly infer density, Young’s modulus, or any clinical parameter. See [DATA_AND_ETHICS.md](DATA_AND_ETHICS.md).
