# Missing-region reconstruction on Chirp recordings

This experiment specifically asks whether estimated values inside an erased interval are closer to the original recording than a blank or interpolated interval. It extends the earlier image filters and short-context CNN.

## Why the earlier CNN left long gaps flat

The five convolutional layers in `run_chirp_learned_restoration.py` have kernel size 9 and dilations 1, 2, 4, 2, 1. Their receptive field is `1 + 8 × (1 + 2 + 4 + 2 + 1) = 81` samples. At the centre of a 180–190-sample erased interval, the model cannot see an observed value through those layers. A high whole-trace correlation therefore did not demonstrate recovery in the gaps.

The new benchmark reports missing-region error separately and includes models that use context outside the missing interval.

## Data and evaluation design

The three local source files are AM 9 dB, Chirp 9 dB and V3. Each `/data` array contains 19,600 traces of 2,048 samples. The same recording supplies disjoint row ranges:

| Purpose | Row range | Number actually used per source |
|---|---|---:|
| Fit waveform priors | 0–10999 | 1,800 evenly spaced rows |
| Select method | 12000–14599 | 24 evenly spaced rows |
| Evaluate selected method | 16000–19599 | 36 evenly spaced rows |

The 36 test traces are reused across five conditions. There are 108 distinct test traces across the three sources and 540 trace/condition cases, not 540 independent recordings.

Training uses a common scale calculated only from training rows: subtract the training median and divide by the median training-row standard deviation. Test targets are never used to normalize inference inputs. Identical raw traces across training/test are checked for and rejected. The split does not establish generalization to an unseen subject, scan or recording.

For every source and condition, the method with the smallest **mean validation gap RMSE** is saved before that condition's test values are read. All methods are also scored on the test data for transparency, but those test scores do not select the featured method.

The source recording is the reference before our added corruption. It is not certified noise-free ground truth.

### Experiment development record

The [first validation-only pilot](outputs/chirp/gap_pilot/) tested 13 global/baseline configurations on Chirp, using 1,200 training rows and 12 validation rows. The [expanded validation run](outputs/chirp/gap_validation/) added local fitting and tested 18 configurations on all three sources using 1,800 training and 24 validation rows. Neither pilot read test values.

The final benchmark adds actual raster-image inputs and evaluates the validation-selected methods on the separate test rows. Pilot metrics are retained as development records; their different sample counts should not be compared as though they were the same evaluation cohort.

## Corruption conditions

Intervals below use Python's half-open convention: `[start, end)` includes the start and excludes the end. They were fixed before test evaluation and do not depend on test peaks.

| Condition | Missing sample intervals | Additional damage |
|---|---|---|
| `short_gaps` | [1030, 1062), [1560, 1592) | None |
| `long_gaps` | [1010, 1106), [1512, 1640) | None |
| `mixed_damage` | [1020, 1084), [1540, 1604) | 2× numerical down/up sampling, Gaussian blur sigma 0.8, gain 0.8, noise sigma 0.04 |
| `severe_4x` | [1020, 1084), [1540, 1604) | 4× numerical down/up sampling, Gaussian blur sigma 1.2, gain 0.65, noise sigma 0.08 |
| `raster_image` | [1030, 1062), [1560, 1592) | Render an image, apply image blur sigma 0.6 and pixel noise sigma 2, erase columns, then extract the visible trace |

Numerical noise is in training-normalized signal units. Pixel noise is in 8-bit image units. Blur and resolution loss are applied before erasure; some neighbouring pixels/samples may therefore retain spread information from the original acquisition. The numerical priors receive the declared degradation operator, not an estimated unknown blur kernel.

## The 18 tested configurations

| Family | Configurations | Information used at inference |
|---|---|---|
| Baselines | Zero fill, linear interpolation, PCHIP interpolation | Observed trace and mask |
| Autoregression | Bidirectional order-24 local autoregression | Up to 256 observed samples on either side |
| Global low-rank prior | Rank 32/64/96 with ridge 0.1; rank 64 with ridge 0.01 or 1 | Training waveform basis fitted to visible test samples |
| Global template/dictionary | One template; combinations of 8, 24 or 48 templates | Training traces selected by observed-sample fit |
| Local low-rank prior | Rank 8, 16 or 32 | Training basis for each gap and 160 samples of context on either side |
| Local dictionary | Combinations of 8 or 24 templates | Local observed context and training traces |

The priors are learned from data, but they are not GANs or newly trained neural networks. This branch requires no PyTorch.

### How the low-rank estimate is calculated

Training traces are decomposed into a mean waveform and a small set of recurring waveform patterns. At inference, coefficients are chosen to match **only observed samples**, with a penalty that limits unstable coefficients. The fitted combination supplies the missing values.

For a basis `B`, mean `mu`, observed indices `O`, declared linear degradation `H`, and observation `y`, the global fit minimizes:

```text
|| (H B)[O] z - (y[O] - (H mu)[O]) ||² + lambda ||z||²

estimated waveform = mu + B z
```

Local fits restrict this procedure to a gap and its surrounding context. Dictionaries instead fit combinations of actual training traces chosen by agreement on visible samples. Matching similar traces does not mean retrieving the test reference; evaluation traces are excluded from the training bank.

A method can produce a plausible estimate that is wrong. In particular, an entirely erased event with weak surrounding information can be unidentifiable.

Background documentation: [PCA and matrix decomposition](https://scikit-learn.org/stable/modules/decomposition.html#principal-component-analysis-pca) and [SciPy least-squares fitting](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html). The masked-fitting and benchmark code in this repository is a project implementation.

## Direct image-input experiment

The raster branch actually reads an image, rather than secretly passing its original waveform to the restorer:

```text
Real HDF5 trace
  -> fixed-scale blue-trace image
  -> image blur + pixel noise + erased columns
  -> visible blue-pixel extraction
  -> observed amplitudes + known acquisition mask
  -> mask-aware reconstruction
  -> replace missing image columns with the estimated trace
```

`damage_raster` creates the input image. `extract_trace` accepts image pixels, the mask and the amplitude calibration; its API does not accept the reference waveform. The restored image preserves the damaged image outside known missing columns. Its missing columns contain a re-rendered estimate, not recovered original pixels.

This adapter assumes a known blue trace, axes calibration and plot geometry. Its amplitude range is determined from training data. Clipping, line rasterization, blur and digitization can introduce error. The numerical prior is an approximation for this image observation process; it is not an exact image-blur inverse. Arbitrary screenshots, unknown scales, overlapping curves and missing labels are not supported yet.

## How to read the results

The main score is `gap_rmse`: root-mean-square error on erased samples only. Also recorded are gap MAE, gap error normalized by reference gap RMS, visible-region error, whole-trace error and correlations.

`selected_test_summary.csv` compares the validation-selected method with zero fill and linear interpolation. A negative error reduction means the selected method made the gap worse. If zero fill wins validation, it is an explicit **no demonstrated recovery** result, not a successful enhancement.

For each source/condition, the visual report includes:

- Best, median-ranked and worst test examples, ranked retrospectively by per-trace gap-error reduction relative to zero fill.
- Full traces on common vertical axes and two close-ups covering the missing intervals.
- A fixed first-test-row comparison of all 18 configurations.
- For raster examples, original/input/enhanced PNG files and a matching before/after crop.

Every test measurement is retained, including failures. The featured example is not an estimate of average performance.

## Reproduce

From the repository root:

```powershell
python -m pip install -r requirements-gap.txt
python -m unittest discover -s tests -v
python run_chirp_gap_benchmark.py --data-root "C:\Users\DELL\Desktop\hs lit theiory\Chirp data"
python render_chirp_gap_report.py
python verify_chirp_gap_results.py
```

Use `--data-root` for a different laptop. `--output` chooses a separate result directory. Training and inference run on CPU. The exact row indices, normalization, source-training checksums, code checksums and configuration are recorded in `manifest.json`. Floating-point/library differences can cause small changes.

To inspect results without the raw files, open `outputs/chirp/gap_reconstruction/index.html` locally or use [the review notebook](notebooks/chirp_gap_review.ipynb). Optional notebook rerunning is off by default. GitHub displays the PNG/CSV results directly; download the HTML gallery to use it locally.

`predictions.npz` contains reference, observed, restored, mask and row arrays for independent scoring. References in these result archives are evaluation evidence and must not be used as restoration inputs. `first_row_methods.npz` preserves all configurations' outputs for the fixed comparison row.

The verifier independently recomputes gap RMSE for all 540 selected predictions, checks validation-only method choices, disjoint row sets and source checksums, and checks that visible image columns were preserved. The unit tests additionally check that altering hidden samples/pixels cannot influence inference.

## Limitations and next experiments

The work demonstrates a controlled reconstruction workflow on the supplied recordings. It does not establish complete missing-data recovery, blind enhancement without calibration, paired optical–thermal biomedical validity, or estimation of density/Young's modulus.

Next priorities are recording-level holdouts, varied/random gap locations, unknown-mask estimation, separate amplitude/phase feature errors, and longer gaps with genuinely informative additional measurements. Increasing model complexity alone cannot supply evidence that an entirely absent event was reconstructed correctly.
