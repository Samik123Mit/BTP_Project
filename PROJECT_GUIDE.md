# Project guide (no AI background needed)

## The one-sentence project

â€œI am testing whether restoration of blurry/noisy optical and thermal images keeps the **scientifically useful signal** more reliable, not merely whether the image looks sharper.â€

## The problem in everyday language

Think of a thermal or optical camera like a phone camera taking an image through a dirty window. Blur, noise, low contrast, compression, and missing lines can hide small intensity changes. In biomedical video, tiny changes can form a waveform; one example is a blood-volume-pulse (BVP) waveform, from which heart rate may be estimated.

Usually we do not know the perfect image in a hospital/field setting. To test methods honestly, this project starts with a known clean frame and deliberately damages a copy. The untouched image is kept hidden while the damaged image is restored. We can then compare the restoration to the original. This is called a **controlled degradation experiment**.

## What is â€œopticalâ€ and what is â€œthermalâ€ here?

- Optical/RGB: ordinary visible-light camera images. The green channel often carries a useful pulse-related intensity signal.
- Thermal: infrared images where brightness represents emitted infrared intensity (and, for a calibrated radiometric device, can be related to temperature).
- Waveform: one number per frame from a selected region. Over time, those numbers form a line graph.

The included demo uses simulated paired frames so it runs for anyone. The recommended follow-up dataset is iBVP, which has real synchronized RGB/thermal videos and ear-PPG reference waveforms.

## What happens in one run

```text
clean paired frames
       |
       v
artificial damage: lower resolution + blur + low contrast + noise + missing rows + JPEG compression
       |
       +-----------------------+------------------------+------------------------+
       |                       |                        |
Gaussian + CLAHE       Non-local means + CLAHE   Non-local means + CLAHE + sharpen
       |                       |                        |
       +-----------------------+------------------------+
                               |
                               v
Compare with hidden clean reference + plot the waveform from the same ROI
```

## What each method means

- **Gaussian blur:** gentle averaging to reduce random grain/noise. It may also smooth fine detail.
- **CLAHE:** â€œadaptive contrastâ€ â€” it improves local visibility in dark/flat regions. Too much can make noise look like detail.
- **Non-local means (NLM):** looks for similar-looking patches elsewhere before averaging, often retaining edges better than simple blur. It is slower.
- **Unsharp mask:** boosts edges by subtracting a blurred version. It can make an image look impressive while creating misleading edge halos, so it must be scored, not trusted visually.

## What the numbers mean

| Number | Plain-English interpretation | Desired direction |
|---|---|---|
| PSNR (dB) | How close pixels are to the hidden clean image | Higher |
| SSIM (0â€“1) | Similarity of brightness, contrast, and structure | Higher; 1 is identical |
| MAE | Average absolute pixel error | Lower |
| RMS contrast | Amount of brightness variation | Context-dependent: higher can also mean amplified noise |
| BVP correlation (-1â€“1) | Agreement between the extracted waveform and known reference | Closer to +1 |
| Estimated BPM | Dominant pulse rate from a waveform | Compare with reference BPM |

**The most important lesson:** a higher contrast score is not automatically a better biomedical result. The restored image must also preserve waveform accuracy, temperature accuracy, or some external clinical measurement.

## Exactly what to click/run in a meeting

1. Open PowerShell in this repository.
2. Run `python -m pip install -r requirements.txt` once.
3. Run `python run_demo.py`.
4. Open `outputs/optical_comparison.png`, then `outputs/thermal_comparison.png`.
5. Open `outputs/waveform_comparison.png` and `outputs/metrics.csv`.
6. If you want an interactive explanation, run `jupyter notebook notebooks/demo.ipynb`.

The command uses fixed random seed 7; the same computer or another laptop should reproduce the same output figures and CSV values (small library-version differences are possible).

## 90-second speaking script

â€œI selected the iBVP dataset direction because it has aligned optical and thermal video, and crucially it includes a known PPG waveform. Since the full dataset requires an academic licence, this repository includes a fully reproducible simulated sample for a first demo. I make a clean image bad in six documented ways, then compare four non-AI restoration choices. The left panel is the reference, centre is the damaged input, right is the processed result. I do not call it successful just because it looks clearer. The table measures closeness to the hidden reference, and the graph checks whether the waveform survives. On the included controlled test, Gaussian plus adaptive contrast gives the highest SSIM for both modalities. My next step is to repeat exactly this protocol on held-out real iBVP participants and then compare it with a supervised deep-learning method.â€

## Questions you may be asked â€” honest answers

**â€œIs this AI?â€**  The current benchmark is classical explainable image processing, deliberately chosen as a transparent baseline. A neural method should only be added after baselines and a participant-level split are in place.

**â€œAre those clinical results?â€**  No. The committed output is a synthetic proof-of-workflow. Clinical claims require licensed real data, an external reference measurement, and appropriate approvals.

**â€œCan you calculate density or Youngâ€™s modulus from the restored image?â€**  Not directly. A restoration algorithm can improve inputs to a validated estimator, but material properties need ground-truth calibrationâ€”for example, elastography/mechanical testing paired to the images.

**â€œWhy intentionally damage good images?â€**  It is the only way to calculate objective reconstruction error when naturally degraded images lack a clean reference. It also lets us test each failure type separately.

## Next real-data protocol (write this in your report)

1. Obtain iBVP under its EULA; use only the permitted recordings.
2. Split by **participant** before tuning any parameters. Never let the same person's frames appear in train and test.
3. Hold out clean clips; produce synthetic degradations with registered severity settings.
4. Tune methods only on validation participants.
5. Report the test-participant mean and standard deviation for image, waveform, and heart-rate metrics.
6. Show failures: motion, occlusion, low perfusion, extreme temperature range, and sensor misalignment.
7. For thermal images, do not convert radiometric values through ordinary contrast enhancement before reporting temperature. Keep a calibrated numeric path separately.
