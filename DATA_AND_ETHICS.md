# Data, licensing, and ethics

## Included material

The repository contains code and generated figures from a **synthetic** paired optical–thermal clip. No patient image, thermal measurement, or biometric record is bundled.

## Proposed external dataset

The real-data target is iBVP (Joshi & Cho, 2024), because it provides synchronized optical/RGB frames, thermal-infrared frames, an ear PPG waveform, and signal-quality labels. Access is academic-research-only through the authors’ EULA process; it is not an anonymous click-to-download corpus.

- Dataset access/readme: https://github.com/PhysiologicAILab/iBVP-Dataset
- Paper: https://doi.org/10.3390/electronics13071334

Before using it, obtain approval through the stated process, follow the EULA, protect data locally, and do not upload data or participant-derived results to a public repository unless the licence specifically permits it.

## Thermal-image safeguard

Display-oriented contrast enhancement changes pixel values. If thermal input is radiometric, preserve the original numeric array for temperature calculation. Use the enhanced render only to assist visual inspection/segmentation, unless a validation study proves the processing preserves the intended quantitative measurement.

## Claim boundary

This project is an image-quality and waveform-preservation experiment. It is not a diagnostic device, clinical decision tool, or validated density/Young’s-modulus estimator.
