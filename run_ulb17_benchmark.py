"""Professional multi-example report on real aligned RGB/thermal ULB17-VT data.

Downloads are intentionally separate: see README. This script never alters source data.
"""
from pathlib import Path
import pickle
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import RESTORERS, psnr, rms_contrast, ssim_global

ROOT = Path(__file__).parent
DATA = ROOT / 'data' / 'ULB17-VT.pkl'
OUT = ROOT / 'outputs' / 'ulb17_vt'
# Chosen across the official test split: fixed indices make reports repeatable.
SAMPLE_IDS = [0, 7, 14, 23, 31, 40]


def unit8(x, lo, hi):
    """Fixed reference-range mapping; display only, preserves raw arrays untouched."""
    return np.clip((x - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


def rgb_hwc(x):
    return np.transpose(np.clip(x, 0, 255).astype(np.uint8), (1, 2, 0))


def thermal_plot(ax, image, title):
    ax.imshow(image, cmap='inferno', vmin=0, vmax=255)
    ax.set_title(title, fontsize=9); ax.axis('off')


def main():
    if not DATA.exists():
        raise FileNotFoundError(f'Missing {DATA}. Follow README download instructions.')
    OUT.mkdir(parents=True, exist_ok=True)
    individual = OUT / 'individual_outputs'
    individual.mkdir(exist_ok=True)
    with DATA.open('rb') as f:
        (_, _, _), (_, _, _), (rgb, hr, lr) = pickle.load(f)
    rows = []
    for index in SAMPLE_IDS:
        raw_hr, raw_lr = hr[index, 0], lr[index, 0]
        # Use the high-resolution thermal range consistently for fair output comparisons.
        lo, hi = np.percentile(raw_hr, [1, 99])
        target = unit8(raw_hr, lo, hi)
        lr8 = unit8(raw_lr, lo, hi)
        up = cv2.resize(lr8, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_CUBIC)
        variants = {'bicubic_only': up}
        variants.update({name: fn(up) for name, fn in RESTORERS.items() if name != 'bicubic_only'})

        # Context sheet: visible-light and registered thermal source/target.
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        axes[0].imshow(rgb_hwc(rgb[index])); axes[0].set_title('Registered RGB optical'); axes[0].axis('off')
        thermal_plot(axes[1], target, 'Thermal HR reference')
        thermal_plot(axes[2], lr8, 'Thermal LR source (80 × 60)')
        thermal_plot(axes[3], up, 'Bicubic 4× to 320 × 240')
        fig.suptitle(f'ULB17-VT test example {index}: real paired RGB–thermal data', fontsize=14)
        fig.tight_layout(); fig.savefig(OUT / f'{index:02d}_source_pair.png', dpi=180); plt.close(fig)

        # A separate, screenshot-ready sheet preserves every attempted method.
        fig, axes = plt.subplots(1, 2 + len(variants), figsize=(3.05 * (2 + len(variants)), 3.5))
        thermal_plot(axes[0], target, 'HR reference')
        thermal_plot(axes[1], up, 'LR → bicubic')
        for ax, (name, image) in zip(axes[2:], variants.items()):
            thermal_plot(ax, image, name.replace('_', '\n'))
        fig.suptitle(f'ULB17-VT example {index}: 4× thermal super-resolution / enhancement', fontsize=14)
        fig.tight_layout(); fig.savefig(OUT / f'{index:02d}_all_methods.png', dpi=180); plt.close(fig)

        for name, image in variants.items():
            # Standalone method output: useful for one-method-at-a-time inspection/screenshots.
            fig, ax = plt.subplots(figsize=(7, 5.3))
            thermal_plot(ax, image, f'ULB17-VT example {index} | {name.replace("_", " ")}')
            fig.tight_layout(); fig.savefig(individual / f'{index:02d}_{name}.png', dpi=180); plt.close(fig)
            rows.append({'test_example':index, 'method':name,
                         'PSNR_dB':psnr(target, image), 'SSIM_global':ssim_global(target, image),
                         'MAE_8bit':float(np.mean(np.abs(target.astype(float)-image.astype(float)))),
                         'RMS_contrast':rms_contrast(image)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'metrics_per_example.csv', index=False)
    summary = metrics.groupby('method', as_index=False)[['PSNR_dB','SSIM_global','MAE_8bit','RMS_contrast']].agg(['mean','std'])
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns.to_flat_index()]
    summary = summary.sort_values('SSIM_global_mean', ascending=False)
    summary.to_csv(OUT / 'metrics_summary.csv', index=False)
    print('Saved', len(SAMPLE_IDS), 'real-data source sheets and method comparison sheets to', OUT)
    print(summary.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
