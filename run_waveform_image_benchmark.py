"""Multi-example waveform IMAGE enhancement benchmark.

Creates paired clean optical/thermal waveform images, then simulates lower resolution,
blur, noise, contrast loss and missing trace sections. All attempts are saved, including
methods that fail to reconstruct a missing trace. No AI model is hidden in this script.
"""
from pathlib import Path
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import (RESTORERS, psnr, rms_contrast, ssim_global)

ROOT = Path(__file__).parent
OUT = ROOT / 'outputs' / 'waveform_images'
W, H = 1400, 440
SAMPLES = [101, 202, 303, 404, 505, 606]
EDSR_MODEL = ROOT / 'models' / 'EDSR_x2.pb'


def clean_waveform(seed: int, modality: str):
    """Clean, labelled waveform-image ground truth; same physiological event, two sensors."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 12, W)
    rate = 1.08 + rng.uniform(-.12, .12)
    # A PPG-like pulse: narrow systolic peak + small dicrotic feature; varying amplitude.
    phase = (t * rate) % 1.0
    beat = np.exp(-((phase-.18)/.065)**2) + .33*np.exp(-((phase-.48)/.10)**2)
    drift = .08*np.sin(2*np.pi*.08*t + rng.uniform(0, 6.28))
    y = beat + drift
    if modality == 'thermal':
        y = .78*y + .10*np.sin(2*np.pi*.22*t) # lower-amplitude sensor appearance
        color = (42, 82, 235)                  # BGR red
        subtitle = 'Thermal waveform image (simulated infrared sensor render)'
    else:
        color = (205, 88, 22)                  # BGR blue
        subtitle = 'Optical waveform image (simulated RGB sensor render)'
    canvas = np.full((H, W, 3), 250, np.uint8)
    for gx in range(80, W, 100): cv2.line(canvas, (gx, 55), (gx, H-55), (224,224,224), 1)
    for gy in range(70, H-55, 70): cv2.line(canvas, (70, gy), (W-35, gy), (224,224,224), 1)
    yy = (H-85 - (y-y.min())/(y.max()-y.min())*(H-155)).astype(int)
    points = np.column_stack((np.arange(W), yy)).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, color, 3, cv2.LINE_AA)
    cv2.putText(canvas, 'Ground-truth waveform', (70, 35), cv2.FONT_HERSHEY_SIMPLEX, .85, (35,35,35), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (70, H-18), cv2.FONT_HERSHEY_SIMPLEX, .46, (75,75,75), 1, cv2.LINE_AA)
    return canvas, yy, color


def degrade_waveform(clean: np.ndarray, seed: int):
    """Known test degradation: 2x resolution loss, blur, low contrast, noise and missing sections."""
    rng = np.random.default_rng(seed + 999)
    small = cv2.resize(clean, (W//2, H//2), interpolation=cv2.INTER_AREA)
    x = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    x = cv2.GaussianBlur(x, (0,0), 1.4).astype(float)
    x = np.clip((x - 128)*.58 + 170 + rng.normal(0, 9, x.shape), 0, 255).astype(np.uint8)
    gaps = [(250 + seed % 100, 335 + seed % 100), (780 + seed % 70, 900 + seed % 70)]
    for left, right in gaps:
        # White missing bands erase part of waveform and grid, similar to lost tiles/corruption.
        x[55:H-50, left:right] = 250
        cv2.putText(x, 'missing', (left+8, 82), cv2.FONT_HERSHEY_SIMPLEX, .42, (130,130,130), 1, cv2.LINE_AA)
    ok, enc = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 36])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else x, gaps


def mask_assisted_inpaint(x, gaps):
    """Classical Telea inpainting when a missing-area mask is supplied by acquisition metadata."""
    mask = np.zeros(x.shape[:2], np.uint8)
    for l, r in gaps: mask[55:H-50, l:r] = 255
    return cv2.inpaint(x, mask, 5, cv2.INPAINT_TELEA)


def waveform_aware_reconstruction(x, expected_color):
    """Blind, explainable recovery: find coloured trace per column; interpolate missing columns.

    It redraws only the waveform trace and is intentionally kept separate from generic filters.
    """
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV)
    # Saturated coloured trace is distinguished from grey grid/background.
    mask = (hsv[:,:,1] > 40) & (hsv[:,:,2] > 45)
    y = np.full(W, np.nan)
    for col in range(W):
        candidates = np.flatnonzero(mask[55:H-50, col]) + 55
        if candidates.size: y[col] = np.median(candidates)
    valid = np.flatnonzero(~np.isnan(y))
    if valid.size >= 8:
        # Interpolate only bounded holes; never invent outside the observed time range.
        interp = PchipInterpolator(valid, y[valid], extrapolate=False)
        missing = np.isnan(y)
        bounded = missing & (np.arange(W) > valid[0]) & (np.arange(W) < valid[-1])
        y[bounded] = interp(np.flatnonzero(bounded))
    out = x.copy()
    finite = np.flatnonzero(~np.isnan(y))
    # Remove large 'missing' annotation area then put a reconstructed trace on top.
    for col in finite:
        cv2.circle(out, (int(col), int(y[col])), 1, expected_color, -1, cv2.LINE_AA)
    return out


def edsr_x2_from_lr(x):
    """Actual pre-trained EDSR neural super-resolution inference, not a GAN.

    The model was trained on natural images; therefore it is an exploratory baseline,
    not a validated biomedical model. It runs on representative cases by default.
    """
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(str(EDSR_MODEL))
    sr.setModel('edsr', 2)
    low_resolution = cv2.resize(x, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    return sr.upsample(low_resolution)


def label(image, text):
    out = image.copy()
    cv2.rectangle(out, (8, 8), (8 + min(560, 11*len(text)), 41), (255,255,255), -1)
    cv2.putText(out, text, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, .62, (20,20,20), 2, cv2.LINE_AA)
    return out


def write_sheet(path, title, items):
    cols = 2
    rows = int(np.ceil(len(items)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.9*rows))
    axes = np.ravel(axes)
    for ax, (name, im) in zip(axes, items):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(name, fontsize=12); ax.axis('off')
    for ax in axes[len(items):]: ax.axis('off')
    fig.suptitle(title, fontsize=17); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    individual = OUT / 'individual_outputs'; individual.mkdir(exist_ok=True)
    rows = []
    for seed in SAMPLES:
        for modality in ('optical', 'thermal'):
            clean, _, color = clean_waveform(seed, modality)
            damaged, gaps = degrade_waveform(clean, seed)
            methods = {'bicubic_degraded': damaged}
            methods.update({name: fn(damaged) for name, fn in RESTORERS.items() if name != 'bicubic_only'})
            methods['mask_assisted_telea_inpaint'] = mask_assisted_inpaint(damaged, gaps)
            methods['blind_waveform_interpolation'] = waveform_aware_reconstruction(damaged, color)
            # CPU EDSR is intentionally retained as a real neural-network comparison.
            # Default: two representative cases; set RUN_EDSR_ALL=1 for all 12 cases.
            if EDSR_MODEL.exists() and (seed == SAMPLES[0] or os.getenv('RUN_EDSR_ALL') == '1'):
                methods['pretrained_edsr_x2_neural_sr'] = edsr_x2_from_lr(damaged)
            prefix = f'{seed}_{modality}'
            cv2.imwrite(str(individual / f'{prefix}_00_clean.png'), clean)
            cv2.imwrite(str(individual / f'{prefix}_01_degraded.png'), damaged)
            for number, (name, im) in enumerate(methods.items(), start=2):
                cv2.imwrite(str(individual / f'{prefix}_{number:02d}_{name}.png'), label(im, name.replace('_', ' ')))
                rows.append({'sample':seed, 'modality':modality, 'method':name,
                             'PSNR_dB':psnr(clean, im), 'SSIM_global':ssim_global(clean, im),
                             'MAE_8bit':float(np.abs(clean.astype(float)-im.astype(float)).mean()),
                             'RMS_contrast':rms_contrast(im)})
            write_sheet(OUT / f'{prefix}_all_methods.png', f'{modality.title()} waveform image | sample {seed}',
                        [('Clean ground truth', clean), ('Degraded input', damaged), *[(k.replace('_',' '),v) for k,v in methods.items()]])
    table = pd.DataFrame(rows)
    table.to_csv(OUT / 'metrics_per_waveform_image.csv', index=False)
    summary = table.groupby(['modality','method'], as_index=False)[['PSNR_dB','SSIM_global','MAE_8bit','RMS_contrast']].mean()
    summary.to_csv(OUT / 'metrics_summary.csv', index=False)
    print(f'Created {len(SAMPLES)*2} waveform comparison sheets and {len(list(individual.glob("*.png")))} individual PNG files in {OUT}')
    print(summary.sort_values(['modality','SSIM_global'], ascending=[True,False]).round(3).to_string(index=False))


if __name__ == '__main__': main()
