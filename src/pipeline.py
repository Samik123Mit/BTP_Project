"""Small, transparent image-restoration benchmark for paired optical/thermal videos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class DegradationConfig:
    downsample: int = 3
    blur_sigma: float = 1.3
    noise_sigma: float = 13.0
    contrast: float = 0.55
    brightness: float = -24.0
    dropout_fraction: float = 0.07
    jpeg_quality: int = 35
    seed: int = 7


def degrade(image: np.ndarray, cfg: DegradationConfig) -> np.ndarray:
    """Apply a fixed, documented combination of realistic acquisition artifacts."""
    rng = np.random.default_rng(cfg.seed)
    h, w = image.shape[:2]
    small = cv2.resize(image, (w // cfg.downsample, h // cfg.downsample), interpolation=cv2.INTER_AREA)
    x = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    x = cv2.GaussianBlur(x, (0, 0), cfg.blur_sigma)
    x = x.astype(np.float32) * cfg.contrast + cfg.brightness
    x += rng.normal(0, cfg.noise_sigma, x.shape)
    # Short horizontal missing-data bands emulate dropped lines/occlusion.
    for _ in range(max(1, int(h * cfg.dropout_fraction / 3))):
        y = int(rng.integers(0, h - 3))
        x[y:y + int(rng.integers(1, 4)), :, ...] = np.nan
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED) if ok else x


def _clahe(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(x)
    lab = cv2.cvtColor(x, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def restore_gaussian_clahe(x: np.ndarray) -> np.ndarray:
    return _clahe(cv2.GaussianBlur(x, (0, 0), 0.8))


def restore_nlm_clahe(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        den = cv2.fastNlMeansDenoising(x, None, 10, 7, 21)
    else:
        den = cv2.fastNlMeansDenoisingColored(x, None, 8, 8, 7, 21)
    return _clahe(den)


def restore_nlm_only(x: np.ndarray) -> np.ndarray:
    """Conservative NLM denoising; avoids contrast amplification on noisy images."""
    if x.ndim == 2:
        return cv2.fastNlMeansDenoising(x, None, 7, 7, 21)
    return cv2.fastNlMeansDenoisingColored(x, None, 6, 6, 7, 21)


def restore_median_clahe(x: np.ndarray) -> np.ndarray:
    return _clahe(cv2.medianBlur(x, 3))


def restore_bilateral_clahe(x: np.ndarray) -> np.ndarray:
    return _clahe(cv2.bilateralFilter(x, 7, 40, 40))


def restore_bilateral_only(x: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(x, 7, 35, 35)


def restore_unsharp_nlm(x: np.ndarray) -> np.ndarray:
    base = restore_nlm_clahe(x)
    blur = cv2.GaussianBlur(base, (0, 0), 1.3)
    return cv2.addWeighted(base, 1.35, blur, -0.35, 0)


RESTORERS = {
    'bicubic_only': lambda x: x,
    'nlm_only_conservative': restore_nlm_only,
    'bilateral_only_conservative': restore_bilateral_only,
    'gaussian_clahe': restore_gaussian_clahe,
    'median_clahe': restore_median_clahe,
    'bilateral_clahe': restore_bilateral_clahe,
    'nlm_clahe': restore_nlm_clahe,
    'nlm_clahe_unsharp': restore_unsharp_nlm,
}


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    return float(cv2.PSNR(reference, test))


def ssim_global(reference: np.ndarray, test: np.ndarray) -> float:
    """Global SSIM; intentionally dependency-light for student reproducibility."""
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    b = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY) if test.ndim == 3 else test
    a, b = a.astype(float), b.astype(float)
    c1, c2 = 6.5025, 58.5225
    ma, mb = a.mean(), b.mean()
    va, vb, cab = a.var(), b.var(), ((a-ma)*(b-mb)).mean()
    return float(((2*ma*mb+c1)*(2*cab+c2))/((ma*ma+mb*mb+c1)*(va+vb+c2)))


def rms_contrast(x: np.ndarray) -> float:
    """RMS contrast after gray conversion; higher is not automatically better."""
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x
    return float(gray.astype(float).std())


def waveform_from_frames(frames: Iterable[np.ndarray], modality: str) -> np.ndarray:
    """Simple transparent baseline: mean intensity in a central skin/face ROI per frame."""
    values = []
    for f in frames:
        h, w = f.shape[:2]
        roi = f[int(.25*h):int(.78*h), int(.28*w):int(.72*w)]
        if modality == 'optical' and roi.ndim == 3:
            values.append(float(roi[:, :, 1].mean()))  # green channel is common rPPG baseline
        else:
            values.append(float(roi.mean()))
    x = np.asarray(values)
    return signal.detrend(x)


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = signal.detrend(a), signal.detrend(b)
    return float(np.corrcoef(a, b)[0, 1])


def estimate_bpm(x: np.ndarray, fps: float) -> float:
    # Zero-padding improves peak interpolation for a short demo clip; it does not add information.
    freqs, power = signal.periodogram(signal.detrend(x), fs=fps, nfft=max(4096, len(x)))
    band = (freqs >= 0.7) & (freqs <= 3.5)
    return float(freqs[band][np.argmax(power[band])] * 60)


def make_demo_pair(n_frames: int = 180, size: Tuple[int, int] = (192, 192), fps: int = 30):
    """Create clearly-labelled *synthetic* paired frames and a known 72 BPM reference.

    This is a smoke-test/demo only. The same pipeline accepts iBVP frames when licensed.
    """
    h, w = size
    yy, xx = np.mgrid[:h, :w]
    face = ((xx-w/2)/(w*.31))**2 + ((yy-h*.49)/(h*.40))**2 < 1
    t = np.arange(n_frames) / fps
    bvp = np.sin(2*np.pi*1.2*t) + .22*np.sin(2*np.pi*2.4*t+.4)
    optical, thermal = [], []
    rng = np.random.default_rng(10)
    for p in bvp:
        rgb = np.zeros((h,w,3), np.uint8)
        rgb[:] = (30, 25, 20)
        rgb[face] = (100, 145 + int(5*p), 185)
        cv2.circle(rgb, (72, 86), 7, (35, 45, 60), -1); cv2.circle(rgb, (120, 86), 7, (35,45,60),-1)
        cv2.ellipse(rgb, (96, 130), (27, 10), 0, 5, 175, (45,55,90), 2)
        rgb = np.clip(rgb.astype(float)+rng.normal(0,1.5,rgb.shape),0,255).astype(np.uint8)
        temp = np.full((h,w), 55, np.uint8)
        temp[face] = np.clip(160 + 8*p + ((yy[face]-h*.49)*-.12),0,255)
        temp = cv2.GaussianBlur(temp, (0,0), 1.2)
        optical.append(rgb); thermal.append(temp)
    return np.asarray(optical), np.asarray(thermal), bvp


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
