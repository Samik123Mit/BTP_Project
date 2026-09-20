"""Image-restoration benchmark using waveform images rendered from supplied Chirp HDF5 data.

The HDF5 recordings are read-only. Results are written to a separate
``Chirp_enhancement_results`` directory next to the data.
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import RESTORERS, psnr, rms_contrast, ssim_global

DATA_ROOT = Path(r"C:\Users\DELL\Desktop\hs lit theiory\Chirp data")
OUT = DATA_ROOT / "Chirp_enhancement_results"
FILES = {
    "am_9db": "ORB3H_p1350_tr60_AM_9dB_v1.h5",
    "chirp_9db": "ORB3H_p1350_tr60_Ch_9dB_v1.h5",
    "reference_v3": "ORB3H_p1350_tr60_v3.h5",
}
ROWS = (500, 5000)
W, H = 1600, 520


def render_waveform(signal: np.ndarray, label: str) -> np.ndarray:
    """Render one genuine HDF5 signal as a clean waveform image."""
    signal = signal.astype(np.float32)
    canvas = np.full((H, W, 3), 250, dtype=np.uint8)
    for gx in range(80, W, 160):
        cv2.line(canvas, (gx, 58), (gx, H - 62), (224, 224, 224), 1)
    for gy in range(80, H - 62, 80):
        cv2.line(canvas, (66, gy), (W - 30, gy), (224, 224, 224), 1)

    lo, hi = np.percentile(signal, [0.5, 99.5])
    y = H - 85 - (signal - lo) / max(hi - lo, 1e-6) * (H - 170)
    y = np.clip(y, 60, H - 63).astype(np.int32)
    x = np.linspace(66, W - 30, len(y)).astype(np.int32)
    points = np.column_stack((x, y)).reshape(-1, 1, 2)
    cv2.polylines(canvas, [points], False, (190, 75, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Chirp waveform | {label}", (66, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (35, 35, 35), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Source: original HDF5 trace (2048 samples)", (66, H - 23), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (78, 78, 78), 1, cv2.LINE_AA)
    return canvas


def degrade_waveform(image: np.ndarray, seed: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Fixed mixed corruption: 4x loss, blur, contrast loss, noise, holes, JPEG."""
    rng = np.random.default_rng(seed)
    small = cv2.resize(image, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    x = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
    x = cv2.GaussianBlur(x, (0, 0), 1.5).astype(np.float32)
    x = x * 0.56 + 90 + rng.normal(0, 10, x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    gaps = [(360 + seed % 80, 470 + seed % 80), (1040 + seed % 65, 1170 + seed % 65)]
    for left, right in gaps:
        x[58 : H - 62, left:right] = 250
        cv2.putText(x, "missing", (left + 8, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (132, 132, 132), 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 35])
    return (cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else x), gaps


def mask_assisted_inpaint(image: np.ndarray, gaps: list[tuple[int, int]]) -> np.ndarray:
    """Telea repair with a supplied acquisition-loss mask; output remains an estimate."""
    mask = np.zeros(image.shape[:2], np.uint8)
    for left, right in gaps:
        mask[58 : H - 62, left:right] = 255
    return cv2.inpaint(image, mask, 5, cv2.INPAINT_TELEA)


def trace_interpolation(image: np.ndarray) -> np.ndarray:
    """Blind trace reconstruction from observed blue waveform pixels.

    It infers one y-value per x-position and interpolates only internal gaps.
    The result is an image-based estimate, not recovery of unseen original values.
    """
    out = image.copy()
    b, g, r = cv2.split(image)
    trace = (b.astype(np.int16) > g.astype(np.int16) + 20) & (b.astype(np.int16) > r.astype(np.int16) + 40)
    y = np.full(W, np.nan)
    for column in range(W):
        candidates = np.flatnonzero(trace[58 : H - 62, column]) + 58
        if candidates.size:
            y[column] = np.median(candidates)
    valid = np.flatnonzero(np.isfinite(y))
    if valid.size > 10:
        model = PchipInterpolator(valid, y[valid], extrapolate=False)
        positions = np.arange(W)
        fill = (~np.isfinite(y)) & (positions > valid[0]) & (positions < valid[-1])
        y[fill] = model(positions[fill])
    points = np.column_stack((np.flatnonzero(np.isfinite(y)), y[np.isfinite(y)].astype(np.int32))).reshape(-1, 1, 2)
    cv2.polylines(out, [points], False, (190, 75, 20), 2, cv2.LINE_AA)
    return out


def draw_before_after(clean: np.ndarray, bad: np.ndarray, restored: np.ndarray, title: str, method: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, name, image in zip(axes, ["Clean source waveform", "Degraded waveform image", f"Best restored: {method}"], [clean, bad, restored]):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=14)
        ax.axis("off")
    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_all_methods(clean: np.ndarray, variants: dict[str, np.ndarray], title: str, path: Path) -> None:
    items = [("clean source", clean), *variants.items()]
    cols, rows = 3, int(np.ceil(len(items) / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4.8 * rows))
    for ax, (name, image) in zip(np.ravel(axes), items):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(name.replace("_", " "), fontsize=10)
        ax.axis("off")
    for ax in np.ravel(axes)[len(items) :]:
        ax.axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    featured = OUT / "featured_before_after"
    featured.mkdir(exist_ok=True)
    all_methods = OUT / "all_methods"
    all_methods.mkdir(exist_ok=True)
    individual = OUT / "individual_outputs"
    individual.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    for dataset_name, filename in FILES.items():
        source = DATA_ROOT / filename
        if not source.exists():
            raise FileNotFoundError(source)
        with h5py.File(source, "r") as h5:
            for row in ROWS:
                clean = render_waveform(h5["data"][row], f"{dataset_name}, row {row}")
                degraded, gaps = degrade_waveform(clean, row)
                methods: dict[str, np.ndarray] = {"degraded_input": degraded}
                methods.update({name: fn(degraded) for name, fn in RESTORERS.items() if name != "bicubic_only"})
                methods["mask_assisted_telea_inpaint"] = mask_assisted_inpaint(degraded, gaps)
                methods["blind_trace_interpolation"] = trace_interpolation(degraded)
                key = f"{dataset_name}_row{row}"
                cv2.imwrite(str(individual / f"{key}_00_clean.png"), clean)
                for name, image in methods.items():
                    cv2.imwrite(str(individual / f"{key}_{name}.png"), image)
                    rows.append(
                        {
                            "source_file": filename,
                            "row": row,
                            "method": name,
                            "PSNR_dB": psnr(clean, image),
                            "SSIM_global": ssim_global(clean, image),
                            "MAE_8bit": float(np.abs(clean.astype(float) - image.astype(float)).mean()),
                            "RMS_contrast": rms_contrast(image),
                        }
                    )
                best_name, best_image = max(methods.items(), key=lambda item: ssim_global(clean, item[1]))
                draw_before_after(clean, degraded, best_image, f"Chirp data | {dataset_name} | row {row}", best_name.replace("_", " "), featured / f"{key}_before_after.png")
                draw_all_methods(clean, methods, f"Chirp data restoration trials | {dataset_name} | row {row}", all_methods / f"{key}_all_methods.png")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics_per_waveform.csv", index=False)
    summary = metrics.groupby(["source_file", "method"], as_index=False)[["PSNR_dB", "SSIM_global", "MAE_8bit", "RMS_contrast"]].mean()
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    print(f"Saved {len(list(featured.glob('*.png')))} before/after panels and {len(list(all_methods.glob('*.png')))} all-method sheets to {OUT}")


if __name__ == "__main__":
    main()
