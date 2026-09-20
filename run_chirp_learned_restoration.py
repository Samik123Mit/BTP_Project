"""Data-driven denoising and missing-segment reconstruction on real Chirp HDF5 traces.

This script uses a within-file row split: 6,000 rows sampled from 0..14999 train
the model; rows 16000, 17200, 18400 and 19500 are used for evaluation.
Original traces are normalized before synthetic corruption. The target signal
is not a model input during prediction; the corrupted trace and mask are inputs.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import random

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DATA_ROOT = Path(r"C:\Users\DELL\Desktop\hs lit theiory\Chirp data")
SOURCES = {
    "chirp_9db": "ORB3H_p1350_tr60_Ch_9dB_v1.h5",
    "am_9db": "ORB3H_p1350_tr60_AM_9dB_v1.h5",
    "v3": "ORB3H_p1350_tr60_v3.h5",
}
SEED = 29
LENGTH = 2048


class SignalRestorer(nn.Module):
    """Small residual 1D CNN: corrupted signal + valid-sample mask -> clean signal."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 24, 9, padding=4),
            nn.GELU(),
            nn.Conv1d(24, 32, 9, padding=8, dilation=2),
            nn.GELU(),
            nn.Conv1d(32, 32, 9, padding=16, dilation=4),
            nn.GELU(),
            nn.Conv1d(32, 24, 9, padding=8, dilation=2),
            nn.GELU(),
            nn.Conv1d(24, 1, 9, padding=4),
        )

    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return signal + self.net(torch.cat([signal, mask], dim=1))


def corruption(clean: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Known realistic degradation applied only to a copy of a clean real trace."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    low = F.interpolate(clean, size=LENGTH // 4, mode="linear", align_corners=False)
    x = F.interpolate(low, size=LENGTH, mode="linear", align_corners=False)
    x = F.avg_pool1d(x, kernel_size=7, stride=1, padding=3)
    x = 0.62 * x + 0.10 * torch.randn(x.shape, generator=generator)
    mask = torch.ones_like(x)
    batch = x.shape[0]
    for i in range(batch):
        # Different deterministic missing spans for every training/test trace.
        offset = int(torch.randint(0, 140, (1,), generator=generator))
        for left, right in ((320 + offset, 500 + offset), (1240 - offset // 2, 1430 - offset // 2)):
            x[i, :, left:right] = 0.0
            mask[i, :, left:right] = 0.0
    return x, mask


def normalized_rows(data: np.ndarray) -> np.ndarray:
    x = data.astype(np.float32)
    x -= np.median(x, axis=1, keepdims=True)
    scale = np.std(x, axis=1, keepdims=True) + 1e-6
    return x / scale


def linear_gap_baseline(signal: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Interpolation baseline using only visible samples and the supplied loss mask."""
    y = signal.copy()
    positions = np.arange(len(y))
    good = np.flatnonzero(mask > 0.5)
    y[mask <= 0.5] = np.interp(positions[mask <= 0.5], positions[good], y[good])
    return y


def signal_metrics(target: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((target - estimate) ** 2))),
        "mae": float(np.mean(np.abs(target - estimate))),
        "correlation": float(np.corrcoef(target, estimate)[0, 1]),
    }


def plot_result(clean: np.ndarray, degraded: np.ndarray, baseline: np.ndarray, restored: np.ndarray, row: int, path: Path) -> None:
    x = np.arange(LENGTH)
    fig, axes = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
    items = [
        ("Clean held-out Chirp trace (ground truth used only for evaluation)", clean, "#1558c0"),
        ("Degraded input: blur + 4× resolution loss + noise + missing segments", degraded, "#bb3e03"),
        ("Classical linear gap-fill baseline", baseline, "#6a4c93"),
        ("Learned 1D restoration: denoising + reconstruction", restored, "#16803a"),
    ]
    for ax, (title, y, color) in zip(axes, items):
        ax.plot(x, y, color=color, linewidth=0.8)
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Sample index")
    fig.suptitle(f"Real Chirp HDF5 held-out row {row}: signal-domain restoration", fontsize=16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(source_key: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    source = DATA_ROOT / SOURCES[source_key]
    out = DATA_ROOT / "Chirp_enhancement_results" / "learned_signal_restoration" / source_key
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as h5:
        # Training and test row ranges never overlap.
        train_rows = np.linspace(0, 14999, 6000, dtype=int)
        test_rows = np.array([16000, 17200, 18400, 19500], dtype=int)
        train = normalized_rows(h5["data"][train_rows])
        test = normalized_rows(h5["data"][test_rows])

    device = torch.device("cpu")
    model = SignalRestorer().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    train_tensor = torch.from_numpy(train).unsqueeze(1)
    batch_size, epochs = 64, 14

    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(train_tensor))
        epoch_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = train_tensor[order[start : start + batch_size]]
            bad, mask = corruption(batch, seed=SEED + epoch * 1000 + start)
            predicted = model(bad, mask)
            # Missing samples matter most; visible samples prevent unrealistic drift.
            weight = 1.0 + 3.0 * (1.0 - mask)
            loss = torch.mean(weight * (predicted - batch) ** 2)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            epoch_loss += float(loss.detach())
        print(f"epoch {epoch + 1:02d}/{epochs}: weighted MSE={epoch_loss / max(1, len(order) // batch_size):.5f}")

    model.eval()
    records = []
    with torch.no_grad():
        for index, row in enumerate(test_rows):
            clean = torch.from_numpy(test[index : index + 1]).unsqueeze(1)
            bad, mask = corruption(clean, seed=SEED + int(row))
            restored = model(bad, mask).squeeze().numpy()
            clean_np, bad_np, mask_np = clean.squeeze().numpy(), bad.squeeze().numpy(), mask.squeeze().numpy()
            baseline = linear_gap_baseline(bad_np, mask_np)
            metrics = {
                "row": int(row),
                **{f"degraded_{k}": v for k, v in signal_metrics(clean_np, bad_np).items()},
                **{f"linear_baseline_{k}": v for k, v in signal_metrics(clean_np, baseline).items()},
                **{f"learned_restoration_{k}": v for k, v in signal_metrics(clean_np, restored).items()},
            }
            records.append(metrics)
            plot_result(clean_np, bad_np, baseline, restored, int(row), out / f"row_{row}_signal_restoration.png")

    with (out / "heldout_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    torch.save({"model_state": model.state_dict(), "seed": SEED, "epochs": epochs, "source": source.name}, out / f"{source_key}_restorer.pt")
    print(f"Saved held-out restoration plots, metrics, and model to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(SOURCES), default="chirp_9db")
    main(parser.parse_args().source)
