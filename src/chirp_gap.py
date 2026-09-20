"""Mask-aware reconstruction using training-only waveform priors.

Reconstructors receive an observed trace, acquisition mask and declared
degradation profile. No reference trace is accepted by their inference API.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve, svd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator


@dataclass(frozen=True)
class Profile:
    name: str
    gaps: tuple[tuple[int, int], ...]
    noise: float = 0.0
    blur: float = 0.0
    downsample: int = 1
    gain: float = 1.0
    image_input: bool = False


# Fixed before validation/test: includes a region near the recurrent late echo.
# No test-trace peak detector is used to locate these intervals.
PROFILES = (
    Profile("short_gaps", ((1030, 1062), (1560, 1592))),
    Profile("long_gaps", ((1010, 1106), (1512, 1640))),
    Profile("mixed_damage", ((1020, 1084), (1540, 1604)), .04, .8, 2, .8),
    Profile("severe_4x", ((1020, 1084), (1540, 1604)), .08, 1.2, 4, .65),
    Profile("raster_image", ((1030, 1062), (1560, 1592)), image_input=True),
)


def forward(signal: np.ndarray, profile: Profile) -> np.ndarray:
    """Declared linear corruption, with noise and missingness handled separately."""
    x = np.asarray(signal, dtype=np.float64).copy()
    if profile.downsample > 1:
        n = x.shape[-1]
        positions = np.arange(n)
        low_positions = np.linspace(0, n - 1, n // profile.downsample)
        shape = x.shape
        flat = x.reshape(-1, n)
        x = np.stack([
            np.interp(positions, low_positions, np.interp(low_positions, positions, row))
            for row in flat
        ]).reshape(shape)
    if profile.blur:
        x = gaussian_filter1d(x, profile.blur, axis=-1, mode="reflect")
    return profile.gain * x


def corrupt(reference: np.ndarray, profile: Profile, seed: int):
    rng = np.random.default_rng(seed)
    observed = forward(reference, profile)
    observed += rng.normal(0, profile.noise, observed.shape)
    mask = np.ones(observed.shape, dtype=bool)
    for start, end in profile.gaps:
        mask[start:end] = False
    observed[~mask] = 0
    return observed, mask


def linear_fill(observed: np.ndarray, mask: np.ndarray):
    x = np.arange(observed.size)
    return np.interp(x, x[mask], observed[mask])


def pchip_fill(observed: np.ndarray, mask: np.ndarray):
    x = np.arange(observed.size)
    result = observed.copy()
    result[~mask] = PchipInterpolator(x[mask], observed[mask], extrapolate=True)(x[~mask])
    return result


def ar_fill(observed: np.ndarray, mask: np.ndarray, order=24):
    """Bidirectional local autoregression with a regularized fit and bounded output."""
    result = observed.copy()
    limits = np.flatnonzero(np.diff(np.r_[True, mask, True].astype(int)))
    bound = max(1., 4 * float(np.max(np.abs(observed[mask]))))

    def predict(context, steps):
        if context.size < 2 * order:
            return np.repeat(context[-1] if context.size else 0., steps)
        windows = np.lib.stride_tricks.sliding_window_view(context, order + 1)
        design, target = windows[:, :-1], windows[:, -1]
        gram = design.T @ design
        regularizer = .02 * max(float(np.trace(gram) / order), 1e-6)
        coeff = solve(gram + regularizer * np.eye(order), design.T @ target, assume_a="pos")
        state = list(context[-order:])
        forecast = []
        for _ in range(steps):
            value = float(np.clip(np.dot(state[-order:], coeff), -bound, bound))
            forecast.append(value)
            state.append(value)
        return np.array(forecast)

    for start, end in limits.reshape(-1, 2):
        left = observed[max(0, start - 256):start]
        left_mask = mask[max(0, start - 256):start]
        if (~left_mask).any():
            left = left[np.flatnonzero(~left_mask)[-1] + 1:]
        right = observed[end:min(observed.size, end + 256)]
        right_mask = mask[end:min(observed.size, end + 256)]
        if (~right_mask).any():
            right = right[:np.flatnonzero(~right_mask)[0]]
        count = end - start
        a = predict(left, count)
        b = predict(right[::-1], count)[::-1]
        weight = np.arange(1, count + 1) / (count + 1)
        result[start:end] = (1 - weight) * a + weight * b
    return result


class TrainingPrior:
    """Global low-rank and nearest-template priors, fitted only on training rows."""

    def __init__(self, training: np.ndarray, rank: int = 96, seed: int = 4198):
        self.training = np.asarray(training, dtype=np.float64)
        self.mean = self.training.mean(axis=0)
        centered = self.training - self.mean
        rank = min(rank, min(centered.shape) - 1)
        rng = np.random.default_rng(seed)
        projection = rng.normal(size=(centered.shape[1], rank + 12))
        q, _ = np.linalg.qr(centered @ projection, mode="reduced")
        for _ in range(2):
            q, _ = np.linalg.qr(centered @ (centered.T @ q), mode="reduced")
        _, singular, vectors = svd(q.T @ centered, full_matrices=False)
        self.basis = (vectors[:rank].T * (singular[:rank] / np.sqrt(len(training) - 1)))
        self.cache = {}
        self.local_cache = {}

    def operator(self, profile):
        if profile not in self.cache:
            self.cache[profile] = (
                forward(self.mean, profile),
                forward(self.basis.T, profile).T,
                forward(self.training, profile),
            )
        return self.cache[profile]

    def pca(self, observed, mask, profile, rank=64, ridge=.1):
        mean_observed, basis_observed, _ = self.operator(profile)
        design = basis_observed[mask, :rank]
        target = observed[mask] - mean_observed[mask]
        coeff = solve(
            design.T @ design + ridge * np.eye(design.shape[1]),
            design.T @ target, assume_a="pos",
        )
        return self.mean + self.basis[:, :rank] @ coeff

    def dictionary(self, observed, mask, profile, neighbours=12, ridge=.1):
        """Select training neighbours by visible-sample fit; fit their combination."""
        _, _, degraded_training = self.operator(profile)
        design = degraded_training[:, mask]
        target = observed[mask]
        energy = np.sum(design * design, axis=1) + 1e-10
        dot = design @ target
        gain = dot / energy
        distance = np.sum(target * target) - dot * gain
        selected = np.argsort(distance)[:neighbours]
        if neighbours == 1:
            return gain[selected[0]] * self.training[selected[0]]
        local = design[selected].T
        gram = local.T @ local
        penalty = ridge * max(float(np.trace(gram) / neighbours), 1e-6)
        coeff = solve(
            gram + penalty * np.eye(neighbours), local.T @ target, assume_a="pos"
        )
        return self.training[selected].T @ coeff

    def local(self, observed, mask, profile, kind="pca", rank=12):
        """Fit each gap from its surrounding context, excluding other erased samples."""
        result = linear_fill(observed, mask)
        for gap_start, gap_end in profile.gaps:
            start = max(0, gap_start - 160)
            end = min(observed.size, gap_end + 160)
            cache_key = (profile, start, end)
            if cache_key not in self.local_cache:
                # Transform full traces before cropping so convolution boundaries match.
                original = self.training[:, start:end]
                degraded = self.operator(profile)[2][:, start:end]
                mean_original = original.mean(axis=0)
                centered = original - mean_original
                # Small local bases restrict the degrees of freedom inside a blank span.
                _, singular, vectors = svd(centered, full_matrices=False)
                basis = vectors[:32].T * (singular[:32] / np.sqrt(len(original) - 1))
                # Regression maps local original PCA coefficients to observed context.
                coefficients = centered @ vectors[:32].T
                coefficients /= np.maximum(singular[:32] / np.sqrt(len(original)-1), 1e-8)
                mean_degraded = degraded.mean(axis=0)
                observed_basis = np.linalg.lstsq(
                    coefficients, degraded - mean_degraded, rcond=None
                )[0].T
                self.local_cache[cache_key] = (
                    original, degraded, mean_original, mean_degraded, basis, observed_basis,
                )
            original, degraded, mu, degraded_mu, basis, observed_basis = self.local_cache[cache_key]
            visible = mask[start:end]
            target = observed[start:end][visible]
            if kind == "pca":
                design = observed_basis[visible, :rank]
                coeff = solve(
                    design.T @ design + .1 * np.eye(rank),
                    design.T @ (target - degraded_mu[visible]), assume_a="pos",
                )
                predicted = mu + basis[:, :rank] @ coeff
            else:
                design = degraded[:, visible]
                dot = design @ target
                energy = np.sum(design**2, axis=1) + 1e-10
                distance = np.sum(target**2) - dot**2 / energy
                indices = np.argsort(distance)[:rank]
                local = design[indices].T
                gram = local.T @ local
                coeff = solve(
                    gram + .02 * max(float(np.trace(gram)/rank), 1e-6) * np.eye(rank),
                    local.T @ target, assume_a="pos",
                )
                predicted = original[indices].T @ coeff
            result[gap_start:gap_end] = predicted[gap_start-start:gap_end-start]
        return result


METHODS = (
    "zero_fill", "linear", "pchip", "autoregression",
    "pca_r32_l01", "pca_r64_l01", "pca_r96_l01",
    "pca_r64_l001", "pca_r64_l1",
    "template_1", "dictionary_8", "dictionary_24", "dictionary_48",
    "local_pca_8", "local_pca_16", "local_pca_32",
    "local_dictionary_8", "local_dictionary_24",
)


def reconstruct(method, observed, mask, profile, prior):
    if mask.all():
        return observed.copy()
    if not mask.any():
        raise ValueError("No observed samples: reconstruction is unidentified.")
    if method == "zero_fill":
        return observed.copy()
    if method == "linear":
        return linear_fill(observed, mask)
    if method == "pchip":
        return pchip_fill(observed, mask)
    if method == "autoregression":
        return ar_fill(observed, mask)
    if method.startswith("local_"):
        _, kind, rank = method.split("_")
        return prior.local(observed, mask, profile, kind, int(rank))
    if method.startswith("pca_"):
        _, rank, regularizer = method.split("_")
        penalty = {"l01": .1, "l001": .01, "l1": 1.}[regularizer]
        result = prior.pca(observed, mask, profile, int(rank[1:]), penalty)
    elif method.startswith("dictionary_") or method == "template_1":
        result = prior.dictionary(observed, mask, profile, int(method.split("_")[-1]))
    else:
        raise ValueError(method)
    # Do not alter genuinely intact samples on gap-only tasks.
    if not profile.noise and not profile.blur and profile.downsample == 1 and profile.gain == 1:
        result[mask] = observed[mask]
    return result


def metrics(reference, output, mask):
    error = np.asarray(output) - reference
    missing = ~mask
    gap_rms = np.sqrt(np.mean(reference[missing] ** 2))
    return {
        "gap_rmse": float(np.sqrt(np.mean(error[missing] ** 2))),
        "gap_nrmse": float(np.sqrt(np.mean(error[missing] ** 2)) / max(gap_rms, 1e-10)),
        "gap_mae": float(np.mean(np.abs(error[missing]))),
        "visible_rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
        "whole_rmse": float(np.sqrt(np.mean(error ** 2))),
        "gap_correlation": correlation(reference[missing], output[missing]),
        "whole_correlation": correlation(reference, output),
    }


def correlation(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
