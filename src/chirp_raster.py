"""Calibrated blue-trace raster input experiment, using actual image pixels.

This adapter is deliberately limited to its declared plot colour and coordinates.
It is not an arbitrary screenshot or biomedical image digitizer.
"""
import cv2
import numpy as np

HEIGHT = 512
PIXELS_PER_SAMPLE = 2
TRACE_BGR = (190, 75, 20)


def render_trace(signal, amplitude_limit):
    width = len(signal) * PIXELS_PER_SAMPLE
    canvas = np.full((HEIGHT, width, 3), 248, dtype=np.uint8)
    x = np.arange(len(signal)) * PIXELS_PER_SAMPLE
    y = np.rint((HEIGHT - 1) / 2 - np.asarray(signal) * (HEIGHT - 32) / (2 * amplitude_limit))
    points = np.column_stack([x, np.clip(y, 2, HEIGHT-3)]).astype(np.int32)
    cv2.polylines(canvas, [points.reshape(-1, 1, 2)], False, TRACE_BGR, 1, cv2.LINE_AA)
    return canvas


def damage_raster(reference, profile, amplitude_limit, seed):
    """Return only image pixels and an acquisition mask to the inference adapter."""
    image = render_trace(reference, amplitude_limit)
    image = cv2.GaussianBlur(image, (0, 0), .6)
    rng = np.random.default_rng(seed)
    image = np.clip(image.astype(float) + rng.normal(0, 2, image.shape), 0, 255).astype(np.uint8)
    mask = np.ones(len(reference), dtype=bool)
    for start, end in profile.gaps:
        image[:, start*PIXELS_PER_SAMPLE:end*PIXELS_PER_SAMPLE] = 248
        mask[start:end] = False
    return image, mask


def extract_trace(image, mask, amplitude_limit):
    """Recover sample amplitudes from blue-pixel weights; erased pixels are ignored."""
    b, g, r = cv2.split(image.astype(np.float64))
    weights = np.maximum(b-r-35, 0) * ((b-g) > 20)
    weights = weights[:, ::PIXELS_PER_SAMPLE]
    mass = weights.sum(axis=0)
    valid = mask & (mass > 1e-8)
    y = (np.arange(HEIGHT)[:, None] * weights).sum(axis=0) / np.maximum(mass, 1e-8)
    signal = ((HEIGHT-1)/2-y) * (2*amplitude_limit)/(HEIGHT-32)
    signal[~valid] = 0
    return signal, valid


def observe_image(reference, profile, amplitude_limit, seed):
    image, mask = damage_raster(reference, profile, amplitude_limit, seed)
    signal, detected = extract_trace(image, mask, amplitude_limit)
    return signal, detected
