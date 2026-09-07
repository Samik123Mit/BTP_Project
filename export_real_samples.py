"""Export a small, visible working subset from ULB17-VT.pkl.

The original 512 MB public dataset remains in data/ULB17-VT.pkl. This script writes
six real aligned RGB/thermal sample triplets so the data used by the demo is visible
in File Explorer and can be versioned separately from the full download.
"""
from pathlib import Path
import pickle
import cv2
import numpy as np

ROOT = Path(__file__).parent
SOURCE = ROOT / 'data' / 'ULB17-VT.pkl'
DEST = ROOT / 'data' / 'samples' / 'ulb17_vt_test'
INDICES = [0, 7, 14, 23, 31, 40]


def render(x, lo, hi):
    return np.clip((x-lo)*255/(hi-lo), 0, 255).astype(np.uint8)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError('Download ULB17-VT.pkl first; see README.md.')
    DEST.mkdir(parents=True, exist_ok=True)
    with SOURCE.open('rb') as f:
        _, _, test = pickle.load(f)
    rgb, hr, lr = test
    for i in INDICES:
        high = hr[i, 0]
        lo, hi = np.percentile(high, [1, 99])
        optical = np.transpose(np.clip(rgb[i], 0, 255).astype(np.uint8), (1, 2, 0))
        thermal_hr = render(high, lo, hi)
        thermal_lr = render(lr[i, 0], lo, hi)
        cv2.imwrite(str(DEST / f'{i:02d}_optical_rgb.png'), optical)
        cv2.imwrite(str(DEST / f'{i:02d}_thermal_hr.png'), thermal_hr)
        cv2.imwrite(str(DEST / f'{i:02d}_thermal_lr_80x60.png'), thermal_lr)
    print(f'Exported {len(INDICES)} real RGB/thermal sample triplets to {DEST}')

if __name__ == '__main__':
    main()
