"""Run validation-only exploration or the locked gap-restoration benchmark."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
import json
import hashlib
from pathlib import Path
import sys
import platform
import time
import h5py
import numpy as np
import pandas as pd

from src.chirp_gap import METHODS, PROFILES, TrainingPrior, corrupt, reconstruct, metrics
from src.chirp_raster import observe_image

SOURCES = {
    "chirp_9db": "ORB3H_p1350_tr60_Ch_9dB_v1.h5",
    "am_9db": "ORB3H_p1350_tr60_AM_9dB_v1.h5",
    "v3": "ORB3H_p1350_tr60_v3.h5",
}
ROOT = Path(__file__).resolve().parent


def run(args):
    if not (100 <= args.train_count <= 11000 and 2 <= args.validation_count <= 2600
            and 2 <= args.test_count <= 3600):
        raise ValueError("Counts must fit the disjoint row ranges; at least 100 training rows are required.")
    args.output.mkdir(parents=True, exist_ok=True)
    all_records = []
    manifest = {
        "seed": 4198, "command": sys.argv, "python": platform.python_version(),
        "numpy": np.__version__, "sources": {},
        "scope": "Synthetic known-mask / known-operator within-recording row holdout.",
        "methods": list(METHODS),
        "profiles": [vars(profile) for profile in PROFILES],
        "code_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT/"src/chirp_gap.py", ROOT/"src/chirp_raster.py")
        },
    }
    for key in args.sources:
        start_time = time.perf_counter()
        training_rows = np.linspace(0, 10999, args.train_count, dtype=int)
        validation_rows = np.linspace(12000, 14599, args.validation_count, dtype=int)
        test_rows = np.linspace(16000, 19599, args.test_count, dtype=int)
        assert not (set(training_rows) & set(validation_rows) | set(training_rows) & set(test_rows))
        source = args.data_root / SOURCES[key]
        with h5py.File(source, "r") as file:
            training_raw = file["data"][training_rows].astype(np.float64)
            validation_raw = file["data"][validation_rows].astype(np.float64)
        center = float(np.median(training_raw))
        scale = float(np.median(np.std(training_raw, axis=1)))
        training = (training_raw - center) / scale
        validation = (validation_raw - center) / scale
        prior = TrainingPrior(training)
        amplitude_limit = float(1.5 * np.quantile(np.abs(training), .9999))
        def observation(reference, profile, seed):
            if profile.image_input:
                return observe_image(reference, profile, amplitude_limit, seed)
            return corrupt(reference, profile, seed)
        manifest["sources"][key] = {
            "filename": source.name, "shape": [19600, 2048],
            "training_rows": training_rows.tolist(),
            "validation_rows": validation_rows.tolist(),
            "test_rows": [] if args.pilot else test_rows.tolist(),
            "training_center": center, "training_scale": scale,
            "raster_amplitude_limit": amplitude_limit,
            "training_sha256": hashlib.sha256(training_raw.tobytes()).hexdigest(),
        }
        source_out = args.output / key
        source_out.mkdir(exist_ok=True)
        selected = {}
        source_records = []
        for profile_id, profile in enumerate(PROFILES):
            validation_records = []
            for row, reference in zip(validation_rows, validation):
                observed, mask = observation(reference, profile, 4198 + int(row) + profile_id * 100000)
                for method in METHODS:
                    output = reconstruct(method, observed, mask, profile, prior)
                    record = dict(
                        source=key, split="validation", profile=profile.name,
                        row=int(row), method=method, **metrics(reference, output, mask),
                    )
                    validation_records.append(record)
            scores = pd.DataFrame(validation_records).groupby("method").gap_rmse.mean()
            winner = str(scores.idxmin())
            selected[profile.name] = winner
            (source_out / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
            print(f"{key}/{profile.name}: validation winner={winner}; "
                  f"gap RMSE={scores[winner]:.4f}; linear={scores['linear']:.4f}; "
                  f"zero={scores['zero_fill']:.4f}", flush=True)
            source_records.extend(validation_records)
            if not args.pilot:
                # Selection above is written to disk before test values are read.
                with h5py.File(source, "r") as file:
                    test_raw = file["data"][test_rows].astype(np.float64)
                test = (test_raw - center) / scale
                training_hashes = {hashlib.sha256(row.tobytes()).digest() for row in training_raw}
                assert not any(hashlib.sha256(row.tobytes()).digest() in training_hashes for row in test_raw)
                profile_out = source_out / profile.name
                profile_out.mkdir(exist_ok=True)
                references, observations, estimates, masks = [], [], [], []
                first_outputs = {}
                for row, reference in zip(test_rows, test):
                    observed, mask = observation(reference, profile, 9198 + int(row) + profile_id * 100000)
                    for method in METHODS:
                        output = reconstruct(method, observed, mask, profile, prior)
                        source_records.append(dict(
                            source=key, split="test", profile=profile.name, row=int(row),
                            method=method, selected_on_validation=(method == winner),
                            **metrics(reference, output, mask),
                        ))
                        if method == winner:
                            estimates.append(output.copy())
                        if row == test_rows[0]:
                            first_outputs[method] = output.copy()
                    references.append(reference)
                    observations.append(observed)
                    masks.append(mask)
                # Full arrays support consistent plots, error crops and independent scoring.
                np.savez_compressed(
                    profile_out / "predictions.npz", rows=test_rows,
                    reference=references, observed=observations,
                    restored=estimates, mask=masks, selected_method=winner,
                )
                np.savez_compressed(profile_out / "first_row_methods.npz", **first_outputs)
        frame = pd.DataFrame(source_records)
        frame.to_csv(source_out / "metrics.csv", index=False)
        (source_out / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        all_records.extend(source_records)
        print(f"Finished {key} in {time.perf_counter()-start_time:.1f}s", flush=True)
    frame = pd.DataFrame(all_records)
    frame.to_csv(args.output / "metrics.csv", index=False)
    frame.groupby(["source", "split", "profile", "method"]).mean(numeric_only=True).to_csv(
        args.output / "summary.csv"
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\DELL\Desktop\hs lit theiory\Chirp data"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chirp/gap_reconstruction")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--train-count", type=int, default=1800)
    parser.add_argument("--validation-count", type=int, default=24)
    parser.add_argument("--test-count", type=int, default=36)
    parser.add_argument("--pilot", action="store_true", help="Use validation rows only; never read test data.")
    run(parser.parse_args())
