"""Independently verify saved predictions, selection, splits and raster pixel preservation."""
from pathlib import Path
import json
import hashlib
import argparse
import numpy as np
import pandas as pd
import cv2

ROOT = Path(__file__).resolve().parent


def verify(root):
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(root/"metrics.csv")
    verified = 0
    for name, expected in manifest["code_sha256"].items():
        actual = hashlib.sha256((ROOT/name.replace("\\", "/")).read_bytes()).hexdigest()
        assert actual == expected, f"Benchmark source changed since this run: {name}"
    for source, details in manifest["sources"].items():
        train, validation, test = map(set, (
            details["training_rows"], details["validation_rows"], details["test_rows"]
        ))
        assert not (train & validation or train & test or validation & test)
        selection = json.loads((root/source/"selection.json").read_text(encoding="utf-8"))
        for profile, method in selection.items():
            rows = metrics[(metrics.source == source) & (metrics.profile == profile)]
            means = rows[rows.split == "validation"].groupby("method").gap_rmse.mean()
            assert means.idxmin() == method, f"Selection was not validation-derived: {source}/{profile}"
            predictions = np.load(root/source/profile/"predictions.npz", allow_pickle=False)
            assert str(predictions["selected_method"]) == method
            assert set(predictions["rows"].tolist()) == test
            assert predictions["reference"].shape == predictions["observed"].shape == predictions["restored"].shape
            for index, row in enumerate(predictions["rows"]):
                target = predictions["reference"][index]
                output = predictions["restored"][index]
                mask = predictions["mask"][index]
                measured = float(np.sqrt(np.mean((target[~mask]-output[~mask])**2)))
                record = rows[(rows.split == "test") & (rows.method == method) & (rows.row == row)]
                assert len(record) == 1
                np.testing.assert_allclose(measured, record.iloc[0].gap_rmse, atol=1e-10)
                verified += 1
            if profile == "raster_image":
                for image_path in (root/source/profile).glob("*/image_enhanced.png"):
                    input_image = cv2.imread(str(image_path.parent/"image_input.png"))
                    output_image = cv2.imread(str(image_path))
                    row = int(image_path.parent.name.split("row")[-1])
                    index = int(np.flatnonzero(predictions["rows"] == row)[0])
                    valid_columns = np.repeat(predictions["mask"][index], 2)
                    np.testing.assert_array_equal(
                        input_image[:, valid_columns], output_image[:, valid_columns]
                    )
    print(f"PASS: independently rescored {verified} selected predictions.")
    print("PASS: disjoint splits, validation-only method selection, source checksums, "
          "and unchanged observed image columns.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"outputs/chirp/gap_reconstruction")
    verify(parser.parse_args().input)
