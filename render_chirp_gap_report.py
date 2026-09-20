"""Render saved benchmark predictions; no model fitting or test-time selection."""
from pathlib import Path
import argparse
import html
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from src.chirp_gap import PROFILES, metrics
from src.chirp_raster import damage_raster, render_trace, PIXELS_PER_SAMPLE

ROOT = Path(__file__).resolve().parent
COLORS = {"reference": "#344054", "input": "#ca5c19", "estimate": "#087e72"}


def shade(ax, profile):
    for left, right in profile.gaps:
        ax.axvspan(left, right, color="#e5a000", alpha=.16, lw=0)
    ax.grid(alpha=.15)


def panel(path, reference, observed, restored, mask, profile, title, method, role):
    x = np.arange(len(reference))
    before = metrics(reference, observed, mask)
    after = metrics(reference, restored, mask)
    fig = plt.figure(figsize=(15, 7.5), facecolor="white")
    grid = fig.add_gridspec(2, 6, height_ratios=[1, 1.25])
    all_values = np.concatenate([reference, observed, restored])
    bound = max(1., float(np.max(np.abs(all_values)))) * 1.08
    for column, (name, values, color) in enumerate([
        ("Reference (evaluation only)", reference, COLORS["reference"]),
        ("Degraded input — shaded gaps", observed, COLORS["input"]),
        (f"Reconstruction — {method}", restored, COLORS["estimate"]),
    ]):
        ax = fig.add_subplot(grid[0, column*2:column*2+2])
        if column == 1:
            values = values.copy()
            values[~mask] = np.nan
        ax.plot(x, values, color=color, lw=.7)
        if column == 2:
            gaps = restored.copy()
            gaps[mask] = np.nan
            ax.plot(x, gaps, color="#b42318", lw=1.1, label="Estimated missing samples")
            ax.legend(fontsize=7, loc="upper left")
        shade(ax, profile)
        ax.set(title=name, xlabel="Sample index", ylim=(-bound, bound))
    for i, (left, right) in enumerate(profile.gaps):
        ax = fig.add_subplot(grid[1, i*3:i*3+3])
        lo, hi = max(0, left-45), min(len(x), right+45)
        ax.plot(x[lo:hi], reference[lo:hi], "--", color=COLORS["reference"], lw=1.2,
                label="Reference (hidden from inference)")
        visible = observed.copy()
        visible[~mask] = np.nan
        ax.plot(x[lo:hi], visible[lo:hi], color=COLORS["input"], lw=.9, label="Observed samples")
        ax.plot(x[lo:hi], restored[lo:hi], color=COLORS["estimate"], lw=1,
                label="Reconstructed waveform")
        shade(ax, profile)
        ax.set(title=f"Missing-region detail: samples {left}–{right-1}", xlabel="Sample index",
               ylabel="Training-normalized amplitude", xlim=(lo, hi))
        ax.legend(fontsize=7)
    fig.suptitle(
        f"{title} | {role} example\n"
        f"Gap RMSE: {before['gap_rmse']:.3f} → {after['gap_rmse']:.3f}  |  "
        f"Whole-trace RMSE: {before['whole_rmse']:.3f} → {after['whole_rmse']:.3f}",
        fontsize=13, y=.995,
    )
    fig.text(.5, .008, "Method selected on validation rows. Example ranked retrospectively; all test scores are published. "
             "Shading marks estimated regions.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .025, 1, .94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def all_methods(path, bank, reference, observed, mask, profile, title):
    fig, axes = plt.subplots(6, 3, figsize=(16, 17))
    left, right = profile.gaps[-1]
    lo, hi = left-40, right+40
    x = np.arange(len(reference))
    bound = max(1., float(np.max(np.abs(np.concatenate([
        reference[lo:hi], *[bank[key][lo:hi] for key in bank.files]
    ]))))) * 1.08
    for ax, method in zip(axes.flat, bank.files):
        estimate = bank[method]
        score = metrics(reference, estimate, mask)["gap_rmse"]
        ax.plot(x[lo:hi], reference[lo:hi], "--", color=COLORS["reference"], lw=.8)
        ax.plot(x[lo:hi], estimate[lo:hi], color=COLORS["estimate"], lw=.8)
        shade(ax, profile)
        ax.set(title=f"{method} | all-gap RMSE {score:.3f}", ylim=(-bound, bound), xlim=(lo, hi))
    fig.suptitle(title + "\nFixed first test row; dashed = reference, green = method output", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .965))
    fig.savefig(path, dpi=115)
    plt.close(fig)


def raster_panel(folder, reference, restored, profile, amplitude_limit, seed):
    clean = render_trace(reference, amplitude_limit)
    damaged, mask = damage_raster(reference, profile, amplitude_limit, seed)
    estimate = render_trace(restored, amplitude_limit)
    result = damaged.copy()
    missing_columns = np.repeat(~mask, PIXELS_PER_SAMPLE)
    result[:, missing_columns] = estimate[:, missing_columns]
    for name, image in (("reference", clean), ("input", damaged), ("enhanced", result)):
        cv2.imwrite(str(folder / f"image_{name}.png"), image)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    # Zoom on the same fixed late-echo interval in every panel.
    left, right = 1490*PIXELS_PER_SAMPLE, 1660*PIXELS_PER_SAMPLE
    for ax, image, label in zip(axes, (clean, damaged, result),
                               ("Reference image", "Actual degraded image input", "Image with estimated trace fill")):
        ax.imshow(cv2.cvtColor(image[:, left:right], cv2.COLOR_BGR2RGB), aspect="auto")
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle("Image → visible-trace extraction → mask-aware reconstruction → image", fontsize=13)
    fig.text(.5, .025, "Same crop and amplitude mapping. Only known missing columns are replaced; "
             "the fill is an estimate.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(folder / "image_before_after.png", dpi=150)
    plt.close(fig)


def run(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(root / "metrics.csv")
    test = frame[frame.split == "test"]
    selected_summary = []
    gallery = []
    for source, details in manifest["sources"].items():
        chosen = json.loads((root / source / "selection.json").read_text(encoding="utf-8"))
        for profile_id, profile in enumerate(PROFILES):
            directory = root / source / profile.name
            if not (directory / "predictions.npz").exists():
                continue
            data = np.load(directory / "predictions.npz", allow_pickle=False)
            method = str(data["selected_method"])
            scores = test[(test.source == source) & (test.profile == profile.name)]
            selected = scores[scores.method == method].sort_values("row")
            zero = scores[scores.method == "zero_fill"].sort_values("row")
            linear = scores[scores.method == "linear"].sort_values("row")
            gains = 1 - selected.gap_rmse.to_numpy()/np.maximum(zero.gap_rmse.to_numpy(), 1e-12)
            selected_summary.append({
                "source": source, "profile": profile.name, "selected_method": method,
                "test_traces": len(selected),
                "zero_gap_rmse": zero.gap_rmse.mean(),
                "linear_gap_rmse": linear.gap_rmse.mean(),
                "selected_gap_rmse": selected.gap_rmse.mean(),
                "mean_gap_rmse_reduction_percent": 100*(1-selected.gap_rmse.mean()/zero.gap_rmse.mean()),
                "fraction_better_than_zero": float(np.mean(gains > 1e-9)),
                "median_per_trace_reduction_percent": float(np.median(100*gains)),
                "worst_per_trace_reduction_percent": float(np.min(100*gains)),
            })
            order = np.argsort(gains, kind="stable")
            examples = (("best", int(order[-1])), ("median", int(order[len(order)//2])),
                        ("worst", int(order[0])))
            all_methods(
                directory / "all_methods_row16000.png",
                np.load(directory / "first_row_methods.npz", allow_pickle=False),
                data["reference"][0], data["observed"][0], data["mask"][0], profile,
                f"{source} / {profile.name}: all 18 configurations",
            )
            for role, index in examples:
                row = int(data["rows"][index])
                folder = directory / f"{role}_row{row}"
                folder.mkdir(exist_ok=True)
                panel(folder/"comparison.png", data["reference"][index], data["observed"][index],
                      data["restored"][index], data["mask"][index], profile,
                      f"{source} / {profile.name} / row {row}", method, role)
                if profile.image_input:
                    raster_panel(folder, data["reference"][index], data["restored"][index], profile,
                                 details["raster_amplitude_limit"], 9198+row+profile_id*100000)
                gallery.append(dict(source=source, profile=profile.name, role=role, row=row,
                                    method=method, reduction_percent=100*gains[index],
                                    image=str((folder/"comparison.png").relative_to(root)).replace("\\", "/")))
    summary = pd.DataFrame(selected_summary)
    summary.to_csv(root / "selected_test_summary.csv", index=False)
    pd.DataFrame(gallery).to_csv(root / "gallery.csv", index=False)
    rows = []
    for item in selected_summary:
        rows.append("<tr>"+"".join(f"<td>{html.escape(str(item[key]))}</td>" for key in (
            "source", "profile", "selected_method"
        )) + f"<td>{item['zero_gap_rmse']:.3f}</td><td>{item['selected_gap_rmse']:.3f}</td>"
                    f"<td>{item['mean_gap_rmse_reduction_percent']:.1f}%</td></tr>")
    cards = []
    for item in gallery:
        cards.append(f"<article><h3>{item['source']} / {item['profile']} / {item['role']} "
                     f"(row {item['row']})</h3><a href='{item['image']}'><img loading='lazy' "
                     f"src='{item['image']}' alt='Gap reconstruction comparison'></a></article>")
    (root / "index.html").write_text(
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><title>Chirp gap reconstruction</title>"
        "<style>body{max-width:1200px;margin:40px auto;padding:0 20px;font:16px system-ui;color:#223}"
        "table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}"
        "img{width:100%;height:auto}article{margin:35px 0;border:1px solid #ddd;padding:15px;border-radius:10px}</style>"
        "<h1>Chirp missing-region reconstruction</h1>"
        "<p>18 configurations, 3 recordings, 5 conditions, 36 test traces per condition. "
        "Methods are selected using separate validation rows. Known masks; synthetic degradation; "
        "within-recording evaluation. Short-gap improvements do not imply complete long-gap recovery.</p>"
        "<p>Best, median and worst examples below are ranked retrospectively by reduction in gap RMSE "
        "relative to zero fill. All test measurements are in <a href='metrics.csv'>metrics.csv</a>.</p>"
        "<table><tr><th>Source</th><th>Condition</th><th>Selected method</th><th>Zero gap RMSE</th>"
        "<th>Selected gap RMSE</th><th>Reduction</th></tr>"+"".join(rows)+"</table>"+"".join(cards)+"</html>",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"Rendered {len(gallery)} comparison panels and 15 all-method sheets in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"outputs/chirp/gap_reconstruction")
    run(parser.parse_args().input)
