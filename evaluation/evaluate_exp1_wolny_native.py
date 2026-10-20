"""Evaluate reconstructed Wolny segmentations on complete native volumes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def dice_per_class(prediction, target, num_classes=4):
    scores = {}
    for cls in range(num_classes):
        pred_cls = prediction == cls
        target_cls = target == cls
        denominator = pred_cls.sum() + target_cls.sum()
        scores[cls] = 1.0 if denominator == 0 else float(2.0 * np.logical_and(pred_cls, target_cls).sum() / denominator)
    return scores


def hd95_asd(prediction, target, cls, spacing=(10.0, 1.5, 1.5)):
    pred_cls = prediction == cls
    target_cls = target == cls
    if not pred_cls.any() or not target_cls.any():
        return float("nan"), float("nan")
    structure = generate_binary_structure(3, 1)
    pred_surface = np.logical_xor(pred_cls, binary_erosion(pred_cls, structure=structure, border_value=0))
    target_surface = np.logical_xor(target_cls, binary_erosion(target_cls, structure=structure, border_value=0))
    distance_to_target = distance_transform_edt(~target_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate((distance_to_target[pred_surface], distance_to_pred[target_surface]))
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def save_csv(rows, path):
    if not rows:
        raise ValueError("No rows to save")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_h5", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    prediction_paths = sorted(args.predictions.glob("*_predictions.h5"))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files found in {args.predictions}")
    for prediction_path in prediction_paths:
        source_stem = prediction_path.name.removesuffix("_predictions.h5")
        source = args.source_h5 / f"{source_stem}.h5"
        if not source.exists():
            raise FileNotFoundError(source)
        with h5py.File(source) as file:
            target = np.asarray(file["label"], dtype=np.int64)
            spacing = tuple(map(float, file.attrs.get("spacing_zyx_mm", (10.0, 1.5, 1.5))))
        with h5py.File(prediction_path) as file:
            prediction = np.asarray(file["predictions"], dtype=np.int64)
        if prediction.shape != target.shape:
            raise ValueError(f"Shape mismatch {source.name}: {prediction.shape} vs {target.shape}")
        patient, phase = source.stem.rsplit("_", 1)
        dice = dice_per_class(prediction, target)
        row = {
            "patient": patient, "phase": phase,
            "RV_Dice": dice[1], "MYO_Dice": dice[2], "LV_Dice": dice[3],
            "Mean_Dice": float(np.mean([dice[1], dice[2], dice[3]])),
        }
        for cls, name in ((1, "RV"), (2, "MYO"), (3, "LV")):
            hd95, asd = hd95_asd(prediction, target, cls, spacing=spacing)
            row[f"{name}_HD95"] = hd95
            row[f"{name}_ASD"] = asd
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    save_csv(rows, args.output / "native_volume_results.csv")
    metrics = [key for key in rows[0] if key not in {"patient", "phase"}]
    summary = {
        metric: {
            "mean": float(np.nanmean([row[metric] for row in rows])),
            "std": float(np.nanstd([row[metric] for row in rows], ddof=1)),
        }
        for metric in metrics
    }
    with (args.output / "native_volume_summary.json").open("w") as file:
        json.dump({"num_volumes": len(rows), "metrics": summary}, file, indent=2)
    print(f"Native full-volume mean Dice: {summary['Mean_Dice']['mean']:.4f}")


if __name__ == "__main__":
    main()
