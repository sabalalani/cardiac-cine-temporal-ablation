"""Full-volume validation for Experiment 2 with Wolny-style patch halos."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from evaluate_exp1_wolny_native import dice_per_class, hd95_asd
from train_exp2_3dconvlstm import (
    CLASSES, PATCH_SHAPE, STRIDE, StaticUNet3DConvLSTM, UNet3DConvLSTM, starts,
)


HALO = (2, 16, 16)
SPACING_DHW = (10.0, 1.5, 1.5)
METRICS = (
    "RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice",
    "RV_HD95", "RV_ASD", "MYO_HD95", "MYO_ASD", "LV_HD95", "LV_ASD",
)


def load_patient(patient: str, temporal_root: Path, spatial_root: Path):
    temporal_dir = temporal_root / patient
    spatial_dir = spatial_root / patient
    xyzt = nib.load(temporal_dir / f"{patient}_cine_T30_EDaligned.nii.gz").get_fdata(dtype=np.float32)
    cine = np.ascontiguousarray(np.transpose(xyzt, (3, 2, 1, 0)))  # T,D,H,W
    with (temporal_dir / "temporal_metadata.json").open() as file:
        es_index = int(json.load(file)["ES_normalized_index"])
    labels = []
    for phase in ("ED", "ES"):
        xyz = nib.load(spatial_dir / f"{patient}_{phase}_gt.nii.gz").get_fdata()
        labels.append(np.ascontiguousarray(np.transpose(np.rint(xyz).astype(np.int64), (2, 1, 0))))
    labels = np.stack(labels)
    if cine.shape[0] != 30 or labels.shape[0] != 2 or labels.shape[1:] != cine.shape[1:]:
        raise ValueError(f"Unexpected shapes for {patient}: cine={cine.shape}, labels={labels.shape}")
    return cine, labels, np.asarray([[0, es_index]], dtype=np.int64)


def retained_slice(start: int, patch: int, size: int, halo: int) -> tuple[slice, slice]:
    left = 0 if start == 0 else halo
    right = patch if start + patch == size else patch - halo
    return slice(left, right), slice(start + left, start + right)


@torch.inference_mode()
def predict_volume(model, cine: np.ndarray, phase_indices: np.ndarray, device: torch.device) -> np.ndarray:
    """Return ED/ES labels [2,D,H,W] after halo-cropped overlap averaging."""
    _, depth, height, width = cine.shape
    dz, dy, dx = PATCH_SHAPE
    accum = np.zeros((2, CLASSES, depth, height, width), dtype=np.float32)
    counts = np.zeros((depth, height, width), dtype=np.float32)
    indices = torch.from_numpy(phase_indices).to(device)
    model.eval()

    for z in starts(depth, dz, STRIDE[0]):
        for y in starts(height, dy, STRIDE[1]):
            for x in starts(width, dx, STRIDE[2]):
                patch = cine[:, z:z + dz, y:y + dy, x:x + dx].copy()
                patch = (patch - float(patch.mean())) / max(float(patch.std()), 1e-6)
                tensor = torch.from_numpy(patch[None, :, None]).to(device)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    probabilities = model(tensor, indices).softmax(dim=2)[0].float().cpu().numpy()

                pz, vz = retained_slice(z, dz, depth, HALO[0])
                py, vy = retained_slice(y, dy, height, HALO[1])
                px, vx = retained_slice(x, dx, width, HALO[2])
                accum[:, :, vz, vy, vx] += probabilities[:, :, pz, py, px]
                counts[vz, vy, vx] += 1.0

    if np.any(counts == 0):
        raise RuntimeError(f"Patch reconstruction left {int((counts == 0).sum())} uncovered voxels")
    return np.argmax(accum / counts[None, None], axis=1).astype(np.uint8)


def metric_rows(patient: str, prediction: np.ndarray, target: np.ndarray,
                spacing_dhw=SPACING_DHW) -> list[dict]:
    rows = []
    for phase_index, phase in enumerate(("ED", "ES")):
        dice = dice_per_class(prediction[phase_index], target[phase_index], CLASSES)
        row = {
            "patient": patient,
            "phase": phase,
            "RV_Dice": dice[1],
            "MYO_Dice": dice[2],
            "LV_Dice": dice[3],
            "Mean_Dice": float(np.mean([dice[1], dice[2], dice[3]])),
        }
        for cls, name in ((1, "RV"), (2, "MYO"), (3, "LV")):
            hd95, asd = hd95_asd(prediction[phase_index], target[phase_index], cls, spacing_dhw)
            row[f"{name}_HD95"] = hd95
            row[f"{name}_ASD"] = asd
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["patient", "phase", *METRICS])
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> dict:
    def group_summary(group: list[dict]) -> dict:
        ddof = 1 if len(group) > 1 else 0
        return {
            metric: {
                "mean": float(np.nanmean([row[metric] for row in group])),
                "std": float(np.nanstd([row[metric] for row in group], ddof=ddof)),
            }
            for metric in METRICS
        }

    return {
        "num_patients": len({row["patient"] for row in rows}),
        "num_volumes": len(rows),
        "overall": group_summary(rows),
        "by_phase": {
            phase: group_summary([row for row in rows if row["phase"] == phase])
            for phase in ("ED", "ES")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split_json", type=Path, default=Path("configs/acdc_5fold_splits_seed42.json"))
    parser.add_argument("--temporal_root", type=Path, default=Path("acdc_temporal/training"))
    parser.add_argument("--spatial_root", type=Path, default=Path("acdc_preprocessed/training"))
    parser.add_argument("--output_root", type=Path, default=Path("exp2_3dconvlstm"))
    parser.add_argument("--max_patients", type=int, help="Development smoke test only")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--spacing_dhw", nargs=3, type=float, default=SPACING_DHW)
    parser.add_argument("--static_input", action="store_true",
                        help="Evaluate the E2-Static repeated-frame control.")
    args = parser.parse_args()

    checkpoint = args.checkpoint or args.output_root / f"fold_{args.fold}" / "best_checkpoint.pytorch"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    with args.split_json.open() as file:
        patients = json.load(file)["folds"][f"fold_{args.fold}"]["val"]
    if args.max_patients is not None:
        if args.max_patients < 1:
            raise ValueError("--max_patients must be positive")
        patients = patients[:args.max_patients]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_class = StaticUNet3DConvLSTM if args.static_input else UNet3DConvLSTM
    model = model_class().to(device)
    model.load_state_dict(state["model_state_dict"])
    output = args.output_root / f"fold_{args.fold}" / "validation_metrics"
    prediction_dir = args.output_root / f"fold_{args.fold}" / "validation_predictions"
    output.mkdir(parents=True, exist_ok=True)
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for number, patient in enumerate(patients, start=1):
        cine, target, phase_indices = load_patient(patient, args.temporal_root, args.spatial_root)
        prediction = predict_volume(model, cine, phase_indices, device)
        rows.extend(metric_rows(patient, prediction, target, tuple(args.spacing_dhw)))
        patient_dice = np.mean([row["Mean_Dice"] for row in rows[-2:]])
        print(f"[{number:02d}/{len(patients):02d}] {patient}: mean Dice={patient_dice:.4f}", flush=True)
        if args.save_predictions:
            np.savez_compressed(prediction_dir / f"{patient}_ED_ES.npz", prediction=prediction)

    summary = summarize(rows)
    write_csv(rows, output / "native_volume_results.csv")
    with (output / "native_volume_summary.json").open("w") as file:
        json.dump(summary, file, indent=2, allow_nan=True)
    print(f"Full-volume mean Dice: {summary['overall']['Mean_Dice']['mean']:.4f}")
    print(f"Saved {len(rows)} ED/ES rows to {output.resolve()}")


if __name__ == "__main__":
    main()
