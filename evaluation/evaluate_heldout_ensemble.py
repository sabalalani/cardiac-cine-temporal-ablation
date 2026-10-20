"""Frozen five-fold ensemble evaluation for ACDC patients 101--150.

Use --check_only for a label-blind readiness audit during development. Actual
held-out inference requires --confirm_frozen_test_evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from evaluate_exp1_wolny_native import dice_per_class, hd95_asd
from evaluate_exp2_3dconvlstm import HALO, SPACING_DHW, load_patient, retained_slice
from train_exp2_3dconvlstm import (
    CLASSES, PATCH_SHAPE, STRIDE, StaticUNet3DConvLSTM, UNet3DConvLSTM, starts,
)
from train_exp3_factorized import FactorizedUNet

MODELS = {
    "Exp1": ("exp1_wolny_native", None),
    "Exp2": ("exp2_3dconvlstm", UNet3DConvLSTM),
    "E2-Static": ("exp2_static_convlstm", StaticUNet3DConvLSTM),
    "Exp3": ("exp3_factorized", FactorizedUNet),
}
METRICS = (
    "RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice",
    "RV_HD95", "RV_ASD", "MYO_HD95", "MYO_ASD", "LV_HD95", "LV_ASD",
)


def checkpoint(root: Path, experiment: str, fold: int) -> Path:
    if experiment == "Exp1":
        return root / f"fold_{fold}" / "checkpoints" / "best_checkpoint.pytorch"
    return root / f"fold_{fold}" / "best_checkpoint.pytorch"


def expected_patients(root: Path) -> list[str]:
    actual = sorted(path.name for path in root.glob("patient*") if path.is_dir())
    expected = [f"patient{number:03d}" for number in range(101, 151)]
    if actual != expected:
        raise RuntimeError(
            f"Held-out cohort mismatch: missing={sorted(set(expected)-set(actual))}, "
            f"extra={sorted(set(actual)-set(expected))}"
        )
    return actual


def audit(args: argparse.Namespace) -> dict:
    temporal = expected_patients(args.temporal_root)
    spatial = expected_patients(args.spatial_root)
    if temporal != spatial:
        raise RuntimeError("Spatial and temporal held-out patient lists differ")
    report: dict[str, object] = {
        "cohort": "patient101--patient150", "patients": 50,
        "labels_examined": False, "experiments": {},
    }
    for experiment in args.experiments:
        root = args.model_roots[experiment]
        paths = [checkpoint(root, experiment, fold) for fold in range(1, 6)]
        report["experiments"][experiment] = {
            "complete_folds": [fold for fold, path in enumerate(paths, 1) if path.exists()],
            "missing_checkpoints": [str(path) for path in paths if not path.exists()],
        }
    if "Exp1" in args.experiments:
        count = len(list(args.exp1_h5.glob("patient*_??.h5")))
        report["experiments"]["Exp1"]["input_volumes"] = count
        if count != 100:
            raise RuntimeError(f"Expected 100 Exp1 held-out HDF5 volumes, found {count}")
    return report


@torch.inference_mode()
def predict_probabilities(model, cine: np.ndarray, phase_indices: np.ndarray,
                          device: torch.device) -> np.ndarray:
    """Return ED/ES probabilities [2,4,D,H,W] with validation patch halos."""
    _, depth, height, width = cine.shape
    dz, dy, dx = PATCH_SHAPE
    accum = np.zeros((2, CLASSES, depth, height, width), dtype=np.float32)
    counts = np.zeros((depth, height, width), dtype=np.float32)
    indices = torch.from_numpy(phase_indices).to(device)
    model.eval()
    for z in starts(depth, dz, STRIDE[0]):
        for y in starts(height, dy, STRIDE[1]):
            for x in starts(width, dx, STRIDE[2]):
                patch = cine[:, z:z+dz, y:y+dy, x:x+dx].copy()
                patch = (patch - float(patch.mean())) / max(float(patch.std()), 1e-6)
                tensor = torch.from_numpy(patch[None, :, None]).to(device)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    probs = model(tensor, indices).softmax(dim=2)[0].float().cpu().numpy()
                pz, vz = retained_slice(z, dz, depth, HALO[0])
                py, vy = retained_slice(y, dy, height, HALO[1])
                px, vx = retained_slice(x, dx, width, HALO[2])
                accum[:, :, vz, vy, vx] += probs[:, :, pz, py, px]
                counts[vz, vy, vx] += 1
    if np.any(counts == 0):
        raise RuntimeError("Patch reconstruction left uncovered voxels")
    return accum / counts[None, None]


def metric_rows(patient: str, prediction: np.ndarray, target: np.ndarray) -> list[dict]:
    rows = []
    for phase_index, phase in enumerate(("ED", "ES")):
        dice = dice_per_class(prediction[phase_index], target[phase_index], CLASSES)
        row = {
            "patient": patient, "phase": phase, "RV_Dice": dice[1],
            "MYO_Dice": dice[2], "LV_Dice": dice[3],
            "Mean_Dice": float(np.mean([dice[1], dice[2], dice[3]])),
        }
        for cls, name in ((1, "RV"), (2, "MYO"), (3, "LV")):
            hd95, asd = hd95_asd(prediction[phase_index], target[phase_index], cls, SPACING_DHW)
            row[f"{name}_HD95"], row[f"{name}_ASD"] = hd95, asd
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for phase in ("combined", "ED", "ES"):
        selected = rows if phase == "combined" else [r for r in rows if r["phase"] == phase]
        summary: dict[str, object] = {"phase": phase, "n_volumes": len(selected)}
        for metric in METRICS:
            values = np.asarray([r[metric] for r in selected], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            summary[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        summaries.append(summary)
    return summaries


def evaluate_temporal(args: argparse.Namespace, experiment: str, patients: list[str]) -> None:
    _, model_class = MODELS[experiment]
    experiment_root = args.model_roots[experiment]
    output_root = experiment_root / "heldout_ensemble"
    probability_root = output_root / "fold_probabilities"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for fold in range(1, 6):
        fold_dir = probability_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        missing = [p for p in patients if not (fold_dir / f"{p}.npz").exists()]
        if not missing:
            continue
        state = torch.load(checkpoint(experiment_root, experiment, fold), map_location="cpu", weights_only=False)
        model = model_class().to(device)
        model.load_state_dict(state["model_state_dict"])
        for number, patient in enumerate(missing, 1):
            cine, _, indices = load_patient(patient, args.temporal_root, args.spatial_root)
            probs = predict_probabilities(model, cine, indices, device)
            np.savez_compressed(fold_dir / f"{patient}.npz", probabilities=probs.astype(np.float16))
            print(f"{experiment} fold {fold}: {number}/{len(missing)} {patient}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = []
    prediction_dir = output_root / "ensemble_predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for number, patient in enumerate(patients, 1):
        _, target, _ = load_patient(patient, args.temporal_root, args.spatial_root)
        total = None
        for fold in range(1, 6):
            with np.load(probability_root / f"fold_{fold}" / f"{patient}.npz") as archive:
                probs = archive["probabilities"].astype(np.float32)
            total = probs if total is None else total + probs
        prediction = np.argmax(total / 5, axis=1).astype(np.uint8)
        rows.extend(metric_rows(patient, prediction, target))
        np.savez_compressed(prediction_dir / f"{patient}_ED_ES.npz", prediction=prediction)
        print(f"{experiment} ensemble: {number}/50 {patient}", flush=True)
    metric_dir = output_root / "metrics"
    summary_rows = summarize(rows)
    write_csv(metric_dir / "heldout_volume_results.csv", rows)
    write_csv(metric_dir / "heldout_summary.csv", summary_rows)
    with (metric_dir / "heldout_summary.json").open("w") as file:
        json.dump({"experiment": experiment, "patients": 50, "volumes": 100,
                   "ensemble_folds": [1, 2, 3, 4, 5], "summaries": summary_rows},
                  file, indent=2, allow_nan=True)


def evaluate_exp1(args: argparse.Namespace) -> None:
    subprocess.run([
        sys.executable, str(args.root / "ensemble_exp1_wolny_test.py"),
        "--experiment_root", str(args.model_roots["Exp1"]),
        "--source_h5", str(args.exp1_h5),
        "--pytorch3dunet", str(args.root / "pytorch-3dunet"),
        "--gpu", str(args.gpu), "--run_models", "--confirm_frozen_test_evaluation",
    ], check=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--root", type=Path, default=repo_root)
    parser.add_argument("--spatial_root", type=Path, default=repo_root / "acdc_preprocessed/testing")
    parser.add_argument("--temporal_root", type=Path, default=repo_root / "acdc_temporal/testing")
    parser.add_argument("--exp1_h5", type=Path, default=repo_root / "exp1_wolny_native_h5/testing")
    parser.add_argument("--exp1_model_root", type=Path, default=repo_root / "exp1_wolny_native")
    parser.add_argument("--exp2_model_root", type=Path, default=repo_root / "exp2_3dconvlstm")
    parser.add_argument("--exp2_static_model_root", type=Path,
                        default=repo_root / "exp2_static_convlstm")
    parser.add_argument("--exp3_model_root", type=Path, default=repo_root / "exp3_factorized")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--confirm_frozen_test_evaluation", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.model_roots = {
        "Exp1": args.exp1_model_root.resolve(),
        "Exp2": args.exp2_model_root.resolve(),
        "E2-Static": args.exp2_static_model_root.resolve(),
        "Exp3": args.exp3_model_root.resolve(),
    }
    report = audit(args)
    print(json.dumps(report, indent=2))
    if args.check_only:
        return
    if not args.confirm_frozen_test_evaluation:
        raise SystemExit("Held-out evaluation is locked. Use --check_only during development, or "
                         "--confirm_frozen_test_evaluation after all model choices are frozen.")
    incomplete = {name: details["missing_checkpoints"] for name, details in
                  report["experiments"].items() if details["missing_checkpoints"]}
    if incomplete:
        raise RuntimeError(f"Cannot evaluate incomplete experiments: {incomplete}")
    patients = expected_patients(args.temporal_root)
    for experiment in args.experiments:
        evaluate_exp1(args) if experiment == "Exp1" else evaluate_temporal(args, experiment, patients)


if __name__ == "__main__":
    main()
