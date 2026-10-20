"""Full-volume validation for Experiment 3 with identical Exp2 reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_exp2_3dconvlstm import (
    load_patient,
    metric_rows,
    predict_volume,
    summarize,
    write_csv,
)
from train_exp3_factorized import FactorizedUNet, StaticFactorizedUNet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(1, 6))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split_json", type=Path, default=Path("configs/acdc_5fold_splits_seed42.json"))
    parser.add_argument("--temporal_root", type=Path, default=Path("acdc_temporal/training"))
    parser.add_argument("--spatial_root", type=Path, default=Path("acdc_preprocessed/training"))
    parser.add_argument("--output_root", type=Path, default=Path("exp3_factorized"))
    parser.add_argument("--max_patients", type=int, help="Development smoke test only")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--spacing_dhw", nargs=3, type=float, default=(10.0, 1.5, 1.5))
    parser.add_argument("--static_input", action="store_true")
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
    model_class = StaticFactorizedUNet if args.static_input else FactorizedUNet
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
