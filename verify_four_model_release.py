"""Read-only integrity audit for the four completed manuscript models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from training.train_exp2_3dconvlstm import StaticUNet3DConvLSTM, UNet3DConvLSTM
from training.train_exp3_factorized import FactorizedUNet


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "E1": {
        "root": ROOT / "exp1_wolny_native", "parameters": 983_254,
        "final": [93, 76, 109, 101, 88], "best": [65, 48, 81, 82, 76],
    },
    "E2-Full": {
        "root": ROOT / "exp2_3dconvlstm", "parameters": 4_027_332,
        "final": [110, 88, 126, 150, 113], "best": [83, 61, 99, 139, 86],
    },
    "E2-Static": {
        "root": ROOT / "exp2_static_convlstm", "parameters": 4_027_332,
        "final": [91, 91, 88, 67, 82], "best": [64, 64, 70, 40, 55],
    },
    "E3": {
        "root": ROOT / "exp3_factorized", "parameters": 1_517_124,
        "final": [108, 101, 102, 85, 62], "best": [99, 83, 75, 58, 44],
    },
}

RESULTS = {
    "E1": {
        "oof": ROOT / "exp1_wolny_native/cv_metrics/oof_volume_results.csv",
        "test": ROOT / "exp1_wolny_native/heldout_ensemble/metrics/native_volume_results.csv",
        "mms": ROOT / "mms_external_validation/Exp1/metrics/external_volume_results.csv",
    },
    "E2-Full": {
        "oof": ROOT / "exp2_3dconvlstm/cv_metrics/oof_volume_results.csv",
        "test": ROOT / "exp2_3dconvlstm/heldout_ensemble/metrics/heldout_volume_results.csv",
        "mms": ROOT / "mms_external_validation/Exp2/metrics/external_volume_results.csv",
    },
    "E2-Static": {
        "oof": ROOT / "exp2_static_convlstm/cv_metrics/oof_volume_results.csv",
        "test": ROOT / "exp2_static_convlstm/frozen_ensemble/heldout_ensemble/metrics/heldout_volume_results.csv",
        "mms": ROOT / "exp2_static_convlstm/mms_external_validation/E2-Static/metrics/external_volume_results.csv",
    },
    "E3": {
        "oof": ROOT / "exp3_factorized/cv_metrics/oof_volume_results.csv",
        "test": ROOT / "exp3_factorized/heldout_ensemble/metrics/heldout_volume_results.csv",
        "mms": ROOT / "mms_external_validation/Exp3/metrics/external_volume_results.csv",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit_split() -> dict:
    split = json.loads((ROOT / "configs/acdc_5fold_splits_seed42.json").read_text())
    development = set(split["development_patients"])
    test = set(split["official_test_patients"])
    require(len(development) == 100 and len(test) == 50, "Expected 100/50 ACDC cohorts")
    require(not development & test, "ACDC development/test overlap")
    validation_union: set[str] = set()
    fold_rows = []
    for number in range(1, 6):
        fold = split["folds"][f"fold_{number}"]
        train, val = set(fold["train"]), set(fold["val"])
        require(len(train) == 80 and len(val) == 20, f"Fold {number}: expected 80/20")
        require(not train & val and train | val == development, f"Fold {number}: invalid partition")
        require(not validation_union & val, f"Fold {number}: repeated validation patient")
        validation_union |= val
        fold_rows.append({"fold": number, "train": len(train), "validation": len(val)})
    require(validation_union == development, "Validation folds do not cover development cohort once")
    return {"development": 100, "test": 50, "folds": fold_rows}


def audit_parameters() -> dict[str, int]:
    counts = {
        "E2-Full": sum(p.numel() for p in UNet3DConvLSTM().parameters()),
        "E2-Static": sum(p.numel() for p in StaticUNet3DConvLSTM().parameters()),
        "E3": sum(p.numel() for p in FactorizedUNet().parameters()),
    }
    e1 = torch.load(
        EXPECTED["E1"]["root"] / "fold_1/checkpoints/best_checkpoint.pytorch",
        map_location="cpu", weights_only=False,
    )
    counts["E1"] = sum(value.numel() for value in e1["model_state_dict"].values())
    for model, count in counts.items():
        require(count == EXPECTED[model]["parameters"], f"{model}: parameter mismatch {count}")
    return counts


def audit_training() -> dict:
    report = {}
    for model, expected in EXPECTED.items():
        rows = []
        for fold in range(1, 6):
            if model == "E1":
                directory = expected["root"] / f"fold_{fold}/checkpoints"
                best_state = torch.load(directory / "best_checkpoint.pytorch", map_location="cpu", weights_only=False)
                last_state = torch.load(directory / "last_checkpoint.pytorch", map_location="cpu", weights_only=False)
                final_epoch = int(last_state["num_epochs"])
                best_epoch = int(best_state["num_epochs"])
                best_score = float(best_state["best_eval_score"])
            else:
                directory = expected["root"] / f"fold_{fold}"
                history = json.loads((directory / "history.json").read_text())
                best_row = max(history, key=lambda row: row["val_miou"])
                final_epoch, best_epoch = len(history), int(best_row["epoch"])
                best_score = float(best_row["val_miou"])
                best_state = torch.load(directory / "best_checkpoint.pytorch", map_location="cpu", weights_only=False)
                require(int(best_state["epoch"]) == best_epoch, f"{model} fold {fold}: checkpoint is not history argmax")
                require(abs(float(best_state["best_eval_score"]) - best_score) < 1e-12,
                        f"{model} fold {fold}: best score mismatch")
                config = best_state.get("config", {})
                require(int(config.get("seed", 42)) == 42, f"{model} fold {fold}: seed mismatch")
                if model == "E2-Static":
                    require(config.get("static_input") is True, f"E2-Static fold {fold}: static_input not saved")
                    require(config.get("augmentation_policy") == "basic",
                            f"E2-Static fold {fold}: augmentation is not basic")
            require(final_epoch == expected["final"][fold-1], f"{model} fold {fold}: final epoch {final_epoch}")
            require(best_epoch == expected["best"][fold-1], f"{model} fold {fold}: best epoch {best_epoch}")
            rows.append({"fold": fold, "final_epoch": final_epoch,
                         "best_epoch": best_epoch, "best_val_miou": best_score})
        report[model] = rows
    return report


def audit_results() -> dict:
    expected = {"oof": (100, 200), "test": (50, 100), "mms": (320, 640)}
    report = {}
    for model, cohorts in RESULTS.items():
        report[model] = {}
        for cohort, path in cohorts.items():
            frame = pd.read_csv(path)
            patients, rows = expected[cohort]
            require({"patient", "phase", "Mean_Dice"}.issubset(frame.columns),
                    f"{model} {cohort}: missing required columns")
            require(len(frame) == rows and frame.patient.nunique() == patients,
                    f"{model} {cohort}: expected {patients}/{rows}, found {frame.patient.nunique()}/{len(frame)}")
            require(not frame.duplicated(["patient", "phase"]).any(),
                    f"{model} {cohort}: duplicate patient-phase row")
            require(set(frame.phase) == {"ED", "ES"}, f"{model} {cohort}: phase mismatch")
            report[model][cohort] = {
                "patients": patients, "volumes": rows,
                "mean_dice": float(frame.Mean_Dice.mean()),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print the full JSON audit")
    args = parser.parse_args()
    report = {
        "status": "PASS", "scope": "completed E1, E2-Full, E2-Static, and E3 only",
        "split": audit_split(), "parameters": audit_parameters(),
        "training": audit_training(), "results": audit_results(),
        "known_ablation_difference": (
            "E2-Full used shuffled/per-epoch workers; completed E2-Static used "
            "patient-grouped/persistent workers"
        ),
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("PASS: four-model release artifacts are internally consistent")
        print(json.dumps({"parameters": report["parameters"], "results": report["results"]}, indent=2))


if __name__ == "__main__":
    main()
