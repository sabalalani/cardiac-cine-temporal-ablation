"""Evaluate and pool Experiment 2 five-fold out-of-fold results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


METRICS = (
    "RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice",
    "RV_HD95", "RV_ASD", "MYO_HD95", "MYO_ASD", "LV_HD95", "LV_ASD",
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for metric in METRICS:
            row[metric] = float(row[metric])
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        result = {name: value for name, value in zip(group_keys, key)}
        result["n_volumes"] = len(group)
        for metric in METRICS:
            values = np.asarray([row[metric] for row in group], dtype=float)
            result[f"{metric}_mean"] = float(np.nanmean(values))
            result[f"{metric}_std"] = float(np.nanstd(values, ddof=1 if len(values) > 1 else 0))
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--experiment_root", type=Path, default=Path("exp2_3dconvlstm"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--evaluate", action="store_true", help="Run full-volume inference first")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--temporal_root", type=Path)
    parser.add_argument("--spatial_root", type=Path)
    parser.add_argument("--spacing_dhw", nargs=3, type=float)
    parser.add_argument("--static_input", action="store_true",
                        help="Evaluate E2-Static checkpoints with repeated target frames.")
    args = parser.parse_args()
    if invalid := sorted(set(args.folds) - {1, 2, 3, 4, 5}):
        raise ValueError(f"Invalid folds: {invalid}")

    script_dir = Path(__file__).resolve().parent
    root = (script_dir.parent / args.experiment_root).resolve()
    evaluator = script_dir / "evaluate_exp2_3dconvlstm.py"
    pooled = []
    for fold in args.folds:
        checkpoint = root / f"fold_{fold}" / "best_checkpoint.pytorch"
        result_csv = root / f"fold_{fold}" / "validation_metrics" / "native_volume_results.csv"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Fold {fold} has not finished training: {checkpoint}")
        if args.evaluate:
            command = [
                sys.executable, str(evaluator), "--fold", str(fold),
                "--checkpoint", str(checkpoint), "--output_root", str(root),
            ]
            if args.save_predictions:
                command.append("--save_predictions")
            if args.static_input:
                command.append("--static_input")
            if args.temporal_root is not None:
                command.extend(["--temporal_root", str(args.temporal_root)])
            if args.spatial_root is not None:
                command.extend(["--spatial_root", str(args.spatial_root)])
            if args.spacing_dhw is not None:
                command.extend(["--spacing_dhw", *map(str, args.spacing_dhw)])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = args.gpu
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, check=True, env=env)
        if not result_csv.exists():
            raise FileNotFoundError(f"Fold {fold} has no full-volume metrics; use --evaluate")
        rows = read_csv(result_csv)
        if len(rows) != 40:
            raise RuntimeError(f"Fold {fold}: expected 40 ED/ES rows, found {len(rows)}")
        for row in rows:
            pooled.append({"fold": fold, **row})

    keys = [(row["patient"], row["phase"]) for row in pooled]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate patient/phase rows across folds")
    expected = 40 * len(args.folds)
    if len(pooled) != expected:
        raise RuntimeError(f"Expected {expected} pooled rows, found {len(pooled)}")

    output = root / "cv_metrics"
    write_csv(output / "oof_volume_results.csv", pooled, ["fold", "patient", "phase", *METRICS])
    fold_summary = summarize(pooled, ("fold",))
    phase_summary = summarize(pooled, ("phase",))
    fold_phase_summary = summarize(pooled, ("fold", "phase"))
    overall = summarize(pooled, tuple())[0]
    write_csv(output / "fold_summary.csv", fold_summary, list(fold_summary[0]))
    write_csv(output / "phase_summary.csv", phase_summary, list(phase_summary[0]))
    write_csv(output / "fold_phase_summary.csv", fold_phase_summary, list(fold_phase_summary[0]))
    with (output / "oof_summary.json").open("w") as file:
        json.dump({
            "folds": args.folds,
            "num_patients": len(set(row["patient"] for row in pooled)),
            "num_volumes": len(pooled),
            "overall": overall,
            "by_phase": phase_summary,
            "by_fold": fold_summary,
            "by_fold_and_phase": fold_phase_summary,
        }, file, indent=2, allow_nan=True)
    print(f"Pooled {len(pooled)} OOF volumes from {len(args.folds)} fold(s)")
    print(f"Mean Dice: {overall['Mean_Dice_mean']:.4f} +/- {overall['Mean_Dice_std']:.4f}")
    print(f"Saved summaries to {output}")


if __name__ == "__main__":
    main()
