"""Finalize Wolny Exp 1 predictions and aggregate five-fold OOF metrics."""

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
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], group_fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple[str, ...], list[dict]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(groups.items()):
        result = dict(zip(group_fields, key, strict=True))
        result["n_volumes"] = len(group)
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            result[f"{metric}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            result[f"{metric}_std"] = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        output.append(result)
    return output


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--experiment_root", type=Path, default=Path("exp1_wolny_native"))
    parser.add_argument("--source_h5", type=Path, default=Path("exp1_wolny_native_h5/training"))
    parser.add_argument("--pytorch3dunet", type=Path, default=Path("pytorch-3dunet"))
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value used for sequential prediction")
    parser.add_argument("--predict", action="store_true", help="Run Wolny reconstruction before evaluation")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--force_predict", action="store_true")
    args = parser.parse_args()

    invalid = sorted(set(args.folds) - {1, 2, 3, 4, 5})
    if invalid:
        raise ValueError(f"Invalid folds: {invalid}")

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    experiment_root = (repo_root / args.experiment_root).resolve()
    source_h5 = (repo_root / args.source_h5).resolve()
    package_root = (repo_root / args.pytorch3dunet).resolve()
    evaluator = script_dir / "evaluate_exp1_wolny_native.py"

    pooled = []
    for fold in args.folds:
        fold_root = experiment_root / f"fold_{fold}"
        checkpoint = fold_root / "checkpoints" / "best_checkpoint.pytorch"
        prediction_dir = fold_root / "validation_predictions"
        metric_dir = fold_root / "validation_metrics"
        result_csv = metric_dir / "native_volume_results.csv"
        config = experiment_root / "configs" / f"fold_{fold}_val_predict.yaml"

        if not checkpoint.exists():
            raise FileNotFoundError(f"Fold {fold} has no best checkpoint: {checkpoint}")

        prediction_files = sorted(prediction_dir.glob("*_predictions.h5"))
        should_predict = args.predict and (args.force_predict or len(prediction_files) != 40)
        if should_predict:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            old_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = str(package_root) if not old_pythonpath else f"{package_root}{os.pathsep}{old_pythonpath}"
            run([sys.executable, "-m", "pytorch3dunet.predict", "--config", str(config)], env=env)

        prediction_files = sorted(prediction_dir.glob("*_predictions.h5"))
        if len(prediction_files) != 40:
            raise RuntimeError(f"Fold {fold}: expected 40 prediction files, found {len(prediction_files)}")

        if not args.skip_evaluation:
            run([
                sys.executable, str(evaluator),
                "--source_h5", str(source_h5),
                "--predictions", str(prediction_dir),
                "--output", str(metric_dir),
            ])
        if not result_csv.exists():
            raise FileNotFoundError(result_csv)

        rows = read_csv(result_csv)
        if len(rows) != 40:
            raise RuntimeError(f"Fold {fold}: expected 40 metric rows, found {len(rows)}")
        for row in rows:
            row = {key: value for key, value in row.items()}
            row["fold"] = fold
            pooled.append(row)

    keys = [(row["patient"], row["phase"]) for row in pooled]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate patient/phase rows found across folds")
    expected = 40 * len(args.folds)
    if len(pooled) != expected:
        raise RuntimeError(f"Expected {expected} pooled rows, found {len(pooled)}")

    output_dir = experiment_root / "cv_metrics"
    ordered_fields = ["fold", "patient", "phase", *METRICS]
    write_csv(output_dir / "oof_volume_results.csv", pooled, ordered_fields)
    fold_summary = summarize(pooled, ("fold",))
    phase_summary = summarize(pooled, ("phase",))
    fold_phase_summary = summarize(pooled, ("fold", "phase"))
    overall_summary = summarize(pooled, tuple())[0]
    write_csv(output_dir / "fold_summary.csv", fold_summary)
    write_csv(output_dir / "phase_summary.csv", phase_summary)
    write_csv(output_dir / "fold_phase_summary.csv", fold_phase_summary)
    with (output_dir / "oof_summary.json").open("w") as file:
        json.dump({
            "folds": args.folds,
            "num_patients": len(set(row["patient"] for row in pooled)),
            "num_volumes": len(pooled),
            "overall": overall_summary,
            "by_phase": phase_summary,
            "by_fold": fold_summary,
            "by_fold_and_phase": fold_phase_summary,
        }, file, indent=2, allow_nan=True)

    print(f"Pooled {len(pooled)} OOF volumes from {len(args.folds)} fold(s)")
    print(f"Mean Dice: {overall_summary['Mean_Dice_mean']:.4f} +/- {overall_summary['Mean_Dice_std']:.4f}")
    print(f"Saved summaries to {output_dir}")


if __name__ == "__main__":
    main()
