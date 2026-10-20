"""Prepare native-spacing ACDC HDF5 files and Wolny five-fold configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import yaml


MODEL = {
    "name": "UNet3D",
    "in_channels": 1,
    "out_channels": 4,
    "layer_order": "gcr",
    "f_maps": [32, 64, 128],
    "num_groups": 8,
    "final_sigmoid": False,
    "is_segmentation": True,
}


def phase_paths(data_root: Path, split: str, patient: str, phase: str):
    directory = data_root / split / patient
    return (
        directory / f"{patient}_{phase}.nii.gz",
        directory / f"{patient}_{phase}_gt.nii.gz",
    )


def convert_one(image_path: Path, mask_path: Path, output_path: Path, force: bool):
    if output_path.exists() and not force:
        try:
            with h5py.File(output_path, "r") as file:
                valid = (
                    "raw" in file
                    and "label" in file
                    and file["raw"].shape == (16, 192, 192)
                    and file["label"].shape == (16, 192, 192)
                )
                if valid:
                    # Force decompression now so truncated/corrupt datasets are
                    # detected instead of failing later inside a worker.
                    _ = file["raw"][:]
                    _ = file["label"][:]
                    return False
        except (OSError, KeyError):
            pass
        print(f"Rebuilding corrupt/incomplete file: {output_path}")
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    image = image_nii.get_fdata(dtype=np.float32)
    mask = np.rint(mask_nii.get_fdata()).astype(np.int64)
    if image.shape != (192, 192, 16) or mask.shape != image.shape:
        raise ValueError(f"Unexpected shape for {image_path}: image={image.shape}, mask={mask.shape}")
    if not set(np.unique(mask)).issubset({0, 1, 2, 3}):
        raise ValueError(f"Unexpected labels in {mask_path}: {np.unique(mask)}")

    # NIfTI XYZ -> Wolny DHW (ZYX).
    image = np.ascontiguousarray(np.transpose(image, (2, 1, 0)))
    mask = np.ascontiguousarray(np.transpose(mask, (2, 1, 0)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as file:
        file.create_dataset("raw", data=image, compression="gzip")
        file.create_dataset("label", data=mask, compression="gzip")
        file.attrs["spacing_zyx_mm"] = np.asarray([10.0, 1.5, 1.5], dtype=np.float32)
        file.attrs["source_image"] = str(image_path)
        file.attrs["source_mask"] = str(mask_path)
    return True


def h5_paths(pool: Path, source_split: str, patients: list[str]) -> list[str]:
    return [
        str((pool / source_split / f"{patient}_{phase}.h5").resolve())
        for patient in patients
        for phase in ("ED", "ES")
    ]


def train_config(fold: int, train_paths: list[str], val_paths: list[str], output_root: Path):
    checkpoint_dir = (output_root / f"fold_{fold}" / "checkpoints").resolve()
    return {
        "manual_seed": 42,
        "model": MODEL,
        "loss": {
            "name": "CrossEntropyLoss",
            "weight": [0.05, 1.3, 2.0, 1.2],
            "ignore_index": 255,
        },
        "optimizer": {"learning_rate": 0.0002, "weight_decay": 0.00001},
        "eval_metric": {"name": "MeanIoU", "ignore_index": 255},
        "lr_scheduler": {"name": "ReduceLROnPlateau", "mode": "max", "factor": 0.2, "patience": 8},
        "trainer": {
            "checkpoint_dir": str(checkpoint_dir),
            "resume": None,
            "pre_trained": None,
            "validate_after_iters": 1000,
            "log_after_iters": 500,
            "max_num_epochs": 150,
            "max_num_iterations": 150000,
            "eval_score_higher_is_better": True,
        },
        "loaders": {
            "dataset": "StandardHDF5Dataset",
            "batch_size": 2,
            "num_workers": 4,
            "raw_internal_path": "raw",
            "label_internal_path": "label",
            "global_normalization": True,
            "train": {
                "file_paths": train_paths,
                "slice_builder": {
                    "name": "FilterSliceBuilder",
                    "patch_shape": [8, 128, 128],
                    "stride_shape": [4, 64, 64],
                    "threshold": 0.01,
                    "slack_acceptance": 0.01,
                },
                "transformer": {
                    "raw": [
                        {"name": "Standardize"},
                        {"name": "RandomFlip"},
                        {"name": "RandomRotate90"},
                        {"name": "ToTensor", "expand_dims": True},
                    ],
                    "label": [
                        {"name": "RandomFlip"},
                        {"name": "RandomRotate90"},
                        {"name": "ToTensor", "expand_dims": False, "dtype": "int64"},
                    ],
                },
            },
            "val": {
                "file_paths": val_paths,
                "slice_builder": {
                    "name": "FilterSliceBuilder",
                    "patch_shape": [8, 128, 128],
                    "stride_shape": [4, 64, 64],
                    "threshold": 0.01,
                    "slack_acceptance": 0.01,
                },
                "transformer": {
                    "raw": [{"name": "Standardize"}, {"name": "ToTensor", "expand_dims": True}],
                    "label": [{"name": "ToTensor", "expand_dims": False, "dtype": "int64"}],
                },
            },
        },
    }


def prediction_config(fold: int, paths: list[str], output_root: Path, name: str):
    fold_root = output_root / f"fold_{fold}"
    return {
        "manual_seed": 42,
        "model_path": str((fold_root / "checkpoints" / "best_checkpoint.pytorch").resolve()),
        "model": MODEL,
        "predictor": {
            "name": "StandardPredictor",
            "save_segmentation": True,
            "performance_metric": "mean_iou",
            "gt_internal_path": "label",
        },
        "loaders": {
            "dataset": "StandardHDF5Dataset",
            "output_dir": str((fold_root / f"{name}_predictions").resolve()),
            "batch_size": 2,
            "num_workers": 4,
            "raw_internal_path": "raw",
            "global_normalization": True,
            "test": {
                "file_paths": paths,
                "slice_builder": {
                    "name": "SliceBuilder",
                    "patch_shape": [8, 128, 128],
                    "stride_shape": [4, 64, 64],
                    "halo_shape": [2, 16, 16],
                },
                "transformer": {
                    "raw": [{"name": "Standardize"}, {"name": "ToTensor", "expand_dims": True}]
                },
            },
        },
    }


def write_yaml(path: Path, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        yaml.safe_dump(config, file, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("acdc_preprocessed"))
    parser.add_argument("--split_json", type=Path, default=Path("configs/acdc_5fold_splits_seed42.json"))
    parser.add_argument("--h5_root", type=Path, default=Path("exp1_wolny_native_h5"))
    parser.add_argument("--output_root", type=Path, default=Path("exp1_wolny_native"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--augmentation_policy",
        choices=("basic", "moderate_v1", "moderate_v2", "strong_v1"),
        default="basic",
    )
    args = parser.parse_args()

    with args.split_json.open() as file:
        splits = json.load(file)
    created = 0
    for source_split, patients in (
        ("training", splits["development_patients"]),
        ("testing", splits["official_test_patients"]),
    ):
        for patient in patients:
            for phase in ("ED", "ES"):
                image, mask = phase_paths(args.data_root, source_split, patient, phase)
                output = args.h5_root / source_split / f"{patient}_{phase}.h5"
                created += int(convert_one(image, mask, output, args.force))

    official_test = h5_paths(args.h5_root, "testing", splits["official_test_patients"])
    config_dir = args.output_root / "configs"
    for fold in range(1, 6):
        fold_data = splits["folds"][f"fold_{fold}"]
        train = h5_paths(args.h5_root, "training", fold_data["train"])
        val = h5_paths(args.h5_root, "training", fold_data["val"])
        config = train_config(fold, train, val, args.output_root)
        if args.augmentation_policy == "moderate_v1":
            # Appearance-focused domain randomization. Preserve the basic
            # geometry and perturb scanner contrast/noise/resolution mildly.
            config["augmentation_policy"] = "moderate_v1"
            config["loaders"]["train"]["transformer"]["raw"] = [
                {"name": "Standardize"},
                {"name": "RandomFlip"},
                {"name": "RandomRotate90"},
                {"name": "GaussianBlur3D", "sigma": [0.2, 0.5], "execution_probability": 0.10},
                {"name": "AdditiveGaussianNoise", "scale": [0.0, 0.05], "execution_probability": 0.20},
                {"name": "SignedGamma", "gamma": [0.85, 1.20], "execution_probability": 0.25},
                {"name": "ToTensor", "expand_dims": True},
            ]
        elif args.augmentation_policy == "moderate_v2":
            config["augmentation_policy"] = "moderate_v2"
            config["augmentation_specification"] = {
                "random_in_plane_rotation": {"probability": 0.30, "angle_degrees": [-10.0, 10.0]},
                "intensity_quality": {
                    "overall_probability": 0.30,
                    "mutually_exclusive": True,
                    "choice_probabilities": {"gamma": 0.50, "noise": 0.30, "blur": 0.20},
                    "gamma": [0.85, 1.20],
                    "noise_sd": [0.0, 0.03],
                    "blur_sigma": [0.3, 0.5],
                },
            }
            config["loaders"]["train"]["transformer"]["raw"] = [
                {"name": "Standardize"},
                {"name": "RandomFlip"},
                {"name": "RandomRotate90"},
                {"name": "RandomInPlaneRotate", "angle_spectrum": 10.0, "execution_probability": 0.30,
                 "mode": "reflect", "order": 1},
                {"name": "RandomChoiceMRIQuality", "execution_probability": 0.30,
                 "choice_probabilities": [0.50, 0.30, 0.20], "gamma": [0.85, 1.20],
                 "noise_scale": [0.0, 0.03], "blur_sigma": [0.3, 0.5]},
                {"name": "ToTensor", "expand_dims": True},
            ]
            config["loaders"]["train"]["transformer"]["label"] = [
                {"name": "RandomFlip"},
                {"name": "RandomRotate90"},
                {"name": "RandomInPlaneRotate", "angle_spectrum": 10.0, "execution_probability": 0.30,
                 "mode": "reflect", "order": 0},
                {"name": "ToTensor", "expand_dims": False, "dtype": "int64"},
            ]
        elif args.augmentation_policy == "strong_v1":
            config["augmentation_policy"] = "strong_v1"
            config["loaders"]["train"]["transformer"]["raw"] = [
                {"name": "Standardize"},
                {"name": "RandomFlip"},
                {"name": "RandomRotate90"},
                {"name": "RandomRotate", "angle_spectrum": 20, "axes": [[1, 2]], "mode": "reflect", "order": 1},
                {"name": "GaussianBlur3D", "sigma": [0.3, 0.8], "execution_probability": 0.20},
                {"name": "AdditiveGaussianNoise", "scale": [0.0, 0.10], "execution_probability": 0.25},
                {"name": "SignedGamma", "gamma": [0.7, 1.5], "execution_probability": 0.30},
                {"name": "ToTensor", "expand_dims": True},
            ]
            config["loaders"]["train"]["transformer"]["label"] = [
                {"name": "RandomFlip"},
                {"name": "RandomRotate90"},
                {"name": "RandomRotate", "angle_spectrum": 20, "axes": [[1, 2]], "mode": "reflect", "order": 0},
                {"name": "ToTensor", "expand_dims": False, "dtype": "int64"},
            ]
        write_yaml(config_dir / f"fold_{fold}_train.yaml", config)
        write_yaml(config_dir / f"fold_{fold}_val_predict.yaml", prediction_config(fold, val, args.output_root, "validation"))
        write_yaml(config_dir / f"fold_{fold}_test_predict.yaml", prediction_config(fold, official_test, args.output_root, "official_test"))

    print(f"Created {created} HDF5 files; reusable pool contains 300 files")
    print(f"Configs written to {config_dir.resolve()}")


if __name__ == "__main__":
    main()
