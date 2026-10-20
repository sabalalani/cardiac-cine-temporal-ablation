from pathlib import Path
import json
import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import zoom


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_SPACING = (1.5, 1.5, 10.0)   # X, Y, Z mm
TARGET_HW = (192, 192)
TARGET_Z = 16

IMAGE_DTYPE = np.float32
MASK_DTYPE = np.uint8


# ============================================================
# BASIC HELPERS
# ============================================================

def read_info_cfg(info_path):
    info = {}

    with open(info_path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()

    if "ED" not in info or "ES" not in info:
        raise ValueError(f"Missing ED/ES in {info_path}")

    return {
        "ED": int(info["ED"]),
        "ES": int(info["ES"]),
        "Group": info.get("Group"),
        "Height": info.get("Height"),
        "Weight": info.get("Weight"),
        "NbFrame": int(info["NbFrame"]) if "NbFrame" in info else None,
    }


def load_canonical(path):
    """Load a phase image or mask in canonical orientation."""
    original_nii = nib.load(str(path))
    original_spacing = tuple(
        float(v)
        for v in original_nii.header.get_zooms()[:3]
    )
    canonical_nii = nib.as_closest_canonical(original_nii)
    data = canonical_nii.get_fdata(dtype=np.float32)
    return canonical_nii, data, original_spacing


def load_cine_like_phase(cine_path, phase_reference_path):
    """Orient 4D cine with the phase-file transform, not its inconsistent affine."""
    cine_nii = nib.load(str(cine_path))
    phase_nii = nib.load(str(phase_reference_path))
    spacing = tuple(float(v) for v in cine_nii.header.get_zooms()[:3])
    source_orientation = nib.orientations.io_orientation(phase_nii.affine)
    target_orientation = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(source_orientation, target_orientation)
    cine = nib.orientations.apply_orientation(
        cine_nii.get_fdata(dtype=np.float32), transform
    )
    return cine, spacing

def spatial_zoom_factors(current_spacing):
    return tuple(
        current_spacing[i] / TARGET_SPACING[i]
        for i in range(3)
    )

def mask_stats(mask, name):
    labels, counts = np.unique(mask, return_counts=True)

    print(
        f"    {name}: shape={mask.shape}, "
        f"labels={dict(zip(labels.astype(int), counts))}, "
        f"foreground={(mask > 0).sum()}"
    )
# ============================================================
# RESAMPLING
# ============================================================

def resample_3d(volume, current_spacing, order):
    """
    volume: [X,Y,Z]

    order:
      1 = linear image interpolation
      0 = nearest-neighbor mask interpolation
    """

    factors = spatial_zoom_factors(current_spacing)

    return zoom(
        volume,
        zoom=factors,
        order=order,
        mode="nearest",
        prefilter=False
    )


def resample_4d(cine, current_spacing):
    """
    cine: [X,Y,Z,T]

    Spatial axes only are resampled.
    Temporal dimension remains unchanged.
    """

    fx, fy, fz = spatial_zoom_factors(current_spacing)

    result = zoom(
        cine,
        zoom=(fx, fy, fz, 1.0),
        order=1,
        mode="nearest",
        prefilter=False
    )

    return result.astype(np.float32)


# ============================================================
# FIXED SPATIAL SIZE
# ============================================================

def center_crop_pad_xy(array, target_x=192, target_y=192):
    """
    Works with:
      [X,Y,Z]
      [X,Y,Z,T]
    """

    x, y = array.shape[:2]

    # crop X
    if x > target_x:
        start = (x - target_x) // 2
        array = array[start:start + target_x, ...]

    # crop Y
    y = array.shape[1]

    if y > target_y:
        start = (y - target_y) // 2
        array = array[:, start:start + target_y, ...]

    # pad if needed
    x, y = array.shape[:2]

    missing_x = target_x - x
    missing_y = target_y - y

    before_x = missing_x // 2
    after_x = missing_x - before_x

    before_y = missing_y // 2
    after_y = missing_y - before_y

    pad_width = [
        (before_x, after_x),
        (before_y, after_y)
    ]

    for _ in range(array.ndim - 2):
        pad_width.append((0, 0))

    return np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=0
    )


def pad_z(array, target_z=16):
    """
    Preserve complete Z coverage.

    No silent Z cropping.

    Works with:
      [X,Y,Z]
      [X,Y,Z,T]
    """

    current_z = array.shape[2]

    if current_z > target_z:
        raise ValueError(
            f"Resampled Z={current_z} > TARGET_Z={target_z}. "
            "Refusing to crop automatically."
        )

    missing = target_z - current_z

    before = missing // 2
    after = missing - before

    pad_width = [
        (0, 0),
        (0, 0),
        (before, after)
    ]

    for _ in range(array.ndim - 3):
        pad_width.append((0, 0))

    result = np.pad(
        array,
        pad_width=pad_width,
        mode="constant",
        constant_values=0
    )

    return result, before, after


# ============================================================
# NORMALIZATION
# ============================================================

def foreground_zscore(cine):
    """
    Normalize using all non-zero voxels in the full cine.

    Background remains zero.
    """

    cine = cine.astype(np.float32)

    foreground = cine != 0
    values = cine[foreground]

    if values.size == 0:
        raise ValueError("No foreground voxels found.")

    mean = float(values.mean())
    std = float(values.std())

    if std < 1e-8:
        raise ValueError("Near-zero intensity standard deviation.")

    output = np.zeros_like(cine, dtype=np.float32)

    output[foreground] = (
        cine[foreground] - mean
    ) / std

    return output, mean, std


def normalize_phase(image, mean, std):
    output = np.zeros_like(image, dtype=np.float32)

    foreground = image != 0

    output[foreground] = (
        image[foreground] - mean
    ) / std

    return output


# ============================================================
# NIFTI SAVING
# ============================================================

def make_target_affine():
    """
    Build a simple canonical affine corresponding to:

        1.5 x 1.5 x 10 mm

    Since data have already been converted to canonical orientation,
    the spatial axes are positive X/Y/Z here.
    """

    affine = np.eye(4, dtype=np.float64)

    affine[0, 0] = TARGET_SPACING[0]
    affine[1, 1] = TARGET_SPACING[1]
    affine[2, 2] = TARGET_SPACING[2]

    return affine


def save_nifti(data, output_path, is_mask=False):
    affine = make_target_affine()

    if is_mask:
        data = data.astype(np.uint8)
    else:
        data = data.astype(np.float32)

    nii = nib.Nifti1Image(
        data,
        affine=affine
    )

    header = nii.header

    if data.ndim == 4:
        header.set_zooms(
            (
                TARGET_SPACING[0],
                TARGET_SPACING[1],
                TARGET_SPACING[2],
                1.0
            )
        )
    else:
        header.set_zooms(TARGET_SPACING)

    if is_mask:
        header.set_data_dtype(np.uint8)
    else:
        header.set_data_dtype(np.float32)

    nib.save(
        nii,
        str(output_path)
    )


# ============================================================
# MASK QA
# ============================================================

def check_mask_labels(mask, patient_id, phase):
    labels = np.unique(mask).astype(int)

    allowed = {0, 1, 2, 3}

    if not set(labels.tolist()).issubset(allowed):
        raise ValueError(
            f"{patient_id} {phase}: unexpected labels {labels}"
        )

    return labels.tolist()


# ============================================================
# QA IMAGE
# ============================================================

def save_qa(
    patient_id,
    ed_image,
    ed_mask,
    es_image,
    es_mask,
    output_path
):
    ed_area = (ed_mask > 0).sum(axis=(0, 1))
    es_area = (es_mask > 0).sum(axis=(0, 1))

    ed_z = int(np.argmax(ed_area))
    es_z = int(np.argmax(es_area))

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(
        ed_image[:, :, ed_z].T,
        cmap="gray",
        origin="lower"
    )
    axes[0, 0].set_title(f"{patient_id} ED")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        ed_image[:, :, ed_z].T,
        cmap="gray",
        origin="lower"
    )
    axes[0, 1].imshow(
        ed_mask[:, :, ed_z].T,
        alpha=0.4,
        origin="lower",
        interpolation="nearest"
    )
    axes[0, 1].set_title("ED + GT")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(
        es_image[:, :, es_z].T,
        cmap="gray",
        origin="lower"
    )
    axes[1, 0].set_title(f"{patient_id} ES")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(
        es_image[:, :, es_z].T,
        cmap="gray",
        origin="lower"
    )
    axes[1, 1].imshow(
        es_mask[:, :, es_z].T,
        alpha=0.4,
        origin="lower",
        interpolation="nearest"
    )
    axes[1, 1].set_title("ES + GT")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# PATIENT PROCESSING
# ============================================================

def preprocess_patient(
    patient_dir,
    output_dir,
    qa_dir=None,
    make_qa=False,
    cine_only=False,
):
    patient_id = patient_dir.name

    print(f"\nProcessing {patient_id}")

    info = read_info_cfg(
        patient_dir / "Info.cfg"
    )

    ed = info["ED"]
    es = info["ES"]

    cine_path = patient_dir / f"{patient_id}_4d.nii.gz"

    ed_image_path = (
        patient_dir /
        f"{patient_id}_frame{ed:02d}.nii.gz"
    )

    ed_mask_path = (
        patient_dir /
        f"{patient_id}_frame{ed:02d}_gt.nii.gz"
    )

    es_image_path = (
        patient_dir /
        f"{patient_id}_frame{es:02d}.nii.gz"
    )

    es_mask_path = (
        patient_dir /
        f"{patient_id}_frame{es:02d}_gt.nii.gz"
    )

    for path in [
        cine_path,
        ed_image_path,
        ed_mask_path,
        es_image_path,
        es_mask_path
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    # ========================================================
    # CINE
    # ========================================================

    cine, spacing = load_cine_like_phase(cine_path, ed_image_path)

    if cine.ndim != 4:
        raise ValueError(
            f"Expected 4D cine, got {cine.shape}"
        )

    original_shape = tuple(int(v) for v in cine.shape)
    original_spacing = spacing

    print(
        f"  Raw cine: {original_shape}, "
        f"spacing={original_spacing}"
    )

    cine = resample_4d(
        cine,
        original_spacing
    )

    resampled_shape = tuple(
        int(v) for v in cine.shape
    )

    print(
        f"  After spatial resampling: {resampled_shape}"
    )

    cine = center_crop_pad_xy(
        cine,
        TARGET_HW[0],
        TARGET_HW[1]
    )

    cine, z_before, z_after = pad_z(
        cine,
        TARGET_Z
    )

    print(
        f"  After crop/pad: {cine.shape}"
    )

    cine, norm_mean, norm_std = foreground_zscore(
        cine
    )

    # ========================================================
    # PHASE PROCESSOR
    # ========================================================

    def process_phase(
        image_path,
        mask_path,
        phase_name
    ):
        _, image, image_spacing = load_canonical(
            image_path
        )

        _, mask, mask_spacing = load_canonical(mask_path)

        mask_stats(mask, f"{phase_name} ORIGINAL")

        if not np.allclose(
            image_spacing,
            mask_spacing,
            atol=1e-5
        ):
            raise ValueError(
                f"{patient_id} {phase_name}: "
                "image/mask spacing mismatch"
            )

        image = resample_3d(
            image,
            image_spacing,
            order=1
        )

        mask = resample_3d(
            mask,
            mask_spacing,
            order=0
        )

        image = center_crop_pad_xy(
            image,
            TARGET_HW[0],
            TARGET_HW[1]
        )

        mask = center_crop_pad_xy(
            mask,
            TARGET_HW[0],
            TARGET_HW[1]
        )

        image, _, _ = pad_z(
            image,
            TARGET_Z
        )

        mask, _, _ = pad_z(
            mask,
            TARGET_Z
        )

        mask_stats(mask, f"{phase_name} AFTER SINGLE RESAMPLE/CROP/PAD")

        image = normalize_phase(
            image,
            norm_mean,
            norm_std
        )

        mask = np.rint(mask).astype(
            np.uint8
        )

        labels = check_mask_labels(
            mask,
            patient_id,
            phase_name
        )

        return image, mask, labels

    ed_image, ed_mask, ed_labels = process_phase(
        ed_image_path,
        ed_mask_path,
        "ED"
    )

    es_image, es_mask, es_labels = process_phase(
        es_image_path,
        es_mask_path,
        "ES"
    )

    # ========================================================
    # FINAL SHAPE CHECKS
    # ========================================================

    expected_xyz = (
        TARGET_HW[0],
        TARGET_HW[1],
        TARGET_Z
    )

    if cine.shape[:3] != expected_xyz:
        raise ValueError(
            f"Cine spatial shape {cine.shape[:3]} "
            f"!= expected {expected_xyz}"
        )

    for name, arr in [
        ("ED image", ed_image),
        ("ED mask", ed_mask),
        ("ES image", es_image),
        ("ES mask", es_mask)
    ]:
        if arr.shape != expected_xyz:
            raise ValueError(
                f"{name} shape {arr.shape} "
                f"!= expected {expected_xyz}"
            )

    # ========================================================
    # SAVE NIFTI
    # ========================================================

    patient_output = output_dir / patient_id
    patient_output.mkdir(
        parents=True,
        exist_ok=True
    )

    cine_output = (
        patient_output /
        f"{patient_id}_cine_spatial.nii.gz"
    )

    ed_image_output = (
        patient_output /
        f"{patient_id}_ED.nii.gz"
    )

    ed_mask_output = (
        patient_output /
        f"{patient_id}_ED_gt.nii.gz"
    )

    es_image_output = (
        patient_output /
        f"{patient_id}_ES.nii.gz"
    )

    es_mask_output = (
        patient_output /
        f"{patient_id}_ES_gt.nii.gz"
    )

    save_nifti(
        cine,
        cine_output,
        is_mask=False
    )

    if not cine_only:
        save_nifti(ed_image, ed_image_output, is_mask=False)
        save_nifti(ed_mask, ed_mask_output, is_mask=True)
        save_nifti(es_image, es_image_output, is_mask=False)
        save_nifti(es_mask, es_mask_output, is_mask=True)

    # ========================================================
    # METADATA
    # ========================================================

    metadata = {
        "patient": patient_id,
        "group": info["Group"],

        "ED_original_frame": int(ed),
        "ES_original_frame": int(es),

        "original_shape_XYZT": list(
            original_shape
        ),

        "original_spacing_XYZ": [
            float(v) for v in original_spacing
        ],

        "target_spacing_XYZ": list(
            TARGET_SPACING
        ),

        "resampled_shape_before_crop_XYZT": list(
            resampled_shape
        ),

        "final_cine_shape_XYZT": [
            int(v) for v in cine.shape
        ],

        "final_phase_shape_XYZ": [
            int(v) for v in ed_image.shape
        ],

        "target_hw": list(TARGET_HW),
        "target_z": TARGET_Z,

        "z_padding_before": int(z_before),
        "z_padding_after": int(z_after),

        "normalization": (
            "foreground z-score calculated "
            "over full spatially standardized cine"
        ),

        "normalization_mean": float(norm_mean),
        "normalization_std": float(norm_std),

        "image_interpolation": "linear",
        "mask_interpolation": "nearest-neighbor",

        "ED_mask_labels": ed_labels,
        "ES_mask_labels": es_labels,

        "temporal_resampling_applied": False
    }

    with open(
        patient_output / "metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    # ========================================================
    # QA
    # ========================================================

    if make_qa and qa_dir is not None and not cine_only:
        qa_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_qa(
            patient_id,
            ed_image,
            ed_mask,
            es_image,
            es_mask,
            qa_dir / f"{patient_id}_qa.png"
        )

    print(
        f"  Final cine: {cine.shape}"
    )

    print(
        f"  ED: {ed_image.shape}, "
        f"labels={ed_labels}"
    )

    print(
        f"  ES: {es_image.shape}, "
        f"labels={es_labels}"
    )

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        default="data/ACDC"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="acdc_preprocessed"
    )

    parser.add_argument(
        "--split",
        choices=[
            "training",
            "testing",
            "both"
        ],
        default="training"
    )

    parser.add_argument(
        "--patients",
        nargs="*",
        default=None
    )

    parser.add_argument(
        "--qa",
        action="store_true"
    )

    parser.add_argument(
        "--cine_only",
        action="store_true",
        help="Rewrite corrected cine and metadata without changing Exp 1 phase files",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    splits = (
        ["training", "testing"]
        if args.split == "both"
        else [args.split]
    )

    processed = []
    failures = []

    for split_name in splits:
        split_input = input_root / split_name
        split_output = output_root / split_name
        qa_output = output_root / "qa" / split_name

        if args.patients:
            patient_dirs = [
                split_input / patient
                for patient in args.patients
                if (split_input / patient).exists()
            ]
        else:
            patient_dirs = sorted(
                p for p in split_input.iterdir()
                if p.is_dir()
                and p.name.startswith("patient")
            )

        print("\n======================================")
        print(f"Split: {split_name}")
        print(f"Patients: {len(patient_dirs)}")
        print("======================================")

        for patient_dir in patient_dirs:
            try:
                metadata = preprocess_patient(
                    patient_dir,
                    split_output,
                    qa_dir=qa_output,
                    make_qa=args.qa,
                    cine_only=args.cine_only,
                )

                metadata["split"] = split_name
                processed.append(metadata)

            except Exception as e:
                print(
                    f"\nFAILED {patient_dir.name}: {e}"
                )

                failures.append({
                    "split": split_name,
                    "patient": patient_dir.name,
                    "error": str(e)
                })

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {
        "target_spacing_XYZ": list(
            TARGET_SPACING
        ),

        "target_HW": list(
            TARGET_HW
        ),

        "target_Z": TARGET_Z,

        "processed": processed,
        "failures": failures
    }

    with open(
        output_root /
        "preprocessing_summary.json",
        "w"
    ) as f:
        json.dump(
            summary,
            f,
            indent=4
        )

    print("\n======================================")
    print("PREPROCESSING COMPLETE")
    print("======================================")

    print(
        f"Successful: {len(processed)}"
    )

    print(
        f"Failures: {len(failures)}"
    )

    if failures:
        for failure in failures:
            print(
                failure["patient"],
                "->",
                failure["error"]
            )


if __name__ == "__main__":
    main()
