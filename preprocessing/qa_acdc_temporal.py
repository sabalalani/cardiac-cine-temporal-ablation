from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_nifti(path):
    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)
    return nii, data


def largest_foreground_slice(volume):
    """
    Choose a reasonable visualization slice from a 3D volume.

    Uses image magnitude rather than GT masks so this works
    independently of segmentation labels.
    """
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")

    slice_energy = np.sum(np.abs(volume), axis=(0, 1))
    return int(np.argmax(slice_energy))


def normalize_for_display(image):
    """
    Robust display normalization only.
    Does NOT alter saved/preprocessed data.
    """
    image = np.asarray(image, dtype=np.float32)

    nonzero = image[np.isfinite(image)]

    if nonzero.size == 0:
        return np.zeros_like(image)

    lo, hi = np.percentile(nonzero, [1, 99])

    if hi <= lo:
        return image

    image = np.clip(image, lo, hi)

    return (image - lo) / (hi - lo)


# ============================================================
# TEMPORAL VALIDATION
# ============================================================

def validate_patient(
    patient_id,
    spatial_cine,
    temporal_cine,
    spatial_meta,
    temporal_meta
):
    errors = []

    if spatial_cine.ndim != 4:
        errors.append(
            f"Spatial cine is not 4D: {spatial_cine.shape}"
        )

    if temporal_cine.ndim != 4:
        errors.append(
            f"Temporal cine is not 4D: {temporal_cine.shape}"
        )

    if temporal_cine.shape != (192, 192, 16, 30):
        errors.append(
            f"Unexpected temporal shape: {temporal_cine.shape}"
        )

    if spatial_cine.shape[:3] != temporal_cine.shape[:3]:
        errors.append(
            "Spatial dimensions differ between "
            "spatial and temporal cine."
        )

    if not np.isfinite(temporal_cine).all():
        errors.append(
            "Temporal cine contains NaN or Inf."
        )

    ed_original_frame = int(
        spatial_meta["ED_original_frame"]
    )

    es_original_frame = int(
        spatial_meta["ES_original_frame"]
    )

    original_t = spatial_cine.shape[-1]

    ed_idx = ed_original_frame - 1
    es_idx = es_original_frame - 1

    # Temporal frame 0 should be exactly the genuine ED frame.
    ed_original = spatial_cine[..., ed_idx]
    ed_temporal = temporal_cine[..., 0]

    ed_mae = float(
        np.mean(np.abs(ed_original - ed_temporal))
    )

    if not np.allclose(
        ed_original,
        ed_temporal,
        atol=1e-5
    ):
        errors.append(
            f"Normalized frame 0 does not exactly "
            f"match original ED. MAE={ed_mae:.6g}"
        )

    expected_ed_normalized = int(
        temporal_meta["ED_normalized_index"]
    )

    if expected_ed_normalized != 0:
        errors.append(
            f"ED normalized index should be 0, "
            f"got {expected_ed_normalized}"
        )

    es_normalized_idx = int(
        temporal_meta["ES_normalized_index"]
    )

    if not (0 <= es_normalized_idx < 30):
        errors.append(
            f"Invalid ES normalized index: "
            f"{es_normalized_idx}"
        )

    relative_es = (
        es_idx - ed_idx
    ) % original_t

    expected_phase = (
        relative_es / original_t
    )

    stored_phase = float(
        temporal_meta["ES_normalized_phase"]
    )

    if not np.isclose(
        expected_phase,
        stored_phase,
        atol=1e-6
    ):
        errors.append(
            f"ES phase mismatch: expected "
            f"{expected_phase:.6f}, "
            f"stored {stored_phase:.6f}"
        )

    return {
        "patient": patient_id,
        "original_T": original_t,
        "original_ED": ed_original_frame,
        "original_ES": es_original_frame,
        "normalized_ED_index": 0,
        "normalized_ES_index": es_normalized_idx,
        "normalized_ES_phase": stored_phase,
        "ED_MAE": ed_mae,
        "errors": errors,
    }


# ============================================================
# STATIC QA FIGURE
# ============================================================

def save_comparison_figure(
    patient_id,
    spatial_cine,
    temporal_cine,
    spatial_meta,
    temporal_meta,
    output_path
):
    ed_frame = int(
        spatial_meta["ED_original_frame"]
    )

    es_frame = int(
        spatial_meta["ES_original_frame"]
    )

    original_t = spatial_cine.shape[-1]

    ed_idx = ed_frame - 1
    es_idx = es_frame - 1

    normalized_es_idx = int(
        temporal_meta["ES_normalized_index"]
    )

    # --------------------------------------------------------
    # Choose Z slice based on original ED
    # --------------------------------------------------------

    z_idx = largest_foreground_slice(
        spatial_cine[..., ed_idx]
    )

    # --------------------------------------------------------
    # Select temporal comparison points
    # --------------------------------------------------------

    original_mid_idx = (
        ed_idx +
        (
            (es_idx - ed_idx) % original_t
        ) // 2
    ) % original_t

    normalized_mid_idx = (
        normalized_es_idx // 2
    )

    normalized_late_idx = 24

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8)
    )

    # ---------------- Original row ----------------

    original_indices = [
        ed_idx,
        original_mid_idx,
        es_idx,
        (ed_idx - 1) % original_t,
    ]

    original_titles = [
        f"Original ED\nframe {ed_frame}",
        f"Original mid-systole\nindex {original_mid_idx}",
        f"Original ES\nframe {es_frame}",
        f"Original late cycle\nindex {(ed_idx - 1) % original_t}",
    ]

    for col, (idx, title) in enumerate(
        zip(original_indices, original_titles)
    ):
        img = spatial_cine[:, :, z_idx, idx]

        axes[0, col].imshow(
            normalize_for_display(img).T,
            cmap="gray",
            origin="lower"
        )

        axes[0, col].set_title(title)
        axes[0, col].axis("off")

    # ---------------- Normalized row ----------------

    normalized_indices = [
        0,
        normalized_mid_idx,
        normalized_es_idx,
        normalized_late_idx,
    ]

    normalized_titles = [
        "T30 ED\nindex 0",
        f"T30 mid-systole\nindex {normalized_mid_idx}",
        f"T30 mapped ES\nindex {normalized_es_idx}",
        f"T30 late cycle\nindex {normalized_late_idx}",
    ]

    for col, (idx, title) in enumerate(
        zip(normalized_indices, normalized_titles)
    ):
        img = temporal_cine[:, :, z_idx, idx]

        axes[1, col].imshow(
            normalize_for_display(img).T,
            cmap="gray",
            origin="lower"
        )

        axes[1, col].set_title(title)
        axes[1, col].axis("off")

    fig.suptitle(
        f"{patient_id} | "
        f"original T={original_t} | "
        f"ED={ed_frame} | "
        f"ES={es_frame} | "
        f"mapped ES={normalized_es_idx} | "
        f"Z={z_idx}",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# GIF
# ============================================================

def save_temporal_gif(
    patient_id,
    temporal_cine,
    temporal_meta,
    output_path,
    fps=5
):
    es_idx = int(
        temporal_meta["ES_normalized_index"]
    )

    # choose slice from ED
    z_idx = largest_foreground_slice(
        temporal_cine[..., 0]
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    first = normalize_for_display(
        temporal_cine[:, :, z_idx, 0]
    )

    im = ax.imshow(
        first.T,
        cmap="gray",
        origin="lower"
    )

    ax.axis("off")

    title = ax.set_title(
        f"{patient_id} | t=0 | ED"
    )

    def update(frame):
        image = temporal_cine[
            :, :, z_idx, frame
        ]

        im.set_data(
            normalize_for_display(image).T
        )

        if frame == 0:
            phase_label = "ED"

        elif frame == es_idx:
            phase_label = "mapped ES"

        else:
            phase_label = ""

        title.set_text(
            f"{patient_id} | "
            f"t={frame:02d}/29 "
            f"{phase_label}"
        )

        return im, title

    animation = FuncAnimation(
        fig,
        update,
        frames=30,
        interval=200,
        blit=False
    )

    writer = PillowWriter(
        fps=fps
    )

    animation.save(
        output_path,
        writer=writer
    )

    plt.close(fig)


# ============================================================
# PATIENT QA
# ============================================================

def qa_patient(
    patient_id,
    spatial_root,
    temporal_root,
    split_name,
    output_root,
    make_gif=False
):
    spatial_dir = (
        spatial_root /
        split_name /
        patient_id
    )

    temporal_dir = (
        temporal_root /
        split_name /
        patient_id
    )

    spatial_cine_path = (
        spatial_dir /
        f"{patient_id}_cine_spatial.nii.gz"
    )

    spatial_meta_path = (
        spatial_dir /
        "metadata.json"
    )

    temporal_cine_path = (
        temporal_dir /
        f"{patient_id}_cine_T30_EDaligned.nii.gz"
    )

    temporal_meta_path = (
        temporal_dir /
        "temporal_metadata.json"
    )

    required = [
        spatial_cine_path,
        spatial_meta_path,
        temporal_cine_path,
        temporal_meta_path
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    _, spatial_cine = load_nifti(
        spatial_cine_path
    )

    _, temporal_cine = load_nifti(
        temporal_cine_path
    )

    spatial_meta = load_json(
        spatial_meta_path
    )

    temporal_meta = load_json(
        temporal_meta_path
    )

    result = validate_patient(
        patient_id,
        spatial_cine,
        temporal_cine,
        spatial_meta,
        temporal_meta
    )

    print(
        f"\n{patient_id}"
    )

    print(
        f"  Original T: "
        f"{result['original_T']}"
    )

    print(
        f"  ED: "
        f"{result['original_ED']} "
        f"-> normalized 0"
    )

    print(
        f"  ES: "
        f"{result['original_ES']} "
        f"-> normalized "
        f"{result['normalized_ES_index']}"
    )

    print(
        f"  ES phase: "
        f"{result['normalized_ES_phase']:.4f}"
    )

    print(
        f"  ED exact-match MAE: "
        f"{result['ED_MAE']:.8f}"
    )

    if result["errors"]:
        print("  STATUS: FAIL")

        for error in result["errors"]:
            print(
                f"    - {error}"
            )

    else:
        print("  STATUS: PASS")

    output_split = (
        output_root /
        split_name
    )

    output_split.mkdir(
        parents=True,
        exist_ok=True
    )

    save_comparison_figure(
        patient_id,
        spatial_cine,
        temporal_cine,
        spatial_meta,
        temporal_meta,
        output_split /
        f"{patient_id}_comparison.png"
    )

    if make_gif:
        save_temporal_gif(
            patient_id,
            temporal_cine,
            temporal_meta,
            output_split /
            f"{patient_id}_T30.gif"
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--spatial_root",
        type=str,
        default="acdc_preprocessed"
    )

    parser.add_argument(
        "--temporal_root",
        type=str,
        default="acdc_temporal"
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="acdc_temporal_qa"
    )

    parser.add_argument(
        "--split",
        choices=[
            "training",
            "testing"
        ],
        default="training"
    )

    parser.add_argument(
        "--patients",
        nargs="+",
        required=True
    )

    parser.add_argument(
        "--gif",
        action="store_true"
    )

    args = parser.parse_args()

    spatial_root = Path(
        args.spatial_root
    )

    temporal_root = Path(
        args.temporal_root
    )

    output_root = Path(
        args.output_root
    )

    results = []

    failures = []

    for patient_id in args.patients:

        try:
            result = qa_patient(
                patient_id,
                spatial_root,
                temporal_root,
                args.split,
                output_root,
                make_gif=args.gif
            )

            results.append(
                result
            )

            if result["errors"]:
                failures.append(
                    patient_id
                )

        except Exception as e:
            print(
                f"\n{patient_id}: "
                f"QA SCRIPT ERROR -> {e}"
            )

            failures.append(
                patient_id
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        output_root /
        f"{args.split}_qa_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:
        json.dump(
            {
                "split": args.split,
                "patients_checked": args.patients,
                "results": results,
                "failed_patients": failures
            },
            f,
            indent=4
        )

    print("\n======================================")
    print("QA COMPLETE")
    print("======================================")

    print(
        f"Checked: {len(args.patients)}"
    )

    print(
        f"Passed: "
        f"{len(args.patients) - len(failures)}"
    )

    print(
        f"Failed: {len(failures)}"
    )

    if failures:
        print(
            "Failed patients:",
            failures
        )

    print(
        f"\nSaved QA summary: "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()