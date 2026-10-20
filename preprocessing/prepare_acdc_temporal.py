from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np


TARGET_T = 30


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def periodic_temporal_resample(cine, target_t):
    """
    Periodically resample a cine sequence along the last axis.

    Input:
        cine shape = [X, Y, Z, T]

    Output:
        [X, Y, Z, target_t]

    The sequence is treated as cyclic:
        final frame connects back to first frame.
    """

    if cine.ndim != 4:
        raise ValueError(
            f"Expected 4D cine [X,Y,Z,T], got {cine.shape}"
        )

    original_t = cine.shape[-1]

    if original_t < 2:
        raise ValueError(
            f"Temporal length must be >= 2, got T={original_t}"
        )

    # Positions in original cyclic cardiac sequence.
    #
    # Example T=12:
    # 0,1,2,...,11
    #
    # We append frame 0 again at temporal position 12
    # so interpolation across the cycle boundary is continuous.
    old_positions = np.arange(
        original_t + 1,
        dtype=np.float32
    )

    extended = np.concatenate(
        [
            cine,
            cine[..., 0:1]
        ],
        axis=-1
    )

    # Sample target phases uniformly over [0, T)
    new_positions = np.linspace(
        0,
        original_t,
        target_t,
        endpoint=False,
        dtype=np.float32
    )

    # Find neighboring temporal indices.
    left = np.floor(new_positions).astype(int)
    right = left + 1

    alpha = (
        new_positions - left
    ).astype(np.float32)

    output_shape = (
        cine.shape[0],
        cine.shape[1],
        cine.shape[2],
        target_t
    )

    output = np.empty(
        output_shape,
        dtype=np.float32
    )

    for i in range(target_t):
        output[..., i] = (
            (1.0 - alpha[i]) * extended[..., left[i]]
            +
            alpha[i] * extended[..., right[i]]
        )

    return output, new_positions



def ed_es_anchored_temporal_resample(shifted_cine, relative_es, target_t, es_target_idx):
    """Resample cyclic time while preserving the exact ED and ES volumes."""
    original_t = shifted_cine.shape[-1]
    if not 0 < relative_es < original_t:
        raise ValueError(f"ES must differ from ED, got relative ES={relative_es}")
    if not 0 < es_target_idx < target_t:
        raise ValueError(f"Invalid target ES index {es_target_idx} for T={target_t}")

    sampled_positions = np.empty(target_t, dtype=np.float32)
    sampled_positions[: es_target_idx + 1] = np.linspace(
        0.0, float(relative_es), es_target_idx + 1, endpoint=True, dtype=np.float32
    )
    sampled_positions[es_target_idx:] = np.linspace(
        float(relative_es), float(original_t), target_t - es_target_idx,
        endpoint=False, dtype=np.float32
    )

    extended = np.concatenate([shifted_cine, shifted_cine[..., :1]], axis=-1)
    left = np.floor(sampled_positions).astype(np.int64)
    right = left + 1
    alpha = sampled_positions - left
    output = np.empty((*shifted_cine.shape[:3], target_t), dtype=np.float32)
    for index in range(target_t):
        output[..., index] = (
            (1.0 - alpha[index]) * extended[..., left[index]]
            + alpha[index] * extended[..., right[index]]
        )
    return output, sampled_positions

def circular_shift_to_ed(cine, ed_frame_1based):
    """
    Circularly shift cine so ED becomes temporal index 0.

    ACDC ED value is 1-based.

    Example:
        T = 12
        ED = 4

    Original:
        1 2 3 4 5 ... 12

    Shifted:
        4 5 ... 12 1 2 3
    """

    original_t = cine.shape[-1]

    ed_idx = ed_frame_1based - 1

    if ed_idx < 0 or ed_idx >= original_t:
        raise ValueError(
            f"Invalid ED={ed_frame_1based} for T={original_t}"
        )

    shifted = np.roll(
        cine,
        shift=-ed_idx,
        axis=-1
    )

    return shifted, ed_idx


def compute_relative_es_index(
    original_t,
    ed_frame_1based,
    es_frame_1based
):
    """
    Compute ES location after ED alignment.

    Returns the ES position in the ED-aligned original cycle,
    using zero-based temporal coordinates.
    """

    ed_idx = ed_frame_1based - 1
    es_idx = es_frame_1based - 1

    if not (0 <= ed_idx < original_t):
        raise ValueError(
            f"Invalid ED={ed_frame_1based} for T={original_t}"
        )

    if not (0 <= es_idx < original_t):
        raise ValueError(
            f"Invalid ES={es_frame_1based} for T={original_t}"
        )

    relative_es = (
        es_idx - ed_idx
    ) % original_t

    return int(relative_es)


def map_original_position_to_target(
    original_position,
    original_t,
    target_t
):
    """
    Map a temporal position from the original cycle
    to the nearest target temporal index.
    """

    normalized_phase = (
        float(original_position) / float(original_t)
    )

    mapped_float = (
        normalized_phase * target_t
    )

    mapped_idx = int(
        np.round(mapped_float)
    ) % target_t

    return mapped_idx, normalized_phase, mapped_float


def validate_temporal_result(
    original_cine,
    shifted_cine,
    normalized_cine,
    expected_t
):
    if shifted_cine.shape != original_cine.shape:
        raise ValueError(
            "Circular shift changed cine shape unexpectedly."
        )

    expected_shape = (
        original_cine.shape[0],
        original_cine.shape[1],
        original_cine.shape[2],
        expected_t
    )

    if normalized_cine.shape != expected_shape:
        raise ValueError(
            f"Temporal output shape {normalized_cine.shape} "
            f"!= expected {expected_shape}"
        )

    if not np.isfinite(normalized_cine).all():
        raise ValueError(
            "Temporal cine contains NaN or Inf values."
        )


def save_nifti_like(
    data,
    source_nii,
    output_path
):
    """
    Save using the spatial affine/header of the
    spatially preprocessed cine.

    Spatial spacing is preserved.
    Temporal zoom is set to 1 because we are working
    in normalized cardiac phase rather than physical seconds.
    """

    affine = source_nii.affine.copy()
    header = source_nii.header.copy()

    output = nib.Nifti1Image(
        data.astype(np.float32),
        affine=affine,
        header=header
    )

    spatial_zooms = source_nii.header.get_zooms()[:3]

    output.header.set_zooms(
        (
            float(spatial_zooms[0]),
            float(spatial_zooms[1]),
            float(spatial_zooms[2]),
            1.0
        )
    )

    output.header.set_data_dtype(
        np.float32
    )

    nib.save(
        output,
        str(output_path)
    )


def preprocess_patient(
    patient_dir,
    output_root,
    split_name
):
    patient_id = patient_dir.name

    print(f"\nProcessing {patient_id}")

    cine_path = (
        patient_dir /
        f"{patient_id}_cine_spatial.nii.gz"
    )

    metadata_path = (
        patient_dir /
        "metadata.json"
    )

    if not cine_path.exists():
        raise FileNotFoundError(
            f"Missing cine: {cine_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata: {metadata_path}"
        )

    metadata = read_json(
        metadata_path
    )

    ed_frame = int(
        metadata["ED_original_frame"]
    )

    es_frame = int(
        metadata["ES_original_frame"]
    )

    source_nii = nib.load(
        str(cine_path)
    )

    cine = source_nii.get_fdata(
        dtype=np.float32
    )

    if cine.ndim != 4:
        raise ValueError(
            f"{patient_id}: expected 4D cine, got {cine.shape}"
        )

    original_t = cine.shape[-1]

    print(
        f"  Input cine: {cine.shape}"
    )

    print(
        f"  Original ED={ed_frame}, "
        f"ES={es_frame}, "
        f"T={original_t}"
    )

    # ========================================================
    # STEP 1 — ED ALIGNMENT
    # ========================================================

    shifted_cine, original_ed_idx = (
        circular_shift_to_ed(
            cine,
            ed_frame
        )
    )

    relative_es = (
        compute_relative_es_index(
            original_t,
            ed_frame,
            es_frame
        )
    )

    print(
        f"  ED aligned to index 0"
    )

    print(
        f"  ES position after ED alignment: "
        f"{relative_es}/{original_t}"
    )

    (
        es_normalized_idx,
        es_normalized_phase,
        es_target_float
    ) = map_original_position_to_target(
        relative_es,
        original_t,
        TARGET_T
    )

    # Piecewise ED-to-ES and ES-to-next-ED resampling preserves both
    # genuinely annotated frames at their supervised temporal indices.
    temporal_cine, sampled_positions = ed_es_anchored_temporal_resample(
        shifted_cine,
        relative_es,
        TARGET_T,
        es_normalized_idx,
    )

    ed_normalized_idx = 0

    print(
        f"  Output cine: {temporal_cine.shape}"
    )

    print(
        f"  ED normalized index: "
        f"{ed_normalized_idx}"
    )

    print(
        f"  ES normalized index: "
        f"{es_normalized_idx}"
    )

    print(
        f"  ES normalized phase: "
        f"{es_normalized_phase:.4f}"
    )

    # ========================================================
    # QA
    # ========================================================

    validate_temporal_result(
        cine,
        shifted_cine,
        temporal_cine,
        TARGET_T
    )

    # Important sanity check:
    # shifted cine frame 0 should exactly equal
    # the original ED frame.
    original_ed_volume = (
        cine[..., original_ed_idx]
    )

    if not np.allclose(
        shifted_cine[..., 0],
        original_ed_volume,
        atol=1e-6
    ):
        raise ValueError(
            f"{patient_id}: ED alignment check failed"
        )

    # Because temporal sampling starts at position 0,
    # normalized frame 0 must also be EXACT ED.
    if not np.allclose(
        temporal_cine[..., 0],
        original_ed_volume,
        atol=1e-6
    ):
        raise ValueError(
            f"{patient_id}: normalized frame 0 "
            "does not exactly equal ED"
        )


    original_es_volume = shifted_cine[..., relative_es]
    if not np.allclose(
        temporal_cine[..., es_normalized_idx],
        original_es_volume,
        atol=1e-6,
    ):
        raise ValueError(
            f"{patient_id}: normalized ES frame does not exactly equal ES"
        )

    # ========================================================
    # SAVE
    # ========================================================

    patient_output = (
        output_root /
        split_name /
        patient_id
    )

    patient_output.mkdir(
        parents=True,
        exist_ok=True
    )

    output_cine_path = (
        patient_output /
        f"{patient_id}_cine_T30_EDaligned.nii.gz"
    )

    save_nifti_like(
        temporal_cine,
        source_nii,
        output_cine_path
    )

    output_metadata = {
        "patient": patient_id,
        "split": split_name,

        "source_cine": str(
            cine_path
        ),

        "spatial_shape_XYZ": [
            int(v)
            for v in cine.shape[:3]
        ],

        "original_T": int(
            original_t
        ),

        "target_T": int(
            TARGET_T
        ),

        "ED_original_frame_1based": int(
            ed_frame
        ),

        "ES_original_frame_1based": int(
            es_frame
        ),

        "ED_original_index_0based": int(
            original_ed_idx
        ),

        "ES_relative_position_after_ED_alignment": int(
            relative_es
        ),

        "ED_normalized_index": int(
            ed_normalized_idx
        ),

        "ES_normalized_index": int(
            es_normalized_idx
        ),

        "ES_normalized_phase": float(
            es_normalized_phase
        ),

        "ES_target_continuous_position": float(
            es_target_float
        ),

        "temporal_alignment": (
            "circular shift so ED is temporal index 0"
        ),

        "temporal_interpolation": (
            "piecewise periodic linear interpolation anchored at exact ED and ES"
        ),

        "temporal_endpoint_behavior": (
            "cycle treated as periodic; "
            "last frame interpolates toward first frame"
        ),

        "segmentation_masks_temporally_interpolated": False,
        "ED_exact_frame_preserved": True,
        "ES_exact_frame_preserved": True,

        "output_shape_XYZT": [
            int(v)
            for v in temporal_cine.shape
        ],

        "sampled_original_temporal_positions": [
            float(v)
            for v in sampled_positions
        ],

        "spatial_spacing_XYZ": [
            float(v)
            for v
            in source_nii.header.get_zooms()[:3]
        ],

        "temporal_axis": (
            "normalized cardiac phase; "
            "not physical time in seconds"
        )
    }

    save_json(
        output_metadata,
        patient_output /
        "temporal_metadata.json"
    )

    return output_metadata


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_root",
        type=str,
        default="acdc_preprocessed",
        help=(
            "Spatially preprocessed ACDC root "
            "containing training/ and testing/"
        )
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default="acdc_temporal",
        help="Temporal output root"
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
        default=None,
        help=(
            "Optional patient IDs, e.g. "
            "--patients patient001 patient090"
        )
    )

    args = parser.parse_args()

    input_root = Path(
        args.input_root
    )

    output_root = Path(
        args.output_root
    )

    split_names = (
        ["training", "testing"]
        if args.split == "both"
        else [args.split]
    )

    successful = []
    failures = []

    for split_name in split_names:

        split_input = (
            input_root /
            split_name
        )

        if not split_input.exists():
            raise FileNotFoundError(
                split_input
            )

        if args.patients:
            patient_dirs = [
                split_input / patient
                for patient in args.patients
                if (
                    split_input /
                    patient
                ).exists()
            ]

        else:
            patient_dirs = sorted(
                [
                    p
                    for p
                    in split_input.iterdir()
                    if p.is_dir()
                    and p.name.startswith(
                        "patient"
                    )
                ]
            )

        print("\n======================================")
        print(
            f"Split: {split_name}"
        )
        print(
            f"Patients: {len(patient_dirs)}"
        )
        print("======================================")

        for patient_dir in patient_dirs:

            try:
                result = preprocess_patient(
                    patient_dir,
                    output_root,
                    split_name
                )

                successful.append(
                    result
                )

            except Exception as e:

                print(
                    f"\nFAILED "
                    f"{patient_dir.name}: "
                    f"{e}"
                )

                failures.append(
                    {
                        "split": split_name,
                        "patient": patient_dir.name,
                        "error": str(e)
                    }
                )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    summary = {
        "target_T": TARGET_T,

        "method": (
            "ED-aligned periodic "
            "linear temporal interpolation"
        ),

        "processed": successful,

        "failures": failures
    }

    save_json(
        summary,
        output_root /
        "temporal_preprocessing_summary.json"
    )

    print("\n======================================")
    print("TEMPORAL PREPROCESSING COMPLETE")
    print("======================================")

    print(
        f"Successful: {len(successful)}"
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