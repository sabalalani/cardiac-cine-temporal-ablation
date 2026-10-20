"""Create frozen-ACDC-compatible M&Ms spatial, temporal, and HDF5 inputs."""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import zoom

from preprocess_acdc_spatial import (
    TARGET_HW, TARGET_SPACING, TARGET_Z, center_crop_pad_xy,
    foreground_zscore, make_target_affine, pad_z, resample_4d,
)
from prepare_acdc_temporal import (
    TARGET_T, ed_es_anchored_temporal_resample,
    map_original_position_to_target,
)

# M&Ms 1=LV, 2=MYO, 3=RV; frozen ACDC model 1=RV, 2=MYO, 3=LV.
LABEL_MAP = np.asarray([0, 3, 2, 1], dtype=np.uint8)
ALLOWED = {0, 1, 2, 3}


def dump_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(rows)


def save_nii(array, path, mask=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = array.astype(np.uint8 if mask else np.float32)
    image = nib.Nifti1Image(array, make_target_affine())
    image.header.set_data_dtype(array.dtype)
    image.header.set_zooms((*TARGET_SPACING, 1.0) if array.ndim == 4 else TARGET_SPACING)
    nib.save(image, str(path))


def save_h5(image, mask, path, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.ascontiguousarray(image.transpose(2, 1, 0))
    mask = np.ascontiguousarray(mask.transpose(2, 1, 0))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("raw", data=image, compression="gzip")
        handle.create_dataset("label", data=mask, compression="gzip")
        handle.attrs["spacing_zyx_mm"] = [10.0, 1.5, 1.5]
        handle.attrs["dataset"] = "M&Ms zero-shot external validation"
        handle.attrs["label_mapping"] = "M&Ms 1=LV,2=MYO,3=RV to ACDC 1=RV,2=MYO,3=LV"
        for key in ("patient", "source_partition", "vendor", "centre", "pathology"):
            handle.attrs[key] = str(meta[key])


def load_table(path):
    table = pd.read_csv(path)
    required = {"External code", "VendorName", "Vendor", "Centre", "ED", "ES", "Pathology"}
    if required - set(table):
        raise ValueError(f"Missing metadata columns: {sorted(required-set(table))}")
    if table["External code"].duplicated().any():
        raise ValueError("Duplicate External code in metadata")
    return table.set_index("External code", drop=False)


def source_paths(directory):
    patient = directory.name
    return directory/f"{patient}_sa.nii.gz", directory/f"{patient}_sa_gt.nii.gz"


def partition_name(directory):
    for path in (directory, *directory.parents):
        if path.name.lower() in {"training", "validation", "testing"}:
            return path.name.lower()
    raise ValueError(f"Cannot determine source partition for {directory}")


def resolve_phases(row, gt):
    """Resolve ED/ES, including the 25 records with withheld 0/0 metadata."""
    t = gt.shape[-1]
    ed, es = int(row.ED), int(row.ES)
    annotated = [index for index in range(t) if np.any(gt[..., index] > 0)]
    if ed != es:
        if not (0 <= ed < t and 0 <= es < t):
            raise ValueError(f"ED/ES={ed}/{es} outside T={t}")
        if set(annotated) != {ed, es}:
            raise ValueError(f"annotated={annotated}, ED/ES={ed}/{es}")
        return ed, es, annotated, "metadata_csv"
    if len(annotated) != 2:
        raise ValueError(f"withheld ED/ES but annotated frames={annotated}")
    lv_counts = {index: int(np.count_nonzero(gt[..., index] == 1)) for index in annotated}
    if len(set(lv_counts.values())) != 2:
        raise ValueError(f"cannot distinguish ED/ES from LV volumes {lv_counts}")
    ed = max(lv_counts, key=lv_counts.get)
    es = min(lv_counts, key=lv_counts.get)
    return ed, es, annotated, "inferred_from_two_mask_frames_by_LV_volume"


def inspect_case(directory, row):
    patient = directory.name
    cine_path, gt_path = source_paths(directory)
    base = {"patient": patient, "source_partition": partition_name(directory),
            "vendor": str(row.VendorName), "centre": str(row.Centre),
            "pathology": str(row.Pathology), "ED_0based": int(row.ED),
            "ES_0based": int(row.ES), "error": ""}
    if not cine_path.exists() or not gt_path.exists():
        return base | {"status": "incomplete_copy"}
    try:
        cine, gt = nib.load(cine_path), nib.load(gt_path)
        if len(cine.shape) != 4 or cine.shape != gt.shape:
            raise ValueError(f"nonmatching 4D shapes {cine.shape}/{gt.shape}")
        data = np.asanyarray(gt.dataobj)
        labels = set(np.unique(data).astype(int))
        if not labels.issubset(ALLOWED): raise ValueError(f"labels={sorted(labels)}")
        ed, es, frames, phase_source = resolve_phases(row, data)
        return base | {"status": "ready", "shape": "x".join(map(str, cine.shape)),
                       "spacing_xyz": ",".join(map(str, cine.header.get_zooms()[:3])),
                       "source_labels": ",".join(map(str, sorted(labels))),
                       "resolved_ED_0based": ed, "resolved_ES_0based": es,
                       "phase_index_source": phase_source}
    except Exception as error:
        return base | {"status": "invalid", "error": str(error)}


def preprocess(directory, row, spatial_root, temporal_root, h5_root,
               inplane_rotate_180=False):
    patient, partition = directory.name, partition_name(directory)
    cine_path, gt_path = source_paths(directory)
    # M&Ms short-axis volumes are oblique. Their third array axis is the
    # through-plane stack, but canonicalization can permute that axis into X
    # or Y. Preserve the native voxel-axis convention and verify paired
    # affines; output uses the frozen ACDC target grid.
    cine_nii = nib.load(cine_path)
    gt_nii = nib.load(gt_path)
    cine = cine_nii.get_fdata(dtype=np.float32)
    gt = np.rint(gt_nii.get_fdata(dtype=np.float32)).astype(np.uint8)
    spacing = tuple(map(float, cine_nii.header.get_zooms()[:3]))
    if not np.allclose(cine_nii.affine, gt_nii.affine, atol=1e-5):
        raise ValueError(f"{patient}: cine/ground-truth affine mismatch")
    if cine.shape != gt.shape or cine.ndim != 4:
        raise ValueError(f"{patient}: incompatible input")
    if not set(np.unique(gt).tolist()).issubset(ALLOWED):
        raise ValueError(f"{patient}: invalid labels")
    ed, es, frames, phase_source = resolve_phases(row, gt)

    cine = resample_4d(cine, spacing)
    factors = tuple(spacing[i]/TARGET_SPACING[i] for i in range(3))
    masks = [zoom(gt[..., phase], factors, order=0, mode="nearest", prefilter=False)
             for phase in (ed, es)]
    cine = center_crop_pad_xy(cine, *TARGET_HW)
    masks = [center_crop_pad_xy(mask, *TARGET_HW) for mask in masks]
    cine, before, after = pad_z(cine, TARGET_Z)
    masks = [pad_z(mask, TARGET_Z)[0] for mask in masks]
    cine, mean, std = foreground_zscore(cine)
    ed_mask, es_mask = [LABEL_MAP[np.rint(mask).astype(np.uint8)] for mask in masks]
    # Lossless fixed rotation on the finalized, normalized model grid.
    if inplane_rotate_180:
        cine = np.flip(cine, axis=(0, 1)).copy()
        ed_mask = np.flip(ed_mask, axis=(0, 1)).copy()
        es_mask = np.flip(es_mask, axis=(0, 1)).copy()
    ed_image, es_image = cine[..., ed], cine[..., es]

    relative_es = (es-ed) % cine.shape[-1]
    shifted = np.roll(cine, -ed, axis=-1)
    es_out, phase, continuous = map_original_position_to_target(relative_es, cine.shape[-1], TARGET_T)
    temporal, positions = ed_es_anchored_temporal_resample(
        shifted, relative_es, TARGET_T, es_out)
    if not np.array_equal(temporal[..., 0], ed_image): raise ValueError("ED anchor mismatch")
    if not np.array_equal(temporal[..., es_out], es_image): raise ValueError("ES anchor mismatch")

    meta = {"patient": patient, "source_partition": partition,
            "vendor": str(row.VendorName), "vendor_code": str(row.Vendor),
            "centre": str(row.Centre), "pathology": str(row.Pathology),
            "ED_original_index_0based": ed, "ES_original_index_0based": es,
            "phase_index_source": phase_source,
            "original_shape_XYZT": list(map(int, cine_nii.shape)),
            "original_spacing_XYZ_mm": list(spacing),
            "output_shape_XYZT": list(map(int, cine.shape)),
            "target_spacing_XYZ_mm": list(TARGET_SPACING),
            "normalization_mean": mean, "normalization_std": std,
            "z_padding_before": before, "z_padding_after": after,
            "label_mapping": {"M&Ms_1_LV": 3, "M&Ms_2_MYO": 2, "M&Ms_3_RV": 1},
            "inplane_rotation_degrees": 180 if inplane_rotate_180 else 0,
            "inplane_rotation_axes_XY": bool(inplane_rotate_180),
            "orientation_harmonization": (
                "fixed_flip_axes_0_1_to_match_ACDC_model_facing_RAS_conversion"
                if inplane_rotate_180 else "none"
            ),
            "target_T": TARGET_T, "ED_normalized_index": 0,
            "ES_normalized_index": es_out, "ES_normalized_phase": phase,
            "ES_target_continuous_position": continuous,
            "sampled_positions_after_ED_shift": list(map(float, positions)),
            "ED_exact_frame_preserved": True, "ES_exact_frame_preserved": True,
            "zero_shot_external_validation": True}
    out = spatial_root/partition/patient
    save_nii(cine, out/f"{patient}_cine_spatial.nii.gz")
    for phase_name, image, mask in (("ED", ed_image, ed_mask), ("ES", es_image, es_mask)):
        save_nii(image, out/f"{patient}_{phase_name}.nii.gz")
        save_nii(mask, out/f"{patient}_{phase_name}_gt.nii.gz", True)
        save_h5(image, mask, h5_root/partition/f"{patient}_{phase_name}.h5", meta)
    dump_json(meta, out/"metadata.json")
    tout = temporal_root/partition/patient
    save_nii(temporal, tout/f"{patient}_cine_T30_EDaligned.nii.gz")
    dump_json(meta, tout/"temporal_metadata.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=Path, default=Path("data/MMS"))
    parser.add_argument("--metadata_csv", type=Path,
        default=Path("data/MMS/211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"))
    parser.add_argument("--spatial_output", type=Path, default=Path("mms_preprocessed"))
    parser.add_argument("--temporal_output", type=Path, default=Path("mms_temporal"))
    parser.add_argument("--h5_output", type=Path, default=Path("mms_exp1_h5"))
    parser.add_argument("--audit_output", type=Path, default=Path("mms_preprocessing_audit.csv"))
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--inplane_rotate_180", action="store_true",
        help="Flip native M&Ms axes 0 and 1 before resampling to match the ACDC model-facing orientation.",
    )
    args = parser.parse_args()
    table = load_table(args.metadata_csv)
    partitions = {d.name.lower(): d for d in args.input_root.iterdir()
                  if d.is_dir() and d.name.lower() in {"training", "validation", "testing"}}
    found = {}
    unlabeled_root = args.input_root / "Training" / "Unlabeled"
    excluded_unlabeled = {d.name for d in unlabeled_root.iterdir() if d.is_dir()} if unlabeled_root.is_dir() else set()
    for partition in partitions.values():
        for cine_path in partition.rglob("*_sa.nii.gz"):
            directory = cine_path.parent
            patient = directory.name
            if cine_path.name != f"{patient}_sa.nii.gz":
                continue
            if patient in excluded_unlabeled:
                continue
            if patient in found: raise ValueError(f"duplicate patient {patient}")
            found[patient] = directory
    audit = []
    for patient, row in table.iterrows():
        if patient in excluded_unlabeled:
            audit.append({"patient": patient, "source_partition": "training",
                          "status": "excluded_unlabeled", "error": "",
                          "exclusion_reason": "original M&Ms Training/Unlabeled subset; ED/ES metadata withheld"})
        elif patient in found:
            audit.append(inspect_case(found[patient], row))
        else:
            audit.append({"patient": patient, "status": "not_copied", "error": ""})
    write_csv(audit, args.audit_output)
    counts = pd.Series([row["status"] for row in audit]).value_counts().to_dict()
    print(json.dumps({"metadata": len(table), "discovered": len(found),
                      "partitions": sorted(partitions), "status": counts}, indent=2))
    if args.audit_only: return
    failures, processed = [], 0
    ready = [row["patient"] for row in audit if row["status"] == "ready"]
    for index, patient in enumerate(ready, 1):
        try:
            print(f"[{index}/{len(ready)}] {patient}", flush=True)
            preprocess(found[patient], table.loc[patient], args.spatial_output,
                       args.temporal_output, args.h5_output,
                       args.inplane_rotate_180); processed += 1
        except Exception as error:
            failures.append({"patient": patient, "error": str(error)})
            print(f"FAILED {patient}: {error}", file=sys.stderr)
    dump_json({"status": "complete" if not failures else "completed_with_failures",
               "metadata_subjects": len(table), "processed_subjects": processed,
               "excluded_unlabeled_subjects": len(excluded_unlabeled),
               "audit_status": counts, "failures": failures,
               "diagnostic_inclusion": "all pathologies among the 320 explicitly phase-labeled subjects",
               "cohort_rule": "Training/Labeled + Validation + Testing; exclude Training/Unlabeled",
               "target_spacing_XYZ_mm": list(TARGET_SPACING),
               "target_shape_XYZ": [*TARGET_HW, TARGET_Z], "target_T": TARGET_T,
               "inplane_rotation_degrees": 180 if args.inplane_rotate_180 else 0,
               "orientation_harmonization_sensitivity_analysis": bool(args.inplane_rotate_180)},
              args.spatial_output/"preprocessing_manifest.json")
    expected = {"ready": len(table) - len(excluded_unlabeled),
                "excluded_unlabeled": len(excluded_unlabeled)}
    if failures or (args.strict and counts != expected):
        raise RuntimeError(f"incomplete preprocessing: audit={counts}, failures={len(failures)}")


if __name__ == "__main__": main()
