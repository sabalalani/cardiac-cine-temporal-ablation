"""Audit M&Ms cohort provenance, phase matching, and evaluation alignment."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = Path("data/MMS")
METADATA = RAW / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
SPATIAL = ROOT / "mms_preprocessed"
TEMPORAL = ROOT / "mms_temporal"
H5_ROOT = ROOT / "mms_exp1_h5"
OUTPUT = ROOT / "mms_provenance_alignment_audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collection_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: str(value)):
        digest.update(str(path.relative_to(RAW)).encode())
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def raw_cases() -> dict[str, tuple[str, Path]]:
    cases: dict[str, tuple[str, Path]] = {}
    locations = {
        "training_labeled": RAW / "Training" / "Labeled",
        "training_unlabeled": RAW / "Training" / "Unlabeled",
        "validation": RAW / "Validation",
        "testing": RAW / "Testing",
    }
    for partition, base in locations.items():
        for directory in sorted(path for path in base.iterdir() if path.is_dir()):
            patient = directory.name
            if patient in cases:
                raise RuntimeError(f"Duplicate raw subject ID: {patient}")
            cases[patient] = (partition, directory)
    return cases


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(METADATA)
    if table["External code"].duplicated().any():
        raise RuntimeError("Duplicate IDs in metadata")
    table = table.set_index("External code", drop=False)
    cases = raw_cases()
    if set(cases) != set(table.index):
        raise RuntimeError(
            f"Raw/metadata ID mismatch: raw-only={sorted(set(cases)-set(table.index))}, "
            f"metadata-only={sorted(set(table.index)-set(cases))}"
        )

    included = {p for p, (part, _) in cases.items() if part != "training_unlabeled"}
    excluded = set(cases) - included
    records = []
    gt_paths = []
    phase_sources = Counter()
    for patient in sorted(cases):
        partition, directory = cases[patient]
        cine_path = directory / f"{patient}_sa.nii.gz"
        gt_path = directory / f"{patient}_sa_gt.nii.gz"
        has_cine, has_gt = cine_path.exists(), gt_path.exists()
        record = {"patient": patient, "partition": partition,
                  "cine_present": has_cine, "gt_present": has_gt}
        if patient in included:
            if not (has_cine and has_gt):
                raise RuntimeError(f"Missing included raw input: {patient}")
            gt_paths.append(gt_path)
            cine_nii, gt_nii = nib.load(cine_path), nib.load(gt_path)
            if cine_nii.shape != gt_nii.shape or len(cine_nii.shape) != 4:
                raise RuntimeError(f"Raw cine/GT shape mismatch: {patient}")
            if not np.allclose(cine_nii.affine, gt_nii.affine, atol=1e-5):
                raise RuntimeError(f"Raw cine/GT affine mismatch: {patient}")
            gt = np.rint(np.asanyarray(gt_nii.dataobj)).astype(np.uint8)
            labels = set(np.unique(gt).tolist())
            if not labels.issubset({0, 1, 2, 3}):
                raise RuntimeError(f"Unexpected labels for {patient}: {labels}")
            annotated = [i for i in range(gt.shape[-1]) if np.any(gt[..., i] > 0)]
            ed, es = int(table.loc[patient, "ED"]), int(table.loc[patient, "ES"])
            if ed == es:
                if len(annotated) != 2:
                    raise RuntimeError(f"Cannot resolve phases for {patient}: {annotated}")
                counts = {i: int(np.count_nonzero(gt[..., i] == 1)) for i in annotated}
                if len(set(counts.values())) != 2:
                    raise RuntimeError(f"Tied LV phase volumes for {patient}: {counts}")
                ed, es = max(counts, key=counts.get), min(counts, key=counts.get)
                phase_source = "inferred_from_two_GT_frames_by_LV_volume"
            else:
                if set(annotated) != {ed, es}:
                    raise RuntimeError(
                        f"Metadata/annotated-frame mismatch {patient}: ED/ES={ed}/{es}, GT={annotated}"
                    )
                phase_source = "metadata_csv"
            phase_sources[phase_source] += 1
            record.update({"shape": "x".join(map(str, cine_nii.shape)),
                           "affine_match": True, "labels": ",".join(map(str, sorted(labels))),
                           "ED": ed, "ES": es, "annotated_frames": str(annotated),
                           "phase_source": phase_source})
        records.append(record)

    processed = {path.parent.name: path.parent for path in SPATIAL.glob("*/*/metadata.json")}
    temporal = {path.parent.name: path.parent for path in TEMPORAL.glob("*/*/temporal_metadata.json")}
    if set(processed) != included or set(temporal) != included:
        raise RuntimeError("Processed spatial/temporal IDs do not exactly equal included raw IDs")

    processed_checks = Counter()
    for patient in sorted(included):
        directory = processed[patient]
        meta = json.loads((directory / "metadata.json").read_text())
        temporal_meta = json.loads((temporal[patient] / "temporal_metadata.json").read_text())
        cine = nib.load(directory / f"{patient}_cine_spatial.nii.gz").get_fdata(dtype=np.float32)
        cine_t = nib.load(temporal[patient] / f"{patient}_cine_T30_EDaligned.nii.gz").get_fdata(dtype=np.float32)
        if cine.shape != (192, 192, 16, cine.shape[-1]) or cine_t.shape != (192, 192, 16, 30):
            raise RuntimeError(f"Processed shape mismatch: {patient}, {cine.shape}, {cine_t.shape}")
        es_out = int(temporal_meta["ES_normalized_index"])
        for phase, temporal_index in (("ED", 0), ("ES", es_out)):
            image_nii = nib.load(directory / f"{patient}_{phase}.nii.gz")
            gt_nii = nib.load(directory / f"{patient}_{phase}_gt.nii.gz")
            image = image_nii.get_fdata(dtype=np.float32)
            gt = np.rint(gt_nii.get_fdata()).astype(np.uint8)
            if image.shape != (192, 192, 16) or gt.shape != image.shape:
                raise RuntimeError(f"Processed image/GT shape mismatch: {patient} {phase}")
            if not np.allclose(image_nii.affine, gt_nii.affine, atol=1e-6):
                raise RuntimeError(f"Processed image/GT affine mismatch: {patient} {phase}")
            if not np.array_equal(image, cine[..., int(meta[f"{phase}_original_index_0based"])]):
                raise RuntimeError(f"Spatial phase-image mismatch: {patient} {phase}")
            if not np.array_equal(image, cine_t[..., temporal_index]):
                raise RuntimeError(f"Temporal anchor mismatch: {patient} {phase}")
            if not set(np.unique(gt).tolist()).issubset({0, 1, 2, 3}):
                raise RuntimeError(f"Processed label mismatch: {patient} {phase}")
            h5_path = H5_ROOT / meta["source_partition"] / f"{patient}_{phase}.h5"
            with h5py.File(h5_path) as handle:
                if not np.array_equal(np.asarray(handle["label"]), gt.transpose(2, 1, 0)):
                    raise RuntimeError(f"HDF5/GT mismatch: {patient} {phase}")
                if not np.array_equal(np.asarray(handle["raw"]), image.transpose(2, 1, 0)):
                    raise RuntimeError(f"HDF5/image mismatch: {patient} {phase}")
            processed_checks["phase_volumes_aligned"] += 1

    result_files = {
        "Exp1": ROOT / "mms_external_validation/Exp1/metrics/external_volume_results.csv",
        "Exp2": ROOT / "mms_external_validation/Exp2/metrics/external_volume_results.csv",
        "E2-Static": ROOT / "exp2_static_convlstm/mms_external_validation/E2-Static/metrics/external_volume_results.csv",
        "Exp3": ROOT / "mms_external_validation/Exp3/metrics/external_volume_results.csv",
    }
    expected_pairs = {(p, phase) for p in included for phase in ("ED", "ES")}
    result_summary = {}
    for model, path in result_files.items():
        frame = pd.read_csv(path)
        pairs = set(zip(frame.patient.astype(str), frame.phase.astype(str)))
        if len(frame) != 640 or frame.duplicated(["patient", "phase"]).any() or pairs != expected_pairs:
            raise RuntimeError(f"Result cohort mismatch: {model}")
        result_summary[model] = {"rows": len(frame), "subjects": frame.patient.nunique(),
                                 "unique_patient_phase_pairs": len(pairs)}

    acdc_split = json.loads((ROOT / "acdc_5fold_splits_seed42.json").read_text())
    acdc_ids = set(acdc_split["development_patients"]) | set(acdc_split["official_test_patients"])
    literal_id_overlap = sorted(acdc_ids & included)

    partition_counts = Counter(part for part, _ in cases.values())
    summary = {
        "status": "PASS",
        "raw_dataset_identification": {
            "root": str(RAW),
            "formal_release_version_found_locally": None,
            "metadata_filename": METADATA.name,
            "metadata_sha256": sha256(METADATA),
            "included_gt_collection_sha256": collection_hash(gt_paths),
            "note": "Hashes identify the local copy; they do not establish who supplied the files.",
        },
        "subjects": {
            "metadata_total": len(table), "raw_total": len(cases),
            "partition_counts": dict(sorted(partition_counts.items())),
            "included": len(included), "excluded_training_unlabeled": len(excluded),
        },
        "ground_truth": {
            "included_gt_files": len(gt_paths),
            "testing_gt_files_locally_present": sum(
                (directory / f"{patient}_sa_gt.nii.gz").exists()
                for patient, (part, directory) in cases.items() if part == "testing"
            ),
            "testing_subjects": partition_counts["testing"],
            "phase_index_sources": dict(phase_sources),
            "raw_label_semantics": {"0": "background", "1": "LV", "2": "MYO", "3": "RV"},
            "model_label_mapping": {"0": "background", "1": "RV", "2": "MYO", "3": "LV"},
        },
        "alignment": {
            "raw_cine_gt_shape_affine_checks": len(included),
            "processed_phase_volumes_aligned": processed_checks["phase_volumes_aligned"],
            "spatial_temporal_h5_subject_sets_identical": True,
            "result_sets": result_summary,
        },
        "leakage_checks": {
            "literal_ACDC_MMs_ID_overlap": literal_id_overlap,
            "MMs_used_for_training_or_checkpoint_selection": False,
            "limitation": "Anonymized IDs cannot rule out the same real person appearing across independently distributed datasets.",
        },
    }
    (OUTPUT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(records).to_csv(OUTPUT / "raw_subject_manifest.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
