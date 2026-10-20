"""Vendor-stratified paired Dice and HD95 effects for E2-Full vs E2-Static."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "mms_e2_vendor_paired_shuffle_matched_analysis"
PATHS = {
    "E2-Full": ROOT / "mms_external_validation/Exp2/metrics/external_volume_results.csv",
    "E2-Static": ROOT / "exp2_static_shuffle_matched/mms_external_validation/E2-Static/metrics/external_volume_results.csv",
}
VENDORS = ("Canon", "GE", "Philips", "Siemens")
OUTCOMES = {
    "Mean_Dice": ("Mean_Dice",),
    "Mean_HD95_mm": ("RV_HD95", "MYO_HD95", "LV_HD95"),
    "RV_HD95_mm": ("RV_HD95",),
    "MYO_HD95_mm": ("MYO_HD95",),
    "LV_HD95_mm": ("LV_HD95",),
}
REPLICATES = 100_000
SEED = 42


def load() -> dict[str, pd.DataFrame]:
    frames = {}
    reference = None
    required = {"patient", "phase", "vendor", "Mean_Dice", "RV_HD95", "MYO_HD95", "LV_HD95"}
    for model, path in PATHS.items():
        frame = pd.read_csv(path)
        if missing := required - set(frame):
            raise ValueError(f"{model} missing {sorted(missing)}")
        if len(frame) != 640 or frame.patient.nunique() != 320:
            raise ValueError(f"{model}: expected 640 rows from 320 patients")
        if frame.duplicated(["patient", "phase"]).any() or set(frame.phase) != {"ED", "ES"}:
            raise ValueError(f"{model}: invalid patient-phase rows")
        keys = set(zip(frame.patient, frame.phase))
        if reference is None:
            reference = keys
        elif keys != reference:
            raise ValueError("Model patient-phase sets differ")
        frames[model] = frame
    maps = [frame.drop_duplicates("patient").set_index("patient").vendor.sort_index()
            for frame in frames.values()]
    if not maps[0].equals(maps[1]):
        raise ValueError("Vendor assignments differ")
    return frames


def patient_outcome(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Require all requested structures at both ED and ES before averaging."""
    selected = frame[["patient", "phase", *columns]].copy()
    finite = np.isfinite(selected[list(columns)].to_numpy(dtype=float)).all(axis=1)
    selected = selected[finite]
    complete = selected.groupby("patient").phase.nunique()
    selected = selected[selected.patient.isin(complete[complete == 2].index)]
    selected["value"] = selected[list(columns)].mean(axis=1)
    return selected.groupby("patient").value.mean()


def bootstrap(values: np.ndarray, wins: np.ndarray, rng: np.random.Generator) -> tuple[float, ...]:
    means = np.empty(REPLICATES)
    win_props = np.empty(REPLICATES)
    chunk = 10_000
    for start in range(0, REPLICATES, chunk):
        stop = min(start + chunk, REPLICATES)
        indices = rng.integers(0, len(values), size=(stop-start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
        win_props[start:stop] = wins[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    win_low, win_high = np.percentile(win_props, [2.5, 97.5])
    return float(low), float(high), float(win_low), float(win_high)


def main() -> None:
    frames = load()
    rows, patient_rows = [], []
    rng = np.random.default_rng(SEED)
    for vendor in VENDORS:
        subsets = {model: frame[frame.vendor == vendor] for model, frame in frames.items()}
        vendor_n = subsets["E2-Full"].patient.nunique()
        for outcome, columns in OUTCOMES.items():
            full = patient_outcome(subsets["E2-Full"], columns).rename("E2_Full")
            static = patient_outcome(subsets["E2-Static"], columns).rename("E2_Static")
            paired = pd.concat([full, static], axis=1, join="inner").dropna().sort_index()
            difference = paired.E2_Full.to_numpy() - paired.E2_Static.to_numpy()
            lower_is_better = outcome.endswith("HD95_mm")
            wins = difference < 0 if lower_is_better else difference > 0
            ci_low, ci_high, win_low, win_high = bootstrap(difference, wins, rng)
            for patient, values in paired.iterrows():
                patient_rows.append({
                    "vendor": vendor, "patient": patient, "outcome": outcome,
                    "E2_Full": values.E2_Full, "E2_Static": values.E2_Static,
                    "difference_Full_minus_Static": values.E2_Full-values.E2_Static,
                    "E2_Full_better": bool(
                        values.E2_Full < values.E2_Static if lower_is_better
                        else values.E2_Full > values.E2_Static
                    ),
                })
            rows.append({
                "vendor": vendor, "vendor_total_patients": vendor_n, "outcome": outcome,
                "direction": "lower is better" if lower_is_better else "higher is better",
                "n_complete_paired_patients": len(paired),
                "E2_Full_mean": paired.E2_Full.mean(), "E2_Full_std": paired.E2_Full.std(ddof=1),
                "E2_Full_median": paired.E2_Full.median(),
                "E2_Full_q1": paired.E2_Full.quantile(.25), "E2_Full_q3": paired.E2_Full.quantile(.75),
                "E2_Static_mean": paired.E2_Static.mean(), "E2_Static_std": paired.E2_Static.std(ddof=1),
                "E2_Static_median": paired.E2_Static.median(),
                "E2_Static_q1": paired.E2_Static.quantile(.25), "E2_Static_q3": paired.E2_Static.quantile(.75),
                "mean_difference_Full_minus_Static": difference.mean(),
                "bootstrap_95ci_lower": ci_low, "bootstrap_95ci_upper": ci_high,
                "patients_Full_better": int(wins.sum()),
                "proportion_Full_better": wins.mean(),
                "bootstrap_win_95ci_lower": win_low, "bootstrap_win_95ci_upper": win_high,
            })
    summary = pd.DataFrame(rows)
    patients = pd.DataFrame(patient_rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "vendor_paired_dice_hd95_summary.csv", index=False)
    patients.to_csv(OUTPUT / "vendor_patient_level_values.csv", index=False)
    priority = summary[summary.outcome.isin(["Mean_Dice", "Mean_HD95_mm"])].copy()
    priority.to_csv(OUTPUT / "vendor_priority_table.csv", index=False)
    manifest = {
        "scope": "Post hoc vendor-stratified analysis of all 320 labeled M&M open-dataset subjects",
        "models": list(PATHS), "vendors": list(VENDORS),
        "vendor_sample_sizes": {vendor: int(frames["E2-Full"][frames["E2-Full"].vendor == vendor].patient.nunique()) for vendor in VENDORS},
        "inferential_unit": "patient",
        "phase_handling": "ED and ES averaged within patient before comparison",
        "Mean_Dice": "arithmetic mean of RV, MYO, and LV Dice per phase, then ED/ES mean",
        "Mean_HD95_mm": "arithmetic mean of RV, MYO, and LV HD95 per phase, then ED/ES mean; all six values required",
        "structure_HD95_mm": "ED/ES mean for the named structure; both phases required",
        "missingness": "Complete paired outcomes only; n is reported separately for every vendor/outcome",
        "paired_difference": "E2-Full minus E2-Static; positive favors Full for Dice and negative favors Full for HD95",
        "bootstrap": {"unit": "patient", "method": "paired percentile", "replicates": REPLICATES, "seed": SEED, "confidence_level": 0.95},
        "classification": "Post hoc subgroup analysis; not an untouched external-validation claim",
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(priority.to_string(index=False))


if __name__ == "__main__":
    main()
