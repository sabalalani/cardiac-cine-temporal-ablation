"""Patient-paired statistics for original E2 versus the E2-Static ablation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parent.parent
MODELS = ("E2-Temporal", "E2-Static")
STRUCTURES = ("RV", "MYO", "LV")
METRICS = ("Dice", "HD95", "ASD")
GROUPS = ("vendor", "centre", "pathology", "source_partition")
MATCHED_DIAGNOSES = ("DCM", "HCM", "NOR", "ARV")
BOOTSTRAP_REPLICATES = 100_000
BOOTSTRAP_SEED = 42


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    difference = x - y
    difference = difference[difference != 0]
    if not len(difference):
        return 0.0
    ranks = rankdata(np.abs(difference))
    return float((ranks[difference > 0].sum() - ranks[difference < 0].sum()) / ranks.sum())


def load(path: Path, model: str, patients: int, require_groups: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"patient", "phase", "Mean_Dice", *(
        f"{structure}_{metric}" for structure in STRUCTURES for metric in METRICS
    )}
    if missing := required - set(frame):
        raise ValueError(f"{model} missing columns: {sorted(missing)}")
    if len(frame) != patients * 2 or frame.patient.nunique() != patients:
        raise ValueError(f"{model}: expected {patients * 2} rows from {patients} patients")
    if frame.duplicated(["patient", "phase"]).any() or set(frame.phase) != {"ED", "ES"}:
        raise ValueError(f"{model}: invalid patient/phase pairing")
    if require_groups and (missing := set(GROUPS) - set(frame)):
        raise ValueError(f"{model} missing subgroup columns: {sorted(missing)}")
    frame.insert(0, "model", model)
    return frame


def verify_pairing(frames: dict[str, pd.DataFrame]) -> None:
    keys = [{(row.patient, row.phase) for row in frame.itertuples()} for frame in frames.values()]
    if keys[0] != keys[1]:
        raise ValueError("Temporal and static patient/phase observations differ")


def outcome(frame: pd.DataFrame, phase: str, structure: str, metric: str) -> pd.Series:
    column = "Mean_Dice" if structure == "Mean" and metric == "Dice" else None
    if structure == "Mean" and metric != "Dice":
        values = frame[[f"{name}_{metric}" for name in STRUCTURES]].mean(axis=1)
    else:
        values = frame[column or f"{structure}_{metric}"]
    selected = frame.assign(value=values)
    if phase == "All":
        return selected.groupby("patient").value.mean()
    return selected[selected.phase == phase].set_index("patient").value


def compare(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for phase in ("All", "ED", "ES"):
        for structure in ("Mean", *STRUCTURES):
            for metric in METRICS:
                if structure == "Mean" and metric in ("HD95", "ASD"):
                    pass
                x = outcome(frames[MODELS[0]], phase, structure, metric)
                y = outcome(frames[MODELS[1]], phase, structure, metric)
                paired = pd.concat([x.rename("temporal"), y.rename("static")], axis=1, join="inner").dropna()
                a, b = paired.temporal.to_numpy(), paired.static.to_numpy()
                if np.allclose(a, b, equal_nan=True):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                difference = a - b
                rows.append({
                    "family": f"{phase}_{structure}_{metric}", "phase": phase,
                    "structure": structure, "metric": metric, "n_pairs": len(paired),
                    "model_a": MODELS[0], "model_b": MODELS[1],
                    "mean_difference_a_minus_b": float(np.mean(difference)),
                    "median_difference_a_minus_b": float(np.median(difference)),
                    "rank_biserial_a_minus_b": rank_biserial(a, b),
                    "wilcoxon_statistic": statistic, "p_value_raw": p_value,
                    # One planned contrast per family: Holm-adjusted p equals raw p.
                    "p_value_holm": p_value, "significant_0p05": p_value < 0.05,
                })
    return pd.DataFrame(rows)


def descriptives(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        for phase in ("All", "ED", "ES"):
            for structure in ("Mean", *STRUCTURES):
                for metric in METRICS:
                    values = outcome(frame, phase, structure, metric)
                    rows.append({"model": model, "phase": phase, "structure": structure,
                                 "metric": metric, "n": len(values), "mean": values.mean(),
                                 "std": values.std(), "median": values.median(),
                                 "q1": values.quantile(.25), "q3": values.quantile(.75)})
    return pd.DataFrame(rows)


def subgroup_tests(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        common = sorted(set.intersection(*(set(frame[group].astype(str)) for frame in frames.values())))
        for value in common:
            subsets = {model: frame[frame[group].astype(str) == value] for model, frame in frames.items()}
            for phase in ("All", "ED", "ES"):
                x = outcome(subsets[MODELS[0]], phase, "Mean", "Dice")
                y = outcome(subsets[MODELS[1]], phase, "Mean", "Dice")
                paired = pd.concat([x.rename("temporal"), y.rename("static")], axis=1, join="inner").dropna()
                if len(paired) < 5:
                    continue
                a, b = paired.temporal.to_numpy(), paired.static.to_numpy()
                result = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
                rows.append({"group_type": group, "group_value": value, "phase": phase,
                             "n_pairs": len(paired), "model_a": MODELS[0], "model_b": MODELS[1],
                             "mean_difference_a_minus_b": float(np.mean(a-b)),
                             "median_difference_a_minus_b": float(np.median(a-b)),
                             "rank_biserial_a_minus_b": rank_biserial(a, b),
                             "p_value_raw": float(result.pvalue),
                             "p_value_holm": float(result.pvalue),
                             "significant_0p05": result.pvalue < .05})
    return pd.DataFrame(rows)


def bootstrap_patient_effects(
    frames: dict[str, pd.DataFrame], replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap the primary ED/ES-averaged mean-Dice contrast by patient."""
    temporal = outcome(frames[MODELS[0]], "All", "Mean", "Dice")
    static = outcome(frames[MODELS[1]], "All", "Mean", "Dice")
    paired = pd.concat(
        [temporal.rename("E2_Full_Mean_Dice"), static.rename("E2_Static_Mean_Dice")],
        axis=1, join="inner",
    ).dropna().sort_index()
    paired["paired_difference_Full_minus_Static"] = (
        paired.E2_Full_Mean_Dice - paired.E2_Static_Mean_Dice
    )
    difference = paired.paired_difference_Full_minus_Static.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    bootstrap_win_proportions = np.empty(replicates, dtype=np.float64)
    chunk = 10_000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        sampled = difference[indices]
        bootstrap_means[start:stop] = sampled.mean(axis=1)
        bootstrap_win_proportions[start:stop] = (sampled > 0).mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    win_lower, win_upper = np.percentile(bootstrap_win_proportions, [2.5, 97.5])
    wins = int(np.count_nonzero(difference > 0))
    ties = int(np.count_nonzero(difference == 0))
    losses = int(np.count_nonzero(difference < 0))
    summary = pd.DataFrame([{
        "outcome": "patient ED/ES-averaged foreground Mean Dice",
        "contrast": "E2-Full minus E2-Static",
        "n_patients": len(difference),
        "mean_paired_difference": float(difference.mean()),
        "bootstrap_95ci_lower": float(lower),
        "bootstrap_95ci_upper": float(upper),
        "median_paired_difference": float(np.median(difference)),
        "patients_full_better": wins,
        "patients_tied": ties,
        "patients_static_better": losses,
        "proportion_full_better": wins / len(difference),
        "bootstrap_win_proportion_95ci_lower": float(win_lower),
        "bootstrap_win_proportion_95ci_upper": float(win_upper),
        "bootstrap_method": "paired nonparametric percentile bootstrap at patient level",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }])
    return paired.reset_index(), summary


def save_analysis(name: str, frames: dict[str, pd.DataFrame], output: Path,
                  subgroup: bool = False) -> None:
    verify_pairing(frames)
    output.mkdir(parents=True, exist_ok=True)
    descriptive = descriptives(frames)
    tests = compare(frames)
    descriptive.to_csv(output / "descriptive_summary.csv", index=False)
    tests.to_csv(output / "pairwise_wilcoxon_holm.csv", index=False)
    patient_effects, bootstrap_summary = bootstrap_patient_effects(frames)
    patient_effects.to_csv(output / "patient_level_paired_mean_dice.csv", index=False)
    bootstrap_summary.to_csv(output / "patient_level_bootstrap_effect.csv", index=False)
    if subgroup:
        subgroup_tests(frames).to_csv(output / "subgroup_mean_dice_tests.csv", index=False)
        matched = {model: frame[frame.pathology.isin(MATCHED_DIAGNOSES)].copy()
                   for model, frame in frames.items()}
        verify_pairing(matched)
        descriptives(matched).to_csv(output / "diagnosis_matched_descriptive_summary.csv", index=False)
        compare(matched).to_csv(output / "diagnosis_matched_pairwise_wilcoxon_holm.csv", index=False)
    manifest = {
        "analysis": name, "models": list(MODELS),
        "patients": int(next(iter(frames.values())).patient.nunique()),
        "volumes_per_model": int(len(next(iter(frames.values())))),
        "primary_pairing_unit": "patient; ED and ES averaged within patient for All-phase tests",
        "test": "paired two-sided Wilcoxon signed-rank with Pratt zero handling",
        "effect_size": "matched-pairs rank-biserial correlation; positive favors E2-Temporal",
        "primary_magnitude_estimate": "mean paired E2-Full minus E2-Static Dice difference",
        "bootstrap": {
            "unit": "patient",
            "ED_ES_handling": "ED and ES Mean Dice averaged within patient before resampling",
            "method": "nonparametric paired percentile bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "multiple_testing": "Holm within each outcome family; one planned contrast, so adjusted equals raw p",
        "omnibus_test": "not applicable for the two-model ablation",
        "alpha": 0.05,
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    base = ROOT / "exp2_temporal_vs_static_analysis"
    save_analysis("ACDC five-fold out-of-fold", {
        MODELS[0]: load(ROOT / "exp2_3dconvlstm/cv_metrics/oof_volume_results.csv", MODELS[0], 100),
        MODELS[1]: load(ROOT / "exp2_static_convlstm/cv_metrics/oof_volume_results.csv", MODELS[1], 100),
    }, base / "oof")
    save_analysis("ACDC frozen held-out ensemble", {
        MODELS[0]: load(ROOT / "exp2_3dconvlstm/heldout_ensemble/metrics/heldout_volume_results.csv", MODELS[0], 50),
        MODELS[1]: load(ROOT / "exp2_static_convlstm/frozen_ensemble/heldout_ensemble/metrics/heldout_volume_results.csv", MODELS[1], 50),
    }, base / "heldout")
    save_analysis("M&Ms post hoc external robustness evaluation", {
        MODELS[0]: load(ROOT / "mms_external_validation/Exp2/metrics/external_volume_results.csv", MODELS[0], 320, True),
        MODELS[1]: load(ROOT / "exp2_static_convlstm/mms_external_validation/E2-Static/metrics/external_volume_results.csv", MODELS[1], 320, True),
    }, base / "mms", subgroup=True)
    print(f"Saved E2 temporal-versus-static paired analyses to {base}")


if __name__ == "__main__":
    main()
