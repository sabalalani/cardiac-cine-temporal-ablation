"""Paired statistical comparison of Experiments 1, 2, and 3."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


MODELS = ("Exp1", "Exp2", "Exp3")
STRUCTURES = ("RV", "MYO", "LV")
METRICS = ("Dice", "HD95", "ASD")
REQUIRED_COLUMNS = {
    "fold", "patient", "phase", "Mean_Dice",
    *(f"{structure}_{metric}" for structure in STRUCTURES for metric in METRICS),
}


def load_oof(path: Path, model: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model} OOF results not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{model} is missing columns: {sorted(missing)}")
    if len(frame) != 200 or frame["patient"].nunique() != 100:
        raise ValueError(
            f"{model} is incomplete: expected 200 ED/ES rows and 100 patients, "
            f"found {len(frame)} rows and {frame['patient'].nunique()} patients"
        )
    if frame.duplicated(["patient", "phase"]).any():
        raise ValueError(f"{model} contains duplicate patient/phase rows")
    if set(frame["phase"]) != {"ED", "ES"}:
        raise ValueError(f"{model} must contain both ED and ES")
    frame.insert(0, "model", model)
    frame["observation"] = frame["patient"] + "_" + frame["phase"]
    return frame


def verify_pairing(frames: dict[str, pd.DataFrame]) -> None:
    reference = set(frames[MODELS[0]]["observation"])
    for model in MODELS[1:]:
        current = set(frames[model]["observation"])
        if current != reference:
            raise ValueError(
                f"Patient/phase pairing differs for {model}: "
                f"missing={sorted(reference-current)[:5]}, extra={sorted(current-reference)[:5]}"
            )


def to_long(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, frame in frames.items():
        for record in frame.to_dict("records"):
            base = {
                "model": model, "fold": int(record["fold"]),
                "patient": record["patient"], "phase": record["phase"],
                "observation": record["observation"],
            }
            # Mean structure values are the per-volume primary outcomes.
            rows.extend([
                {**base, "structure": "Mean", "metric": "Dice", "value": record["Mean_Dice"]},
                {**base, "structure": "Mean", "metric": "HD95",
                 "value": np.nanmean([record[f"{s}_HD95"] for s in STRUCTURES])},
                {**base, "structure": "Mean", "metric": "ASD",
                 "value": np.nanmean([record[f"{s}_ASD"] for s in STRUCTURES])},
            ])
            for structure in STRUCTURES:
                for metric in METRICS:
                    rows.append({
                        **base, "structure": structure, "metric": metric,
                        "value": record[f"{structure}_{metric}"],
                    })
    return pd.DataFrame(rows)


def descriptive_table(long: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for phase in ("All", "ED", "ES"):
        subset = long if phase == "All" else long[long["phase"] == phase]
        grouped = subset.groupby(["model", "structure", "metric"])["value"]
        result = grouped.agg(n="count", mean="mean", std="std", median="median",
                             q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)).reset_index()
        result.insert(1, "phase", phase)
        pieces.append(result)
    return pd.concat(pieces, ignore_index=True)


def paired_matrix(long: pd.DataFrame, phase: str, structure: str, metric: str) -> pd.DataFrame:
    subset = long[(long["structure"] == structure) & (long["metric"] == metric)]
    if phase == "All":
        # Average ED and ES so the inferential unit remains the patient.
        subset = subset.groupby(["patient", "model"], as_index=False)["value"].mean()
    else:
        subset = subset[subset["phase"] == phase]
    return subset.pivot(index="patient", columns="model", values="value")[list(MODELS)].dropna()


def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    differences = x - y
    differences = differences[differences != 0]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences))
    return float((ranks[differences > 0].sum() - ranks[differences < 0].sum()) / ranks.sum())


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def statistical_tests(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    pairwise_rows = []
    for phase in ("All", "ED", "ES"):
        for structure in ("Mean", *STRUCTURES):
            for metric in METRICS:
                matrix = paired_matrix(long, phase, structure, metric)
                if len(matrix) < 2:
                    raise ValueError(f"Insufficient complete pairs: {phase}/{structure}/{metric}")
                arrays = [matrix[model].to_numpy() for model in MODELS]
                if all(np.allclose(arrays[0], values, equal_nan=True) for values in arrays[1:]):
                    statistic, p_value = 0.0, 1.0
                else:
                    statistic, p_value = friedmanchisquare(*arrays)
                family = f"{phase}_{structure}_{metric}"
                omnibus_rows.append({
                    "family": family, "phase": phase, "structure": structure,
                    "metric": metric, "n_pairs": len(matrix),
                    "friedman_chi2": statistic, "p_value": p_value,
                })
                family_pairs = []
                for model_a, model_b in combinations(MODELS, 2):
                    a = matrix[model_a].to_numpy()
                    b = matrix[model_b].to_numpy()
                    if np.allclose(a, b, equal_nan=True):
                        w_statistic, raw_p = 0.0, 1.0
                    else:
                        result = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
                        w_statistic, raw_p = float(result.statistic), float(result.pvalue)
                    family_pairs.append({
                        "family": family, "phase": phase, "structure": structure,
                        "metric": metric, "n_pairs": len(matrix),
                        "model_a": model_a, "model_b": model_b,
                        "mean_difference_a_minus_b": float(np.mean(a - b)),
                        "median_difference_a_minus_b": float(np.median(a - b)),
                        "rank_biserial_a_minus_b": rank_biserial(a, b),
                        "wilcoxon_statistic": w_statistic, "p_value_raw": raw_p,
                        "friedman_p_value": p_value,
                    })
                adjusted = holm_adjust([row["p_value_raw"] for row in family_pairs])
                for row, corrected in zip(family_pairs, adjusted):
                    row["p_value_holm"] = corrected
                    row["significant_0p05"] = corrected < 0.05
                    pairwise_rows.append(row)
    return pd.DataFrame(omnibus_rows), pd.DataFrame(pairwise_rows)


def paper_table(descriptive: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for phase in ("All", "ED", "ES"):
            row = {"model": model, "phase": phase}
            selected = descriptive[(descriptive["model"] == model) & (descriptive["phase"] == phase)]
            for structure in ("Mean", *STRUCTURES):
                for metric in METRICS:
                    match = selected[(selected["structure"] == structure) & (selected["metric"] == metric)]
                    values = match.iloc[0]
                    row[f"{structure}_{metric}"] = f"{values['mean']:.4f} +/- {values['std']:.4f}"
            rows.append(row)
    return pd.DataFrame(rows)


def save_figures(long: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

    dice = long[(long["structure"] == "Mean") & (long["metric"] == "Dice")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.boxplot(data=dice, x="model", y="value", hue="phase", order=MODELS,
                hue_order=["ED", "ES"], palette=["#3B82A0", "#D9822B"],
                showfliers=False, ax=ax)
    ax.set(xlabel="Architecture", ylabel="Mean Dice", ylim=(0, 1.02),
           title="Out-of-fold segmentation performance")
    ax.legend(title="Phase", frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"model_phase_dice.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)

    class_dice = long[(long["structure"] != "Mean") & (long["metric"] == "Dice")]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(data=class_dice, x="structure", y="value", hue="model",
                order=STRUCTURES, hue_order=MODELS, showfliers=False,
                palette=["#4C78A8", "#F58518", "#54A24B"], ax=ax)
    ax.set(xlabel="Cardiac structure", ylabel="Dice", ylim=(0, 1.02),
           title="Class-wise out-of-fold Dice")
    ax.legend(title="Architecture", frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"model_class_dice.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp1", type=Path,
                        default=Path("exp1_wolny_native/cv_metrics/oof_volume_results.csv"))
    parser.add_argument("--exp2", type=Path,
                        default=Path("exp2_3dconvlstm/cv_metrics/oof_volume_results.csv"))
    parser.add_argument("--exp3", type=Path,
                        default=Path("exp3_factorized/cv_metrics/oof_volume_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("cross_experiment_analysis"))
    args = parser.parse_args()

    frames = {
        "Exp1": load_oof(args.exp1, "Exp1"),
        "Exp2": load_oof(args.exp2, "Exp2"),
        "Exp3": load_oof(args.exp3, "Exp3"),
    }
    verify_pairing(frames)
    long = to_long(frames)
    descriptive = descriptive_table(long)
    omnibus, pairwise = statistical_tests(long)
    args.output.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.output / "paired_long_metrics.csv", index=False)
    descriptive.to_csv(args.output / "descriptive_summary.csv", index=False)
    omnibus.to_csv(args.output / "friedman_tests.csv", index=False)
    pairwise.to_csv(args.output / "pairwise_wilcoxon_holm.csv", index=False)
    paper_table(descriptive).to_csv(args.output / "paper_results_table.csv", index=False)
    with (args.output / "analysis_manifest.json").open("w") as file:
        json.dump({
            "models": list(MODELS), "patients": 100, "volumes_per_model": 200,
            "primary_pairing_unit": "patient; ED and ES averaged within patient for All-phase tests",
            "omnibus_test": "Friedman",
            "posthoc_test": "paired two-sided Wilcoxon signed-rank",
            "multiple_testing": "Holm correction across three model pairs within each outcome family",
            "alpha": 0.05,
        }, file, indent=2)
    save_figures(long, args.output)
    print(f"Compared {len(frames['Exp1'])} paired OOF volumes per model")
    print(f"Saved statistical tables and figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
