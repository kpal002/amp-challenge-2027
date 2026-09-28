"""Diagnose and refine the challenger locally, without reading final-test outcomes.

Run: python -m amp_challenge_2027.challenger_analysis

Existing predictor weights are frozen. New selection models learn only from
calibration families; grouped cross-validation chooses their regularization.
Validation is used only for the subsequent, explicitly exploratory comparison.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import joblib
import Levenshtein
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .research.data import aggregate_labels, atomic_json, digest, file_digest, load_prepared
from .research.embeddings import read_vectors
from .research.experiment import (POTENCY_THRESHOLD, choose_penalty, optional_auc,
                                 paired_family_interval, selection_metrics, sequence_order)
from .research.models import error_auditor, warning_features

REGULARIZATION = (0.01, 0.1, 1.0, 10.0)
TASKS = ("E. coli", "S. aureus", "P. aeruginosa")


def development_labels(frame: pd.DataFrame, splits: pd.DataFrame, species: str) -> pd.DataFrame:
    """Exclude final-test records BEFORE aggregating any activity values."""
    allowed = splits.loc[splits.split.ne("test"), "sequence"]
    labels = aggregate_labels(frame[frame.sequence.isin(allowed)], species)
    result = splits[splits.split.ne("test") & splits.sequence.isin(labels.index)].copy()
    result["label"] = result.sequence.map(labels)
    return result


def active_probability(model, X: np.ndarray) -> np.ndarray:
    classes = model.classes_.tolist()
    return model.predict_proba(X)[:, classes.index(1)] if 1 in classes else np.zeros(len(X))


def classification_model(X: np.ndarray, y: np.ndarray, strength: float):
    estimator = (LogisticRegression(C=strength, max_iter=2000, solver="lbfgs")
                 if len(np.unique(y)) > 1 else DummyClassifier(strategy="prior"))
    return make_pipeline(StandardScaler(), estimator).fit(X, y)


def grouped_folds(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    count = min(5, len(np.unique(groups)))
    if count < 2:
        raise ValueError("Need at least two calibration families for cross-validation")
    return list(GroupKFold(n_splits=count).split(np.zeros(len(groups)), groups=groups))


def fit_threshold_model(X: np.ndarray, labels: np.ndarray, groups: np.ndarray):
    """All scaling, fitting, and C selection use calibration only."""
    target = (labels <= POTENCY_THRESHOLD).astype(int)
    folds = grouped_folds(groups)
    trials = []
    predictions = {}
    for strength in REGULARIZATION:
        oof = np.full(len(labels), np.nan)
        for train, holdout in folds:
            model = classification_model(X[train], target[train], strength)
            oof[holdout] = active_probability(model, X[holdout])
        if not np.isfinite(oof).all():
            raise AssertionError("Calibration OOF predictions are incomplete")
        loss = float(log_loss(target, oof, labels=[0, 1]))
        trials.append({"C": strength, "family_oof_log_loss": loss})
        predictions[strength] = oof
    chosen = min(trials, key=lambda row: (row["family_oof_log_loss"], row["C"]))["C"]
    return classification_model(X, target, chosen), predictions[chosen], {
        "C": chosen, "selection_criterion": "calibration family-OOF log loss",
        "trials": trials,
        "calibration_family_oof_brier": float(brier_score_loss(target, predictions[chosen])),
        "note": "OOF statistics selected C and are development diagnostics, not unbiased final estimates"}


def tail_weights(predictions: np.ndarray) -> np.ndarray:
    """Fixed emphasis on high predicted potency, with no measured labels involved."""
    return 1.0 + 4.0 * (predictions <= np.quantile(predictions, 0.2))


def fit_tail_auditor(features: np.ndarray, mean: np.ndarray, labels: np.ndarray,
                     groups: np.ndarray, sequences: list[str], k: int):
    residual = np.maximum(0, labels - mean)
    oof = np.full(len(labels), np.nan)
    for train, holdout in grouped_folds(groups):
        model = error_auditor().fit(features[train], residual[train],
                                   ridge__sample_weight=tail_weights(mean[train]))
        oof[holdout] = np.maximum(0, model.predict(features[holdout]))
    penalty = choose_penalty(mean, oof, labels, sequences, k)
    model = error_auditor().fit(features, residual, ridge__sample_weight=tail_weights(mean))
    return model, oof, {"penalty": penalty,
                       "selection_criterion": "calibration family-OOF selected activity yield",
                       "weighting": "5x weight for lowest predicted-MIC quintile; 1x otherwise"}


def predictor_features(predictions: dict[str, np.ndarray], bootstrap: np.ndarray,
                       novelty: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[str]], np.ndarray]:
    mean = bootstrap.mean(axis=1)
    scenarios = {name: values for name, values in predictions.items()
                 if name not in ("ridge", "mlp") and not name.startswith("bootstrap_")}
    # Every source predictor is available to every richer method. No credibility
    # screen based on held-out fold labels precedes grouped hyperparameter tuning.
    raw_names = ["ensemble_prediction", "embedding_ridge"]
    raw = np.column_stack([mean, predictions["embedding_ridge"]])
    source_names = sorted(name for name in scenarios if name != "embedding_ridge")
    stacked = np.column_stack([raw, *[predictions[name] for name in source_names]])
    warnings, warning_names = warning_features(
        mean, predictions["ridge"], predictions["mlp"], bootstrap, scenarios, novelty)
    features = {"threshold_only": mean[:, None],
                "representation_stack": raw,
                "source_stack": stacked,
                "threshold_auditor": np.column_stack([stacked, warnings])}
    names = {"threshold_only": ["ensemble_prediction"], "representation_stack": raw_names,
             "source_stack": raw_names + source_names,
             "threshold_auditor": raw_names + source_names + warning_names}
    return features, names, warnings


def swaps(labels: np.ndarray, challenger: np.ndarray, comparator: np.ndarray,
          sequences: list[str], k: int) -> dict:
    a = set(sequence_order(challenger, sequences, k))
    b = set(sequence_order(comparator, sequences, k))
    gained, lost = sorted(a - b), sorted(b - a)
    active = labels <= POTENCY_THRESHOLD
    return {"gained_candidates": len(gained), "gained_active": int(active[gained].sum()),
            "lost_candidates": len(lost), "lost_active": int(active[lost].sum())}


def nearest_distance(sequences: list[str], references: list[str]) -> np.ndarray:
    return np.array([1 - max(Levenshtein.ratio(s, r) for r in references) for s in sequences])


def inspect_species(work: Path, output: Path, species: str) -> dict:
    frame, splits, manifest = load_prepared(work)
    slug = species.lower().replace(".", "").replace(" ", "_")
    original = work / "results/esm" / slug
    original_config = json.loads((original / "config.json").read_text())
    frozen = json.loads((original / "frozen_policy.json").read_text())
    original_report = json.loads((original / "development_report.json").read_text())
    if frozen["identity"] != digest(original_config) or original_report["identity"] != frozen["identity"]:
        raise ValueError("Original configuration/policy/report identities disagree")
    if original_config["dataset"] != digest(manifest):
        raise ValueError("Original model dataset/split identity no longer matches")
    research_dir = Path(__file__).parent / "research"
    for name, expected in original_config["code"].items():
        if file_digest(research_dir / name) != expected:
            raise ValueError(f"Original experiment source changed: {name}")
    for name, expected in frozen["model_hashes"].items():
        if file_digest(original / "models" / name) != expected:
            raise ValueError(f"Original model changed: {name}")
    vectors_path = work / "embeddings/vectors.npz"
    if file_digest(vectors_path) != original_config["embedding_sha256"]:
        raise ValueError("Original embeddings changed")
    descriptor_path = work / "descriptors.npz"
    with np.load(descriptor_path, allow_pickle=False) as cached:
        expected_descriptor_identity = digest({"sequences": splits.sequence.tolist(),
                                                "code": original_config["features_code"]})
        if str(cached["identity"]) != expected_descriptor_identity:
            raise ValueError("Original descriptor cache provenance changed")
    config = {"original_identity": frozen["identity"], "dataset": digest(manifest),
              "analysis_source_sha256": file_digest(Path(__file__)),
              "original_report_sha256": file_digest(original / "development_report.json"),
              "original_prediction_sha256": file_digest(original / "validation_predictions.csv"),
              "descriptors_sha256": file_digest(descriptor_path),
              "regularization_grid": list(REGULARIZATION), "top_k": original_config["top_k"],
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "pandas", "scikit-learn", "scipy")}}
    dest = output / slug
    dest.mkdir(parents=True, exist_ok=True)
    config_path = dest / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Analysis configuration changed; choose a new output directory")
    atomic_json(config_path, config)
    if (dest / "report.json").exists():
        return json.loads((dest / "report.json").read_text())

    # Final-test values are excluded here and never enter a downstream aggregate.
    development = development_labels(frame, splits, species)
    allowed = splits.loc[splits.split.ne("test"), "sequence"]
    frame = frame[frame.sequence.isin(allowed)].copy()
    all_sequences = splits.sequence.tolist()
    lookup = {s: i for i, s in enumerate(all_sequences)}
    descriptors = read_vectors(work / "descriptors.npz", all_sequences)
    embeddings = read_vectors(vectors_path, all_sequences)
    models = {p.stem: joblib.load(p) for p in sorted((original / "models").glob("*.joblib"))
              if p.stem != "auditor"}
    train_sequences = splits.loc[splits.split.eq("train"), "sequence"]
    refs = sorted(frame.loc[frame.sequence.isin(train_sequences) & frame.bacterium.eq(species), "sequence"].unique())
    members = original_config["ensemble_members"]
    k = original_config["top_k"]
    prepared = {}
    for stage in ("calibration", "validation"):
        part = development[development.split.eq(stage)].copy()
        indices = np.array([lookup[s] for s in part.sequence])
        predictions = {name: model.predict((embeddings if name == "embedding_ridge" else descriptors)[indices])
                       for name, model in models.items()}
        bootstrap = np.column_stack([0.5 * (predictions[f"bootstrap_ridge_{i}"] +
                                            predictions[f"bootstrap_mlp_{i}"]) for i in range(members)])
        novelty = nearest_distance(part.sequence.tolist(), refs)
        features, names, warnings = predictor_features(predictions, bootstrap, novelty)
        prepared[stage] = {"part": part, "predictions": predictions, "bootstrap": bootstrap,
                           "mean": bootstrap.mean(axis=1), "novelty": novelty,
                           "features": features, "names": names, "warnings": warnings}
    cal, val = prepared["calibration"], prepared["validation"]
    y_cal = cal["part"].label.to_numpy()
    groups = cal["part"].family.to_numpy()
    fit_metadata, new_models = {}, {}
    cal_oof = cal["part"][["sequence", "family", "label"]].reset_index(drop=True)

    # Fit and freeze every variant before accessing validation outcomes below.
    for name, X in cal["features"].items():
        print(f"{species}: calibration-family CV for {name}", flush=True)
        model, oof, fit_metadata[name] = fit_threshold_model(X, y_cal, groups)
        fit_metadata[name]["features"] = cal["names"][name]
        new_models[name] = model
        cal_oof[name] = oof
    tail_model, tail_oof, fit_metadata["tail_error_auditor"] = fit_tail_auditor(
        cal["warnings"], cal["mean"], y_cal, groups, cal["part"].sequence.tolist(), k)
    new_models["tail_error_auditor"] = tail_model
    cal_oof["tail_error_auditor_warning"] = tail_oof
    for name, model in new_models.items():
        path = dest / f"{name}.joblib"
        temporary = path.with_suffix(".tmp")
        joblib.dump(model, temporary)
        temporary.replace(path)
    atomic_json(dest / "frozen_variants.json", {
        "identity": digest(config), "fit_partition": "calibration",
        "fit_sequence_sha256": digest(cal["part"].sequence.tolist()),
        "policy": fit_metadata,
        "model_hashes": {p.name: file_digest(p) for p in sorted(dest.glob("*.joblib"))}})
    cal_oof.to_csv(dest / "calibration_oof.csv", index=False)

    sequences = val["part"].sequence.tolist()
    y_val = val["part"].label.to_numpy()
    base_detail = pd.read_csv(original / "validation_predictions.csv")
    if base_detail.sequence.tolist() != sequences or not np.allclose(base_detail.label, y_val):
        raise ValueError("Original validation cohort does not match prepared data")
    scores = {name.removeprefix("score_"): base_detail[name].to_numpy()
              for name in base_detail if name.startswith("score_")}
    metrics = {}
    for name, model in new_models.items():
        if name == "tail_error_auditor":
            warning = np.maximum(0, model.predict(val["warnings"]))
            scores[name] = val["mean"] + fit_metadata[name]["penalty"] * warning
        else:
            scores[name] = -active_probability(model, val["features"][name])
    detail = val["part"][["sequence", "family", "label"]].reset_index(drop=True)
    detail["distance_to_training"] = val["novelty"]
    for name, values in scores.items():
        metrics[name], selected = selection_metrics(y_val, values, sequences, val["novelty"], k, 42)
        detail[f"score_{name}"] = values
        detail[f"selected_{name}"] = np.isin(np.arange(len(detail)), selected)
        if name in cal["features"]:
            metrics[name]["validation_log_loss"] = float(log_loss(y_val <= POTENCY_THRESHOLD, -values, labels=[0, 1]))
            metrics[name]["validation_brier"] = float(brier_score_loss(y_val <= POTENCY_THRESHOLD, -values))

    # Paired comparisons isolate task alignment from access to extra predictors.
    comparisons = {}
    for challenger, comparator in (("threshold_auditor", "source_stack"),
                                   ("threshold_auditor", "representation_stack"),
                                   ("threshold_auditor", "representation_matched_uncertainty"),
                                   ("source_stack", "representation_matched_uncertainty"),
                                   ("source_stack", "challenger"),
                                   ("tail_error_auditor", "challenger")):
        comparisons[f"{challenger}_minus_{comparator}"] = {
            "hit_rate_delta": metrics[challenger]["hit_rate"] - metrics[comparator]["hit_rate"],
            "conditional_family_bootstrap_interval": paired_family_interval(
                y_val, scores[challenger], scores[comparator], val["part"].family.to_numpy(), sequences, k, 42),
            "swaps": swaps(y_val, scores[challenger], scores[comparator], sequences, k)}

    novelty_strata = {}
    for low, high in ((0.3, 0.4), (0.4, 0.5), (0.5, 1.01)):
        indices = np.flatnonzero((val["novelty"] >= low - 1e-12) &
                                 (val["novelty"] < high - 1e-12))
        if len(indices) < 10:
            continue
        count = min(k, max(5, len(indices) // 5))
        cohort_sequences = [sequences[i] for i in indices]
        novelty_strata[f"{low:.1f}-{min(high, 1):.1f}"] = {
            "pool_size": len(indices), "selected": count,
            "hit_rates": {name: float(np.mean(y_val[indices[sequence_order(
                values[indices], cohort_sequences, count)]] <= POTENCY_THRESHOLD))
                for name, values in scores.items()}}

    original_auditor = joblib.load(original / "models/auditor.joblib")
    original_coefficients = dict(zip(original_report["calibration"]["warning_features"],
                                     original_auditor[-1].coef_.tolist()))
    # Raw warning transfer: predictors trained on train, no learned auditor fit
    # on the calibration labels being used to assess each raw signal here.
    warning_names = cal["names"]["threshold_auditor"][len(cal["names"]["source_stack"]):]
    signal_transfer = {}
    for j, name in enumerate(warning_names):
        signal_transfer[name] = {}
        for stage, data in prepared.items():
            event = data["part"].label.to_numpy() - data["mean"] >= np.log10(4)
            pool = sequence_order(data["mean"], data["part"].sequence.tolist(),
                                  min(len(event), max(2 * k, len(event) // 5)))
            signal_transfer[name][f"{stage}_error_auc_top_pool"] = optional_auc(event[pool], data["warnings"][pool, j])

    report = {"identity": digest(config), "species": species, "stage": "development_validation",
              "metrics": metrics, "fit_metadata": fit_metadata, "comparisons": comparisons,
              "novelty_strata": novelty_strata,
              "original_warning_coefficients_standardized": original_coefficients,
              "original_challenger_swaps": swaps(y_val, scores["challenger"],
                                                   scores["representation_matched_uncertainty"], sequences, k),
              "raw_warning_transfer": signal_transfer,
              "counts": {"calibration": len(y_cal), "validation": len(y_val),
                         "calibration_families": len(np.unique(groups))},
              "limitations": [
                  "Variants were motivated by earlier validation results; these remain development experiments.",
                  "All new hyperparameters were selected on calibration-family OOF predictions, never validation labels.",
                  "OOF diagnostics used to choose hyperparameters are not unbiased final estimates.",
                  "Historical species-median MIC labels do not establish activity on competition strains.",
                  "No final-test outcome was aggregated, fitted, scored, or reported.",
                  "Class probabilities are model outputs; their calibration under design distribution shift is unproven.",
                  "Conditional bootstrap intervals omit training and method-selection uncertainty.",
                  "No generator or submission checkpoint was changed."]}
    detail.to_csv(dest / "validation_predictions.csv", index=False)
    atomic_json(dest / "report.json", report)
    return report


def write_overview(output: Path, reports: list[dict]) -> None:
    methods = ["potency_blend", "representation_matched_uncertainty", "challenger",
               "threshold_only", "representation_stack", "source_stack",
               "threshold_auditor", "tail_error_auditor"]
    lines = ["# Challenger investigation", "", "Development data only; final test remains unopened.", "",
             "Each cell is measured activity yield among 50 selected historical candidates.", "",
             "| Method | " + " | ".join(r["species"] for r in reports) + " |",
             "|---|" + "---:|" * len(reports)]
    for method in methods:
        lines.append("| " + method + " | " + " | ".join(
            f"{r['metrics'][method]['hit_rate']:.0%}" for r in reports) + " |")
    lines += ["", "New variants use calibration-family cross-validation for all settings. "
              "The original challenger/control penalties were previously tuned on validation. "
              "This table remains exploratory because earlier validation results motivated the variants.", "",
              "- `threshold_only`: calibrate the descriptor ensemble prediction to activity probability.",
              "- `representation_stack`: combine descriptor and ESM predictions using an activity classifier.",
              "- `source_stack`: also include source/modification-sensitivity model predictions.",
              "- `threshold_auditor`: add disagreement, novelty, and uncertainty warning features to that same stack.",
              "- `tail_error_auditor`: retain positive-error regression, weighting the highest-scoring calibration quintile 5x.", "",
              "## Paired comparisons", ""]
    for report in reports:
        lines.append(f"### {report['species']}")
        lines.append("")
        for name, comparison in report["comparisons"].items():
            lo, hi = comparison["conditional_family_bootstrap_interval"]
            lines.append(f"- {name}: {comparison['hit_rate_delta']:+.0%}; exploratory interval [{lo:+.0%}, {hi:+.0%}].")
        lines.append("")
    lines += ["## Limits", "", *[f"- {s}" for s in reports[0]["limitations"]]]
    (output / "OVERVIEW.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("research_runs/challenger"))
    parser.add_argument("--output", type=Path, default=Path("research_runs/challenger_analysis"))
    parser.add_argument("--species", choices=TASKS, nargs="+", default=list(TASKS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        reports = [inspect_species(args.work, args.output, species) for species in args.species]
    write_overview(args.output, reports)
    print((args.output / "OVERVIEW.md").read_text())


if __name__ == "__main__":
    main()
