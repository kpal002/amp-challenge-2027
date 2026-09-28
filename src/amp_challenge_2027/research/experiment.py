"""Development comparison and explicitly unlocked, frozen-policy final evaluation."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from amp_challenge_2027 import features as feature_module
from amp_challenge_2027.features import featurize_many
from .data import aggregate_labels, atomic_json, digest, file_digest, load_prepared
from .embeddings import atomic_npz, read_vectors
from .models import (bootstrap_family_indices, error_auditor, fit_cached, mlp,
                     ridge, warning_features)

POTENCY_THRESHOLD = float(np.log10(16))
PENALTIES = (0.0, 0.5, 1.0, 2.0)


def sequence_order(scores: np.ndarray, sequences: list[str], k: int) -> np.ndarray:
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite selection scores")
    return np.array(sorted(range(len(scores)), key=lambda i: (scores[i], sequences[i]))[:k])


def choose_penalty(mean: np.ndarray, warning: np.ndarray, measured: np.ndarray,
                   sequences: list[str], k: int) -> float:
    """Development-only choice. Highest hit rate, then lowest median MIC, then lambda."""
    def objective(value):
        selected = sequence_order(mean + value * warning, sequences, k)
        return (-float(np.mean(measured[selected] <= POTENCY_THRESHOLD)),
                float(np.median(measured[selected])), value)
    return float(min(PENALTIES, key=objective))


def optional_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None


def selection_metrics(y: np.ndarray, scores: np.ndarray, sequences: list[str],
                      novelty: np.ndarray, k: int, seed: int) -> tuple[dict, np.ndarray]:
    selected = sequence_order(scores, sequences, k)
    observed = y[selected]
    active = observed <= POTENCY_THRESHOLD
    rng = np.random.default_rng(seed)
    batch_size = min(25, k)
    draws = [float(active[rng.choice(k, batch_size, replace=False)].mean()) for _ in range(2000)]
    return {"selected": k, "hit_rate": float(active.mean()),
            "median_measured_log10_mic": float(np.median(observed)),
            "median_distance_to_training": float(np.median(novelty[selected])),
            "random_draw_size": batch_size,
            "random_draw_hit_rate_p10": float(np.quantile(draws, 0.1)),
            "random_draw_hit_rate_mean": float(np.mean(draws))}, selected


def paired_family_interval(y: np.ndarray, left: np.ndarray, right: np.ndarray,
                           groups: np.ndarray, sequences: list[str], k: int,
                           seed: int, repeats: int = 500) -> list[float]:
    """Exploratory paired family-bootstrap interval for difference in selected hit rate.

    Models/policies remain fixed. This omits training variability, and is not a
    prospective confidence guarantee or a replacement for experimental evidence.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = bootstrap_family_indices(groups, rng)
        if len(indices) < k:
            continue
        seqs = [sequences[i] for i in indices]
        a = indices[sequence_order(left[indices], seqs, k)]
        b = indices[sequence_order(right[indices], seqs, k)]
        values.append(np.mean(y[a] <= POTENCY_THRESHOLD) - np.mean(y[b] <= POTENCY_THRESHOLD))
    return np.quantile(values, [0.025, 0.975]).tolist() if values else [0.0, 0.0]


def _run(work: Path, species: str, embeddings: Path | None, k: int,
         members: int, iterations: int, seed: int, final: bool) -> dict:
    if k < 1 or members < 2 or iterations < 1:
        raise ValueError("Positive top-k/iterations and at least two ensemble members required")
    frame, splits, manifest = load_prepared(work)
    all_sequences = splits.sequence.tolist()
    lookup = {s: i for i, s in enumerate(all_sequences)}
    mode = "esm" if embeddings is not None else "descriptors"
    slug = species.lower().replace(".", "").replace(" ", "_").replace("/", "_")
    output = work / "results" / mode / slug
    output.mkdir(parents=True, exist_ok=True)
    config = {"dataset": digest(manifest), "species": species, "top_k": k,
              "ensemble_members": members, "mlp_iterations": iterations, "seed": seed,
              "embedding_sha256": file_digest(embeddings) if embeddings else None,
              "code": {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "features_code": file_digest(Path(feature_module.__file__)),
              "versions": {name: importlib.metadata.version(name) for name in
                           ("numpy", "scipy", "scikit-learn", "pandas", "levenshtein")}}
    identity = digest(config)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError(f"Experiment configuration changed; use a new work directory: {output}")
    if final and not config_path.exists():
        raise ValueError("Run development and freeze its policy before final evaluation")
    atomic_json(config_path, config)
    policy_path = output / "frozen_policy.json"
    if final and not policy_path.exists():
        raise ValueError("No frozen development policy; run development first")
    report_path = output / ("test_report.json" if final else "development_report.json")
    if report_path.exists():
        print(f"Reusing completed report: {report_path}", flush=True)
        return json.loads(report_path.read_text())

    descriptor_path = work / "descriptors.npz"
    descriptor_identity = digest({"sequences": all_sequences, "code": config["features_code"]})
    if descriptor_path.exists():
        with np.load(descriptor_path, allow_pickle=False) as z:
            if str(z["identity"]) != descriptor_identity:
                raise ValueError("Descriptor cache stale; use a new work directory")
        X = read_vectors(descriptor_path, all_sequences)
    else:
        X = featurize_many(all_sequences)
        atomic_npz(descriptor_path, sequences=np.array(all_sequences), vectors=X,
                   identity=np.array(descriptor_identity))
    E = read_vectors(embeddings, all_sequences) if embeddings else None

    # Development does not even aggregate test outcomes. Preparation stores them,
    # but all downstream label/model/tuning paths filter them out explicitly.
    allowed = splits.sequence if final else splits.loc[splits.split.ne("test"), "sequence"]
    labels = aggregate_labels(frame[frame.sequence.isin(allowed)], species)
    dataset = splits[splits.sequence.isin(labels.index)].copy()
    dataset["label"] = dataset.sequence.map(labels)
    by_split = {name: dataset[dataset.split.eq(name)].copy()
                for name in ("train", "calibration", "validation", "test")}
    for name in ("train", "calibration", "test" if final else "validation"):
        minimum = 100 if name == "train" else max(40, 2 * k)
        if len(by_split[name]) < minimum:
            raise ValueError(f"{species}: only {len(by_split[name])} strict {name} labels; need {minimum}")
    train = by_split["train"]
    train_idx = np.array([lookup[s] for s in train.sequence])
    y_train = train.label.to_numpy()
    # Unknown-modification and source models also use only train families, even
    # when they contain sequences absent from the strict main training labels.
    train_frame = frame[frame.sequence.isin(splits.loc[splits.split.eq("train"), "sequence"])]
    training_sequences = sorted(train_frame.loc[train_frame.bacterium.eq(species), "sequence"].unique())

    model_dir = output / "models"
    frozen = json.loads(policy_path.read_text()) if final else None
    if final:
        if frozen["identity"] != identity:
            raise ValueError("Frozen policy fingerprint mismatch")
        for filename, expected in frozen["model_hashes"].items():
            if file_digest(model_dir / filename) != expected:
                raise ValueError(f"Frozen model changed: {filename}")

    fitted = {}
    fit_records = {}

    def fit(name, estimator, matrix, target_labels):
        positions = np.array([lookup[s] for s in target_labels.index])
        if not splits.iloc[positions].split.eq("train").all():
            raise AssertionError("Non-training family entered predictor fit")
        fitted[name] = fit_cached(model_dir / f"{name}.joblib", estimator,
                                  matrix[positions], target_labels.to_numpy(), locked=final)
        fit_records[name] = {"n_sequences": len(positions),
                             "sequence_sha256": digest(target_labels.index.tolist()),
                             "convergence_warnings": fitted[name].research_convergence_warnings_}

    strict_train = pd.Series(y_train, index=train.sequence)
    fit("ridge", ridge(), X, strict_train)
    fit("mlp", mlp(seed, iterations), X, strict_train)
    rng = np.random.default_rng(seed)
    for member in range(members):
        indices = bootstrap_family_indices(train.family.to_numpy(), rng)
        boot_labels = strict_train.iloc[indices]
        fit(f"bootstrap_ridge_{member}", ridge(), X, boot_labels)
        fit(f"bootstrap_mlp_{member}", mlp(seed + member + 1, iterations), X, boot_labels)

    scenarios = []
    inclusive = aggregate_labels(train_frame, species, strict=False)
    if len(inclusive) >= 100:
        fit("unknown_modifications_included", ridge(), X, inclusive)
        scenarios.append("unknown_modifications_included")
    for number, source in enumerate(sorted(train_frame.database.unique())):
        targets = aggregate_labels(train_frame, species, omit_source=source)
        if len(targets) >= 100 and not targets.equals(strict_train.sort_index()):
            name = f"omit_source_{number}"
            fit(name, ridge(), X, targets)
            fit_records[name]["omitted_source"] = source
            scenarios.append(name)
    if E is not None:
        fit("embedding_ridge", ridge(), E, strict_train)
        scenarios.append("embedding_ridge")

    def predict(partition):
        sequences = partition.sequence.tolist()
        indices = np.array([lookup[s] for s in sequences])
        predictions = {name: model.predict((E if name == "embedding_ridge" else X)[indices])
                       for name, model in fitted.items()}
        boot = np.column_stack([0.5 * (predictions[f"bootstrap_ridge_{i}"]
                                              + predictions[f"bootstrap_mlp_{i}"])
                                for i in range(members)])
        mean = boot.mean(axis=1)
        novelty = np.array([1 - max(Levenshtein.ratio(s, t) for t in training_sequences)
                            for s in sequences])
        return predictions, boot, mean, novelty

    cal = by_split["calibration"]
    cal_pred, cal_boot, cal_mean, cal_novelty = predict(cal)
    cal_y = cal.label.to_numpy()
    base_rmse = float(np.sqrt(np.mean((cal_y - cal_mean) ** 2)))
    scenario_rmse = {name: float(np.sqrt(np.mean((cal_y - cal_pred[name]) ** 2))) for name in scenarios}
    # Predeclared credibility screen. This uses calibration, never validation/test.
    credible = [name for name in scenarios if scenario_rmse[name] <= 1.5 * base_rmse]
    if final and credible != frozen["credible_scenarios"]:
        raise ValueError("Calibration/model state changed since policy freezing")
    cal_features, feature_names = warning_features(
        cal_mean, cal_pred["ridge"], cal_pred["mlp"], cal_boot,
        {name: cal_pred[name] for name in credible}, cal_novelty)
    auditor = fit_cached(model_dir / "auditor.joblib", error_auditor(), cal_features,
                         np.maximum(0, cal_y - cal_mean), locked=final)

    stage = "test" if final else "validation"
    evaluation = by_split[stage]
    sequences = evaluation.sequence.tolist()
    y = evaluation.label.to_numpy()
    pred, boot, mean, novelty = predict(evaluation)
    signals, _ = warning_features(mean, pred["ridge"], pred["mlp"], boot,
                                 {name: pred[name] for name in credible}, novelty)
    risk = np.maximum(0, auditor.predict(signals))
    sigma = boot.std(axis=1)
    # Representation-matched controls are essential: the auditor must beat an
    # ordinary ensemble with the SAME access to ESM, not merely a weaker input.
    matched_mean, matched_sigma = None, None
    if E is not None:
        embedding_pred = pred["embedding_ridge"]
        matched_mean = 0.5 * (mean + embedding_pred)
        matched_sigma = np.sqrt(0.5 * sigma ** 2 + 0.25 * (mean - embedding_pred) ** 2)
    if final:
        penalties = frozen["penalties"]
    else:
        penalties = {"ensemble_uncertainty": choose_penalty(mean, sigma, y, sequences, k),
                     "challenger": choose_penalty(mean, risk, y, sequences, k)}
        if E is not None:
            penalties["representation_matched_uncertainty"] = choose_penalty(
                matched_mean, matched_sigma, y, sequences, k)
    scores = {"potency_blend": 0.5 * (pred["ridge"] + pred["mlp"]),
              "ensemble_mean": mean,
              "ensemble_uncertainty": mean + penalties["ensemble_uncertainty"] * sigma,
              "challenger": mean + penalties["challenger"] * risk}
    comparator = "ensemble_uncertainty"
    if E is not None:
        scores["embedding_only"] = embedding_pred
        scores["representation_matched_mean"] = matched_mean
        scores["representation_matched_uncertainty"] = (
            matched_mean + penalties["representation_matched_uncertainty"] * matched_sigma)
        comparator = "representation_matched_uncertainty"
    metrics, selections = {}, {}
    for name, values in scores.items():
        metrics[name], selections[name] = selection_metrics(y, values, sequences, novelty, k, seed)
    high_error = y - mean >= np.log10(4)
    top_pool = sequence_order(mean, sequences, min(len(y), max(2 * k, len(y) // 5)))
    report = {"identity": identity, "species": species, "stage": stage,
              "embedding_enabled": embeddings is not None, "top_k": k,
              "counts": {name: len(part) for name, part in by_split.items() if name != "test" or final},
              "pool_hit_rate": float(np.mean(y <= POTENCY_THRESHOLD)),
              "metrics": metrics, "penalties": penalties, "predictor_fit_records": fit_records,
              "uncertainty_comparator": comparator,
              "calibration": {"ensemble_rmse": base_rmse, "scenario_rmse": scenario_rmse,
                              "credible_scenarios": credible, "warning_features": feature_names},
              "diagnostics": {"ensemble_rmse": float(np.sqrt(np.mean((y - mean) ** 2))),
                              "ensemble_spearman": float(spearmanr(y, mean).statistic),
                              "fourfold_overprediction_rate": float(high_error.mean()),
                              "auditor_error_auc": optional_auc(high_error, risk),
                              "bootstrap_error_auc": optional_auc(high_error, sigma),
                              "auditor_error_auc_high_scoring_pool": optional_auc(high_error[top_pool], risk[top_pool]),
                              "bootstrap_error_auc_high_scoring_pool": optional_auc(high_error[top_pool], sigma[top_pool])},
              "challenger_minus_ensemble_hit_rate_interval": paired_family_interval(
                  y, scores["challenger"], scores[comparator],
                  evaluation.family.to_numpy(), sequences, k, seed),
              "limitations": [
                  "Development scores are used for tuning and are not final evidence.",
                  "Uncertainty is an empirical warning signal, not a calibrated probability or coverage guarantee.",
                  "Family-bootstrap interval conditions on fitted models; it omits training variability.",
                  "Species-level median MIC is not activity on the exact competition strain panel.",
                  "Source ablations are not independent laboratories: databases overlap.",
                  "Potency blend baseline is refitted; shipped checkpoint and full ranking pipeline are not evaluated.",
                  "Pretrained ESM may have seen evaluation sequences without activity labels.",
                  "No HC50/selectivity evidence and no prospective activity evidence for generated sequences."]}
    detail = evaluation[["sequence", "family", "label"]].reset_index(drop=True)
    detail["distance_to_training"] = novelty
    detail["ensemble_prediction"] = mean
    detail["bootstrap_std"] = sigma
    detail["auditor_warning"] = risk
    for name, values in scores.items():
        detail[f"score_{name}"] = values
        detail[f"selected_{name}"] = np.isin(np.arange(len(detail)), selections[name])
    # A novelty-stratified view checks whether an apparent gain merely selects
    # candidates closer to the training corpus. Fixed bins; same candidates per method.
    report["novelty_strata"] = {}
    for low, high in ((0.3, 0.4), (0.4, 0.5), (0.5, 1.01)):
        mask = (novelty >= low - 1e-12) & (novelty < high - 1e-12)
        indices = np.flatnonzero(mask)
        if len(indices) < 10:
            continue
        count = min(k, max(5, len(indices) // 5))
        report["novelty_strata"][f"{low:.1f}-{min(high, 1):.1f}"] = {
            "pool_size": len(indices), "top_k": count,
            "hit_rates": {name: float(np.mean(y[indices[sequence_order(
                values[indices], [sequences[i] for i in indices], count)]] <= POTENCY_THRESHOLD))
                for name, values in scores.items()}}
    if not final:
        frozen = {"identity": identity, "penalties": penalties,
                  "credible_scenarios": credible,
                  "model_hashes": {p.name: file_digest(p) for p in sorted(model_dir.glob("*.joblib"))}}
        atomic_json(policy_path, frozen)
    detail.to_csv(output / f"{stage}_predictions.csv", index=False)
    atomic_json(report_path, report)
    write_summary(output / ("TEST.md" if final else "DEVELOPMENT.md"), report)
    print(f"Wrote {report_path}", flush=True)
    return report


def write_summary(path: Path, report: dict) -> None:
    lines = [f"# Challenger experiment: {report['species']}", "",
             f"Stage: **{report['stage']}**. Embeddings: {report['embedding_enabled']}. "
             f"Selected candidates: {report['top_k']}.", "",
             "Validation results below are exploratory when stage is validation. "
             "The final test requires a separate explicit command.", "",
             "| Method | Measured hit rate | Median log10 MIC | Median distance to training |",
             "|---|---:|---:|---:|"]
    for name, row in report["metrics"].items():
        lines.append(f"| {name} | {row['hit_rate']:.1%} | {row['median_measured_log10_mic']:.3f} "
                     f"| {row['median_distance_to_training']:.3f} |")
    lines += ["", f"Full evaluation pool activity rate: {report['pool_hit_rate']:.1%}.", "",
              "Auditor AUC for fourfold overprediction: "
              f"{report['diagnostics']['auditor_error_auc']}; bootstrap disagreement AUC: "
              f"{report['diagnostics']['bootstrap_error_auc']}.", "",
              f"Exploratory paired family-bootstrap interval for challenger minus {report['uncertainty_comparator']} "
              f"hit rate: {report['challenger_minus_ensemble_hit_rate_interval']}.", "",
              "## Limits", "", *[f"- {item}" for item in report["limitations"]]]
    path.write_text("\n".join(lines) + "\n")


def run(work: Path, species: str, embeddings: Path | None = None, k: int = 50,
        members: int = 5, iterations: int = 1200, seed: int = 42, final: bool = False) -> dict:
    # Tiny dense regressions are faster and less variable without BLAS oversubscription.
    with threadpool_limits(limits=1):
        return _run(work, species, embeddings, k, members, iterations, seed, final)
