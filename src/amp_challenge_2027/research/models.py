"""Predictors and an error auditor fitted on disjoint sequence families."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def mlp(seed: int, iterations: int):
    return make_pipeline(StandardScaler(), MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3,
        learning_rate_init=1e-3, max_iter=iterations, early_stopping=True,
        n_iter_no_change=30, random_state=seed,
    ))


def fit_cached(path: Path, model, X: np.ndarray, y: np.ndarray, *, locked: bool = False):
    """Caller must bind the directory to a full input/code/config fingerprint."""
    if path.exists():
        return joblib.load(path)
    if locked:
        raise ValueError(f"Frozen model missing: {path}; do not refit during final evaluation")
    print(f"Fitting {path.stem}: {len(y)} measured sequences", flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
    for message in convergence:
        print(f"  Convergence warning: {message}", flush=True)
    model.research_convergence_warnings_ = convergence
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)
    return model


def bootstrap_family_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def error_auditor():
    # Nonnegative coefficients give monotone penalties in each warning signal.
    # No distribution-free coverage or probability interpretation is claimed.
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0, positive=True))


def warning_features(base: np.ndarray, ridge_pred: np.ndarray, mlp_pred: np.ndarray,
                     bootstrap: np.ndarray, scenarios: dict[str, np.ndarray],
                     novelty: np.ndarray) -> tuple[np.ndarray, list[str]]:
    values = [bootstrap.std(axis=1), np.abs(ridge_pred - mlp_pred), novelty]
    names = ["bootstrap_std", "ridge_mlp_disagreement", "distance_to_training"]
    for name in sorted(scenarios):
        delta = scenarios[name] - base
        values.extend([np.abs(delta), np.maximum(0, delta)])
        names.extend([f"{name}_absolute_delta", f"{name}_pessimistic_delta"])
    return np.column_stack(values), names
