"""NumPy replay of the trained scoring models, and the candidate ranking policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import featurize_many


def _mlp_forward(x: np.ndarray, weights: dict[str, np.ndarray], prefix: str, n_layers: int) -> np.ndarray:
    """ReLU MLP matching sklearn's MLP layer ordering."""
    h = x
    for i in range(n_layers):
        h = h @ weights[f"{prefix}_w{i}"] + weights[f"{prefix}_b{i}"]
        if i < n_layers - 1:
            h = np.maximum(h, 0.0)
    return h


class Scorer:
    """Predicts potency (log10 MIC) and AMP-likeness for peptide sequences."""

    def __init__(self, weights_path: Path) -> None:
        z = np.load(weights_path)
        self.w = {k: z[k] for k in z.files}
        self.pot_layers = int(self.w["pot_n_layers"])
        self.amp_layers = int(self.w["amp_n_layers"])

    def features(self, sequences: list[str]) -> np.ndarray:
        return featurize_many(sequences)

    def predict_log_mic(self, X: np.ndarray) -> np.ndarray:
        """Lower is more potent. Blend of the MLP and ridge models."""
        Xs = (X - self.w["pot_mean"]) / self.w["pot_scale"]
        mlp = _mlp_forward(Xs, self.w, "pot", self.pot_layers).ravel()
        ridge = Xs @ self.w["pot_ridge_w"] + self.w["pot_ridge_b"]
        return 0.5 * (mlp + ridge)

    def predict_amp_likeness(self, X: np.ndarray) -> np.ndarray:
        """Probability in [0, 1] that a sequence looks like a real AMP."""
        Xs = (X - self.w["amp_mean"]) / self.w["amp_scale"]
        logit = _mlp_forward(Xs, self.w, "amp", self.amp_layers).ravel()
        return 1.0 / (1.0 + np.exp(-logit))

    def in_envelope(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask: does each sequence lie within the potent-AMP envelope?

        The bounds are the central 95% range of peptides with *measured* MIC at or
        below 10 uM. Candidates outside it are ones where the potency model is
        extrapolating rather than interpolating.
        """
        if "env_cols" not in self.w:
            return np.ones(X.shape[0], dtype=bool)
        cols = self.w["env_cols"]
        lo, hi = self.w["env_lo"], self.w["env_hi"]
        sub = X[:, cols]
        return np.all((sub >= lo) & (sub <= hi), axis=1)


# --- Ranking policy ---------------------------------------------------------
# The competition's aggregation score is withheld, so the top-100 is not tuned to
# a single oracle. It combines predicted potency, realism, and a selectivity
# proxy, each converted to a rank-normalised score so that no term can dominate
# through scale alone.

# Hemolysis rises with overall hydrophobicity and with long uninterrupted
# hydrophobic stretches, so the therapeutic proxy penalises both. This is a
# published qualitative trend, not a fitted model -- we have no HC50 data locally
# and deliberately do not pretend to.
def _rank_normalise(values: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Map to [0, 1] by rank. Ties share a rank; deterministic ordering."""
    n = len(values)
    if n == 0:
        return values
    if n == 1:
        return np.ones(1)
    order = np.argsort(values if higher_is_better else -values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def selectivity_proxy(X: np.ndarray, feature_index: dict[str, int]) -> np.ndarray:
    """Higher is expected to be less hemolytic (more therapeutically usable)."""
    frac_hydrophobic = X[:, feature_index["frac_hydrophobic"]]
    max_run = X[:, feature_index["max_hydrophobic_run"]]
    charge_density = X[:, feature_index["charge_per_residue"]]
    # Cationicity is associated with bacterial selectivity; bulk hydrophobicity
    # and long hydrophobic runs with mammalian membrane damage.
    return charge_density - 0.8 * frac_hydrophobic - 0.05 * max_run


def composite_score(
    scorer: Scorer,
    sequences: list[str],
    category: str,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (score, components). Higher score ranks better."""
    from .features import FEATURE_NAMES

    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    X = scorer.features(sequences)

    log_mic = scorer.predict_log_mic(X)
    amp_like = scorer.predict_amp_likeness(X)
    selectivity = selectivity_proxy(X, index)

    potency_rank = _rank_normalise(log_mic, higher_is_better=False)
    realism_rank = _rank_normalise(amp_like, higher_is_better=True)
    selectivity_rank = _rank_normalise(selectivity, higher_is_better=True)

    presets = {
        "broad_spectrum": {"potency": 0.50, "realism": 0.35, "selectivity": 0.15},
        "gram_pos": {"potency": 0.50, "realism": 0.35, "selectivity": 0.15},
        "gram_neg": {"potency": 0.50, "realism": 0.35, "selectivity": 0.15},
        "mdr": {"potency": 0.60, "realism": 0.30, "selectivity": 0.10},
        # The therapeutic category is ranked on the safety window, so the
        # selectivity proxy carries far more weight here.
        "therapeutic": {"potency": 0.30, "realism": 0.25, "selectivity": 0.45},
    }
    w = weights or presets[category]

    score = (
        w["potency"] * potency_rank
        + w["realism"] * realism_rank
        + w["selectivity"] * selectivity_rank
    )
    components = {
        "log_mic": log_mic,
        "amp_likeness": amp_like,
        "selectivity": selectivity,
        "potency_rank": potency_rank,
        "realism_rank": realism_rank,
        "selectivity_rank": selectivity_rank,
        "in_envelope": scorer.in_envelope(X),
        # Returned so callers can stratify selection without recomputing features.
        "features": X,
        "feature_index": index,
    }
    return score, components
