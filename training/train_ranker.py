"""Train the two scoring models used to rank candidates.

1. Potency regressor: log10 MIC (uM) from sequence features, fitted only on
   *unmodified* GRAMPA measurements, because the competition forbids terminal
   modifications and ~43% of GRAMPA is C-terminally amidated. Predicting the
   potency of a molecule we cannot submit would be the wrong target.

2. AMP-likeness classifier: real AMPs against composition-matched shuffles and
   composition-sampled random peptides. Shuffled negatives are the important
   ones: they hold amino-acid composition fixed, so the model is forced to learn
   sequence *arrangement* (amphipathicity, cationic patches) rather than simply
   counting lysines. This acts as the realism filter on generated libraries.

Both are exported to a plain .npz for the NumPy inference path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.features import N_FEATURES, featurize_many  # noqa: E402

DATA = ROOT / "data"
CHECKPOINT = ROOT / "checkpoint"

# Fixed probes whose scikit-learn predictions are stored alongside the weights,
# so `check_parity.py` can prove the NumPy replay reproduces the fitted models
# without refitting them.
PROBES = (
    "GIGKFLHSAKKFGKAFVGEIMNS",
    "KRWWKWWRR",
    "FLPIIAKVLSGLL",
    "ACDEFGHIKLMNPQRSTVWY",
    "KKKKKKKKKK",
    "EEEEEEEEEE",
    "GLLSVLGSVAKHVLPHVVPVIAEHL",
)


def make_negatives(sequences: list[str], rng: np.random.Generator) -> list[str]:
    """Composition-matched shuffles plus composition-sampled random peptides."""
    negatives: list[str] = []
    for seq in sequences:
        # permutation on an ndarray, not shuffle on a list: Generator.shuffle
        # would copy a list via asarray and leave the original untouched, which
        # would silently make the negatives identical to the positives.
        negatives.append("".join(rng.permutation(np.array(list(seq)))))

    # Background peptides drawn from the pooled residue frequencies of the whole
    # corpus: same overall amino-acid usage, no peptide-specific structure.
    pool = np.array(list("".join(sequences)))
    lengths = np.array([len(s) for s in sequences])
    for length in rng.choice(lengths, size=len(sequences), replace=True):
        negatives.append("".join(rng.choice(pool, size=int(length), replace=True)))
    return negatives


def cluster_groups(sequences: list[str], threshold: float = 0.6) -> np.ndarray:
    """Greedy single-linkage clustering by Levenshtein ratio.

    A random split over these peptides overstates generalisation: AMP databases
    are full of homologues and truncation series, so a near-identical sequence
    usually sits in the training fold. Grouping similar peptides and splitting by
    group gives a harder, more honest estimate of performance on unrelated
    families.
    """
    import Levenshtein

    groups = np.full(len(sequences), -1, dtype=np.int64)
    representatives: list[tuple[str, int]] = []
    order = sorted(range(len(sequences)), key=lambda i: (-len(sequences[i]), sequences[i]))
    for i in order:
        seq = sequences[i]
        for rep, gid in representatives:
            if Levenshtein.ratio(seq, rep) >= threshold:
                groups[i] = gid
                break
        else:
            gid = len(representatives)
            representatives.append((seq, gid))
            groups[i] = gid
    return groups


def fit_potency(seed: int) -> dict[str, np.ndarray]:
    df = pd.read_csv(DATA / "train_mic.csv")
    sequences = df["sequence"].astype(str).tolist()
    y = df["log10_mic"].to_numpy(dtype=np.float64)
    X = featurize_many(sequences)
    print(f"potency: {X.shape[0]} sequences, {X.shape[1]} features")

    def build_mlp() -> MLPRegressor:
        return MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=1200,
            early_stopping=True,
            n_iter_no_change=30,
            random_state=seed,
        )

    # The scaler is fitted inside each fold, not once over the whole dataset.
    # Fitting it first leaks the test folds' means and variances into training and
    # inflates the reported out-of-fold numbers.
    def evaluate_split(splitter, groups: np.ndarray | None, label: str) -> None:
        oof_mlp = np.zeros_like(y)
        oof_ridge = np.zeros_like(y)
        for train_idx, test_idx in splitter.split(X, y, groups):
            mlp_pipe = make_pipeline(StandardScaler(), build_mlp())
            ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            oof_mlp[test_idx] = mlp_pipe.fit(X[train_idx], y[train_idx]).predict(X[test_idx])
            oof_ridge[test_idx] = ridge_pipe.fit(X[train_idx], y[train_idx]).predict(X[test_idx])
        blend = 0.5 * (oof_mlp + oof_ridge)
        print(f"  [{label}]")
        for name, pred in (("ridge", oof_ridge), ("mlp", oof_mlp), ("blend", blend)):
            rho = spearmanr(pred, y).statistic
            rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
            print(f"    {name:<6} Spearman {rho:+.3f}  RMSE {rmse:.3f} log10 units")

    evaluate_split(KFold(n_splits=5, shuffle=True, random_state=seed), None, "random split")

    groups = cluster_groups(sequences)
    n_groups = len(set(groups.tolist()))
    print(f"  similarity clusters (Levenshtein ratio >= 0.6): {n_groups} groups")
    evaluate_split(GroupKFold(n_splits=5), groups, "clustered split (harder, honest)")

    # Shipped model: scaler fitted on all data, which is correct for the final fit.
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    mlp = build_mlp().fit(Xs, y)
    ridge = Ridge(alpha=10.0).fit(Xs, y)

    out: dict[str, np.ndarray] = {
        "pot_mean": scaler.mean_.astype(np.float32),
        "pot_scale": scaler.scale_.astype(np.float32),
        "pot_ridge_w": ridge.coef_.astype(np.float32),
        "pot_ridge_b": np.float32(ridge.intercept_),
        "pot_n_layers": np.int64(len(mlp.coefs_)),
    }
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        out[f"pot_w{i}"] = w.astype(np.float32)
        out[f"pot_b{i}"] = b.astype(np.float32)

    probe_X = scaler.transform(featurize_many(list(PROBES)))
    out["probe_pot"] = (
        0.5 * (mlp.predict(probe_X) + ridge.predict(probe_X))
    ).astype(np.float64)
    return out


def fit_amp_likeness(seed: int) -> dict[str, np.ndarray]:
    gen = pd.read_csv(DATA / "train_generator.csv")
    positives = sorted(gen["sequence"].astype(str).unique())
    rng = np.random.default_rng(seed)
    negatives = make_negatives(positives, rng)

    X = featurize_many(positives + negatives)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    print(f"amp-likeness: {len(positives)} positives, {len(negatives)} negatives")

    def build_clf() -> MLPClassifier:
        return MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=seed,
        )

    # Scaler fitted per fold to avoid leaking test-fold statistics.
    kf = KFold(n_splits=4, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    oof_lr = np.zeros_like(y)
    for train_idx, test_idx in kf.split(X):
        clf_pipe = make_pipeline(StandardScaler(), build_clf())
        lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        oof[test_idx] = clf_pipe.fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])[:, 1]
        oof_lr[test_idx] = lr_pipe.fit(X[train_idx], y[train_idx]).predict_proba(X[test_idx])[:, 1]
    print(f"  logreg out-of-fold AUC {roc_auc_score(y, oof_lr):.4f}")
    print(f"  mlp    out-of-fold AUC {roc_auc_score(y, oof):.4f}")

    # Report separately against shuffles alone: the harder, order-only task.
    n_pos = len(positives)
    shuffle_mask = np.zeros(len(y), dtype=bool)
    shuffle_mask[:n_pos] = True
    shuffle_mask[n_pos : n_pos + n_pos] = True
    print(
        f"  mlp    AUC vs shuffled-only negatives "
        f"{roc_auc_score(y[shuffle_mask], oof[shuffle_mask]):.4f}"
    )

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = build_clf().fit(Xs, y)
    out: dict[str, np.ndarray] = {
        "amp_mean": scaler.mean_.astype(np.float32),
        "amp_scale": scaler.scale_.astype(np.float32),
        "amp_n_layers": np.int64(len(clf.coefs_)),
    }
    for i, (w, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        out[f"amp_w{i}"] = w.astype(np.float32)
        out[f"amp_b{i}"] = b.astype(np.float32)

    probe_X = scaler.transform(featurize_many(list(PROBES)))
    out["probe_amp"] = clf.predict_proba(probe_X)[:, 1].astype(np.float64)
    return out


# Descriptors that define the plausibility envelope for top-100 candidates.
ENVELOPE_FEATURES = (
    "net_charge",
    "charge_per_residue",
    "hydrophobic_moment",
    "frac_hydrophobic",
    "gravy",
    "length",
    "frac_cationic",
    "boman",
    "isoelectric_point",
    "max_hydrophobic_run",
)
ENVELOPE_LOW_PCT = 2.5
ENVELOPE_HIGH_PCT = 97.5


def fit_envelope() -> dict[str, np.ndarray]:
    """Physicochemical bounds taken from peptides with *measured* potency.

    Ranking on predicted potency alone drives selection into the upper tail of
    the cationic/amphipathic distribution, where the MIC model is extrapolating
    beyond anything it was trained on, and where peptides tend to be hemolytic.
    Constraining candidates to the central range of measured-potent AMPs keeps
    the top-100 in the region where the oracle has evidence, and keeps it
    comparable to known potent peptides -- which is what the aggregation score's
    embedding-based terms reward.
    """
    from amp_challenge_2027.features import FEATURE_NAMES
    from amp_challenge_2027.fasta import read_sequences

    mic = pd.read_csv(DATA / "train_mic.csv")
    reference = set(read_sequences(DATA / "antibacterial.fasta"))
    potent = [s for s, v in zip(mic["sequence"], mic["log10_mic"]) if v <= 1.0 and s in reference]

    X = featurize_many(potent)
    index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    cols = np.array([index[name] for name in ENVELOPE_FEATURES], dtype=np.int64)

    lo = np.percentile(X[:, cols], ENVELOPE_LOW_PCT, axis=0)
    hi = np.percentile(X[:, cols], ENVELOPE_HIGH_PCT, axis=0)
    print(f"envelope from {len(potent)} measured-potent AMPs (MIC <= 10 uM):")
    for name, a, b in zip(ENVELOPE_FEATURES, lo, hi):
        print(f"  {name:<22} [{a:>8.3f}, {b:>8.3f}]")

    return {
        "env_cols": cols,
        "env_lo": lo.astype(np.float64),
        "env_hi": hi.astype(np.float64),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    weights = {"n_features": np.int64(N_FEATURES)}
    weights.update(fit_potency(args.seed))
    weights.update(fit_amp_likeness(args.seed))
    weights.update(fit_envelope())

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    np.savez(CHECKPOINT / "ranker.npz", **weights)
    print(f"\nwrote {CHECKPOINT / 'ranker.npz'}")


if __name__ == "__main__":
    main()
