"""Physicochemical and sequence features, in pure NumPy.

Implemented here rather than pulled from modlamp/biopython so that inference has
no dependency beyond NumPy, and so the exact definitions are pinned in the
repository instead of floating with a third-party release.

A deliberate emphasis on **order-sensitive** descriptors (hydrophobic moment,
reduced-alphabet dipeptides, hydrophobic and cationic runs). Composition-only
features cannot distinguish a real peptide from a shuffled one, which is exactly
the discrimination the AMP-likeness model has to make.
"""

from __future__ import annotations

import numpy as np

from .constants import AMINO_ACIDS

# Kyte & Doolittle hydropathy.
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4, "H": -3.2,
    "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5,
    "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

# Eisenberg consensus hydrophobicity, used for the hydrophobic moment.
EISENBERG = {
    "A": 0.62, "C": 0.29, "D": -0.90, "E": -0.74, "F": 1.19, "G": 0.48,
    "H": -0.40, "I": 1.38, "K": -1.50, "L": 1.06, "M": 0.64, "N": -0.78,
    "P": 0.12, "Q": -0.85, "R": -2.53, "S": -0.18, "T": -0.05, "V": 1.08,
    "W": 0.81, "Y": 0.26,
}

# Boman protein-binding potential (kcal/mol), sign as published.
BOMAN = {
    "A": -1.81, "C": -1.28, "D": 8.72, "E": 6.81, "F": -2.98, "G": -0.94,
    "H": 2.33, "I": -4.92, "K": 5.55, "L": -4.92, "M": -2.35, "N": 6.64,
    "P": 0.0, "Q": 5.54, "R": 14.92, "S": 3.40, "T": 2.57, "V": -4.04,
    "W": -2.33, "Y": -0.14,
}

# pKa values for net-charge and isoelectric-point calculation.
PKA_SIDE_POS = {"K": 10.54, "R": 12.48, "H": 6.04}
PKA_SIDE_NEG = {"D": 3.90, "E": 4.07, "C": 8.37, "Y": 10.46}
PKA_N_TERM = 9.69
PKA_C_TERM = 2.34

# Seven-class reduced alphabet (Murphy-style), for order-sensitive dipeptides.
REDUCED = {
    **{a: 0 for a in "AGV"},
    **{a: 1 for a in "ILFP"},
    **{a: 2 for a in "YMTS"},
    **{a: 3 for a in "HNQW"},
    **{a: 4 for a in "RK"},
    **{a: 5 for a in "DE"},
    "C": 6,
}
N_REDUCED = 7

HYDROPHOBIC = set("AVILMFWCY")
CATIONIC = set("KR")

AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}


def net_charge(seq: str, ph: float = 7.4) -> float:
    """Net charge via Henderson-Hasselbalch, including free termini."""
    charge = 1.0 / (1.0 + 10.0 ** (ph - PKA_N_TERM))
    charge -= 1.0 / (1.0 + 10.0 ** (PKA_C_TERM - ph))
    for residue in seq:
        if residue in PKA_SIDE_POS:
            charge += 1.0 / (1.0 + 10.0 ** (ph - PKA_SIDE_POS[residue]))
        elif residue in PKA_SIDE_NEG:
            charge -= 1.0 / (1.0 + 10.0 ** (PKA_SIDE_NEG[residue] - ph))
    return charge


def isoelectric_point(seq: str) -> float:
    """pI by bisection on the net-charge curve."""
    lo, hi = 0.0, 14.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if net_charge(seq, mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hydrophobic_moment(seq: str, window: int = 11, angle_deg: float = 100.0) -> float:
    """Maximum Eisenberg hydrophobic moment over a sliding window.

    Order-sensitive: measures how well hydrophobic residues segregate onto one
    face of an alpha helix, which is the structural basis of AMP amphipathicity.
    """
    h = np.array([EISENBERG.get(a, 0.0) for a in seq], dtype=np.float64)
    n = len(h)
    if n == 0:
        return 0.0
    w = min(window, n)
    angles = np.deg2rad(angle_deg) * np.arange(w)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    best = 0.0
    for start in range(n - w + 1):
        seg = h[start : start + w]
        mu = np.hypot(seg @ cos_a, seg @ sin_a) / w
        if mu > best:
            best = float(mu)
    return best


def _max_run(seq: str, members: set[str]) -> int:
    best = run = 0
    for residue in seq:
        run = run + 1 if residue in members else 0
        if run > best:
            best = run
    return best


def aliphatic_index(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    a = seq.count("A") / n
    v = seq.count("V") / n
    il = (seq.count("I") + seq.count("L")) / n
    return 100.0 * (a + 2.9 * v + 3.9 * il)


# Feature-vector layout, fixed so exported model weights stay meaningful.
SCALAR_NAMES = (
    "length",
    "net_charge",
    "charge_per_residue",
    "gravy",
    "eisenberg_mean",
    "hydrophobic_moment",
    "aromaticity",
    "aliphatic_index",
    "boman",
    "isoelectric_point",
    "frac_hydrophobic",
    "frac_cationic",
    "frac_anionic",
    "max_hydrophobic_run",
    "max_cationic_run",
    "n_cysteine",
    "frac_glycine",
    "frac_proline",
)
FEATURE_NAMES = (
    SCALAR_NAMES
    + tuple(f"comp_{a}" for a in AMINO_ACIDS)
    + tuple(f"dip_{i}_{j}" for i in range(N_REDUCED) for j in range(N_REDUCED))
)
N_FEATURES = len(FEATURE_NAMES)


def featurize(seq: str) -> np.ndarray:
    """Feature vector for one peptide. Layout matches FEATURE_NAMES."""
    n = len(seq)
    out = np.zeros(N_FEATURES, dtype=np.float64)
    if n == 0:
        return out

    charge = net_charge(seq)
    out[0] = n
    out[1] = charge
    out[2] = charge / n
    out[3] = sum(KD.get(a, 0.0) for a in seq) / n
    out[4] = sum(EISENBERG.get(a, 0.0) for a in seq) / n
    out[5] = hydrophobic_moment(seq)
    out[6] = sum(seq.count(a) for a in "FWY") / n
    out[7] = aliphatic_index(seq)
    out[8] = sum(BOMAN.get(a, 0.0) for a in seq) / n
    out[9] = isoelectric_point(seq)
    out[10] = sum(1 for a in seq if a in HYDROPHOBIC) / n
    out[11] = sum(1 for a in seq if a in CATIONIC) / n
    out[12] = sum(1 for a in seq if a in "DE") / n
    out[13] = _max_run(seq, HYDROPHOBIC)
    out[14] = _max_run(seq, CATIONIC)
    out[15] = seq.count("C")
    out[16] = seq.count("G") / n
    out[17] = seq.count("P") / n

    base = len(SCALAR_NAMES)
    for residue in seq:
        idx = AA_INDEX.get(residue)
        if idx is not None:
            out[base + idx] += 1.0
    out[base : base + 20] /= n

    dip_base = base + 20
    if n >= 2:
        for a, b in zip(seq, seq[1:]):
            i, j = REDUCED.get(a), REDUCED.get(b)
            if i is not None and j is not None:
                out[dip_base + i * N_REDUCED + j] += 1.0
        out[dip_base:] /= n - 1

    return out


def featurize_many(sequences: list[str]) -> np.ndarray:
    if not sequences:
        return np.zeros((0, N_FEATURES), dtype=np.float64)
    return np.stack([featurize(s) for s in sequences])
