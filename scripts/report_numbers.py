"""Emit every number quoted in the documentation, from the saved outputs.

The README tables previously drifted: top-100 median net charge was quoted as
both 3.99 and 8.99 while the saved artifact measured something else again,
because figures were copied by hand from different runs and configurations. This
script is the single source for those numbers. Regenerate the tables from its
output after any change that touches generation or scoring.

    uv run python scripts/report_numbers.py

It reads only committed artifacts (checkpoint/, data/) and the generated output
directories, and reports which commit and which files it measured.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import Levenshtein
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.compliance import ReferenceIndex  # noqa: E402
from amp_challenge_2027.constants import CATEGORIES  # noqa: E402
from amp_challenge_2027.fasta import read_sequences  # noqa: E402
from amp_challenge_2027.features import FEATURE_NAMES  # noqa: E402
from amp_challenge_2027.scoring import Scorer  # noqa: E402

INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + (" (dirty)" if dirty else "")
    except Exception:
        return "unknown"


def describe(scorer: Scorer, sequences: list[str]) -> dict[str, float]:
    X = scorer.features(sequences)
    mic = scorer.predict_log_mic(X)
    row = {
        "n": len(sequences),
        "pred_log_mic_median": float(np.median(mic)),
        "pred_log_mic_top10": float(np.median(mic[:10])) if len(mic) >= 10 else float("nan"),
        "amp_likeness_median": float(np.median(scorer.predict_amp_likeness(X))),
    }
    for key in ("net_charge", "hydrophobic_moment", "frac_hydrophobic", "length"):
        col = X[:, INDEX[key]]
        row[f"{key}_median"] = float(np.median(col))
        row[f"{key}_iqr"] = float(np.percentile(col, 75) - np.percentile(col, 25))
    return row


def main() -> None:
    scorer = Scorer(ROOT / "checkpoint" / "ranker.npz")
    reference = read_sequences(ROOT / "data" / "antibacterial.fasta")
    index = ReferenceIndex(reference)

    print(f"commit: {git_commit()}")
    print(f"reference: data/antibacterial.fasta ({len(reference)} sequences)")
    print(f"ranker: checkpoint/ranker.npz\n")

    dirs = [ROOT / "generate"] + [ROOT / f"generate_{c}" for c in CATEGORIES]
    dirs = [d for d in dirs if (d / "library.fasta").exists()]
    if not dirs:
        raise SystemExit("no generated output found; run `uv run generate` first")

    print("=" * 108)
    print("LIBRARY")
    print("=" * 108)
    header = f"{'output dir':<26}{'n':>7}{'unique':>8}{'len med':>9}{'charge med':>12}{'muH med':>9}{'pred MIC med':>14}"
    print(header)
    print("-" * len(header))
    for d in dirs:
        lib = read_sequences(d / "library.fasta")
        r = describe(scorer, lib)
        print(
            f"{d.name:<26}{len(lib):>7}{len(set(lib)):>8}{r['length_median']:>9.0f}"
            f"{r['net_charge_median']:>12.2f}{r['hydrophobic_moment_median']:>9.3f}"
            f"{r['pred_log_mic_median']:>14.3f}"
        )

    print()
    print("=" * 108)
    print("TOP-100")
    print("=" * 108)
    header = (
        f"{'output dir':<26}{'n':>5}{'maxsim':>8}{'charge med':>12}{'charge IQR':>12}"
        f"{'len med':>9}{'muH med':>9}{'MIC med':>9}{'MIC top10':>11}"
    )
    print(header)
    print("-" * len(header))
    for d in dirs:
        top = read_sequences(d / "top.fasta")
        r = describe(scorer, top)
        sims = [index.max_similarity(s, 0.75) for s in top]
        print(
            f"{d.name:<26}{len(top):>5}{max(sims):>8.3f}{r['net_charge_median']:>12.2f}"
            f"{r['net_charge_iqr']:>12.2f}{r['length_median']:>9.0f}"
            f"{r['hydrophobic_moment_median']:>9.3f}{r['pred_log_mic_median']:>9.3f}"
            f"{r['pred_log_mic_top10']:>11.3f}"
        )

    print()
    print("=" * 108)
    print("NOVELTY AND INTERNAL SIMILARITY (subsampled, seed 0)")
    print("=" * 108)
    rng = np.random.default_rng(0)
    for d in dirs:
        lib = read_sequences(d / "library.fasta")
        top = read_sequences(d / "top.fasta")
        probe = [lib[i] for i in rng.choice(len(lib), size=min(400, len(lib)), replace=False)]
        lib_sims = np.array([index.exact_max_similarity(s) for s in probe])
        top_sims = np.array([index.exact_max_similarity(s) for s in top])
        pair = [
            Levenshtein.ratio(a, b)
            for i, a in enumerate(probe[:300])
            for b in probe[i + 1 : 300]
        ]
        print(
            f"{d.name:<26} library novelty median {np.median(lib_sims):.3f} "
            f"p95 {np.percentile(lib_sims, 95):.3f} max {lib_sims.max():.3f} | "
            f"top-100 max {top_sims.max():.3f} | mean pairwise {np.mean(pair):.3f}"
        )

    print()
    print("Reference cohorts for comparison")
    print("-" * 60)
    import pandas as pd

    mic = pd.read_csv(ROOT / "data" / "train_mic.csv")
    refset = set(reference)
    potent = [s for s, v in zip(mic.sequence, mic.log10_mic) if v <= 1.0 and s in refset]
    weak = [s for s, v in zip(mic.sequence, mic.log10_mic) if v >= 2.0 and s in refset]
    for name, seqs in (
        ("reference AMPs", reference[:6000]),
        ("potent refs (MIC<=10uM)", potent),
        ("weak refs (MIC>=100uM)", weak),
    ):
        r = describe(scorer, seqs)
        print(
            f"  {name:<26} n={r['n']:<6} charge {r['net_charge_median']:>6.2f}  "
            f"muH {r['hydrophobic_moment_median']:.3f}  len {r['length_median']:.0f}  "
            f"pred MIC {r['pred_log_mic_median']:+.3f}"
        )


if __name__ == "__main__":
    main()
