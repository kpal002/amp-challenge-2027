"""Compare a generated library against known AMPs and decoy controls.

The competition's aggregation score is withheld, so we cannot optimise against
it. What we can do is reproduce the discrimination it was reportedly tuned for:
known potent AMPs versus weak ones, versus decoys. If the generated library sits
with the potent AMPs on every axis independently, that is evidence of robustness
rather than of having fitted one hidden weighting.

Reference cohorts:
  reference  known AMPs from the competition's own database
  potent     reference AMPs with measured median MIC <= 10 uM
  weak       reference AMPs with measured median MIC >= 100 uM
  shuffled   the generated library, shuffled per sequence (composition-matched)
  random     random peptides matching the corpus residue frequencies
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.compliance import ReferenceIndex  # noqa: E402
from amp_challenge_2027.fasta import read_sequences  # noqa: E402
from amp_challenge_2027.features import FEATURE_NAMES  # noqa: E402
from amp_challenge_2027.scoring import Scorer  # noqa: E402

INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
REPORT = (
    ("predicted log10 MIC", "log_mic", "lower = more potent"),
    ("AMP-likeness", "amp_like", "higher = more realistic"),
    ("net charge", "net_charge", ""),
    ("hydrophobic moment", "hydrophobic_moment", ""),
    ("frac hydrophobic", "frac_hydrophobic", ""),
    ("length", "length", ""),
)


def describe(scorer: Scorer, name: str, sequences: list[str]) -> dict[str, float]:
    X = scorer.features(sequences)
    row = {
        "n": len(sequences),
        "log_mic": float(np.median(scorer.predict_log_mic(X))),
        "amp_like": float(np.median(scorer.predict_amp_likeness(X))),
    }
    for key in ("net_charge", "hydrophobic_moment", "frac_hydrophobic", "length"):
        row[key] = float(np.median(X[:, INDEX[key]]))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=Path("generate/library.fasta"))
    ap.add_argument("--top", type=Path, default=Path("generate/top.fasta"))
    ap.add_argument("--sample", type=int, default=6000, help="subsample size for speed")
    ap.add_argument("--novelty-sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    scorer = Scorer(ROOT / "checkpoint" / "ranker.npz")

    library = read_sequences(args.library)
    top = read_sequences(args.top)
    reference = read_sequences(ROOT / "data" / "antibacterial.fasta")

    mic = pd.read_csv(ROOT / "data" / "train_mic.csv")
    ref_set = set(reference)
    potent = [s for s, v in zip(mic.sequence, mic.log10_mic) if v <= 1.0 and s in ref_set]
    weak = [s for s, v in zip(mic.sequence, mic.log10_mic) if v >= 2.0 and s in ref_set]

    def subsample(seqs: list[str], k: int) -> list[str]:
        if len(seqs) <= k:
            return list(seqs)
        idx = rng.choice(len(seqs), size=k, replace=False)
        return [seqs[i] for i in idx]

    lib_s = subsample(library, args.sample)
    shuffled = ["".join(rng.permutation(np.array(list(s)))) for s in lib_s]
    pool = np.array(list("".join(subsample(reference, 2000))))
    random_peptides = [
        "".join(rng.choice(pool, size=len(s), replace=True)) for s in lib_s
    ]

    cohorts = {
        "generated library": lib_s,
        "generated top-100": top,
        "reference AMPs": subsample(reference, args.sample),
        "potent refs (MIC<=10uM)": potent,
        "weak refs (MIC>=100uM)": weak,
        "shuffled decoys": shuffled,
        "random peptides": random_peptides,
    }

    rows = {name: describe(scorer, name, seqs) for name, seqs in cohorts.items()}
    table = pd.DataFrame(rows).T

    print("Medians by cohort\n")
    header = f"{'cohort':<26}{'n':>7}" + "".join(f"{label:>22}" for label, _, _ in REPORT)
    print(header)
    print("-" * len(header))
    for name, row in table.iterrows():
        line = f"{name:<26}{int(row['n']):>7}"
        for _, key, _ in REPORT:
            line += f"{row[key]:>22.3f}"
        print(line)

    print("\nNotes:")
    for label, _, note in REPORT:
        if note:
            print(f"  {label}: {note}")

    # Novelty: true maximum identity to the reference database, on a subsample.
    print(f"\nNovelty (exact max Levenshtein ratio vs {len(reference)} reference AMPs)")
    index = ReferenceIndex(reference)
    for name, seqs in (("library", subsample(library, args.novelty_sample)), ("top-100", top)):
        sims = np.array([index.exact_max_similarity(s) for s in seqs])
        print(
            f"  {name:<8} n={len(seqs):<5} median {np.median(sims):.3f}  "
            f"p95 {np.percentile(sims, 95):.3f}  max {sims.max():.3f}"
        )

    # Internal diversity of the library, as mean pairwise distance on a subsample.
    import Levenshtein

    probe = subsample(library, 300)
    pairwise = [
        Levenshtein.ratio(a, b)
        for i, a in enumerate(probe)
        for b in probe[i + 1 :]
    ]
    print(
        f"\nInternal similarity (library subsample n={len(probe)}): "
        f"mean {np.mean(pairwise):.3f}  p99 {np.percentile(pairwise, 99):.3f}  "
        f"max {np.max(pairwise):.3f}"
    )
    print(f"Unique sequences: {len(set(library))}/{len(library)}")
    lengths = np.array([len(s) for s in library])
    print(f"Length: min {lengths.min()}  median {int(np.median(lengths))}  max {lengths.max()}")


if __name__ == "__main__":
    main()
