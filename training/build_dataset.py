"""Build the training corpora from public data.

Inputs (both committed to the repo for full training-data disclosure):
  data/antibacterial.fasta  MarLys reference AMPs, supplied by the organizers
                            (CC-0, Mendeley DOI 10.17632/w4hb5grjwb.3)
  data/grampa.csv           GRAMPA MIC compilation, Witten & Witten
                            (github.com/zswitten/Antimicrobial-Peptides)

Outputs:
  data/train_generator.csv  sequence, category  (one row per applicable category)
  data/train_mic.csv        sequence, log10_mic, n_obs

Only unmodified peptides are kept for the MIC model. 43% of GRAMPA rows carry
C-terminal amidation, and the competition forbids terminal modifications, so a
model fitted on amidated measurements would predict the potency of a molecule we
are not allowed to submit.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amp_challenge_2027.constants import AMINO_ACID_SET, MAX_LENGTH, MIN_LENGTH  # noqa: E402
from amp_challenge_2027.fasta import read_fasta  # noqa: E402

DATA = ROOT / "data"


def _compliant(seq: str) -> bool:
    return MIN_LENGTH <= len(seq) <= MAX_LENGTH and not (set(seq) - AMINO_ACID_SET)


def build_breadth_set() -> set[str]:
    """Sequences measured as potent across several species.

    The reference database's ``anti-gram-`` and ``anti-gram+`` labels turn out to
    be mutually exclusive, so "active against both" never fires and cannot define
    the MDR category. Instead we derive breadth from GRAMPA's measurements:
    peptides assayed against at least three distinct bacteria with a median MIC
    at or below 10 uM. That is grounded in data rather than in a label.
    """
    df = pd.read_csv(DATA / "grampa.csv")
    df["sequence"] = df["sequence"].astype(str).str.upper()
    unmod = df[
        ~df["is_modified"].astype(bool)
        & ~df["has_cterminal_amidation"].astype(bool)
        & ~df["has_unusual_modification"].astype(bool)
    ]
    grouped = unmod.groupby("sequence").agg(
        n_bacteria=("bacterium", "nunique"), median_mic=("value", "median")
    )
    selected = grouped[(grouped.n_bacteria >= 3) & (grouped.median_mic <= 1.0)]
    return {s for s in selected.index if _compliant(s)}


def build_generator_corpus(breadth: set[str]) -> pd.DataFrame:
    headers, sequences = read_fasta(DATA / "antibacterial.fasta")

    rows: list[dict[str, str]] = []
    seen_fasta: set[str] = set()
    for header, seq in zip(headers, sequences):
        if not _compliant(seq):
            continue
        seen_fasta.add(seq)
        match = re.search(r"activity=(\S+)", header)
        labels = set(match.group(1).split("|")) if match else set()

        categories: set[str] = set()
        if "anti-gram-" in labels:
            categories.add("gram_neg")
        if "anti-gram+" in labels:
            categories.add("gram_pos")
        if seq in breadth:
            categories.add("mdr")
        # Everything in this database is antimicrobial, so every sequence is a
        # valid broad-spectrum training example.
        categories.add("broad_spectrum")
        # Peptides without a cysteine cannot form disulfides, so they are the
        # cleanest examples of the strictly linear, free-terminus peptides the
        # therapeutic category should draw from.
        if "C" not in seq:
            categories.add("therapeutic")

        for category in sorted(categories):
            rows.append({"sequence": seq, "category": category})

    # GRAMPA contributes measured-potent peptides that the reference FASTA does
    # not contain; they are valid public AMPs and widen the training signal.
    for seq in sorted(breadth - seen_fasta):
        for category in ("broad_spectrum", "mdr"):
            rows.append({"sequence": seq, "category": category})
        if "C" not in seq:
            rows.append({"sequence": seq, "category": "therapeutic"})

    return pd.DataFrame(rows)


def build_known_amps() -> set[str]:
    """Every known AMP we can see, for novelty exclusion at generation time.

    Only exact matches to ``antibacterial.fasta`` are *disqualifying*, but
    re-emitting any already-published peptide would still cost novelty points, so
    the generator excludes this wider set.
    """
    _, reference = read_fasta(DATA / "antibacterial.fasta")
    known = {s for s in reference if _compliant(s)}
    df = pd.read_csv(DATA / "grampa.csv")
    known |= {
        s for s in df["sequence"].astype(str).str.upper().unique() if _compliant(s)
    }
    return known


def build_mic_corpus() -> pd.DataFrame:
    df = pd.read_csv(DATA / "grampa.csv")
    df["sequence"] = df["sequence"].astype(str).str.upper()

    keep = (
        df["sequence"].map(_compliant)
        & ~df["is_modified"].astype(bool)
        & ~df["has_unusual_modification"].astype(bool)
        & ~df["has_cterminal_amidation"].astype(bool)
    )
    df = df.loc[keep, ["sequence", "value"]]

    # Aggregate replicate measurements by median: GRAMPA pools several databases
    # and strains, so a single sequence can carry dozens of MIC values whose
    # spread reflects assay conditions rather than the peptide.
    grouped = (
        df.groupby("sequence")["value"]
        .agg(log10_mic="median", n_obs="size")
        .reset_index()
        .sort_values("sequence", kind="mergesort")
    )
    return grouped


def main() -> None:
    breadth = build_breadth_set()
    gen = build_generator_corpus(breadth)
    gen = gen.sort_values(["category", "sequence"], kind="mergesort")
    gen.to_csv(DATA / "train_generator.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    mic = build_mic_corpus()
    mic.to_csv(DATA / "train_mic.csv", index=False)

    known = build_known_amps()
    (DATA / "known_amps.txt").write_text("\n".join(sorted(known)) + "\n")

    print(f"breadth (MDR) set: {len(breadth)} sequences")
    print(f"known-AMP exclusion set: {len(known)} sequences")
    print(f"generator corpus: {len(gen)} rows, {gen.sequence.nunique()} unique sequences")
    print(gen.category.value_counts().to_string())
    print(
        f"\nMIC corpus: {len(mic)} unique sequences, "
        f"median log10 MIC {np.median(mic.log10_mic):.3f}, "
        f"{int(mic.n_obs.sum())} measurements"
    )


if __name__ == "__main__":
    main()
