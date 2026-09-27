"""Entry point: generate a compliant AMP library and a ranked top-100 list.

Invoked as a console script. The output directory is named after the entry point
(``Path(sys.argv[0]).stem``), matching the official template, so `generate`,
`generate_mdr` and friends all route through this one implementation and write to
their own directory.

Determinism is a hard requirement -- the organizers run the script twice and
byte-compare both FASTA files. Everything random flows from a single seeded
NumPy Generator, inference is float32 NumPy on CPU, and every ordering step uses
a stable sort with an explicit tie-break.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import Levenshtein
import numpy as np

from .compliance import (
    NoveltyScreen,
    ReferenceIndex,
    dedupe_valid,
    validate_library,
    validate_top,
)
from .constants import (
    CATEGORIES,
    LIBRARY_SIZE,
    TOP_SIMILARITY_CEILING,
    TOP_SIZE,
)
from .fasta import read_sequences, write_fasta
from .sampler import PeptideSampler
from .scoring import Scorer, composite_score

# Sampling temperatures cycled across batches. A single temperature either
# collapses diversity (low) or degrades realism (high); cycling covers more of
# the distribution while keeping the mixture fixed and reproducible.
TEMPERATURE_SCHEDULE = (0.85, 0.95, 1.00, 1.05, 1.15)

# Cap on how much we oversample before giving up, as a multiple of the target.
MAX_OVERSAMPLE = 8.0


def resolve_category(entry_point: str) -> str:
    """Map the invoked script name to a competition category."""
    if entry_point.startswith("generate_"):
        suffix = entry_point.removeprefix("generate_")
        if suffix in CATEGORIES:
            return suffix
        raise SystemExit(f"unknown category in entry point {entry_point!r}")
    return "broad_spectrum"


def build_library(
    sampler: PeptideSampler,
    category: str,
    n_target: int,
    rng: np.random.Generator,
    exclude: frozenset[str],
    batch_size: int,
    novelty: NoveltyScreen | None = None,
    verbose: bool = True,
) -> list[str]:
    """Sample until `n_target` unique, compliant, novel sequences are collected."""
    collected: list[str] = []
    seen: set[str] = set()
    drawn = 0
    rejected_near_duplicate = 0
    budget = int(n_target * MAX_OVERSAMPLE)
    round_index = 0

    while len(collected) < n_target and drawn < budget:
        temperature = TEMPERATURE_SCHEDULE[round_index % len(TEMPERATURE_SCHEDULE)]
        n_draw = min(batch_size, budget - drawn)
        raw = sampler.sample(
            n_draw,
            category=category,
            rng=rng,
            temperature=temperature,
            top_p=0.95,
            batch_size=n_draw,
        )
        drawn += n_draw

        fresh = dedupe_valid(raw, exclude=exclude | frozenset(seen))
        for seq in fresh:
            if len(collected) >= n_target:
                break
            seen.add(seq)
            if novelty is not None and not novelty.is_novel(seq):
                rejected_near_duplicate += 1
                continue
            collected.append(seq)

        if verbose:
            print(
                f"  round {round_index + 1:>3}  T={temperature:.2f}  "
                f"drawn {drawn:>7}  kept {len(collected):>6}/{n_target}"
                + (f"  near-dup dropped {rejected_near_duplicate}" if novelty else ""),
                flush=True,
            )
        round_index += 1

    if len(collected) < n_target:
        raise RuntimeError(
            f"only produced {len(collected)} of {n_target} valid unique sequences "
            f"after drawing {drawn}; raise MAX_OVERSAMPLE or sampling temperature"
        )
    return collected


def select_top(
    sequences: list[str],
    scores: np.ndarray,
    index: ReferenceIndex,
    k: int,
    in_envelope: np.ndarray | None = None,
    max_internal_similarity: float = 0.80,
    verbose: bool = True,
) -> list[str]:
    """Pick the k best-scoring sequences that are novel and mutually distinct.

    Three filters beyond the raw score:

    * reference novelty -- required by the rules;
    * internal diversity -- 25 of the 100 are drawn at random for assay, so a
      cluster of near-identical peptides would turn one design idea into a
      correlated bet. Enforcing mutual distance makes the drawn sample
      informative about the model rather than about a single lucky motif;
    * plausibility envelope -- ranking on predicted potency alone pushes
      selection into the upper tail of the cationic/amphipathic distribution,
      where the potency model extrapolates and peptides tend to be hemolytic.
      Candidates must lie within the central range of measured-potent AMPs.
    """
    # Tie-break on the sequence string so the ordering is total and therefore
    # identical across runs, regardless of how ties fall out of the scorer.
    order = sorted(range(len(sequences)), key=lambda i: (-scores[i], sequences[i]))

    chosen: list[str] = []
    rejected_reference = 0
    rejected_internal = 0
    rejected_envelope = 0

    for i in order:
        if len(chosen) >= k:
            break
        seq = sequences[i]
        if in_envelope is not None and not in_envelope[i]:
            rejected_envelope += 1
            continue
        if index.max_similarity(seq, TOP_SIMILARITY_CEILING) > TOP_SIMILARITY_CEILING:
            rejected_reference += 1
            continue
        if any(Levenshtein.ratio(seq, other) > max_internal_similarity for other in chosen):
            rejected_internal += 1
            continue
        chosen.append(seq)

    if verbose:
        print(
            f"  top-{k} selection: {rejected_envelope} rejected outside the potent "
            f"envelope, {rejected_reference} for reference similarity, "
            f"{rejected_internal} for redundancy",
            flush=True,
        )
    if len(chosen) < k:
        raise RuntimeError(
            f"only {len(chosen)} of {k} candidates passed the novelty and "
            f"diversity filters; generate a larger library"
        )
    return chosen


def main() -> None:
    entry_point = Path(sys.argv[0]).stem

    parser = argparse.ArgumentParser(description="Generate an AMP library and ranked top list.")
    parser.add_argument("--n-sequences", type=int, default=LIBRARY_SIZE)
    parser.add_argument("--top-k", type=int, default=TOP_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    # Part of the reproducibility contract, not a free tuning knob: every batch
    # draws from the same RNG stream, so changing this changes the output. The
    # organizers invoke with no arguments, so the default is what ships.
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--category", default=None, help="Override the inferred category.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=Path("checkpoint/generator.npz"))
    parser.add_argument("--ranker", type=Path, default=Path("checkpoint/ranker.npz"))
    parser.add_argument("--reference", type=Path, default=Path("data/antibacterial.fasta"))
    parser.add_argument("--known", type=Path, default=Path("data/known_amps.txt"))
    parser.add_argument(
        "--novelty-ceiling",
        type=float,
        default=0.90,
        help="Drop library sequences above this similarity to a known AMP (0 disables).",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    verbose = not args.quiet
    category = args.category or resolve_category(entry_point)
    out_dir = args.out_dir or Path(entry_point)

    t0 = time.time()
    if verbose:
        print(f"entry point '{entry_point}' -> category '{category}', seed {args.seed}")

    reference = read_sequences(args.reference)
    index = ReferenceIndex(reference)
    known = frozenset(
        line.strip() for line in args.known.read_text().splitlines() if line.strip()
    ) if args.known.exists() else frozenset(reference)

    sampler = PeptideSampler(args.weights)
    scorer = Scorer(args.ranker)

    # One Generator drives every random draw, so the seed fully determines output.
    rng = np.random.default_rng(args.seed)

    novelty = (
        NoveltyScreen(reference, ceiling=args.novelty_ceiling)
        if args.novelty_ceiling > 0
        else None
    )

    if verbose:
        print(f"sampling (excluding {len(known)} known AMPs)")
    library = build_library(
        sampler,
        category=category,
        n_target=args.n_sequences,
        rng=rng,
        exclude=known,
        batch_size=args.batch_size,
        novelty=novelty,
        verbose=verbose,
    )

    if verbose:
        print("scoring library")
    scores, components = composite_score(scorer, library, category)

    top = select_top(
        library,
        scores,
        index,
        args.top_k,
        in_envelope=components["in_envelope"],
        verbose=verbose,
    )

    # Validate before writing: never emit a file we know to be non-compliant.
    validate_library(library, frozenset(reference), args.n_sequences)
    validate_top(top, frozenset(library), index, args.top_k)

    write_fasta(library, out_dir / "library.fasta")
    write_fasta(top, out_dir / "top.fasta")

    if verbose:
        mic = components["log_mic"]
        top_idx = [library.index(s) for s in top[:10]]
        print(
            f"\nlibrary {len(library)} -> {out_dir/'library.fasta'}\n"
            f"top {len(top)} -> {out_dir/'top.fasta'}\n"
            f"predicted log10 MIC: library median {np.median(mic):+.3f}, "
            f"top-10 median {np.median(mic[top_idx]):+.3f}\n"
            f"elapsed {time.time() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
