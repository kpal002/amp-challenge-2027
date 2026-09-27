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


def stratum_labels(features: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """Assign each candidate to a physicochemical stratum.

    Bins are quantiles of length, net charge and hydrophobic moment over the whole
    library, giving `n_bins ** 3` strata.

    Edges are taken library-wide deliberately. Deriving them from the
    high-scoring selection pool instead was tried and measured worse on four of
    five categories -- broad-spectrum FBD 3.94 -> 8.73 and property conformity
    0.488 -> 0.221. Pool-local edges rescale the bins onto the pool's own narrow
    slice, so covering every stratum stops implying any spread in absolute terms.
    Library-wide edges are what force selection to reach into regions that are
    genuinely far apart.
    """
    label = np.zeros(features.shape[0], dtype=np.int64)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    for column in features.T:
        edges = np.quantile(column, quantiles)
        label = label * n_bins + np.searchsorted(edges, column, side="right")
    return label


def select_top(
    sequences: list[str],
    scores: np.ndarray,
    index: ReferenceIndex,
    k: int,
    in_envelope: np.ndarray | None = None,
    strata_features: np.ndarray | None = None,
    strata_bins: int = 4,
    strata_pool: int = 12000,
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

    Selection is additionally **stratified** over physicochemical space when
    `strata` is supplied. Score-greedy selection inside the envelope still
    collapsed into one narrow corner -- measured with seqme, the top-100 reached
    precision 0.97 but recall 0.17, the signature of a homogeneous cluster.
    Because 25 of the 100 are drawn at random for assay, homogeneity converts
    that draw into a correlated bet even when the sequences differ. Taking the
    best candidate from each stratum in turn spreads the list across charge,
    length and amphipathicity while still preferring high scores within each bin.
    """
    # Tie-break on the sequence string so the ordering is total and therefore
    # identical across runs, regardless of how ties fall out of the scorer.
    order = sorted(range(len(sequences)), key=lambda i: (-scores[i], sequences[i]))

    chosen: list[str] = []
    chosen_idx: list[int] = []
    rejected_reference = 0
    rejected_internal = 0
    rejected_envelope = 0

    def admissible(i: int) -> bool:
        nonlocal rejected_envelope, rejected_reference, rejected_internal
        seq = sequences[i]
        if in_envelope is not None and not in_envelope[i]:
            rejected_envelope += 1
            return False
        if index.max_similarity(seq, TOP_SIMILARITY_CEILING) > TOP_SIMILARITY_CEILING:
            rejected_reference += 1
            return False
        if any(Levenshtein.ratio(seq, other) > max_internal_similarity for other in chosen):
            rejected_internal += 1
            return False
        return True

    if strata_features is None:
        for i in order:
            if len(chosen) >= k:
                break
            if admissible(i):
                chosen.append(sequences[i])
                chosen_idx.append(i)
    else:
        # Stratify *within* the high-scoring region, not across the whole library.
        # Round-robin over every stratum pulls in the best member of weak strata
        # too, which costs most of the potency signal; restricting the pool to the
        # top-scoring candidates first keeps quality and buys spread inside it.
        pool = order[: max(strata_pool, k)]

        strata = stratum_labels(strata_features, n_bins=strata_bins)

        # Round-robin over strata, each holding its candidates in score order.
        queues: dict[int, list[int]] = {}
        for i in pool:
            queues.setdefault(int(strata[i]), []).append(i)
        stratum_ids = sorted(queues)
        cursors = {s: 0 for s in stratum_ids}

        while len(chosen) < k:
            progressed = False
            for s in stratum_ids:
                if len(chosen) >= k:
                    break
                queue, cursor = queues[s], cursors[s]
                while cursor < len(queue):
                    i = queue[cursor]
                    cursor += 1
                    if admissible(i):
                        chosen.append(sequences[i])
                        chosen_idx.append(i)
                        progressed = True
                        break
                cursors[s] = cursor
            if not progressed:
                break

        if len(chosen) < k:
            # The stratified pool was exhausted. Fall back to score order over the
            # rest of the library rather than failing: spread is a preference, but
            # returning a full ranked list is a requirement.
            already = set(chosen_idx)
            for i in order:
                if len(chosen) >= k:
                    break
                if i in already:
                    continue
                if admissible(i):
                    chosen.append(sequences[i])
                    chosen_idx.append(i)
            if verbose:
                print(
                    f"  stratified pool exhausted; topped up to {len(chosen)} "
                    f"by score order",
                    flush=True,
                )

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
    # Stratified round-robin selects in stratum order, not score order, but the
    # submitted top list must be ranked best-first. Re-sort before returning.
    chosen_idx.sort(key=lambda i: (-scores[i], sequences[i]))
    return [sequences[i] for i in chosen_idx]


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
    parser.add_argument(
        "--strata-bins",
        type=int,
        default=4,
        help="Bins per axis for stratified top-k selection (1 disables stratification).",
    )
    parser.add_argument(
        "--strata-pool",
        type=int,
        default=12000,
        help="Stratify within this many top-scoring candidates.",
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

    # Stratify over length, net charge and amphipathicity so the top list spans
    # physicochemical space rather than clustering in one corner of it.
    feature_index = components["feature_index"]
    strata_columns = [
        feature_index[name] for name in ("length", "net_charge", "hydrophobic_moment")
    ]
    strata_features = (
        components["features"][:, strata_columns] if args.strata_bins > 1 else None
    )

    top = select_top(
        library,
        scores,
        index,
        args.top_k,
        in_envelope=components["in_envelope"],
        strata_features=strata_features,
        strata_bins=args.strata_bins,
        strata_pool=args.strata_pool,
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
