"""Hard compliance gate.

Every sequence leaving the generator passes through here. The rules mirror the
official validator, so a library that survives this cannot fail their check.
Compliance is treated as a gate, not a post-hoc audit: a single violation
disqualifies the whole submission.
"""

from __future__ import annotations

from collections import defaultdict

import Levenshtein

from .constants import (
    AMINO_ACID_SET,
    MAX_LENGTH,
    MIN_LENGTH,
    TOP_SIMILARITY_CEILING,
)


def is_valid_sequence(seq: str) -> bool:
    """Alphabet and length rules. Uniqueness is a library-level property."""
    return (
        MIN_LENGTH <= len(seq) <= MAX_LENGTH
        and not (set(seq) - AMINO_ACID_SET)
    )


def dedupe_valid(candidates: list[str], exclude: frozenset[str]) -> list[str]:
    """Keep compliant, unique, non-reference sequences in first-seen order.

    Order is preserved deterministically so repeated runs with the same inputs
    produce the same library.
    """
    seen: set[str] = set()
    kept: list[str] = []
    for seq in candidates:
        if seq in seen or seq in exclude:
            continue
        if not is_valid_sequence(seq):
            continue
        seen.add(seq)
        kept.append(seq)
    return kept


class ReferenceIndex:
    """Reference AMPs bucketed by length, for fast similarity screening.

    ``Levenshtein.ratio(a, b) = 2 * M / (len(a) + len(b))`` where ``M`` is the
    number of matched characters, and ``M <= min(len(a), len(b))``. So a ratio
    above ``c`` requires ``2 * min / (la + lb) > c``, which bounds how different
    the two lengths can be. That lets us skip most references outright instead
    of running the naive 100 x 39,448 double loop.
    """

    def __init__(self, sequences: list[str]) -> None:
        self.by_length: dict[int, list[str]] = defaultdict(list)
        for seq in sequences:
            self.by_length[len(seq)].append(seq)
        self.lengths = sorted(self.by_length)

    def _candidate_lengths(self, n: int, ceiling: float) -> list[int]:
        # Need 2 * min(n, m) / (n + m) > ceiling for a match to be possible.
        lo = int(n * ceiling / (2.0 - ceiling)) - 1
        hi = int(n * (2.0 - ceiling) / ceiling) + 1
        return [m for m in self.lengths if lo <= m <= hi]

    def max_similarity(self, seq: str, ceiling: float) -> float:
        """Screen `seq` against the reference set.

        Contract: the returned value exceeds `ceiling` if and only if the true
        maximum similarity exceeds it. That is what the gate needs, and it is
        exact for the decision.

        Strictly below the ceiling the value is only a lower bound, because the
        length filter may prune the true argmax among already-safe pairs. Use
        `exact_max_similarity` when the number itself is being reported.
        """
        best = 0.0
        n = len(seq)
        for m in self._candidate_lengths(n, ceiling):
            for ref in self.by_length[m]:
                r = Levenshtein.ratio(seq, ref)
                if r > best:
                    best = r
                    if best > ceiling:
                        return best
        return best

    def exact_max_similarity(self, seq: str) -> float:
        """True maximum similarity against every reference. Slow; for reporting."""
        return max(
            (Levenshtein.ratio(seq, ref) for refs in self.by_length.values() for ref in refs),
            default=0.0,
        )

    def is_novel_enough(self, seq: str, ceiling: float = TOP_SIMILARITY_CEILING) -> bool:
        return self.max_similarity(seq, ceiling) <= ceiling


class NoveltyScreen:
    """Rejects sequences that are near-duplicates of known AMPs.

    Only *exact* reference matches are disqualifying, but the competition also
    screens for "near-exact matches" against AMP repositories, so emitting a
    one-or-two-edit variant of a published peptide costs novelty for no benefit.

    A k-mer index provides the fast path. Two sequences above the ceiling differ
    by few edits and so almost always share a long exact substring; sequences
    sharing none are accepted without the expensive comparison. Sequences that do
    share one go through the exact similarity screen.

    This is a quality filter, not a compliance gate, so the fast path being a
    heuristic is acceptable: a small fraction of near-duplicates can pass. The
    hard rules are enforced separately in `validate_library`.
    """

    def __init__(self, reference: list[str], k: int = 10, ceiling: float = 0.90) -> None:
        self.k = k
        self.ceiling = ceiling
        self.index = ReferenceIndex(reference)
        self.kmers: set[str] = set()
        for seq in reference:
            for i in range(len(seq) - k + 1):
                self.kmers.add(seq[i : i + k])

    def shares_kmer(self, seq: str) -> bool:
        k = self.k
        return any(seq[i : i + k] in self.kmers for i in range(len(seq) - k + 1))

    def is_novel(self, seq: str) -> bool:
        if not self.shares_kmer(seq):
            return True
        return self.index.max_similarity(seq, self.ceiling) <= self.ceiling


def validate_library(
    sequences: list[str],
    reference: frozenset[str],
    expected_size: int,
) -> None:
    """Raise ValueError describing every violation, or return silently."""
    errors: list[str] = []

    if len(sequences) != expected_size:
        errors.append(f"expected {expected_size} sequences, got {len(sequences)}")

    seen: set[str] = set()
    for i, seq in enumerate(sequences, start=1):
        if not seq:
            errors.append(f"record {i}: empty sequence")
            continue
        invalid = sorted(set(seq) - AMINO_ACID_SET)
        if invalid:
            errors.append(f"record {i}: invalid characters {invalid}")
        if not MIN_LENGTH <= len(seq) <= MAX_LENGTH:
            errors.append(f"record {i}: length {len(seq)} outside [{MIN_LENGTH}, {MAX_LENGTH}]")
        if seq in seen:
            errors.append(f"record {i}: duplicate sequence")
        seen.add(seq)

    overlap = seen & reference
    if overlap:
        errors.append(f"{len(overlap)} sequence(s) exactly match the reference database")

    if errors:
        shown = errors[:20]
        more = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise ValueError(
            "library validation failed:\n"
            + "\n".join(f"  - {e}" for e in shown)
            + more
        )


def validate_top(
    top: list[str],
    library: frozenset[str],
    index: ReferenceIndex,
    expected_size: int,
    ceiling: float = TOP_SIMILARITY_CEILING,
) -> None:
    errors: list[str] = []

    if len(top) != expected_size:
        errors.append(f"expected {expected_size} sequences, got {len(top)}")

    seen: set[str] = set()
    for i, seq in enumerate(top, start=1):
        if seq not in library:
            errors.append(f"record {i}: not present in the full library")
        if seq in seen:
            errors.append(f"record {i}: duplicate sequence")
        seen.add(seq)
        sim = index.max_similarity(seq, ceiling)
        if sim > ceiling:
            errors.append(f"record {i}: reference similarity {sim:.3f} exceeds {ceiling}")

    if errors:
        raise ValueError(
            "top-list validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )
