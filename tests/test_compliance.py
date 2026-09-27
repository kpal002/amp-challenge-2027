"""Tests for the compliance gate.

The gate is the only thing standing between a good library and disqualification,
so its failure modes are tested explicitly rather than assumed.
"""

from __future__ import annotations

import random

import Levenshtein
import pytest

from amp_challenge_2027.compliance import (
    ReferenceIndex,
    dedupe_valid,
    is_valid_sequence,
    validate_library,
    validate_top,
)
from amp_challenge_2027.constants import MAX_LENGTH, MIN_LENGTH

VALID = "GIGKFLHSAKKFGKAFVGEIMNS"


def test_length_bounds():
    assert is_valid_sequence("A" * MIN_LENGTH)
    assert is_valid_sequence("A" * MAX_LENGTH)
    assert not is_valid_sequence("A" * (MIN_LENGTH - 1))
    assert not is_valid_sequence("A" * (MAX_LENGTH + 1))


@pytest.mark.parametrize("bad", ["GIGKFLHSXKKFGK", "GIGKFLHS-KKFGK", "gigkflhsakkfgk", "GIGKFLHSBKKFGK"])
def test_rejects_non_canonical(bad):
    assert not is_valid_sequence(bad)


def test_dedupe_preserves_first_seen_order():
    candidates = ["KRWWKWWRR", "GIGKFLHSAKKFG", "KRWWKWWRR", "FLPIIAKVLSGLL"]
    kept = dedupe_valid(candidates, exclude=frozenset())
    assert kept == ["KRWWKWWRR", "GIGKFLHSAKKFG", "FLPIIAKVLSGLL"]


def test_dedupe_applies_exclusion_set():
    kept = dedupe_valid(["KRWWKWWRR", VALID], exclude=frozenset({VALID}))
    assert kept == ["KRWWKWWRR"]


def test_dedupe_drops_invalid():
    kept = dedupe_valid(["SHORT", "A" * 60, "XXXXXXXXXX", VALID], exclude=frozenset())
    assert kept == [VALID]


def test_validate_library_catches_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        validate_library([VALID, VALID], frozenset(), expected_size=2)


def test_validate_library_catches_wrong_size():
    with pytest.raises(ValueError, match="expected 5 sequences"):
        validate_library([VALID], frozenset(), expected_size=5)


def test_validate_library_catches_reference_overlap():
    with pytest.raises(ValueError, match="exactly match"):
        validate_library([VALID], frozenset({VALID}), expected_size=1)


def test_validate_top_requires_membership_in_library():
    index = ReferenceIndex([])
    with pytest.raises(ValueError, match="not present in the full library"):
        validate_top([VALID], frozenset({"KRWWKWWRR"}), index, expected_size=1)


def test_validate_top_rejects_too_similar_to_reference():
    index = ReferenceIndex([VALID])
    # One substitution away from a reference sequence: far above any sane ceiling.
    near = VALID[:-1] + ("A" if VALID[-1] != "A" else "C")
    with pytest.raises(ValueError, match="similarity"):
        validate_top([near], frozenset({near}), index, expected_size=1)


def test_similarity_screen_decision_matches_brute_force():
    """The length-bucket prune must never hide a true ceiling violation.

    ratio = 2*LCS/(len_a+len_b) and LCS <= min(len_a, len_b), so pruning on the
    length bound is sound. This checks the implementation of that argument.
    """
    rng = random.Random(7)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    refs = ["".join(rng.choice(alphabet) for _ in range(rng.randint(8, 50))) for _ in range(400)]
    index = ReferenceIndex(refs)

    probes = []
    for _ in range(80):
        probes.append("".join(rng.choice(alphabet) for _ in range(rng.randint(8, 50))))
    for _ in range(80):
        # Mutated references, which should frequently trip the ceiling.
        s = list(rng.choice(refs))
        for _ in range(max(1, len(s) // 8)):
            s[rng.randrange(len(s))] = rng.choice(alphabet)
        probes.append("".join(s))
    probes.extend(rng.choice(refs) for _ in range(10))

    ceiling = 0.75
    tripped = 0
    for probe in probes:
        brute = max(Levenshtein.ratio(probe, r) for r in refs)
        screened = index.max_similarity(probe, ceiling)
        assert (screened > ceiling) == (brute > ceiling), probe
        tripped += brute > ceiling
    # Guard against a vacuous test where nothing ever exceeds the ceiling.
    assert tripped > 0


def test_exact_max_similarity_is_exact():
    refs = ["KRWWKWWRR", VALID, "FLPIIAKVLSGLL"]
    index = ReferenceIndex(refs)
    probe = "KRWWKWWRK"
    assert index.exact_max_similarity(probe) == pytest.approx(
        max(Levenshtein.ratio(probe, r) for r in refs)
    )
