"""End-to-end tests against the shipped checkpoint.

These run the real sampler at small scale. They are the regression guard on the
two properties the submission depends on: every sequence is compliant by
construction, and output depends on the seed alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from amp_challenge_2027.compliance import NoveltyScreen, is_valid_sequence
from amp_challenge_2027.constants import (
    AMINO_ACID_SET,
    AMINO_ACIDS,
    CATEGORIES,
    MAX_LENGTH,
    MIN_LENGTH,
)
from amp_challenge_2027.fasta import read_sequences, write_fasta
from amp_challenge_2027.generate import resolve_category
from amp_challenge_2027.sampler import PeptideSampler
from amp_challenge_2027.scoring import Scorer

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "checkpoint" / "generator.npz"
RANKER = ROOT / "checkpoint" / "ranker.npz"

requires_checkpoint = pytest.mark.skipif(
    not GENERATOR.exists() or not RANKER.exists(),
    reason="trained checkpoints not present; run training/ first",
)


@pytest.fixture(scope="module")
def sampler() -> PeptideSampler:
    return PeptideSampler(GENERATOR)


@pytest.fixture(scope="module")
def scorer() -> Scorer:
    return Scorer(RANKER)


def test_entry_point_names_map_to_categories():
    assert resolve_category("generate") == "broad_spectrum"
    for category in CATEGORIES:
        assert resolve_category(f"generate_{category}") == category
    with pytest.raises(SystemExit):
        resolve_category("generate_nonsense")


@requires_checkpoint
def test_samples_are_compliant_by_construction(sampler):
    out = sampler.sample(256, "broad_spectrum", np.random.default_rng(0), batch_size=256)
    assert len(out) == 256
    for seq in out:
        assert MIN_LENGTH <= len(seq) <= MAX_LENGTH, seq
        assert set(seq) <= AMINO_ACID_SET, seq
        assert is_valid_sequence(seq)


@requires_checkpoint
def test_same_seed_gives_identical_output(sampler):
    a = sampler.sample(128, "mdr", np.random.default_rng(3), batch_size=128)
    b = sampler.sample(128, "mdr", np.random.default_rng(3), batch_size=128)
    assert a == b


@requires_checkpoint
def test_different_seed_gives_different_output(sampler):
    a = sampler.sample(128, "mdr", np.random.default_rng(3), batch_size=128)
    c = sampler.sample(128, "mdr", np.random.default_rng(4), batch_size=128)
    assert a != c


@requires_checkpoint
def test_categories_produce_different_distributions(sampler):
    """The control token must actually change the conditional distribution."""
    a = set(sampler.sample(512, "gram_neg", np.random.default_rng(5), batch_size=512))
    b = set(sampler.sample(512, "gram_pos", np.random.default_rng(5), batch_size=512))
    overlap = len(a & b) / max(len(a), 1)
    assert overlap < 0.5, f"categories collapsed to the same distribution ({overlap:.2f})"


@requires_checkpoint
def test_scorer_ranks_known_potent_above_random(scorer):
    """Sanity check on the direction of the potency model."""
    potent = ["GIGKFLHSAKKFGKAFVGEIMNS", "KRWWKWWRR", "FLPIIAKVLSGLL"]
    rng = np.random.default_rng(0)
    # Iterate the ordered alphabet, not the frozenset: set ordering varies with
    # PYTHONHASHSEED, which would make this test non-deterministic across runs.
    random_peptides = [
        "".join(rng.choice(list(AMINO_ACIDS), size=20)) for _ in range(40)
    ]
    potent_mic = scorer.predict_log_mic(scorer.features(potent))
    random_mic = scorer.predict_log_mic(scorer.features(random_peptides))
    # Lower log MIC means more potent.
    assert potent_mic.mean() < random_mic.mean()


@requires_checkpoint
def test_amp_likeness_separates_real_from_shuffled(scorer):
    reference = read_sequences(ROOT / "data" / "antibacterial.fasta")[:300]
    rng = np.random.default_rng(1)
    shuffled = ["".join(rng.permutation(np.array(list(s)))) for s in reference]
    real = scorer.predict_amp_likeness(scorer.features(reference))
    fake = scorer.predict_amp_likeness(scorer.features(shuffled))
    assert real.mean() > fake.mean() + 0.15


@requires_checkpoint
def test_envelope_excludes_extreme_charge(scorer):
    """A 20-lysine peptide is far outside the measured-potent envelope."""
    X = scorer.features(["K" * 20, "GIGKFLHSAKKFGKAFVGEIMNS"])
    mask = scorer.in_envelope(X)
    assert not mask[0]
    assert mask[1]


def test_novelty_screen_rejects_near_duplicates():
    reference = ["GIGKFLHSAKKFGKAFVGEIMNS", "FLPIIAKVLSGLL"]
    screen = NoveltyScreen(reference, k=10, ceiling=0.90)
    # Identical sequence: similarity 1.0.
    assert not screen.is_novel("GIGKFLHSAKKFGKAFVGEIMNS")
    # Unrelated sequence shares no 10-mer.
    assert screen.is_novel("WWWWWWWWWWWWWWWW")


def test_fasta_roundtrip_is_byte_stable(tmp_path):
    sequences = ["GIGKFLHSAKKFGKAFVGEIMNS", "KRWWKWWRR"]
    a, b = tmp_path / "a.fasta", tmp_path / "b.fasta"
    write_fasta(sequences, a)
    write_fasta(sequences, b)
    assert a.read_bytes() == b.read_bytes()
    assert read_sequences(a) == sequences


def test_stratified_selection_returns_rank_order():
    """Regression: round-robin selects per stratum, but the file must be ranked.

    An earlier version returned the round-robin order, which is not score order,
    so `top.fasta` was not a ranked list as the competition requires.
    """
    from amp_challenge_2027.compliance import ReferenceIndex
    from amp_challenge_2027.generate import select_top, stratum_labels

    rng = np.random.default_rng(0)
    sequences = [
        "".join(rng.choice(list(AMINO_ACIDS), size=int(rng.integers(8, 51))))
        for _ in range(600)
    ]
    sequences = list(dict.fromkeys(sequences))
    scores = rng.random(len(sequences))
    features = np.column_stack(
        [
            np.array([len(s) for s in sequences], dtype=float),
            rng.random(len(sequences)),
            rng.random(len(sequences)),
        ]
    )
    chosen = select_top(
        sequences,
        scores,
        ReferenceIndex([]),
        k=40,
        max_cysteines=None,
        strata_features=features,
        strata_bins=3,
        strata_pool=300,
        verbose=False,
    )
    assert len(chosen) == 40
    assert len(set(chosen)) == 40
    lookup = {s: sc for s, sc in zip(sequences, scores)}
    picked = [lookup[s] for s in chosen]
    assert all(a >= b - 1e-12 for a, b in zip(picked, picked[1:])), "not in rank order"


def test_stratification_spreads_more_than_greedy():
    """Stratified selection must cover more strata than score-greedy selection."""
    from amp_challenge_2027.compliance import ReferenceIndex
    from amp_challenge_2027.generate import select_top, stratum_labels

    rng = np.random.default_rng(1)
    sequences = list(
        dict.fromkeys(
            "".join(rng.choice(list(AMINO_ACIDS), size=int(rng.integers(8, 51))))
            for _ in range(800)
        )
    )
    n = len(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=float)
    # Make score correlate with length so greedy selection concentrates in one bin.
    scores = lengths / lengths.max() + 0.01 * rng.random(n)
    features = np.column_stack([lengths, rng.random(n), rng.random(n)])
    strata = stratum_labels(features, n_bins=3)
    index = ReferenceIndex([])
    lookup = {s: i for i, s in enumerate(sequences)}

    greedy = select_top(sequences, scores, index, k=60, max_cysteines=None, verbose=False)
    strat = select_top(
        sequences,
        scores,
        index,
        k=60,
        max_cysteines=None,
        strata_features=features,
        strata_bins=3,
        strata_pool=400,
        verbose=False,
    )
    # Coverage is measured against library-wide strata, which is what the
    # aggregation score's distributional terms compare against.
    greedy_strata = {int(strata[lookup[s]]) for s in greedy}
    strat_strata = {int(strata[lookup[s]]) for s in strat}
    assert len(strat_strata) > len(greedy_strata)


def test_stratified_selection_fills_k_from_a_tiny_pool():
    """A tiny strata pool must still yield a full ranked list, not raise.

    Note this asserts the outcome, not the top-up code path specifically: the pool
    is `max(strata_pool, k)`, so it always holds at least k candidates, and the
    top-up only runs when admissibility filtering rejects enough of them. That is
    hard to force with an empty reference set.
    """
    from amp_challenge_2027.compliance import ReferenceIndex
    from amp_challenge_2027.generate import select_top, stratum_labels

    rng = np.random.default_rng(2)
    sequences = list(
        dict.fromkeys(
            "".join(rng.choice(list(AMINO_ACIDS), size=int(rng.integers(8, 51))))
            for _ in range(400)
        )
    )
    n = len(sequences)
    scores = rng.random(n)
    features = np.column_stack(
        [np.array([len(s) for s in sequences], dtype=float), rng.random(n), rng.random(n)]
    )
    # Pool smaller than k forces the top-up path.
    chosen = select_top(
        sequences,
        scores,
        ReferenceIndex([]),
        k=50,
        max_cysteines=None,
        strata_features=features,
        strata_bins=2,
        strata_pool=10,
        verbose=False,
    )
    assert len(chosen) == 50
    assert len(set(chosen)) == 50
    lookup = {s: sc for s, sc in zip(sequences, scores)}
    picked = [lookup[s] for s in chosen]
    assert all(a >= b - 1e-12 for a, b in zip(picked, picked[1:]))


def test_cysteine_gate_excludes_multi_cysteine_candidates():
    """Top-100 candidates must be single molecules, not disulfide isomer mixtures.

    Two or more cysteines in a linear, unmodified, free-terminus peptide with no
    controlled oxidation step gives an undefined mixture; an odd count leaves a
    free thiol. Either way the assay result is uninterpretable.
    """
    from amp_challenge_2027.compliance import ReferenceIndex
    from amp_challenge_2027.generate import select_top

    rng = np.random.default_rng(11)
    clean = ["".join(rng.choice(list("AGVILFPYMTSHNQWRKDE"), size=20)) for _ in range(60)]
    # Cysteine-rich candidates given deliberately better scores.
    rich = ["C" + "".join(rng.choice(list("AGVILFKR"), size=18)) + "C" for _ in range(60)]
    sequences = list(dict.fromkeys(rich + clean))
    scores = np.array([2.0 if s.count("C") >= 2 else 1.0 for s in sequences])

    chosen = select_top(sequences, scores, ReferenceIndex([]), k=30, verbose=False)
    assert len(chosen) == 30
    assert all(s.count("C") == 0 for s in chosen), "cysteine-containing candidate selected"

    # The intermediate policy is still available and excludes only multi-cysteine.
    single = select_top(
        sequences, scores, ReferenceIndex([]), k=30, max_cysteines=1, verbose=False
    )
    assert all(s.count("C") <= 1 for s in single)

    # Disabling the gate lets the high-scoring cysteine-rich peptides back in.
    ungated = select_top(
        sequences, scores, ReferenceIndex([]), k=30, max_cysteines=None, verbose=False
    )
    assert any(s.count("C") >= 2 for s in ungated)


def test_assayed_prefix_holds_the_stratified_set():
    """The first `assayed_k` ranks are the operative set and must carry the spread.

    The competition draws its 25 tested peptides from the top 50, so diversity
    optimised across all 100 leaves half the benefit in ranks that are never
    assayed.
    """
    from amp_challenge_2027.compliance import ReferenceIndex
    from amp_challenge_2027.generate import select_top, stratum_labels

    rng = np.random.default_rng(5)
    sequences = list(
        dict.fromkeys(
            "".join(rng.choice(list("AGVILFPYMTSHNQWRKDE"), size=int(rng.integers(10, 40))))
            for _ in range(1200)
        )
    )
    n = len(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=float)
    # Score correlates with length, so score-greedy selection concentrates.
    scores = lengths / lengths.max() + 0.01 * rng.random(n)
    features = np.column_stack([lengths, rng.random(n), rng.random(n)])
    strata = stratum_labels(features, n_bins=3)
    lookup = {s: i for i, s in enumerate(sequences)}
    index = ReferenceIndex([])

    chosen = select_top(
        sequences, scores, index, k=100, assayed_k=50,
        strata_features=features, strata_bins=3, strata_pool=600, verbose=False,
    )
    assert len(chosen) == 100
    assert len(set(chosen)) == 100

    head_strata = {int(strata[lookup[s]]) for s in chosen[:50]}
    # Same budget spread over all 100 instead: the assayed prefix covers less.
    spread_over_100 = select_top(
        sequences, scores, index, k=100, assayed_k=None,
        strata_features=features, strata_bins=3, strata_pool=600, verbose=False,
    )
    baseline_head = {int(strata[lookup[s]]) for s in spread_over_100[:50]}
    assert len(head_strata) >= len(baseline_head), (
        f"assayed prefix covers {len(head_strata)} strata vs {len(baseline_head)} baseline"
    )

    # Each block must still be internally ranked best-first.
    sc = [scores[lookup[s]] for s in chosen]
    assert all(a >= b - 1e-12 for a, b in zip(sc[:50], sc[1:50]))
    assert all(a >= b - 1e-12 for a, b in zip(sc[50:], sc[51:]))
