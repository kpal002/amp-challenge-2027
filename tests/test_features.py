"""Tests for the physicochemical features.

Values are checked against published chemistry for well-characterised peptides,
not against a previous run of this code, so a regression cannot be "confirmed"
by its own output.
"""

from __future__ import annotations

import numpy as np
import pytest

from amp_challenge_2027.features import (
    FEATURE_NAMES,
    N_FEATURES,
    featurize,
    featurize_many,
    hydrophobic_moment,
    isoelectric_point,
    net_charge,
)

MAGAININ_2 = "GIGKFLHSAKKFGKAFVGEIMNS"
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def test_magainin_charge_and_pi_match_literature():
    # Magainin 2 carries a net charge near +3 to +4 at physiological pH with free
    # termini, and is strongly basic.
    assert 2.5 < net_charge(MAGAININ_2) < 4.5
    assert 9.5 < isoelectric_point(MAGAININ_2) < 11.5


def test_charge_extremes():
    assert net_charge("K" * 20) > 18.0
    assert net_charge("E" * 20) < -18.0
    assert isoelectric_point("K" * 20) > 10.0
    assert isoelectric_point("E" * 20) < 4.5


def test_net_charge_is_monotonic_in_ph():
    charges = [net_charge(MAGAININ_2, ph) for ph in (2.0, 5.0, 7.4, 10.0, 13.0)]
    assert all(a > b for a, b in zip(charges, charges[1:]))


def test_hydrophobic_moment_is_order_sensitive():
    """An idealised amphipathic helix must score above its own shuffle.

    Composition is identical, so any difference is structural. This is the
    property that lets the AMP-likeness model beat a composition baseline.
    """
    amphipathic = "KLLKLLLKLLKLLLKLLK"
    clustered = "KKKKKLLLLLLLLLLLLL"
    assert hydrophobic_moment(amphipathic) > hydrophobic_moment(clustered)


def test_composition_invariant_under_shuffle_but_order_features_change():
    rng = np.random.default_rng(0)
    shuffled = "".join(rng.permutation(np.array(list(MAGAININ_2))))
    a, b = featurize(MAGAININ_2), featurize(shuffled)

    comp = slice(INDEX["comp_A"], INDEX["comp_A"] + 20)
    assert np.allclose(a[comp], b[comp]), "composition should be shuffle-invariant"
    assert a[INDEX["hydrophobic_moment"]] != b[INDEX["hydrophobic_moment"]]


def test_composition_sums_to_one():
    f = featurize(MAGAININ_2)
    comp = f[INDEX["comp_A"] : INDEX["comp_A"] + 20]
    assert comp.sum() == pytest.approx(1.0)


def test_dipeptide_block_sums_to_one():
    f = featurize(MAGAININ_2)
    dip = f[INDEX["dip_0_0"] :]
    assert dip.sum() == pytest.approx(1.0)


def test_shape_and_finiteness():
    X = featurize_many([MAGAININ_2, "KRWWKWWRR", "A" * 8, "W" * 50])
    assert X.shape == (4, N_FEATURES)
    assert np.isfinite(X).all()


def test_empty_input_returns_empty_matrix():
    assert featurize_many([]).shape == (0, N_FEATURES)


def test_max_runs():
    f = featurize("KKKAAAAAAKK")
    assert f[INDEX["max_hydrophobic_run"]] == 6.0  # the AAAAAA stretch
    assert f[INDEX["max_cationic_run"]] == 3.0     # leading KKK
