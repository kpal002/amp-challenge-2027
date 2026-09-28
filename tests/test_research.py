"""Research validity checks: transitive splits, provenance, and label isolation."""

import json
from pathlib import Path

import Levenshtein
import numpy as np
import pandas as pd
import pytest

from amp_challenge_2027.research.data import (
    aggregate_labels, assign_splits, connected_families, load_measurements, parse_flag,
)
from amp_challenge_2027.research.embeddings import atomic_npz, read_vectors
from amp_challenge_2027.research.experiment import choose_penalty, sequence_order
from amp_challenge_2027.research.models import bootstrap_family_indices, ridge


def test_transitive_families_prevent_bridge_leakage():
    # A~B and B~C, but A !~ C. Representative-only clustering misses this.
    sequences = ["AAAAAAAAAA", "AAAAAAACCC", "AAAACCCCCC", "WWWWWWWWWW"]
    groups = connected_families(sequences, 0.7)
    assert Levenshtein.ratio(sequences[0], sequences[2]) < 0.7
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_no_threshold_neighbors_cross_partitions():
    rng = np.random.default_rng(19)
    sequences = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 20)) for _ in range(40)]
    sequences += [s[:-1] + "A" for s in sequences[:10]]
    groups = connected_families(sequences, 0.7)
    partitions = assign_splits(groups)
    assert np.array_equal(partitions, assign_splits(groups))
    assert len(set(partitions)) == 4
    for i, a in enumerate(sequences):
        for j, b in enumerate(sequences[:i]):
            if Levenshtein.ratio(a, b) >= 0.7:
                assert partitions[i] == partitions[j]


def test_provenance_unknown_is_not_unmodified(tmp_path):
    records = []
    for source, confirmed, value, bacterium in [
        ("A", True, 2., "E. coli"), ("B", False, -1., "E. coli"),
        ("C", True, 3., "S. aureus"),
    ]:
        records.append(dict(sequence="ACDEFGHIK", bacterium=bacterium, strain="s1",
                            database=source, value=value, unit="uM", is_modified="False",
                            has_cterminal_amidation="False", has_unusual_modification="False",
                            datasource_has_modifications=str(confirmed)))
    path = tmp_path / "measurements.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    frame, audit = load_measurements(path)
    assert audit["unknown_modification_rows"] == 1
    assert aggregate_labels(frame, "E. coli").iloc[0] == 2.
    assert aggregate_labels(frame, "E. coli", strict=False).iloc[0] == .5
    assert aggregate_labels(frame, "S. aureus").iloc[0] == 3.
    assert aggregate_labels(frame, "E. coli", omit_source="A").empty


def test_duplicate_database_records_do_not_reweight_strains():
    frame = pd.DataFrame([
        dict(sequence="ACDEFGHIK", bacterium="E. coli", strain="s1", log_mic=0., confirmed=True, database="A"),
        dict(sequence="ACDEFGHIK", bacterium="E. coli", strain="s1", log_mic=0., confirmed=True, database="B"),
        dict(sequence="ACDEFGHIK", bacterium="E. coli", strain="s2", log_mic=2., confirmed=True, database="A"),
    ])
    assert aggregate_labels(frame, "E. coli").iloc[0] == 1.


def test_unknown_boolean_fails_closed():
    assert parse_flag(pd.Series(["False", "True"])).tolist() == [False, True]
    with pytest.raises(ValueError):
        parse_flag(pd.Series(["False", None]))


def test_scaler_sees_training_only():
    model = ridge().fit(np.array([[0.], [2.], [4.]]), [0., 1., 2.])
    model.predict(np.array([[10000.]]))
    assert model[0].mean_[0] == 2.


def test_embedding_cache_refuses_reordered_sequences(tmp_path):
    path = tmp_path / "vectors.npz"
    atomic_npz(path, sequences=np.array(["AAAA", "CCCC"]), vectors=np.ones((2, 5)))
    assert read_vectors(path, ["AAAA", "CCCC"]).shape == (2, 5)
    with pytest.raises(ValueError, match="order mismatch"):
        read_vectors(path, ["CCCC", "AAAA"])


def test_family_bootstrap_resamples_whole_families():
    groups = np.array([0, 0, 0, 1, 1, 2])
    indices = bootstrap_family_indices(groups, np.random.default_rng(0))
    counts = np.bincount(indices, minlength=len(groups))
    assert counts[0] == counts[1] == counts[2]
    assert counts[3] == counts[4]


def test_selection_ties_are_order_independent():
    assert sequence_order(np.array([1., 1.]), ["CC", "AA"], 1).tolist() == [1]
    assert sequence_order(np.array([1., 1.]), ["AA", "CC"], 1).tolist() == [0]


def test_penalty_tuning_rejects_harmful_warning():
    # Warning penalizes the truly active candidate. Zero penalty must be allowed.
    value = choose_penalty(np.array([0., 1.]), np.array([4., 0.]),
                           np.array([0., 2.]), ["AA", "CC"], 1)
    assert value == 0.


def test_development_cannot_see_test_labels_and_final_does_not_refit(tmp_path, monkeypatch):
    """Poison every test label; development policies/predictions must stay identical."""
    from amp_challenge_2027.research import experiment
    from amp_challenge_2027.research.data import atomic_json, file_digest

    # Exercise the entire orchestration with cheap regressors, not trained fixtures.
    monkeypatch.setattr(experiment, "mlp", lambda *args: ridge())
    monkeypatch.setattr(experiment, "paired_family_interval", lambda *args: [0., 0.])
    rng = np.random.default_rng(57)
    sequences = ["".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), 20)) for _ in range(300)]
    splits = pd.DataFrame({"sequence": sequences, "family": np.arange(300),
                           "split": ["train"] * 120 + ["calibration"] * 60
                           + ["validation"] * 60 + ["test"] * 60})
    values = rng.normal(1, .7, 300)
    outputs = []
    for name, poison in (("ordinary", 0), ("poisoned", 100)):
        work = tmp_path / name
        work.mkdir()
        changed = values.copy()
        changed[240:] += poison
        frame = pd.DataFrame({"sequence": sequences, "bacterium": "E. coli", "strain": "s1",
                              "database": "A", "confirmed": True, "log_mic": changed})
        frame.to_csv(work / "measurements.csv", index=False)
        splits.to_csv(work / "splits.csv", index=False)
        atomic_json(work / "manifest.json", {
            "files": {p: file_digest(work / p) for p in ("measurements.csv", "splits.csv")}})
        report = experiment.run(work, "E. coli", k=5, members=2, iterations=2)
        output = work / "results/descriptors/e_coli"
        assert not (output / "test_report.json").exists()
        assert "test" not in report["counts"]
        outputs.append((report, pd.read_csv(output / "validation_predictions.csv")))
    assert outputs[0][0]["penalties"] == outputs[1][0]["penalties"]
    assert outputs[0][0]["metrics"] == outputs[1][0]["metrics"]
    pd.testing.assert_frame_equal(outputs[0][1], outputs[1][1])

    output = tmp_path / "ordinary/results/descriptors/e_coli"
    before = {p.name: file_digest(p) for p in (output / "models").glob("*.joblib")}
    final = experiment.run(tmp_path / "ordinary", "E. coli", k=5, members=2, iterations=2, final=True)
    assert final["stage"] == "test"
    assert final["penalties"] == outputs[0][0]["penalties"]
    after = {p.name: file_digest(p) for p in (output / "models").glob("*.joblib")}
    assert before == after
    with pytest.raises(ValueError, match="configuration changed"):
        experiment.run(tmp_path / "ordinary", "E. coli", k=6, members=2, iterations=2, final=True)

    # Embedding runs must include controls with the same representation access.
    embedding_path = tmp_path / "embeddings.npz"
    atomic_npz(embedding_path, sequences=np.array(sequences), vectors=rng.normal(size=(300, 8)))
    esm_report = experiment.run(tmp_path / "ordinary", "E. coli", embeddings=embedding_path,
                                k=5, members=2, iterations=2)
    assert "embedding_only" in esm_report["metrics"]
    assert "representation_matched_uncertainty" in esm_report["metrics"]
    assert esm_report["uncertainty_comparator"] == "representation_matched_uncertainty"
    esm_final = experiment.run(tmp_path / "ordinary", "E. coli", embeddings=embedding_path,
                              k=5, members=2, iterations=2, final=True)
    assert esm_final["penalties"] == esm_report["penalties"]
