"""Guard the development-only refinement protocol and its comparison controls."""

import numpy as np
import pandas as pd
import pytest

from amp_challenge_2027 import challenger_analysis as analysis


def test_analysis_excludes_test_values_before_aggregation():
    splits = pd.DataFrame({"sequence": ["AAAAAAAA", "CCCCCCCC", "WWWWWWWW"],
                           "split": ["train", "calibration", "test"], "family": [0, 1, 2]})
    frame = pd.DataFrame({"sequence": splits.sequence, "bacterium": "E. coli",
                          "strain": "s1", "database": "A", "confirmed": True,
                          "log_mic": [0., 1., 1000.]})
    first = analysis.development_labels(frame, splits, "E. coli")
    frame.loc[2, "log_mic"] = -1000.
    second = analysis.development_labels(frame, splits, "E. coli")
    pd.testing.assert_frame_equal(first, second)
    assert first.sequence.tolist() == ["AAAAAAAA", "CCCCCCCC"]


def test_grouped_tuning_never_splits_a_family(monkeypatch):
    groups = np.repeat(np.arange(10), 3)
    X = np.column_stack([np.arange(30), np.sin(np.arange(30))])
    labels = np.tile([0., 1., 2.], 10)
    expected_train_sets = [set(train) for train, held in analysis.grouped_folds(groups)]
    for train, held in analysis.grouped_folds(groups):
        assert set(groups[train]).isdisjoint(groups[held])
    fit = analysis.classification_model
    seen = []

    def traced(features, target, strength):
        seen.append(set(features[:, 0].astype(int)))
        model = fit(features, target, strength)
        np.testing.assert_allclose(model[0].mean_, features.mean(axis=0))
        return model

    monkeypatch.setattr(analysis, "classification_model", traced)
    model, oof, metadata = analysis.fit_threshold_model(X, labels, groups)
    assert len(seen) == len(analysis.REGULARIZATION) * 5 + 1
    assert all(rows in expected_train_sets for rows in seen[:-1])
    assert seen[-1] == set(range(30))
    assert np.isfinite(oof).all() and np.all((oof >= 0) & (oof <= 1))
    assert metadata["C"] in analysis.REGULARIZATION


@pytest.mark.parametrize("constant", [0, 1])
def test_one_class_fold_returns_correct_probability(constant):
    X = np.arange(8).reshape(-1, 1)
    model = analysis.classification_model(X, np.full(8, constant), 1.)
    np.testing.assert_array_equal(analysis.active_probability(model, X), np.full(8, constant))


def test_auditor_has_representation_matched_source_control():
    p = {"ridge": np.array([0., 1.]), "mlp": np.array([.2, 1.2]),
         "embedding_ridge": np.array([.1, .8]), "omit_source_0": np.array([.4, .9])}
    bootstrap = np.array([[.1, .2], [1.1, 1.]])
    f, names, warnings = analysis.predictor_features(p, bootstrap, np.array([.4, .5]))
    count = f["source_stack"].shape[1]
    np.testing.assert_array_equal(f["threshold_auditor"][:, :count], f["source_stack"])
    assert names["threshold_auditor"][:count] == names["source_stack"]
    assert "omit_source_0" in names["source_stack"]
    assert "distance_to_training" in names["threshold_auditor"]


def test_tail_weights_depend_on_predictions_and_emphasize_low_mic():
    weights = analysis.tail_weights(np.arange(10, dtype=float))
    assert weights.tolist() == [5., 5., 1., 1., 1., 1., 1., 1., 1., 1.]


def test_swaps_count_measured_gains_and_losses():
    report = analysis.swaps(np.array([0., 2., 0., 2.]), np.array([0., 3., 1., 2.]),
                            np.array([0., 1., 2., 3.]), ["AA", "CC", "DD", "EE"], 2)
    assert report == {"gained_candidates": 1, "gained_active": 1,
                      "lost_candidates": 1, "lost_active": 0}
