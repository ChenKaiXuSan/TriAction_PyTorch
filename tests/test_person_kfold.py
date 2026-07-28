"""Person-wise K-fold must never place one person on both sides of a split."""

import pytest

from project.cross_validation import DefineCrossValidation
from project.map_config import VideoSample


def _samples(num_persons=10, videos_each=4):
    out = []
    for p in range(num_persons):
        for v in range(videos_each):
            out.append(
                VideoSample(
                    person_id=f"{p:02d}",
                    env_folder=f"env{v}",
                    env_key=f"key{v}",
                    label_path=f"/l/person_{p:02d}_{v}.json",
                    videos={"front": f"/v/{p}/{v}/front.mp4"},
                )
            )
    return out


def test_no_person_appears_in_both_train_and_val():
    samples = _samples()
    for fold in range(5):
        split = DefineCrossValidation.person_kfold_split(samples, 5, fold, seed=42)
        train_p = {s.person_id for s in split["train"]}
        val_p = {s.person_id for s in split["val"]}
        assert val_p, f"fold {fold} has an empty validation set"
        assert not (train_p & val_p), f"fold {fold} leaks persons {train_p & val_p}"


def test_folds_cover_every_person_exactly_once():
    samples = _samples()
    seen = []
    for fold in range(5):
        split = DefineCrossValidation.person_kfold_split(samples, 5, fold, seed=42)
        seen.extend({s.person_id for s in split["val"]})
    assert sorted(seen) == sorted({s.person_id for s in samples})
    assert len(seen) == len(set(seen)), "a person was validated in more than one fold"


def test_folds_are_balanced_within_one_person():
    samples = _samples(num_persons=22)
    sizes = [
        len({s.person_id for s in DefineCrossValidation.person_kfold_split(samples, 5, f, 42)["val"]})
        for f in range(5)
    ]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced folds: {sizes}"


def test_split_is_deterministic_for_a_seed():
    samples = _samples()
    a = DefineCrossValidation.person_kfold_split(samples, 5, 1, seed=7)
    b = DefineCrossValidation.person_kfold_split(samples, 5, 1, seed=7)
    assert [s.person_id for s in a["val"]] == [s.person_id for s in b["val"]]


def test_rejects_more_folds_than_persons():
    with pytest.raises(ValueError, match="exceeds"):
        DefineCrossValidation.person_kfold_split(_samples(num_persons=3), 5, 0, seed=42)


def test_nested_split_keeps_test_disjoint_from_train_and_val():
    samples = _samples(num_persons=12)
    for fold in range(5):
        sp = DefineCrossValidation.person_kfold_split(
            samples, 5, fold, seed=42, nested_val_persons=2
        )
        assert set(sp) == {"train", "val", "test"}
        tr = {s.person_id for s in sp["train"]}
        va = {s.person_id for s in sp["val"]}
        te = {s.person_id for s in sp["test"]}
        assert va, f"fold {fold} inner val is empty"
        assert not (tr & va) and not (tr & te) and not (va & te), (
            f"fold {fold} leaks persons: train&val={tr & va} train&test={tr & te} val&test={va & te}"
        )


def test_nested_test_set_matches_the_non_nested_val_set():
    samples = _samples(num_persons=12)
    plain = DefineCrossValidation.person_kfold_split(samples, 5, 2, seed=42)
    nested = DefineCrossValidation.person_kfold_split(
        samples, 5, 2, seed=42, nested_val_persons=2
    )
    assert {s.person_id for s in plain["val"]} == {s.person_id for s in nested["test"]}


def test_nested_rejects_consuming_every_training_person():
    with pytest.raises(ValueError, match="no training persons"):
        DefineCrossValidation.person_kfold_split(
            _samples(num_persons=6), 5, 0, seed=42, nested_val_persons=99
        )
