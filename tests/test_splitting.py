from types import SimpleNamespace

import pandas as pd
import pytest

from biocarcnet.data import MoleData, MoleDataSet
from biocarcnet.training import tag_conflict_samples
from biocarcnet.utils import (
    chemical_level_split,
    record_level_split,
    save_split_manifest,
)


def test_record_values_require_binary_labels_and_conditions():
    args = SimpleNamespace()
    with pytest.raises(ValueError, match="Labels must be binary"):
        MoleData(["CC", "2", "0", "0"], args)
    with pytest.raises(ValueError, match="species and sex must be binary"):
        MoleData(["CC", "1", "0.5", "0"], args)


class QuietLogger:
    def info(self, _message):
        pass

    def debug(self, _message):
        pass


def make_item(chemical, species, sex, label):
    return SimpleNamespace(
        args=SimpleNamespace(),
        smile=chemical,
        canonical_smile=chemical,
        species=species,
        sex=sex,
        label=[float(label)],
    )


def make_balanced_dataset():
    rows = []
    for chemical_index in range(20):
        chemical = f"chemical-{chemical_index:02d}"
        for species in (0, 1):
            for sex in (0, 1):
                label = (chemical_index + species + sex) % 2
                rows.append(make_item(chemical, species, sex, label))
    return MoleDataSet(rows)


def chemical_sets(*datasets):
    return [
        {item.canonical_smile for item in dataset.data}
        for dataset in datasets
    ]


def test_chemical_split_has_no_group_overlap():
    dataset = make_balanced_dataset()
    train, test, validation = chemical_level_split(
        dataset,
        (0.8, 0.1, 0.1),
        seed=1,
        log=QuietLogger(),
    )
    train_groups, test_groups, validation_groups = chemical_sets(
        train, test, validation
    )
    assert not train_groups & validation_groups
    assert not train_groups & test_groups
    assert not validation_groups & test_groups
    assert len(train) + len(validation) + len(test) == len(dataset)


def test_record_split_preserves_all_records():
    dataset = make_balanced_dataset()
    train, test, validation = record_level_split(
        dataset,
        (0.8, 0.1, 0.1),
        seed=1,
        log=QuietLogger(),
    )
    assert len(train) + len(validation) + len(test) == len(dataset)


def test_manifest_uses_manuscript_subset_names_without_random_state(tmp_path):
    dataset = make_balanced_dataset()
    train, test, validation = record_level_split(
        dataset,
        (0.8, 0.1, 0.1),
        seed=1,
        log=QuietLogger(),
    )
    path = tmp_path / "split_manifest.csv"
    save_split_manifest(path, train, test, validation, "record")
    manifest = pd.read_csv(path)

    assert "seed" not in manifest.columns
    assert manifest["split"].drop_duplicates().tolist() == [
        "train",
        "test",
        "validation",
    ]


def test_conflict_status_uses_only_the_supplied_split():
    train = MoleDataSet([make_item("A", 0, 0, 0)])
    held_out = MoleDataSet([make_item("A", 1, 1, 1)])

    tag_conflict_samples(train, "train")
    tag_conflict_samples(held_out, "held-out")

    assert train.data[0].is_conflict is False
    assert held_out.data[0].is_conflict is False


def test_local_conflict_is_detected_within_training_split():
    train = MoleDataSet(
        [
            make_item("A", 0, 0, 0),
            make_item("A", 1, 1, 1),
        ]
    )
    tag_conflict_samples(train, "train")
    assert all(item.is_conflict for item in train.data)

