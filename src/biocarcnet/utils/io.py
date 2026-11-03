import copy
import csv
import json
import logging
import os
import random

import torch
from sklearn.model_selection import StratifiedGroupKFold

from biocarcnet.data import MoleData, MoleDataSet


def mkdir(path, isdir=True):
    if not isdir:
        path = os.path.dirname(path)
    if path:
        os.makedirs(path, exist_ok=True)


def set_log(name, save_path):
    log = logging.getLogger(name)
    if log.handlers:
        return log

    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(logging.INFO)
    log.addHandler(stream)

    mkdir(save_path)
    file_handler = logging.FileHandler(os.path.join(save_path, "debug.log"), encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )
    file_handler.setLevel(logging.DEBUG)
    log.addHandler(file_handler)
    return log


def get_header(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle))


def get_task_name(path):
    # Release data schema: SMILES, one or more labels, species, sex.
    return get_header(path)[1:-2]


def load_data(path, args):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        records = [MoleData(row, args) for row in reader if row]

    valid_records = [record for record in records if record.mol is not None]
    print(f"There are {len(valid_records)} valid molecular records in total.")
    return MoleDataSet(valid_records)


def record_level_split(data, size, seed, log):
    """Return training, test, and validation record-level partitions."""
    rng = random.Random(seed)
    strata = {}
    # Normalize input order before the seeded within-stratum shuffle so the
    # split is reproducible even if rows are reordered in the source file.
    ordered_items = sorted(
        data.data,
        key=lambda item: (item.smile, str(item.label), item.sex),
    )
    for item in ordered_items:
        key = (item.species, item.sex, tuple(item.label))
        strata.setdefault(key, []).append(item)

    train_set, test_set, validation_set = [], [], []
    for key, items in strata.items():
        items = list(items)
        rng.shuffle(items)
        count = len(items)
        train_end = int(count * size[0])
        test_end = train_end + int(count * size[1])
        train_set.extend(items[:train_end])
        test_set.extend(items[train_end:test_end])
        validation_set.extend(items[test_end:])
        log.debug(
            f"Stratum {key}: total={count}, train={train_end}, "
            f"test={test_end - train_end}, validation={count - test_end}"
        )

    rng.shuffle(train_set)
    rng.shuffle(test_set)
    rng.shuffle(validation_set)
    log.info(
        f"Stratified split: train={len(train_set)}, "
        f"test={len(test_set)}, validation={len(validation_set)}"
    )
    return (
        MoleDataSet(train_set),
        MoleDataSet(test_set),
        MoleDataSet(validation_set),
    )


def chemical_level_split(data, size, seed, log):
    """Split canonical chemicals while approximately preserving record strata.

    Ten stratified group folds are generated. Eight folds form the training
    set, one the test set, and one the validation set. All records belonging to
    the same canonical isomeric SMILES remain in exactly one partition.
    """
    if tuple(size) != (0.8, 0.1, 0.1):
        raise ValueError(
            "chemical-level splitting currently requires split_ratio=(0.8, 0.1, 0.1)."
        )

    ordered_items = sorted(
        data.data,
        key=lambda item: (
            item.canonical_smile,
            item.species,
            item.sex,
            str(item.label),
        ),
    )
    groups = [item.canonical_smile for item in ordered_items]
    unique_groups = set(groups)
    if None in unique_groups:
        raise ValueError("Chemical-level splitting received an invalid molecule.")
    if len(unique_groups) < 10:
        raise ValueError("Chemical-level splitting requires at least 10 chemicals.")

    strata = [
        f"{item.species}|{item.sex}|"
        + ",".join("NA" if value is None else str(value) for value in item.label)
        for item in ordered_items
    ]
    splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=seed,
    )
    fold_indices = [
        list(test_indices)
        for _, test_indices in splitter.split(ordered_items, strata, groups)
    ]
    test_indices = set(fold_indices[0])
    validation_indices = set(fold_indices[1])
    train_indices = set(range(len(ordered_items))) - validation_indices - test_indices

    train_set = [ordered_items[index] for index in sorted(train_indices)]
    test_set = [ordered_items[index] for index in sorted(test_indices)]
    validation_set = [ordered_items[index] for index in sorted(validation_indices)]

    group_sets = [
        {item.canonical_smile for item in split}
        for split in (train_set, test_set, validation_set)
    ]
    if (
        group_sets[0] & group_sets[1]
        or group_sets[0] & group_sets[2]
        or group_sets[1] & group_sets[2]
    ):
        raise RuntimeError("Chemical-level split produced overlapping chemicals.")

    log.info(
        "Chemical-level split: "
        f"train={len(train_set)} records/{len(group_sets[0])} chemicals, "
        f"test={len(test_set)} records/{len(group_sets[1])} chemicals, "
        f"validation={len(validation_set)} records/{len(group_sets[2])} chemicals"
    )
    return (
        MoleDataSet(train_set),
        MoleDataSet(test_set),
        MoleDataSet(validation_set),
    )


def split_data(data, split_type, size, seed, log):
    if len(size) != 3 or abs(sum(size) - 1.0) > 1e-8:
        raise ValueError("split_ratio must contain three values that sum to 1.")
    normalized = split_type.strip().lower().replace("_", "-")
    if normalized in {"record", "record-level", "stratified"}:
        return record_level_split(data, size, seed, log)
    if normalized in {"chemical", "chemical-level"}:
        return chemical_level_split(data, size, seed, log)
    raise ValueError(
        f"Unsupported split type: {split_type}. Use 'record' or 'chemical'."
    )


def save_split_manifest(path, train_data, test_data, validation_data, split_type):
    """Save the exact record assignment used by one training run."""
    mkdir(path, isdir=False)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "split",
                "split_type",
                "smiles",
                "canonical_smiles",
                "species",
                "sex",
                "labels",
            ]
        )
        for split_name, dataset in (
            ("train", train_data),
            ("test", test_data),
            ("validation", validation_data),
        ):
            for item in dataset.data:
                writer.writerow(
                    [
                        split_name,
                        split_type,
                        item.smile,
                        item.canonical_smile,
                        item.species,
                        item.sex,
                        json.dumps(item.label, ensure_ascii=False),
                    ]
                )


def save_model(path, model, scaler, args):
    public_args = copy.copy(args)
    if hasattr(public_args, "seed"):
        delattr(public_args, "seed")
    state = {
        "args": public_args,
        "state_dict": model.state_dict(),
        "data_scaler": (
            {"means": scaler[0], "stds": scaler[1]} if scaler is not None else None
        ),
    }
    torch.save(state, path)


def choose_model(args):
    from biocarcnet.models.biocarcnet import BioCarcNet

    model_type = getattr(args, "model_type", "biocarcnet")
    if model_type != "biocarcnet":
        raise ValueError(f"Unsupported model type: {model_type}")
    return BioCarcNet(args)
