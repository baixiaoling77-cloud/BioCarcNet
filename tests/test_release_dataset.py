import hashlib
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parents[1] / "data" / "biocarcnet.csv"
EXPECTED_SHA256 = "9bb216394c0272c516581020f14e9df00bcbaf1beb7869191c8cd8879cf6b168"
EXTERNAL_PATH = (
    Path(__file__).parents[1] / "data" / "external_isscan.csv"
)
EXPECTED_EXTERNAL_SHA256 = (
    "850ecc229a08511d7a1f742fa14430845eb54e7873f4417ccbbd8c7b8b0c9c6e"
)


def test_release_dataset_schema_and_size():
    frame = pd.read_csv(DATA_PATH)
    assert list(frame.columns) == ["smiles", "label", "species", "sex"]
    assert len(frame) == 5847
    assert frame["smiles"].nunique() == 2314
    assert not frame.duplicated(["smiles", "species", "sex"]).any()


def test_release_dataset_checksum():
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256


def test_external_dataset_schema_size_and_checksum():
    frame = pd.read_csv(EXTERNAL_PATH)
    modeling_frame = pd.read_csv(DATA_PATH)
    assert list(frame.columns) == [
        "CAS",
        "ChemName",
        "smiles",
        "label",
        "species",
        "sex",
    ]
    assert len(frame) == 119
    assert frame["smiles"].nunique() == 52
    assert not frame.duplicated(["smiles", "species", "sex"]).any()
    assert set(frame["smiles"]).isdisjoint(modeling_frame["smiles"])
    assert frame["label"].isin([0, 1]).all()
    assert frame["species"].isin([0, 1]).all()
    assert frame["sex"].isin([0, 1]).all()
    digest = hashlib.sha256(EXTERNAL_PATH.read_bytes()).hexdigest()
    assert digest == EXPECTED_EXTERNAL_SHA256
