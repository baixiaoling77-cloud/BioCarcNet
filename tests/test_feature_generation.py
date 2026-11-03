
import numpy as np
import pytest
import torch
from rdkit import Chem

import biocarcnet.data.molecular_graph as molecular_graph
import biocarcnet.training.engine as engine
from biocarcnet.data.molecular_graph import generate_3d_conformer


def test_fingerprint_failure_is_not_silently_replaced(monkeypatch):
    def fail(_smiles):
        raise ValueError("synthetic fingerprint failure")

    monkeypatch.setattr(engine, "get_RDKitFingerprint", fail)
    with pytest.raises(RuntimeError, match="Failed to generate"):
        engine.extract_fpn_features("CCO")


def test_one_bond_molecule_keeps_its_bond_features():
    bond_feature = torch.arange(37, dtype=torch.float32).reshape(1, 37)
    item = {
        "graph": (
            torch.zeros((2, 136), dtype=torch.float32),
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            bond_feature,
            torch.empty((2, 0), dtype=torch.long),
        ),
        "fpn": torch.zeros(2048, dtype=torch.float32),
        "species": 0,
        "sex": 0,
        "label": [0.0],
        "is_conflict": False,
        "smile": "CC",
    }
    batch = engine.mole_collate_fn([item])
    assert torch.equal(batch["bond_graph"].x, bond_feature)


def test_seeded_3d_fallback_is_reproducible():
    first = generate_3d_conformer(Chem.MolFromSmiles("CCO"), "MMFF", seed=42)
    second = generate_3d_conformer(Chem.MolFromSmiles("CCO"), "MMFF", seed=42)
    first_coordinates = np.asarray(first.GetConformer().GetPositions())
    second_coordinates = np.asarray(second.GetConformer().GetPositions())
    assert np.allclose(first_coordinates, second_coordinates)


def test_conformer_embedding_stages_have_fixed_limits(monkeypatch):
    calls = []

    def fail_embedding(_mol, *args, **kwargs):
        calls.append((args, kwargs))
        return -1

    monkeypatch.setattr(molecular_graph.AllChem, "EmbedMolecule", fail_embedding)
    conformer = generate_3d_conformer(
        Chem.MolFromSmiles("CCO"), "MMFF", seed=42
    )

    assert conformer is None
    assert len(calls) == 3
    assert calls[0][1]["maxAttempts"] == 10
    assert calls[1][1]["maxAttempts"] == 10
    assert calls[1][1]["useRandomCoords"] is True
    assert calls[2][0][0].maxIterations == 10
    assert calls[2][0][0].randomSeed == 42


def test_mmff94s_optimization_has_fixed_iteration_limit(monkeypatch):
    optimization = {}

    monkeypatch.setattr(
        molecular_graph.AllChem,
        "EmbedMolecule",
        lambda _mol, *args, **kwargs: 0,
    )

    def record_optimization(_mol, **kwargs):
        optimization.update(kwargs)
        return 0

    monkeypatch.setattr(
        molecular_graph.AllChem,
        "MMFFOptimizeMolecule",
        record_optimization,
    )
    conformer = generate_3d_conformer(
        Chem.MolFromSmiles("CCO"), "MMFF", seed=42
    )

    assert conformer is not None
    assert optimization == {"mmffVariant": "MMFF94s", "maxIters": 100}

