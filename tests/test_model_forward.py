from types import SimpleNamespace

import torch

from biocarcnet.data import MoleData
from biocarcnet.training import PreFetchDataset, mole_collate_fn
from biocarcnet.utils import choose_model


def test_model_forward_with_zero_and_one_bond_molecules():
    args = SimpleNamespace(
        cuda=False,
        hidden_size=32,
        dropout=0.0,
        task_num=1,
        conformer_seed=42,
        force_field="MMFF",
    )
    records = [
        MoleData(["C", "0", "0", "0"], args),
        MoleData(["CC", "1", "1", "1"], args),
    ]
    dataset = PreFetchDataset(records, args, desc="Model smoke test")
    batch = mole_collate_fn([dataset[0], dataset[1]])
    model = choose_model(args).eval()

    with torch.no_grad():
        logits, details = model(
            batch["atom_graph"],
            batch["bond_graph"],
            batch["fpn"],
            batch["species"],
            batch["sex"],
        )

    assert logits.shape == (2, 1)
    assert details["selected_conditioned_feat"].shape == (2, 32)
    assert torch.isfinite(logits).all()
