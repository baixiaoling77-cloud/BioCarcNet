"""Atom, bond and deterministic 3D features used by BioCarcNet."""

from itertools import combinations_with_replacement

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

ELEMENTS = ["C", "O", "S", "N", "H"]
ELEMENT_COMBINATIONS = [
    pair
    for pair in combinations_with_replacement(ELEMENTS, 2)
    if pair != ("H", "H")
]
ELEMENT_COMBINATIONS.extend(
    [
        ("C", "P"),
        ("C", "F"),
        ("C", "Cl"),
        ("C", "Br"),
        ("C", "I"),
        ("C", "Na"),
        ("C", "K"),
        ("C", "Mg"),
        ("C", "B"),
        ("C", "Se"),
    ]
)
COMBINATION_TO_LABEL = {
    combination: index for index, combination in enumerate(ELEMENT_COMBINATIONS)
}

BOND_FEATURE_DIM = 37
ATOM_FEATURE_DIM = 136
MAX_CONFORMER_ATTEMPTS = 10
MAX_FORCE_FIELD_ITERATIONS = 100
BOND_FEATURES = {
    "bond_type": [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    "element_pair": list(range(len(COMBINATION_TO_LABEL))),
    "in_ring": [True, False],
}
ATOM_FEATURES = {
    "atomic_number": list(range(100)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chirality": [0, 1, 2, 3],
    "hydrogens": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}


def edge_get_bond_features_dim():
    return BOND_FEATURE_DIM


def edge_get_atom_features_dim():
    return ATOM_FEATURE_DIM


def one_hot_unknown(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def _bond_element_label(bond):
    symbols = tuple(
        sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
    )
    return COMBINATION_TO_LABEL.get(symbols, -1)


def _atom_position(mol, atom_index):
    if mol is None or mol.GetNumConformers() == 0:
        return [0.0, 0.0, 0.0]
    position = mol.GetConformer().GetAtomPosition(atom_index)
    return [position.x, position.y, position.z]


def _bond_geometry(mol, bond_index):
    if mol is None or mol.GetNumConformers() == 0:
        return [0.0, 0.0, 0.0], 0.0
    bond = mol.GetBondWithIdx(bond_index)
    begin = np.asarray(_atom_position(mol, bond.GetBeginAtomIdx()))
    end = np.asarray(_atom_position(mol, bond.GetEndAtomIdx()))
    vector = end - begin
    length = float(np.linalg.norm(vector))
    direction = vector / length if length > 0 else np.zeros(3, dtype=float)
    return direction.tolist(), length


def get_bond_feature(mol, bond, use_3d=False):
    feature = (
        one_hot_unknown(bond.GetBondType(), BOND_FEATURES["bond_type"])
        + one_hot_unknown(_bond_element_label(bond), BOND_FEATURES["element_pair"])
        + one_hot_unknown(bond.IsInRing(), BOND_FEATURES["in_ring"])
    )
    if use_3d:
        direction, length = _bond_geometry(mol, bond.GetIdx())
        return feature + direction + [length]
    return feature + [0.0, 0.0, 0.0, 0.0]


def get_atom_feature(mol, atom, use_3d=False):
    feature = (
        one_hot_unknown(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_number"])
        + one_hot_unknown(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + one_hot_unknown(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + one_hot_unknown(int(atom.GetChiralTag()), ATOM_FEATURES["chirality"])
        + one_hot_unknown(int(atom.GetTotalNumHs()), ATOM_FEATURES["hydrogens"])
        + one_hot_unknown(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [int(atom.GetIsAromatic()), atom.GetMass() * 0.01]
    )
    coordinates = _atom_position(mol, atom.GetIdx()) if use_3d else [0.0, 0.0, 0.0]
    return feature + coordinates


def generate_3d_conformer(mol, force_field="MMFF", seed=42):
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(
        mol,
        randomSeed=seed,
        maxAttempts=MAX_CONFORMER_ATTEMPTS,
    )
    if result == -1:
        result = AllChem.EmbedMolecule(
            mol,
            useRandomCoords=True,
            randomSeed=seed,
            maxAttempts=MAX_CONFORMER_ATTEMPTS,
        )
    if result == -1:
        try:
            parameters = AllChem.ETKDG()
            parameters.randomSeed = seed
            parameters.maxIterations = MAX_CONFORMER_ATTEMPTS
            result = AllChem.EmbedMolecule(mol, parameters)
        except Exception:
            result = -1
    if result == -1:
        return None

    try:
        if force_field == "UFF":
            AllChem.UFFOptimizeMolecule(
                mol, maxIters=MAX_FORCE_FIELD_ITERATIONS
            )
        elif force_field == "MMFF":
            AllChem.MMFFOptimizeMolecule(
                mol,
                mmffVariant="MMFF94s",
                maxIters=MAX_FORCE_FIELD_ITERATIONS,
            )
    except Exception:
        # A valid embedded conformer remains useful if optimization fails.
        pass
    return mol


class BondGraphOne:
    def __init__(self, smile, mol, bond_count, use_3d):
        self.smile = smile
        self.bond_num = bond_count
        self.bond_feature = [
            get_bond_feature(mol, bond, use_3d)
            for bond in list(mol.GetBonds())[:bond_count]
        ]


class AtomGraphOne:
    def __init__(self, smile, mol, atom_count, use_3d):
        self.smile = smile
        self.atom_num = atom_count
        self.atom_feature = [
            get_atom_feature(mol, atom, use_3d)
            for atom in list(mol.GetAtoms())[:atom_count]
        ]


class BondGraphBatch:
    def __init__(self, graphs):
        self.bond_index = []
        features = []
        offset = 0
        for graph in graphs:
            features.extend(graph.bond_feature)
            self.bond_index.append((offset, graph.bond_num))
            offset += graph.bond_num
        self.bond_feature = torch.tensor(features, dtype=torch.float32)

    def get_bond_feature(self):
        return self.bond_feature, self.bond_index


class AtomGraphBatch:
    def __init__(self, graphs):
        self.atom_index = []
        features = []
        offset = 0
        for graph in graphs:
            features.extend(graph.atom_feature)
            self.atom_index.append((offset, graph.atom_num))
            offset += graph.atom_num
        self.atom_feature = torch.tensor(features, dtype=torch.float32)

    def get_atom_feature(self):
        return self.atom_feature, self.atom_index


_ATOM_CACHE = {}
_BOND_CACHE = {}


def clear_graph_cache():
    _ATOM_CACHE.clear()
    _BOND_CACHE.clear()


def create_atom_bond_graph(smiles, args, force_field):
    atom_graphs, bond_graphs = [], []
    seed = int(getattr(args, "conformer_seed", 42))

    for smile in smiles:
        cache_key = (smile, force_field, seed)
        if cache_key not in _ATOM_CACHE:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smile}")
            atom_count = mol.GetNumAtoms()
            bond_count = mol.GetNumBonds()
            conformer = generate_3d_conformer(mol, force_field, seed)
            use_3d = conformer is not None
            feature_mol = conformer if use_3d else mol
            _ATOM_CACHE[cache_key] = AtomGraphOne(
                smile, feature_mol, atom_count, use_3d
            )
            _BOND_CACHE[cache_key] = BondGraphOne(
                smile, feature_mol, bond_count, use_3d
            )
        atom_graphs.append(_ATOM_CACHE[cache_key])
        bond_graphs.append(_BOND_CACHE[cache_key])

    return AtomGraphBatch(atom_graphs), BondGraphBatch(bond_graphs)
