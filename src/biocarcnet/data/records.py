import random

import torch
from rdkit import Chem
from torch.utils.data import Dataset


class MoleData:
    """One condition-specific molecular carcinogenicity record."""

    def __init__(self, line, args):
        if len(line) < 4:
            raise ValueError("Each row must contain SMILES, label, species and sex.")

        self.args = args
        self.smile = line[0].strip()
        self.mol = Chem.MolFromSmiles(self.smile)
        self.canonical_smile = (
            Chem.MolToSmiles(self.mol, canonical=True, isomericSmiles=True)
            if self.mol is not None
            else None
        )
        self.label = [
            float(value) if str(value).strip() != "" else None
            for value in line[1:-2]
        ]
        if any(value not in (None, 0.0, 1.0) for value in self.label):
            raise ValueError(
                f"Labels must be binary or empty, got {self.label} for {self.smile}"
            )

        species = float(line[-2])
        sex = float(line[-1])
        if species not in (0.0, 1.0) or sex not in (0.0, 1.0):
            raise ValueError(
                f"species and sex must be binary, got species={species}, "
                f"sex={sex} for {self.smile}"
            )
        self.species = int(species)
        self.sex = int(sex)

    def task_num(self):
        return len(self.label)

    def change_label(self, label):
        self.label = label


class MoleDataSet(Dataset):
    def __init__(self, data):
        self.data = data
        self.args = self.data[0].args if self.data else None
        self.scaler = None

    def smile(self):
        return [item.smile for item in self.data]

    def mol(self):
        return [item.mol for item in self.data]

    def label(self):
        return [item.label for item in self.data]

    def task_num(self):
        return self.data[0].task_num() if self.data else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def random_data(self, seed):
        random.Random(seed).shuffle(self.data)

    def change_label(self, labels):
        if len(self.data) != len(labels):
            raise ValueError("The number of labels must equal the dataset size.")
        for item, label in zip(self.data, labels):
            item.change_label(label)

    def get_conditions(self):
        if not self.data:
            return None, None
        species = torch.tensor([item.species for item in self.data], dtype=torch.long)
        sex = torch.tensor([item.sex for item in self.data], dtype=torch.long)
        return species, sex
