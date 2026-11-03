"""Command-line prediction with a trained BioCarcNet checkpoint."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from biocarcnet.data import MoleData
from biocarcnet.training import PreFetchDataset, mole_collate_fn
from biocarcnet.utils import choose_model

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".txt", ".smi"}
SPECIES_NAMES = {0: "Rat", 1: "Mouse"}
SEX_NAMES = {0: "Male", 1: "Female"}


def build_model_args(cli_args):
    return argparse.Namespace(
        model_type="biocarcnet",
        batch_size=cli_args.batch_size,
        conformer_seed=42,
        hidden_size=128,
        dropout=0.6,
        force_field="MMFF",
        task_num=1,
        cuda=torch.cuda.is_available() and not cli_args.cpu,
    )


def load_checkpoint_model(args, checkpoint_path):
    device = torch.device("cuda" if args.cuda else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
    for name in ("hidden_size", "dropout", "task_num", "conformer_seed", "force_field"):
        if saved_args is not None and hasattr(saved_args, name):
            setattr(args, name, getattr(saved_args, name))

    model = choose_model(args)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def find_column(frame, candidates):
    normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in frame.columns
    }
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def parse_binary_condition(value, kind):
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    aliases = (
        {"0": 0, "1": 1, "rat": 0, "mouse": 1}
        if kind == "species"
        else {"0": 0, "1": 1, "male": 0, "female": 1}
    )
    if text in aliases:
        return aliases[text]
    try:
        number = int(float(text))
    except ValueError as exc:
        raise ValueError(f"Invalid {kind} value: {value}") from exc
    if number not in (0, 1):
        raise ValueError(f"{kind} must be 0 or 1, got {value}")
    return number


def expand_conditions(record, species=None, sex=None):
    species_values = [species] if species is not None else [0, 1]
    sex_values = [sex] if sex is not None else [0, 1]
    return [
        {**record, "species": species_value, "sex": sex_value}
        for species_value in species_values
        for sex_value in sex_values
    ]


def records_from_frame(frame, source_name):
    smiles_column = find_column(
        frame, {"smiles", "smile", "canonical smiles", "canonical smile"}
    )
    if smiles_column is None:
        raise ValueError(f"No SMILES column found in {source_name}")
    species_column = find_column(frame, {"species"})
    sex_column = find_column(frame, {"sex", "gender"})

    records = []
    for row_number, (_, row) in enumerate(frame.iterrows(), start=1):
        smile = str(row[smiles_column]).strip()
        if not smile or smile.lower() == "nan":
            continue
        species = (
            parse_binary_condition(row[species_column], "species")
            if species_column is not None
            else None
        )
        sex = (
            parse_binary_condition(row[sex_column], "sex")
            if sex_column is not None
            else None
        )
        original_fields = {
            str(column).strip(): None if pd.isna(value) else value
            for column, value in row.items()
        }
        base = {
            **original_fields,
            "smiles": smile,
            "source_file": source_name,
            "source_row": row_number,
        }
        records.extend(expand_conditions(base, species, sex))
    return records


def read_input_file(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix == ".tsv":
        frame = pd.read_csv(path, sep="\t")
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif suffix in {".txt", ".smi"}:
        lines = [
            line.strip().split()[0]
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        frame = pd.DataFrame({"smiles": lines})
    else:
        raise ValueError(f"Unsupported input file: {path}")
    return records_from_frame(frame, path.name)


def load_query_records(input_value, species=None, sex=None):
    path = Path(input_value)
    if path.exists() and path.is_dir():
        files = sorted(
            file
            for file in path.rglob("*")
            if file.is_file() and file.suffix.lower() in SUPPORTED_SUFFIXES
        )
        if not files:
            raise FileNotFoundError(f"No supported files found in {path}")
        records = []
        for file in files:
            records.extend(read_input_file(file))
        return records
    if path.exists() and path.is_file():
        return read_input_file(path)

    base = {"smiles": input_value, "source_file": "command_line", "source_row": 1}
    return expand_conditions(base, species, sex)


def build_query_dataset(records, args):
    valid_items = []
    valid_records = []
    invalid_rows = []
    for record in records:
        try:
            item = MoleData(
                [record["smiles"], "", str(record["species"]), str(record["sex"])],
                args,
            )
            if item.mol is None:
                raise ValueError("RDKit could not parse the SMILES")
            valid_items.append(item)
            valid_records.append(record)
        except Exception as exc:
            invalid_rows.append({**record, "status": "invalid", "error": str(exc)})

    dataset = PreFetchDataset(valid_items, args, desc="Query preprocessing")
    kept_records = []
    kept_cache = []
    for record, cached in zip(valid_records, dataset.data_cache):
        if cached["graph"][0] is None:
            invalid_rows.append(
                {
                    **record,
                    "status": "invalid",
                    "error": "Failed to generate molecular features",
                }
            )
        else:
            kept_records.append(record)
            kept_cache.append(cached)
    dataset.data_cache = kept_cache
    return dataset, kept_records, invalid_rows


def predict_records(dataset, records, model, device, args):
    if not records:
        return []
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mole_collate_fn,
        num_workers=0,
    )
    output_rows = []
    record_offset = 0
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch_size = len(batch["smiles"])
            batch_records = records[record_offset : record_offset + batch_size]
            record_offset += batch_size
            logits, info = model(
                batch["atom_graph"].to(device),
                batch["bond_graph"].to(device),
                batch["fpn"].to(device),
                batch["species"].to(device),
                batch["sex"].to(device),
            )
            probabilities = torch.sigmoid(logits).cpu().numpy()[:, 0]
            branch_weights = info["branch_weights"].cpu().numpy()
            for index, record in enumerate(batch_records):
                probability = float(probabilities[index])
                output_rows.append(
                    {
                        **record,
                        "species_name": SPECIES_NAMES[record["species"]],
                        "sex_name": SEX_NAMES[record["sex"]],
                        "carcinogenic_probability": probability,
                        "prediction": int(probability >= 0.5),
                        "branch_atom": float(branch_weights[index, 0]),
                        "branch_bond": float(branch_weights[index, 1]),
                        "branch_fingerprint": float(branch_weights[index, 2]),
                        "status": "ok",
                        "error": "",
                    }
                )
    return output_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Predict carcinogenicity with BioCarcNet.")
    parser.add_argument(
        "input",
        help="A SMILES string, a CSV/TSV/Excel/SMI file, or an input folder.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--species", help="For a single SMILES: 0/rat or 1/mouse.")
    parser.add_argument("--sex", help="For a single SMILES: 0/male or 1/female.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    if not cli_args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {cli_args.checkpoint}")
    args = build_model_args(cli_args)
    model, device = load_checkpoint_model(args, cli_args.checkpoint)
    species = (
        parse_binary_condition(cli_args.species, "species")
        if cli_args.species is not None
        else None
    )
    sex = (
        parse_binary_condition(cli_args.sex, "sex")
        if cli_args.sex is not None
        else None
    )

    records = load_query_records(cli_args.input, species, sex)
    dataset, valid_records, invalid_rows = build_query_dataset(records, args)
    output_rows = predict_records(dataset, valid_records, model, device, args)
    output_rows.extend(invalid_rows)

    cli_args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(cli_args.output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(output_rows)} prediction rows to {cli_args.output}")


if __name__ == "__main__":
    main()
