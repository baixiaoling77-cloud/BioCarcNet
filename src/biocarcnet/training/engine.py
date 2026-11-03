import csv
import os
import random
import secrets
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from biocarcnet.data import create_atom_bond_graph, edge_get_bond_features_dim
from biocarcnet.data.fingerprints import get_RDKitFingerprint
from biocarcnet.utils.io import (
    choose_model,
    get_task_name,
    load_data,
    mkdir,
    save_model,
    save_split_manifest,
    split_data,
)


def tag_conflict_samples(dataset, name="dataset"):
    """Mark label conflicts using only records contained in ``dataset``.

    This function must be called after data splitting. Computing conflict
    status on the complete dataset before splitting would expose held-out
    labels to the training loss through ``conflict_mask``.
    """
    smile_to_labels = defaultdict(set)
    for item in dataset.data:
        label = tuple(-1 if value is None else value for value in item.label)
        smile_to_labels[item.smile].add(label)

    conflict_smiles = {
        smile for smile, labels in smile_to_labels.items() if len(labels) > 1
    }
    for item in dataset.data:
        item.is_conflict = item.smile in conflict_smiles

    conflict_rows = sum(item.is_conflict for item in dataset.data)
    print(
        f"{name}: found {len(conflict_smiles)} conflict SMILES "
        f"covering {conflict_rows}/{len(dataset)} records."
    )
    return dataset


def extract_fpn_features(smile):
    try:
        fingerprint = get_RDKitFingerprint([smile])[0].astype(np.float32)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate the RDKit fingerprint for SMILES: {smile}"
        ) from exc
    if fingerprint.shape != (2048,):
        raise ValueError(
            f"Expected a 2048-dimensional fingerprint for {smile}, "
            f"got shape {fingerprint.shape}."
        )
    if not np.isfinite(fingerprint).all():
        raise ValueError(f"Fingerprint contains non-finite values for {smile}.")
    return fingerprint


def extract_graph_data(smile, args):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None, None, None, None

    try:
        atom_graph, bond_graph = create_atom_bond_graph(
            [smile], args, args.force_field
        )
    except Exception:
        return None, None, None, None

    atom_features, atom_indices = atom_graph.get_atom_feature()
    bond_features, bond_indices = bond_graph.get_bond_feature()

    atom_edges = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        atom_edges.extend(((begin, end), (end, begin)))
    atom_edges = (
        torch.tensor(atom_edges, dtype=torch.long).t().contiguous()
        if atom_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    # The bond branch treats bonds as nodes in a fully connected bond graph.
    bond_count = mol.GetNumBonds()
    bond_edges = []
    for left in range(bond_count):
        for right in range(left + 1, bond_count):
            bond_edges.extend(((left, right), (right, left)))
    bond_edges = (
        torch.tensor(bond_edges, dtype=torch.long).t().contiguous()
        if bond_edges
        else torch.empty((2, 0), dtype=torch.long)
    )

    atom_start, atom_size = atom_indices[0]
    bond_start, bond_size = bond_indices[0]
    one_atom = atom_features[atom_start : atom_start + atom_size]
    one_bond = (
        bond_features[bond_start : bond_start + bond_size]
        if bond_size
        else None
    )
    return one_atom, atom_edges, one_bond, bond_edges


class PreFetchDataset(Dataset):
    """Precompute deterministic graph, 3D and fingerprint features."""

    def __init__(
        self,
        raw_dataset,
        args,
        desc="Pre-processing",
    ):
        self.args = args
        self.data_cache = []
        for item in tqdm(raw_dataset, desc=desc):
            self.data_cache.append(
                {
                    "graph": extract_graph_data(item.smile, args),
                    "fpn": torch.tensor(
                        extract_fpn_features(item.smile),
                        dtype=torch.float32,
                    ),
                    "label": item.label,
                    "species": item.species,
                    "sex": item.sex,
                    "smile": item.smile,
                    "is_conflict": getattr(item, "is_conflict", False),
                }
            )

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, index):
        return self.data_cache[index]


def filter_invalid_prefetched_rows(dataset, log=None, name="dataset"):
    before = len(dataset.data_cache)
    dataset.data_cache = [
        item for item in dataset.data_cache if item["graph"][0] is not None
    ]
    dropped = before - len(dataset.data_cache)
    if dropped and log is not None:
        log.info(f"{name}: dropped {dropped} records with invalid graph features.")
    return dropped


def prefetched_labels(dataset):
    return [item["label"] for item in dataset.data_cache]


def prefetched_conflict_mask(dataset):
    return [item["is_conflict"] for item in dataset.data_cache]


def mole_collate_fn(batch):
    batch = [item for item in batch if item["graph"][0] is not None]
    if not batch:
        return None

    bond_dim = next(
        (
            item["graph"][2].shape[1]
            for item in batch
            if item["graph"][2] is not None
        ),
        edge_get_bond_features_dim(),
    )
    atom_graphs = []
    bond_graphs = []
    for item in batch:
        atom_x, atom_edges, bond_x, bond_edges = item["graph"]
        atom_graphs.append(Data(x=atom_x.float(), edge_index=atom_edges.long()))
        # A molecule with no bonds needs a dummy node. A one-bond molecule has
        # no bond-to-bond edges but still retains its real bond-node features;
        # GCNConv can process the node through its default self-loop.
        if bond_x is None:
            bond_x = torch.zeros((1, bond_dim), dtype=torch.float32)
            bond_edges = torch.empty((2, 0), dtype=torch.long)
        bond_graphs.append(Data(x=bond_x.float(), edge_index=bond_edges.long()))

    return {
        "atom_graph": Batch.from_data_list(atom_graphs),
        "bond_graph": Batch.from_data_list(bond_graphs),
        "fpn": torch.stack([item["fpn"] for item in batch]),
        "species": torch.tensor(
            [item["species"] for item in batch], dtype=torch.long
        ),
        "sex": torch.tensor([item["sex"] for item in batch], dtype=torch.long),
        "labels": [item["label"] for item in batch],
        "conflict_mask": torch.tensor(
            [item["is_conflict"] for item in batch], dtype=torch.float32
        ),
        "smiles": [item["smile"] for item in batch],
    }


def _to_device(batch, device):
    for key in ("atom_graph", "bond_graph", "fpn", "species", "sex", "conflict_mask"):
        batch[key] = batch[key].to(device)
    return batch


def _pairwise_ranking_loss(logits, target, mask):
    """Rank positive records above negative records within each task.

    Only valid positive-negative pairs from the current mini-batch are used.
    The loss is zero for a task (or batch) that has no eligible pair.
    """
    task_losses = []
    for task_index in range(logits.shape[1]):
        valid = mask[:, task_index].bool()
        positive = logits[valid & (target[:, task_index] > 0.5), task_index]
        negative = logits[valid & (target[:, task_index] < 0.5), task_index]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        margins = positive[:, None] - negative[None, :]
        task_losses.append(F.softplus(-margins).mean())

    if not task_losses:
        return logits.sum() * 0.0
    return torch.stack(task_losses).mean()


def _masked_training_loss(
    logits,
    target,
    mask,
    conflict_mask,
    smiles,
    ranking_weight=0.0,
):
    smoothed_target = target * 0.9 + 0.05
    bce = F.binary_cross_entropy_with_logits(
        logits, smoothed_target, reduction="none"
    )
    focal_weight = (1.0 - torch.exp(-bce)).pow(2.0)
    row_weight = 1.0 + conflict_mask.unsqueeze(1).expand_as(target)
    main_loss = (bce * focal_weight * row_weight * mask).sum() / (
        mask.sum() + 1e-8
    )

    contrast_loss = torch.zeros((), device=logits.device)
    unique_smiles, inverse = np.unique(np.asarray(smiles), return_inverse=True)
    if len(unique_smiles) < len(smiles):
        inverse = torch.tensor(inverse, device=logits.device)
        same_molecule = (inverse[:, None] == inverse[None, :]).float()
        same_molecule.fill_diagonal_(0)
        same_molecule = same_molecule.unsqueeze(-1)
        label_distance = torch.abs(target[:, None, :] - target[None, :, :])
        pair_mask = (
            same_molecule
            * label_distance
            * mask[:, None, :]
            * mask[None, :, :]
        )
        if pair_mask.sum() > 0:
            probabilities = torch.sigmoid(logits)
            prediction_distance = torch.abs(
                probabilities[:, None, :] - probabilities[None, :, :]
            )
            pair_loss = (1.0 - prediction_distance).pow(2.0)
            contrast_loss = (pair_loss * pair_mask).sum() / pair_mask.sum()

    ranking_loss = _pairwise_ranking_loss(logits, target, mask)
    return main_loss + 1.5 * contrast_loss + ranking_weight * ranking_loss


def epoch_train(model, train_loader, optimizer, args):
    model.train()
    device = next(model.parameters()).device
    all_labels, all_predictions = [], []

    progress = tqdm(train_loader, desc="Training")
    for batch in progress:
        if batch is None:
            continue
        batch = _to_device(batch, device)
        labels = batch["labels"]
        target = torch.tensor(
            [[0 if value is None else value for value in row] for row in labels],
            dtype=torch.float32,
            device=device,
        )
        mask = torch.tensor(
            [[value is not None for value in row] for row in labels],
            dtype=torch.float32,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(
            batch["atom_graph"],
            batch["bond_graph"],
            batch["fpn"],
            batch["species"],
            batch["sex"],
        )
        loss = _masked_training_loss(
            logits,
            target,
            mask,
            batch["conflict_mask"],
            batch["smiles"],
            ranking_weight=args.ranking_loss_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        all_labels.extend(labels)
        all_predictions.extend(torch.sigmoid(logits).detach().cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return all_predictions, all_labels


def predict(model, loader, args, return_attention=False):
    model.eval()
    device = next(model.parameters()).device
    predictions, species_values, sex_values = [], [], []
    attention_data = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if batch is None:
                continue
            species_values.extend(batch["species"].tolist())
            sex_values.extend(batch["sex"].tolist())
            batch = _to_device(batch, device)
            logits, info = model(
                batch["atom_graph"],
                batch["bond_graph"],
                batch["fpn"],
                batch["species"],
                batch["sex"],
            )
            probabilities = torch.sigmoid(logits).cpu()
            predictions.extend(probabilities.tolist())

            if return_attention:
                attention_data.append(
                    {
                        "smiles": list(batch["smiles"]),
                        "atom_attention": info["atom_self_attn"].cpu(),
                        "bond_importance": info["bond_importance"].cpu(),
                        "branch_weights": info["branch_weights"].cpu(),
                    }
                )

    if return_attention:
        return predictions, species_values, sex_values, attention_data
    return predictions, species_values, sex_values


def compute_score(predictions, labels, log, conflict_mask=None):
    y_true = np.asarray([row[0] for row in labels], dtype=float)
    y_prob = np.asarray([row[0] for row in predictions], dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError("Prediction and label counts do not match.")

    auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_prob >= 0.5)

    if conflict_mask is not None:
        conflict_mask = np.asarray(conflict_mask, dtype=bool)
        if conflict_mask.any():
            conflict_accuracy = accuracy_score(
                y_true[conflict_mask], y_prob[conflict_mask] >= 0.5
            )
            log.info(f"Conflict-subset ACC: {conflict_accuracy:.4f}")
    return [auc, accuracy]


def calculate_binary_metrics(predictions, labels):
    """Return the binary metrics reported in the manuscript."""
    y_true = np.asarray(labels, dtype=float)
    y_prob = np.asarray(predictions, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, _fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = np.unique(y_true).size == 2
    specificity_denominator = tn + fp
    return {
        "n": int(y_true.size),
        "RA": accuracy_score(y_true, y_pred),
        "RBA": balanced_accuracy_score(y_true, y_pred),
        "RP": precision_score(y_true, y_pred, zero_division=0),
        "RTP": recall_score(y_true, y_pred, zero_division=0),
        "RTN": tn / specificity_denominator if specificity_denominator else np.nan,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "AROC": roc_auc_score(y_true, y_prob) if has_both_classes else np.nan,
        "APR": average_precision_score(y_true, y_prob)
        if np.any(y_true == 1)
        else np.nan,
    }


def analyze_subgroups(predictions, labels, species, sex, log):
    y_true = np.asarray([row[0] for row in labels], dtype=float)
    y_prob = np.asarray([row[0] for row in predictions], dtype=float)
    species = np.asarray(species)
    sex = np.asarray(sex)
    groups = {
        "Male Rat": (0, 0),
        "Female Rat": (0, 1),
        "Male Mouse": (1, 0),
        "Female Mouse": (1, 1),
    }
    rows = []
    for name, (species_id, sex_id) in groups.items():
        selected = (species == species_id) & (sex == sex_id)
        if not selected.any():
            continue
        metrics = calculate_binary_metrics(y_prob[selected], y_true[selected])
        rows.append({"endpoint": name, **metrics})
        log.info(
            f"{name} (n={metrics['n']}): AROC={metrics['AROC']:.4f}, "
            f"RA={metrics['RA']:.4f}, APR={metrics['APR']:.4f}"
        )
    return rows


def save_validation_outputs(path, predictions, labels, species, sex, metric_rows):
    """Save final validation predictions and endpoint-specific metrics."""
    prediction_path = os.path.join(path, "validation_predictions.csv")
    with open(prediction_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["probability", "label", "species", "sex"])
        writer.writerows(
            zip(
                (row[0] for row in predictions),
                (row[0] for row in labels),
                species,
                sex,
            )
        )

    metric_path = os.path.join(path, "validation_metrics.csv")
    fieldnames = [
        "endpoint",
        "n",
        "RA",
        "RBA",
        "RP",
        "RTP",
        "RTN",
        "F1",
        "AROC",
        "APR",
    ]
    with open(metric_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


def _make_prefetched_loader(
    raw_data,
    args,
    name,
    log,
    sort_by_smiles=False,
):
    dataset = PreFetchDataset(
        raw_data.data,
        args,
        desc=name,
    )
    filter_invalid_prefetched_rows(dataset, log, name)
    if sort_by_smiles:
        dataset.data_cache.sort(key=lambda item: item["smile"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=mole_collate_fn,
        num_workers=0,
    )
    return dataset, loader


def fold_train(args, log):
    args.task_names = get_task_name(args.data_path)
    data = load_data(args.data_path, args)
    args.task_num = data.task_num()
    train_data, test_data, validation_data = split_data(
        data, args.split_type, args.split_ratio, args.seed, log
    )
    tag_conflict_samples(train_data, "Train set")
    tag_conflict_samples(test_data, "Test set")
    tag_conflict_samples(validation_data, "Validation set")
    save_split_manifest(
        os.path.join(args.save_path, "split_manifest.csv"),
        train_data,
        test_data,
        validation_data,
        args.split_type,
    )

    train_dataset, train_loader = _make_prefetched_loader(
        train_data,
        args,
        "Train set",
        log,
        sort_by_smiles=True,
    )
    test_dataset, test_loader = _make_prefetched_loader(
        test_data,
        args,
        "Test set",
        log,
    )
    validation_dataset, validation_loader = _make_prefetched_loader(
        validation_data,
        args,
        "Validation set",
        log,
    )

    model = choose_model(args)
    base_params, condition_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = condition_params if name.startswith("bio_predictor") else base_params
        target.append(parameter)
    optimizer = AdamW(
        [
            {"params": base_params, "lr": args.init_lr},
            {"params": condition_params, "lr": args.cond_lr},
        ],
        weight_decay=1e-2,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    log.info(
        "Optimizer: %d Cond Params (lr=%g) vs %d Base Params (lr=%g); "
        "CosineAnnealingLR(T_max=%d, eta_min=1e-6)",
        len(condition_params),
        args.cond_lr,
        len(base_params),
        args.init_lr,
        args.epochs,
    )
    log.info(
        "Loss: focal BCE + 1.5 * same-molecule separation + %g * "
        "positive-negative ranking",
        args.ranking_loss_weight,
    )

    best_auc, best_epoch = -float("inf"), -1
    model_path = os.path.join(args.save_path, "model.pt")
    for epoch in range(args.epochs):
        log.info(f"Epoch {epoch + 1}/{args.epochs}")
        train_predictions, train_labels = epoch_train(
            model, train_loader, optimizer, args
        )
        scheduler.step()
        log.info(
            "Learning rates after epoch %d: %s",
            epoch + 1,
            ", ".join(f"{lr:.8g}" for lr in scheduler.get_last_lr()),
        )
        test_predictions, _, _ = predict(model, test_loader, args)
        test_score = compute_score(
            test_predictions,
            prefetched_labels(test_dataset),
            log,
            prefetched_conflict_mask(test_dataset),
        )
        train_score = compute_score(train_predictions, train_labels, log)
        log.info(f"Train AUC={train_score[0]:.4f}, ACC={train_score[1]:.4f}")
        log.info(f"Test AUC={test_score[0]:.4f}, ACC={test_score[1]:.4f}")

        if test_score[0] > best_auc:
            best_auc = test_score[0]
            best_epoch = epoch + 1
            save_model(model_path, model, None, args)

    log.info(f"Best test AUC={best_auc:.4f} at epoch {best_epoch}")
    checkpoint = torch.load(
        model_path, map_location=next(model.parameters()).device
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    validation_predictions, validation_species, validation_sex = predict(
        model, validation_loader, args
    )
    validation_labels = prefetched_labels(validation_dataset)
    validation_score = compute_score(
        validation_predictions,
        validation_labels,
        log,
        prefetched_conflict_mask(validation_dataset),
    )
    log.info(
        f"Validation AUC={validation_score[0]:.4f}, "
        f"ACC={validation_score[1]:.4f}"
    )
    metric_rows = analyze_subgroups(
        validation_predictions,
        validation_labels,
        validation_species,
        validation_sex,
        log,
    )
    save_validation_outputs(
        args.save_path,
        validation_predictions,
        validation_labels,
        validation_species,
        validation_sex,
        metric_rows,
    )
    return validation_score


def _seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def training(args, log):
    args.seed = secrets.randbits(32)
    _seed_everything(args.seed)
    mkdir(args.save_path)
    score = np.asarray(fold_train(args, log), dtype=float)
    log.info(f"Final validation: AUC={score[0]:.4f}, ACC={score[1]:.4f}")
    return score
