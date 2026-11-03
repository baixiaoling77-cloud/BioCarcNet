from argparse import ArgumentParser

import torch

from .io import mkdir


def set_train_argument():
    """Parse the public training options for the BioCarcNet release."""
    parser = ArgumentParser(description="Train BioCarcNet.")
    parser.add_argument("--data-path", dest="data_path", type=str)
    parser.add_argument(
        "--save-path", dest="save_path", type=str, default="outputs/checkpoints"
    )
    parser.add_argument(
        "--log-path", dest="log_path", type=str, default="outputs/logs"
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument(
        "--split-type",
        choices=["record", "chemical"],
        default="record",
        help=(
            "record: stratify individual condition records; chemical: keep all "
            "records of each canonical chemical in one partition."
        ),
    )
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--cond-lr", dest="cond_lr", type=float, default=3e-4)
    parser.add_argument(
        "--ranking-loss-weight",
        dest="ranking_loss_weight",
        type=float,
        default=1.0,
        help=(
            "Weight of the explicit within-batch positive-negative ranking "
            "loss (default: 1.0). Use 0 to disable it."
        ),
    )
    parser.add_argument(
        "--force-field",
        dest="force_field",
        choices=["NO", "UFF", "MMFF"],
        default="MMFF",
    )
    args = parser.parse_args()

    if args.ranking_loss_weight < 0:
        parser.error("--ranking-loss-weight must be non-negative.")

    mkdir(args.save_path)
    args.cuda = torch.cuda.is_available()
    args.model_type = "biocarcnet"
    args.split_ratio = (0.8, 0.1, 0.1)
    args.conformer_seed = 42
    args.init_lr = 1e-4
    return args
