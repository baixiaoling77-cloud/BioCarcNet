import os
import time

# Default to GPU 0 locally while respecting CUDA_VISIBLE_DEVICES supplied by
# an external launcher.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from biocarcnet.training import training
from biocarcnet.utils import set_log, set_train_argument

DATA_PATH = "data/biocarcnet.csv"


def main() -> None:
    args = set_train_argument()

    # BioCarcNet experiment configuration.
    if args.data_path is None:
        args.data_path = DATA_PATH

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"biocarcnet_{timestamp}"
    args.save_path = os.path.join(args.save_path, run_name)
    args.log_path = os.path.join(args.log_path, run_name)

    log = set_log("", args.log_path)
    log.info("BioCarcNet training started")
    log.info(f"Data: {args.data_path}")
    log.info(f"Split: {args.split_type} {args.split_ratio}")

    start_time = time.time()
    training(args, log)
    elapsed_minutes = (time.time() - start_time) / 60
    log.info(f"Training finished in {elapsed_minutes:.2f} minutes")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
