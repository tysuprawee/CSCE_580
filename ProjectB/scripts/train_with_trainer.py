#!/usr/bin/env python3
"""CLI for running the 🤗 Trainer workflow on a manageable IMDB subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from projectb.trainer_workflow import TrainerConfig, run_trainer_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/trainer_metrics.json"), help="Where to write evaluation metrics.")
    parser.add_argument("--limit-train", type=int, default=2000, help="Limit the number of training examples (for faster experimentation).")
    parser.add_argument("--limit-test", type=int, default=1000, help="Limit the number of test examples.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--train-batch-size",
        "--train_batch_size",
        dest="train_batch_size",
        type=int,
        default=None,
        help="Per-device batch size used during training.",
    )
    parser.add_argument(
        "--eval-batch-size",
        "--eval_batch_size",
        dest="eval_batch_size",
        type=int,
        default=None,
        help="Per-device batch size used during evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="(deprecated) Sets both the train and eval batch size.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="imdb",
        help="Dataset builder or repo ID passed to datasets.load_dataset().",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional path to local IMDB parquet files (e.g., hf:// downloads).",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help="Cache directory forwarded to datasets.load_dataset().",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size_arg = args.batch_size
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    if train_batch_size is None:
        train_batch_size = batch_size_arg if batch_size_arg is not None else 16
    if eval_batch_size is None:
        eval_batch_size = batch_size_arg if batch_size_arg is not None else 32

    config = TrainerConfig(
        num_epochs=args.epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        output_dir=str(args.output.parent / "trainer_checkpoints"),
        dataset_name=args.dataset_name,
        dataset_path=str(args.dataset_path) if args.dataset_path else None,
        dataset_cache_dir=str(args.dataset_cache_dir)
        if args.dataset_cache_dir
        else None,
    )
    metrics = run_trainer_pipeline(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
