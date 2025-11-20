#!/usr/bin/env python3
"""CLI for running the custom PyTorch fine-tuning loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the Python path so `src.projectb` imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.projectb.manual_training import ManualTrainingConfig, run_manual_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/manual_metrics.json"),
        help="Where to write the manual training metrics JSON.",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=2000,
        help="Maximum number of training examples to use (None for full train split).",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=1000,
        help="Maximum number of test examples to use (None for full test split).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of epochs for the manual training loop.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-device batch size for training and evaluation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device string (e.g. 'cuda', 'mps', or 'cpu'). "
             "If not set, picks CUDA, then MPS, then CPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ManualTrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        device=args.device,
    )

    metrics = run_manual_training(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))

    # Also echo to stdout so you can see a summary in the terminal
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
