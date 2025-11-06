#!/usr/bin/env python3
"""Train classical ML baselines on the cleaned IMDB dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from projectb.classical import ClassicalConfig, train_classical_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/classical_metrics.json"))
    parser.add_argument("--limit-train", type=int, default=5000)
    parser.add_argument("--limit-test", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ClassicalConfig(limit_train=args.limit_train, limit_test=args.limit_test)
    metrics = train_classical_models(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
