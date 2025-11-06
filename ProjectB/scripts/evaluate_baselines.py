"""Command-line entry point to evaluate base DistilBERT and GPT classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from projectb import BaselineConfig, evaluate_base_distilbert, evaluate_gpt_classifier


def _write_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-base", type=Path, default=Path("artifacts/base_metrics.json"))
    parser.add_argument("--output-gpt", type=Path, default=Path("artifacts/gpt_metrics.json"))
    parser.add_argument("--skip-base", action="store_true", help="Skip DistilBERT base evaluation")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT baseline evaluation")
    parser.add_argument("--limit-train", type=int, default=2000)
    parser.add_argument("--limit-test", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None, help="Force computation device (cpu/cuda)")
    parser.add_argument("--distilbert-model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--gpt-model", type=str, default="openai-community/gpt2")
    args = parser.parse_args()

    config = BaselineConfig(
        distilbert_model_name=args.distilbert_model,
        gpt_model_name=args.gpt_model,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
        device=args.device,
    )

    if not args.skip_base:
        base_results = evaluate_base_distilbert(config)
        _write_json(base_results, args.output_base)

    if not args.skip_gpt:
        gpt_results = evaluate_gpt_classifier(config)
        _write_json(gpt_results, args.output_gpt)


if __name__ == "__main__":  # pragma: no cover
    main()
