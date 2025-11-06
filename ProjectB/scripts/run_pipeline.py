"""Utility script to execute the entire Project B workflow end-to-end.

This helper orchestrates the classical baseline, both DistilBERT fine-tuning
loops, the baseline transformer/GPT evaluations, and finally regenerates the
SVG artefacts.  It is intended for the final rerun on the full IMDB dataset
before packaging your submission.

Example
-------
```bash
python scripts/run_pipeline.py --epochs 3 --train-limit 25000 --test-limit 25000 \
    --batch-size 16 --plots-dir artifacts/plots
```

The script assumes you have already installed all requirements listed in
``requirements.txt``.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from typing import Iterable, List


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
ARTIFACTS_DIR = ROOT / "artifacts"


def run_step(command: Iterable[str], *, cwd: pathlib.Path | None = None) -> None:
    """Execute a command and stream its output."""

    cmd_list: List[str] = list(command)
    print(f"\n[projectb] Running: {' '.join(cmd_list)}")
    subprocess.run(cmd_list, check=True, cwd=cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs for both fine-tuning runs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the manual PyTorch loop.",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of training examples to load. "
            "Omit to use the entire training split."
        ),
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of evaluation examples to load. "
            "Omit for the entire test split."
        ),
    )
    parser.add_argument(
        "--plots-dir",
        type=pathlib.Path,
        default=ARTIFACTS_DIR / "plots",
        help="Directory where generated SVG plots should be stored.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip running the base DistilBERT and GPT-2 baseline evaluations.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip regenerating SVG plots at the end of the pipeline.",
    )
    return parser.parse_args()


def maybe_add_limit(flag: str, value: int | None) -> List[str]:
    if value is None:
        return []
    return [flag, str(value)]


def main() -> None:
    args = parse_args()

    python_bin = sys.executable

    classical_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "train_classical.py"),
        *maybe_add_limit("--limit-train", args.train_limit),
        *maybe_add_limit("--limit-test", args.test_limit),
    ]
    run_step(classical_cmd, cwd=ROOT)

    trainer_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "train_with_trainer.py"),
        "--epochs",
        str(args.epochs),
        *maybe_add_limit("--limit-train", args.train_limit),
        *maybe_add_limit("--limit-test", args.test_limit),
    ]
    run_step(trainer_cmd, cwd=ROOT)

    manual_cmd = [
        python_bin,
        str(SCRIPTS_DIR / "train_manual.py"),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        *maybe_add_limit("--limit-train", args.train_limit),
        *maybe_add_limit("--limit-test", args.test_limit),
    ]
    run_step(manual_cmd, cwd=ROOT)

    if not args.skip_baselines:
        baselines_cmd = [
            python_bin,
            str(SCRIPTS_DIR / "evaluate_baselines.py"),
            *maybe_add_limit("--limit-train", args.train_limit),
            *maybe_add_limit("--limit-test", args.test_limit),
        ]
        run_step(baselines_cmd, cwd=ROOT)

    if args.skip_plots:
        return

    metrics = {
        "--trainer": ARTIFACTS_DIR / "trainer_metrics.json",
        "--manual": ARTIFACTS_DIR / "manual_metrics.json",
        "--classical": ARTIFACTS_DIR / "classical_metrics.json",
        "--base": ARTIFACTS_DIR / "base_metrics.json",
        "--gpt": ARTIFACTS_DIR / "gpt_metrics.json",
    }

    plot_cmd: List[str] = [
        python_bin,
        str(SCRIPTS_DIR / "create_svg_plots.py"),
        "--output-dir",
        str(args.plots_dir),
    ]

    for flag, path in metrics.items():
        if path.exists():
            plot_cmd.extend([flag, str(path)])

    run_step(plot_cmd, cwd=ROOT)


if __name__ == "__main__":
    main()
