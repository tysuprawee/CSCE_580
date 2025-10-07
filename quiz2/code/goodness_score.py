"""Goodness score evaluation for R3 recipe JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

REQUIRED_FIELDS: List[str] = [
    "recipe_name",
    "data_provenance",
    "macronutrients",
    "ingredients",
    "instructions",
]


def evaluate_recipe_file(path: Path) -> Dict[str, int]:
    """Return a dictionary containing the goodness score for ``path``."""

    score = 0
    result = {"file": str(path), "score": 0, "valid_json": False}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        result["valid_json"] = True
        score += 50
    except (json.JSONDecodeError, OSError):
        result["score"] = 0
        return result

    for field in REQUIRED_FIELDS:
        value = payload.get(field)
        if value:
            score += 10

    result["score"] = score
    return result


def evaluate_paths(paths: List[Path]) -> List[Dict[str, int]]:
    return [evaluate_recipe_file(path) for path in paths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate goodness score for R3 JSON files")
    parser.add_argument("paths", nargs="+", help="JSON files or directories to score")
    return parser.parse_args()


def gather_files(inputs: List[str]) -> List[Path]:
    files: List[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        else:
            files.append(path)
    return files


def main() -> None:
    args = parse_args()
    files = gather_files(args.paths)
    scores = evaluate_paths(files)
    for record in scores:
        status = "valid" if record["valid_json"] else "invalid"
        print(f"{record['file']}: {record['score']} ({status})")


if __name__ == "__main__":
    main()
