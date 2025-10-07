"""Approximate GAICO-style comparison for recipe JSON outputs."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def load_instructions(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    instructions = payload.get("instructions", [])
    collected: List[str] = []
    for instruction in instructions:
        text = instruction.get("original_text", "").strip()
        if text:
            collected.append(text)
        for task in instruction.get("task", []):
            action = task.get("action_name")
            if action:
                collected.append(action.strip())
    return collected


def jaccard_similarity(items_a: Iterable[str], items_b: Iterable[str]) -> float:
    set_a = {item.lower() for item in items_a if item}
    set_b = {item.lower() for item in items_b if item}
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def compare_files(paths: List[Path]) -> List[Tuple[str, str, float]]:
    cache: Dict[Path, List[str]] = {path: load_instructions(path) for path in paths}
    comparisons: List[Tuple[str, str, float]] = []
    for left, right in itertools.combinations(paths, 2):
        score = jaccard_similarity(cache[left], cache[right])
        comparisons.append((left.name, right.name, score))
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare R3 JSON instruction similarity")
    parser.add_argument("paths", nargs="+", help="JSON files to compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = [Path(p) for p in args.paths]
    results = compare_files(files)
    for left, right, score in results:
        print(f"{left} vs {right}: {score:.2f}")


if __name__ == "__main__":
    main()
