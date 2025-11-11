"""Run MCAgent on the standard missionary and cannibal scenarios."""
from __future__ import annotations

import time
from typing import Iterable, Tuple

from mc_agent import MCAgent

TestCase = Tuple[int, int]


def execute_tests(test_cases: Iterable[TestCase]) -> None:
    agent = MCAgent()

    for missionaries, cannibals in test_cases:
        start_time = time.perf_counter()
        solution = agent.solve(missionaries, cannibals)
        duration = time.perf_counter() - start_time

        print(
            f"Case {missionaries}M/{cannibals}C -> crossings: {len(solution)}, "
            f"moves: {solution}, time: {duration:.6f}s",
        )


if __name__ == "__main__":
    CASES: Tuple[TestCase, ...] = (
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 3),
        (5, 3),
        (2, 3),
    )
    execute_tests(CASES)
