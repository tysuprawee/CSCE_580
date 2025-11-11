"""Missionaries and Cannibals solver using breadth-first search (BFS)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

Move = Tuple[int, int]
StateTuple = Tuple[int, int, int]


@dataclass(frozen=True)
class State:
    """Immutable representation of a problem configuration.

    Attributes
    ----------
    left_missionaries: int
        Number of missionaries on the left bank.
    left_cannibals: int
        Number of cannibals on the left bank.
    boat_left: int
        1 if the boat is on the left bank, 0 otherwise.
    """

    left_missionaries: int
    left_cannibals: int
    boat_left: int

    def as_tuple(self) -> StateTuple:
        return (self.left_missionaries, self.left_cannibals, self.boat_left)


class MCAgent:
    """Agent that solves the missionaries and cannibals problem via BFS."""

    POSSIBLE_MOVES: Tuple[Move, ...] = (
        (2, 0),
        (0, 2),
        (1, 1),
        (1, 0),
        (0, 1),
    )

    def solve(self, initial_missionaries: int, initial_cannibals: int) -> List[Move]:
        """Compute the sequence of boat crossings to solve the puzzle.

        Parameters
        ----------
        initial_missionaries: int
            Starting number of missionaries on the left bank.
        initial_cannibals: int
            Starting number of cannibals on the left bank.

        Returns
        -------
        List[Move]
            Ordered list of moves (missionaries moved, cannibals moved).
            Returns an empty list if no solution exists.
        """

        totals = (initial_missionaries, initial_cannibals)
        start = State(initial_missionaries, initial_cannibals, 1)
        goal = State(0, 0, 0)

        if not self._is_valid_state(start, totals):
            return []

        frontier: Deque[State] = deque([start])
        came_from: Dict[StateTuple, Optional[Tuple[StateTuple, Move]]] = {
            start.as_tuple(): None
        }

        while frontier:
            current = frontier.popleft()
            current_tuple = current.as_tuple()

            if current_tuple == goal.as_tuple():
                return self._reconstruct_path(came_from, current_tuple)

            for move, successor in self._successors(current, totals):
                successor_tuple = successor.as_tuple()

                if successor_tuple in came_from:
                    continue

                came_from[successor_tuple] = (current_tuple, move)
                frontier.append(successor)

        return []

    def _successors(
        self, state: State, totals: Tuple[int, int]
    ) -> Iterable[Tuple[Move, State]]:
        successors: List[Tuple[Move, State]] = []
        total_missionaries, total_cannibals = totals

        for missionaries, cannibals in self.POSSIBLE_MOVES:
            if state.boat_left:
                new_state = State(
                    state.left_missionaries - missionaries,
                    state.left_cannibals - cannibals,
                    0,
                )
            else:
                new_state = State(
                    state.left_missionaries + missionaries,
                    state.left_cannibals + cannibals,
                    1,
                )

            if self._is_valid_state(new_state, totals):
                # ensure the move is feasible relative to the boat's capacity and availability
                if state.boat_left:
                    if missionaries > state.left_missionaries or cannibals > state.left_cannibals:
                        continue
                else:
                    right_missionaries = total_missionaries - state.left_missionaries
                    right_cannibals = total_cannibals - state.left_cannibals
                    if missionaries > right_missionaries or cannibals > right_cannibals:
                        continue

                if missionaries + cannibals == 0:
                    continue

                successors.append(((missionaries, cannibals), new_state))

        return successors

    def _is_valid_state(self, state: State, totals: Tuple[int, int]) -> bool:
        total_missionaries, total_cannibals = totals
        right_missionaries = total_missionaries - state.left_missionaries
        right_cannibals = total_cannibals - state.left_cannibals

        if min(state.left_missionaries, state.left_cannibals, right_missionaries, right_cannibals) < 0:
            return False

        if state.left_missionaries > 0 and state.left_cannibals > state.left_missionaries:
            return False

        if right_missionaries > 0 and right_cannibals > right_missionaries:
            return False

        if state.boat_left not in (0, 1):
            return False

        return True

    def _reconstruct_path(
        self,
        came_from: Dict[StateTuple, Optional[Tuple[StateTuple, Move]]],
        goal: StateTuple,
    ) -> List[Move]:
        path: List[Move] = []
        current = goal

        while came_from[current] is not None:
            parent, move = came_from[current]
            path.append(move)
            current = parent

        path.reverse()
        return path


__all__ = ["MCAgent"]
