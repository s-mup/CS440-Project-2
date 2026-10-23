"""Strategy interface and helpers shared by all LocatorBot strategies.

``run_trial`` drives a strategy like this:

    strategy.reset(ship, detector, start, rng)
    loop:
        action = strategy.next_action(position)
        Move(target) -> the world moves one adjacent cell (cost: move)
        Sense()      -> the world reads the detector (cost: sense), then
                        strategy.observe(position, outcome)

A strategy only sees public information: the layout, the detector model, its
own position, and the outcomes it has observed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..detectors import Outcome
from ..pathfinding import UnreachableError, shortest_path


@dataclass(frozen=True)
class Move:
    target: int  # index of an adjacent open cell


@dataclass(frozen=True)
class Sense:
    pass


Action = Move | Sense


class Strategy(ABC):
    name: str = "strategy"

    @abstractmethod
    def reset(self, ship, detector, start: int, rng: np.random.Generator) -> None:
        """Start a new episode at cell ``start``."""

    @abstractmethod
    def next_action(self, position: int) -> Action:
        """Choose the next action given the bot's current cell."""

    @abstractmethod
    def observe(self, position: int, outcome) -> None:
        """Receive the outcome of a Sense taken at ``position``."""


class PathFollowingStrategy(Strategy):
    """Shared loop: sense, update the knowledge base, choose a destination,
    walk the whole BFS shortest path to it, sense again.

    Subclasses set ``self.kb`` in ``reset`` and implement ``choose_target``.
    """

    def reset(self, ship, detector, start, rng):
        self.ship = ship
        self.detector = detector
        self.rng = rng
        self.dist = ship.bfs_dist           # computed once per ship, cached
        self.plan: deque[int] = deque()
        self.done = False

    def next_action(self, position: int) -> Action:
        if self.plan:
            return Move(self.plan.popleft())
        return Sense()

    def observe(self, position: int, outcome) -> None:
        self.kb.update(position, outcome)
        self.plan.clear()
        if outcome is Outcome.FOUND:
            self.done = True
            return
        target = self.choose_target(position)
        self.plan.extend(shortest_path(self.ship.neighbors, self.dist,
                                       position, target))

    @abstractmethod
    def choose_target(self, position: int) -> int:
        """Pick the next cell to walk to (and sense at)."""


def nearest_random_tiebreak(cells: np.ndarray, position: int,
                            dist: np.ndarray,
                            rng: np.random.Generator) -> int:
    """Return the cell in ``cells`` closest by BFS distance to ``position``.

    Ties are broken uniformly at random. Unreachable cells are ignored.
    """
    d = dist[position, cells]
    reachable = d >= 0
    if not reachable.any():
        raise UnreachableError("no reachable target cell")
    cells, d = cells[reachable], d[reachable]
    nearest = cells[d == d.min()]
    return int(nearest[rng.integers(len(nearest))])
