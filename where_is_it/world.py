"""Simulation: hidden world state, action execution, costs, and the trial loop.

Information boundary
--------------------
``World`` is the only object that knows the LostBot's location (``_lostbot``)
and owns the detector's random generator. Strategies never receive a World.
``run_trial`` gives a strategy only the following:
  * the ship layout and the detector *model* (both public),
  * its own start cell and current cell,
  * the outcome of each detector reading it requests,
  * its own random generator, used for tie-breaking.
A strategy acts by returning ``Move(target)`` or ``Sense()``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import Outcome
from .strategies.base import Move, Sense, Strategy


@dataclass(frozen=True)
class CostModel:
    move_cost: float
    sense_cost: float
    name: str

    def cost(self, moves: int, senses: int) -> float:
        return self.move_cost * moves + self.sense_cost * senses


# Handout cost: (number of moves) + 1/2 (number of detector actions).
ORIGINAL_COST = CostModel(move_cost=1.0, sense_cost=0.5, name="original")
# Question 6: movement is free, only detector actions cost anything.
FREE_MOVEMENT_COST = CostModel(move_cost=0.0, sense_cost=1.0, name="free_movement")


class IllegalMoveError(RuntimeError):
    pass


class World:
    """The true (hidden) state of one search episode."""

    def __init__(self, ship, detector, start: int, lostbot: int,
                 rng: np.random.Generator):
        for name, idx in (("start", start), ("lostbot", lostbot)):
            if not 0 <= idx < ship.n:
                raise ValueError(f"{name} index {idx} is not an open cell")
        self.ship = ship
        self.detector = detector
        self._lostbot = int(lostbot)
        self._rng = rng
        self.position = int(start)
        self.moves = 0
        self.senses = 0
        self.found = False

    @property
    def actions(self) -> int:
        return self.moves + self.senses

    def move(self, target: int) -> None:
        if self.found:
            raise RuntimeError("search already finished")
        if target not in self.ship.neighbors[self.position]:
            raise IllegalMoveError(
                f"cannot move from cell {self.position} to cell {target}")
        self.position = int(target)
        self.moves += 1

    def sense(self) -> Outcome:
        if self.found:
            raise RuntimeError("search already finished")
        self.senses += 1
        outcome = self.detector.sample(self.ship.cell_of(self.position),
                                       self.ship.cell_of(self._lostbot),
                                       self._rng)
        if outcome is Outcome.FOUND:
            self.found = True
        return outcome


@dataclass(frozen=True)
class TrialResult:
    found: bool
    moves: int
    senses: int
    cost: float
    start: tuple[int, int]
    lostbot: tuple[int, int]
    hit_action_cap: bool


def default_action_cap(ship) -> int:
    """A generous action limit, so a faulty strategy cannot loop forever.

    Both baselines rule out a new cell with every reading and walk at most
    ``diameter`` moves between readings, so they always finish within
    n * (diameter + 1) actions. The cap is 10 times that bound.
    """
    return 10 * ship.n * (ship.diameter + 1)


def run_trial(ship, detector, strategy: Strategy, start: int, lostbot: int, *,
              detector_rng: np.random.Generator,
              strategy_rng: np.random.Generator,
              cost_model: CostModel = ORIGINAL_COST,
              max_actions: int | None = None) -> TrialResult:
    """Run one search episode and return its counters and cost."""
    world = World(ship, detector, start, lostbot, detector_rng)
    strategy.reset(ship, detector, start, strategy_rng)
    cap = default_action_cap(ship) if max_actions is None else max_actions

    while not world.found and world.actions < cap:
        action = strategy.next_action(world.position)
        if isinstance(action, Move):
            world.move(action.target)
        elif isinstance(action, Sense):
            outcome = world.sense()
            strategy.observe(world.position, outcome)
        else:
            raise TypeError(f"strategy returned an invalid action {action!r}")

    return TrialResult(
        found=world.found,
        moves=world.moves,
        senses=world.senses,
        cost=cost_model.cost(world.moves, world.senses),
        start=ship.cell_of(start),
        lostbot=ship.cell_of(lostbot),
        hit_action_cap=not world.found,
    )


def sample_scenario(ship, rng: np.random.Generator) -> tuple[int, int]:
    """Draw (start, lostbot): two distinct open cells, uniformly at random.

    Assumption (the handout does not specify): the LostBot is never placed on
    the LocatorBot's start cell, matching Project 1's distinct placements.
    Strategies use the same assumption for their prior.
    """
    start, lostbot = rng.choice(ship.n, size=2, replace=False)
    return int(start), int(lostbot)
