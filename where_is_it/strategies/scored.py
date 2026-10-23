"""Shared machinery for strategies that score every candidate sensing cell.

Each decision:
  1. Score every open cell a as a possible next sensing position
     (``scoring.SensingScores``) from the current knowledge base.
  2. Choose the best cell by the strategy's rule.
  3. Walk the BFS shortest path there and sense (via PathFollowingStrategy).
     Nothing is learned while walking, so re-planning mid-path would choose
     the same cell again.
"""
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from ..scoring import SensingScores, get_scorer
from .base import PathFollowingStrategy

INFO_EPS = 1e-9


def argbest(values: np.ndarray, position: int, dist: np.ndarray,
            rng: np.random.Generator, minimise: bool = True) -> int:
    """Index of the best finite value. Ties within a relative 1e-9 go to the
    cell nearest by BFS distance, and remaining ties are broken at random."""
    v = values if minimise else -values
    v = np.where(dist[position] >= 0, v, np.inf)
    best = v.min()
    if not np.isfinite(best):
        raise RuntimeError("no admissible sensing position")
    tied = np.flatnonzero(v <= best + 1e-9 * max(1.0, abs(best)))
    d = dist[position, tied]
    nearest = tied[d == d.min()]
    return int(nearest[rng.integers(len(nearest))])


class ScoredStrategy(PathFollowingStrategy):
    """Path-following strategy whose target comes from ``SensingScores``."""

    def reset(self, ship, detector, start, rng):
        super().reset(ship, detector, start, rng)
        self.scorer = get_scorer(ship, detector)

    def current_scores(self) -> SensingScores:
        return self.scorer.score(self.belief_vector())

    @abstractmethod
    def belief_vector(self) -> np.ndarray:
        """Input expected by the scorer: possible-mask (A) or probabilities (B)."""

    def choose_target(self, position: int) -> int:
        return self.select(position, self.current_scores())

    @abstractmethod
    def select(self, position: int, sc: SensingScores) -> int:
        ...


def expected_cost_objective(sc: SensingScores, d_from_here: np.ndarray, *,
                            c: float, gamma: float,
                            move_cost: float, sense_cost: float) -> np.ndarray:
    """One-step lookahead estimate of the total remaining cost if the next
    reading is taken at cell a:

      J(a) = move_cost * [ d(cur, a) + gamma * E d(a, B) ]  travel now + lower
                                                            bound on travel later
           + sense_cost                                     this reading
           + sense_cost * (1 - P(found at a))               >= 1 more reading
                                                            unless found now
           + c * E[H_post(a)]                               remaining uncertainty,
                                                            priced at c per bit

    E d(a, B) is the same after any outcome (law of total expectation), so it
    is a true expected lower bound on the walking still needed after sensing
    at a. With gamma = 1, the movement terms charge exactly for detours off a
    shortest route toward the probability mass.
    """
    return (move_cost * (d_from_here + gamma * sc.expected_distance)
            + sense_cost * (2.0 - sc.p_found)
            + c * sc.expected_entropy)
