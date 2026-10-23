"""Improved Type B strategies.

ImprovedB (main, "B2"): expected-cost lookahead on the Bayesian belief.
    a* = argmin_a J(a) with gamma = 1 (see ``scored.expected_cost_objective``).
    Differences from the baseline, which walks to argmax_b P(b):
      * It may sense at a cell that is not the most likely one, if the
        expected reading there removes more uncertainty per unit cost (for
        example, between two separated clusters of probability).
      * It charges for travel, both now and as a lower bound on later travel,
        so it does not walk across the ship after a single NO_PING the way the
        baseline does.
      * Re-sensing a cell already ruled out is allowed only when the
        reading is worth its cost (c * IG >= sense cost). A noisy detector's
        repeated readings do carry fresh evidence, but often too little.
    gamma is fixed at 1. With gamma < 1, re-sensing in place forever can look
    cheaper than walking to a concentrated belief, so the search might never
    finish.

ProbPerCostB (ablation, "B3"): classic search-theory rule,
    a* = argmax_a P(a) / (move_cost * d(cur, a) + sense_cost).
"""
from __future__ import annotations

import numpy as np

from ..detectors import DetectorB
from ..knowledge_base import BeliefKB
from .base import PathFollowingStrategy
from .scored import ScoredStrategy, argbest, expected_cost_objective


class _TypeB(ScoredStrategy):
    move_cost = 1.0
    sense_cost = 0.5

    def reset(self, ship, detector, start, rng):
        if not isinstance(detector, DetectorB):
            raise TypeError(f"{type(self).__name__} requires a DetectorB")
        super().reset(ship, detector, start, rng)
        self.kb = BeliefKB(ship, detector, excluded=[start])

    def belief_vector(self):
        return self.kb.probabilities()


class ImprovedB(_TypeB):
    name = "improved_b"

    def __init__(self, c: float = 1.0):
        if c < 0:
            raise ValueError("c must be non-negative")
        self.c = float(c)

    def admissible(self, sc) -> np.ndarray:
        """A reading at a must either test a cell that may hold the LostBot
        (P(a) > 0, which removes that possibility if it is not there) or be
        worth its price under the strategy's own valuation (c * IG >= cost).
        Without this, a weak detector makes in-place re-sensing look cheaper
        than any move, and the bot buys readings worth far less than 0.5."""
        return (sc.p_found > 0) | (self.c * sc.info_gain >= self.sense_cost)

    def select(self, position, sc):
        J = expected_cost_objective(
            sc, self.dist[position].astype(np.float64), c=self.c, gamma=1.0,
            move_cost=self.move_cost, sense_cost=self.sense_cost)
        J = np.where(self.admissible(sc), J, np.inf)
        return argbest(J, position, self.dist, self.rng)


class ProbPerCostB(PathFollowingStrategy):
    name = "prob_per_cost_b"
    move_cost = 1.0
    sense_cost = 0.5

    def reset(self, ship, detector, start, rng):
        if not isinstance(detector, DetectorB):
            raise TypeError("ProbPerCostB requires a DetectorB")
        super().reset(ship, detector, start, rng)   # no O(n^2) scorer needed
        self.kb = BeliefKB(ship, detector, excluded=[start])

    def choose_target(self, position):
        p = self.kb.probabilities()
        ratio = p / (self.move_cost * self.dist[position] + self.sense_cost)
        ratio[p <= 0] = -np.inf
        return argbest(-ratio, position, self.dist, self.rng)
