"""Improved Type B strategies.

ImprovedB (main, "B2"): the slides' expected-cost planning (RL-8) on the
Bayesian belief.
    1. Where to head (slides 13-16): sample pairs of hypothetical LostBot
       locations from P, compare the expected cost of searching each pair in
       either order (move = 1, reading = 1/2), and head for the cell that most
       often starts the cheaper order (``planning.plan_destination``).
    2. Where to sense (slide 17): along the walking route to that cell, sense
       at the cell with the lowest J (gamma = 1, see
       ``scored.expected_cost_objective``).
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
from ..planning import plan_destination
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

    def __init__(self, c: float = 1.0, n_pairs: int = 64):
        if c < 0:
            raise ValueError("c must be non-negative")
        if n_pairs < 1:
            raise ValueError("n_pairs must be positive")
        self.c, self.n_pairs = float(c), int(n_pairs)

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
        return self.plan_and_sense_point(position, sc, J, self.n_pairs)


class LookaheadB(ImprovedB):
    """Tried earlier (replaced): sense at the global argmin of J."""
    name = "lookahead_b"

    def __init__(self, c: float = 1.0):
        super().__init__(c)

    def select(self, position, sc):
        J = expected_cost_objective(
            sc, self.dist[position].astype(np.float64), c=self.c, gamma=1.0,
            move_cost=self.move_cost, sense_cost=self.sense_cost)
        J = np.where(self.admissible(sc), J, np.inf)
        return argbest(J, position, self.dist, self.rng)


class SlidesPureB(_TypeB):
    """Tried: the slides' method alone on the Bayesian belief. Head for the
    cell chosen by sampled pairwise expected-cost plans and sense there."""
    name = "slides_pure_b"

    def __init__(self, n_pairs: int = 64):
        self.n_pairs = int(n_pairs)

    def reset(self, ship, detector, start, rng):
        if not isinstance(detector, DetectorB):
            raise TypeError("SlidesPureB requires a DetectorB")
        PathFollowingStrategy.reset(self, ship, detector, start, rng)
        self.kb = BeliefKB(ship, detector, excluded=[start])

    def choose_target(self, position):          # no scoring needed
        return plan_destination(self.kb.probabilities(), position, self.dist,
                                self.rng, self.move_cost, self.sense_cost,
                                self.n_pairs)

    def select(self, position, sc):              # unused
        raise NotImplementedError


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
