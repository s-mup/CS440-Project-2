"""Improved Type A strategies.

ImprovedA (main, "A2"): the slides' expected-cost planning (RL-8).
    1. Where to head (slides 13-16): sample pairs of hypothetical LostBot
       locations from the belief (uniform over S), compare the expected cost
       of searching each pair in either order with the project's action
       costs, and head for the cell that most often starts the cheaper order
       (``planning.plan_destination``).
    2. Where to sense (slide 17): along the walking route to that cell, sense
       at the cell with the lowest estimated total cost J (see
       ``scored.expected_cost_objective``). J adds up the walk to the cell, a
       lower bound on the walking still needed afterwards, the reading itself,
       the further reading needed unless it finds the LostBot, and c times the
       expected remaining uncertainty in bits. A reading on the way can
       therefore replace part of the trip.
    Only cells whose reading can change S (info gain > 0, or a in S) are
    allowed, so every reading shrinks S and the search ends after at most
    |S| readings.

InfoPerCostA (ablation, "A1"): greedy information per unit cost.
    a* = argmax_a IG(a) / (move_cost * d(cur, a) + sense_cost).
"""
from __future__ import annotations

import numpy as np

from ..detectors import DetectorA
from ..knowledge_base import CandidateSetKB
from ..planning import plan_destination
from .scored import INFO_EPS, ScoredStrategy, argbest, expected_cost_objective


class _TypeA(ScoredStrategy):
    move_cost = 1.0
    sense_cost = 0.5

    def reset(self, ship, detector, start, rng):
        if not isinstance(detector, DetectorA):
            raise TypeError(f"{type(self).__name__} requires a DetectorA")
        super().reset(ship, detector, start, rng)
        self.kb = CandidateSetKB(ship, detector, excluded=[start])

    def belief_vector(self):
        return self.kb.possible

    def admissible(self, sc) -> np.ndarray:
        """Cells where a reading would shrink S."""
        return (sc.info_gain > INFO_EPS) | (sc.p_found > 0)


class ImprovedA(_TypeA):
    name = "improved_a"

    def __init__(self, c: float = 1.0, gamma: float = 1.0, n_pairs: int = 64):
        if c < 0 or gamma < 0:
            raise ValueError("c and gamma must be non-negative")
        if n_pairs < 1:
            raise ValueError("n_pairs must be positive")
        self.c, self.gamma, self.n_pairs = float(c), float(gamma), int(n_pairs)

    def select(self, position, sc):
        J = expected_cost_objective(
            sc, self.dist[position].astype(np.float64), c=self.c,
            gamma=self.gamma, move_cost=self.move_cost, sense_cost=self.sense_cost)
        J = np.where(self.admissible(sc), J, np.inf)
        return self.plan_and_sense_point(position, sc, J, self.n_pairs)


class LookaheadA(_TypeA):
    """Tried earlier (replaced): sense at the global argmin of J, walking the
    shortest route straight to it (no sampled planning)."""
    name = "lookahead_a"

    def __init__(self, c: float = 1.0, gamma: float = 1.0):
        if c < 0 or gamma < 0:
            raise ValueError("c and gamma must be non-negative")
        self.c, self.gamma = float(c), float(gamma)

    def select(self, position, sc):
        J = expected_cost_objective(
            sc, self.dist[position].astype(np.float64), c=self.c,
            gamma=self.gamma, move_cost=self.move_cost, sense_cost=self.sense_cost)
        J = np.where(self.admissible(sc), J, np.inf)
        return argbest(J, position, self.dist, self.rng)


class SlidesPureA(_TypeA):
    """Tried: the slides' method alone. Head for the cell chosen by sampled
    pairwise expected-cost plans (belief uniform over S) and sense there."""
    name = "slides_pure_a"

    def __init__(self, n_pairs: int = 64):
        self.n_pairs = int(n_pairs)

    def choose_target(self, position):          # no scoring needed
        p = self.kb.possible.astype(np.float64)
        return plan_destination(p / p.sum(), position, self.dist, self.rng,
                                self.move_cost, self.sense_cost, self.n_pairs)

    def select(self, position, sc):              # unused
        raise NotImplementedError


class InfoPerCostA(_TypeA):
    name = "info_per_cost_a"

    def select(self, position, sc):
        if sc.entropy <= 0:                      # one candidate left: go there
            return argbest(-sc.p_found, position, self.dist, self.rng)
        cost = self.move_cost * self.dist[position] + self.sense_cost
        ratio = np.where(sc.info_gain > INFO_EPS, sc.info_gain / cost, -np.inf)
        return argbest(-ratio, position, self.dist, self.rng)
