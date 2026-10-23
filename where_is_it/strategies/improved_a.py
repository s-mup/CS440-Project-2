"""Improved Type A strategies.

ImprovedA (main, "A2"): expected-cost lookahead.
    Picks the next sensing cell a* = argmin_a J(a), where J (see
    ``scored.expected_cost_objective``) adds up the walk to a, a lower bound
    on the walking still needed afterwards, the reading itself, the further
    reading needed unless a finds the LostBot, and c times the expected
    remaining uncertainty in bits. Under a uniform belief over S, a reading at
    a splits S into FOUND / PING / NO_PING groups. A balanced split leaves
    little expected entropy, so J favours informative cells, while the
    movement terms stop the bot from crossing the ship for a slightly better
    split. Only cells whose reading can change S (info gain > 0, or a in S)
    are allowed, so every reading shrinks S and the search ends after at most
    |S| readings.

InfoPerCostA (ablation, "A1"): greedy information per unit cost.
    a* = argmax_a IG(a) / (move_cost * d(cur, a) + sense_cost).
"""
from __future__ import annotations

import numpy as np

from ..detectors import DetectorA
from ..knowledge_base import CandidateSetKB
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


class InfoPerCostA(_TypeA):
    name = "info_per_cost_a"

    def select(self, position, sc):
        if sc.entropy <= 0:                      # one candidate left: go there
            return argbest(-sc.p_found, position, self.dist, self.rng)
        cost = self.move_cost * self.dist[position] + self.sense_cost
        ratio = np.where(sc.info_gain > INFO_EPS, sc.info_gain / cost, -np.inf)
        return argbest(-ratio, position, self.dist, self.rng)
