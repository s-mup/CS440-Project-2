"""Baseline Strategy Type B (from the handout).

  * Run the detector.
  * Given the result, update the knowledge base of where LostBot is (Bayes).
  * Move to the cell most likely to contain LostBot (breaking ties by
    distance, then at random).
  * Repeat until LostBot becomes FoundBot.

"Distance" in the tie-break is BFS shortest-path distance. "Move to" means the
bot walks the whole shortest path, paying 1 per step, and then runs the
detector again. Probability ties are detected in log space with a tiny
tolerance (``TIE_LOG_TOL``), so rounding noise between cells that are tied in
exact arithmetic does not break the tie arbitrarily.
"""
from __future__ import annotations

from ..detectors import DetectorB
from ..knowledge_base import BeliefKB
from .base import PathFollowingStrategy, nearest_random_tiebreak


class BaselineB(PathFollowingStrategy):
    name = "baseline_b"

    def reset(self, ship, detector, start, rng):
        if not isinstance(detector, DetectorB):
            raise TypeError("BaselineB requires a DetectorB")
        super().reset(ship, detector, start, rng)
        # Prior: uniform over every open cell except the start cell.
        self.kb = BeliefKB(ship, detector, excluded=[start])

    def choose_target(self, position: int) -> int:
        return nearest_random_tiebreak(self.kb.most_likely(), position,
                                       self.dist, self.rng)
