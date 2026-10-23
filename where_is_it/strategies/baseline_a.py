"""Baseline Strategy Type A (from the handout).

  * Run the detector.
  * Update the possible locations of LostBot based on the result.
  * Move to the nearest cell that might contain LostBot (ties at random).
  * Repeat until LostBot becomes FoundBot.

"Nearest" means BFS shortest-path distance through the ship. "Move to" means
the bot walks the whole shortest path, paying 1 per step, and then runs the
detector again.
"""
from __future__ import annotations

from ..detectors import DetectorA
from ..knowledge_base import CandidateSetKB
from .base import PathFollowingStrategy, nearest_random_tiebreak


class BaselineA(PathFollowingStrategy):
    name = "baseline_a"

    def reset(self, ship, detector, start, rng):
        if not isinstance(detector, DetectorA):
            raise TypeError("BaselineA requires a DetectorA")
        super().reset(ship, detector, start, rng)
        # Prior: every open cell except the start cell (see sample_scenario).
        self.kb = CandidateSetKB(ship, detector, excluded=[start])

    def choose_target(self, position: int) -> int:
        return nearest_random_tiebreak(self.kb.candidates(), position,
                                       self.dist, self.rng)
