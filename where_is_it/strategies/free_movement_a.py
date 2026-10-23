"""Type A strategy for the free-movement cost model (Question 6).

When moving costs nothing, the cost of a search is just the number of
readings, so the bot can go to the best cell anywhere on the ship. Each
reading is a three-way question (FOUND / PING / NO_PING) about a uniform
candidate set. Greedy "twenty questions" picks the question with the highest
expected information gain:

    a* = argmax_a IG(a),  IG(a) = log2|S| - E[H(S_post)],

which favours cells whose square splits S as evenly as possible. Distance is
used only to break ties (it costs nothing, but shorter walks are tidier).
Movement still follows legal BFS paths and is still counted. Every chosen
reading shrinks S, so the search ends after at most |S| readings.
"""
from __future__ import annotations

import numpy as np

from .improved_a import _TypeA
from .scored import INFO_EPS, argbest


class FreeMovementA(_TypeA):
    name = "free_movement_a"
    move_cost = 0.0
    sense_cost = 1.0

    def select(self, position, sc):
        if sc.entropy <= 0:                      # one candidate left: go there
            return argbest(-sc.p_found, position, self.dist, self.rng)
        ig = np.where(sc.info_gain > INFO_EPS, sc.info_gain, -np.inf)
        return argbest(-ig, position, self.dist, self.rng)
