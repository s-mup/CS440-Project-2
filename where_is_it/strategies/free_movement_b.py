"""Type B strategy for the free-movement cost model (Question 6).

With free movement, only the number of readings matters, so the bot senses
wherever the expected reduction in uncertainty is largest:

    a* = argmax_a IG(a) = H(P) - sum_o P(o | a) H(P | o),

with outcomes FOUND / PING / NO_PING (H(P | FOUND) = 0). The FOUND outcome
means a highly likely cell scores well on its own. Once the belief is nearly
certain, the cell with the highest IG is the mode, so the strategy ends by
sensing there. If every IG is zero (a point mass), it senses at the mode.
Distance only breaks ties.
"""
from __future__ import annotations

from .improved_b import _TypeB
from .scored import INFO_EPS, argbest


class FreeMovementB(_TypeB):
    name = "free_movement_b"
    move_cost = 0.0
    sense_cost = 1.0

    def select(self, position, sc):
        if sc.info_gain.max() <= INFO_EPS:
            return argbest(-sc.p_found, position, self.dist, self.rng)
        return argbest(-sc.info_gain, position, self.dist, self.rng)
