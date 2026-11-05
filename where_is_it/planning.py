"""Sampled expected-cost planning ("where should we search next?").

This implements the decision method from the course slides (RL-8):

1. **Sample hypothetical LostBot locations** (HLBs) from the current belief
   (slide 13). A sample is a plausible scenario, not a claim about where the
   bot actually is.
2. **Condition on each sampled pair** (HLB1, HLB2) and renormalise within the
   pair (slide 15):
       w1 = P(HLB1) / (P(HLB1) + P(HLB2)),   w2 = 1 - w1.
3. **Compare the two visiting orders by expected cost**, using the project's
   action weights (move = 1, reading = 1/2) instead of counting every action
   as 1 (slides 10 and 16). For the order HLB1 -> HLB2:
       cost(1 -> 2) = [m d(cur, HLB1) + s]  +  w2 [m d(HLB1, HLB2) + s]
   The first leg is always paid. The second leg is paid only if the LostBot
   was not at HLB1, which happens with conditional probability w2. This is
   the cumulative form of slide 10's worked example.
4. **Repeat over many pairs** and head for the cell that most often starts the
   cheaper order. Ties go to the nearer cell, then are broken at random.

With m = 0 (free movement) the comparison reduces to "search the more likely
cell first", the highest-probability-first rule of slides 7-8.

d(., .) is the *walking* distance through the ship (number of moves). Walls
block straight lines, so this is the precomputed shortest-walk table
``ship.bfs_dist``, not Euclidean or Manhattan distance.
"""
from __future__ import annotations

import numpy as np


def sample_pairs(p: np.ndarray, n_pairs: int,
                 rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_pairs`` pairs of distinct cells, each cell drawn with
    probability proportional to the belief ``p`` (cells with p = 0 are never
    drawn). Requires at least two cells with p > 0."""
    support = np.flatnonzero(p > 0)
    if len(support) < 2:
        raise ValueError("need at least two possible cells to form a pair")
    w = p[support]
    q = w / w.sum()
    first = rng.choice(len(support), size=n_pairs, p=q)
    second = rng.choice(len(support), size=n_pairs, p=q)
    # The second cell must differ from the first, i.e. it is drawn from the
    # belief with the first cell removed. Redrawing collisions does exactly
    # that, but can take extremely long when one cell holds almost all the
    # probability. So after a few redraws, the remaining collisions are drawn
    # directly from the belief with that cell excluded.
    for _ in range(4):
        clash = np.flatnonzero(first == second)
        if len(clash) == 0:
            break
        second[clash] = rng.choice(len(support), size=len(clash), p=q)
    for i in np.flatnonzero(first == second):
        rest = w.copy()
        rest[first[i]] = 0.0
        second[i] = rng.choice(len(support), p=rest / rest.sum())
    return support[first], support[second]


def pair_plan_costs(position: int, h1: np.ndarray, h2: np.ndarray,
                    p: np.ndarray, dist: np.ndarray, move_cost: float,
                    sense_cost: float) -> tuple[np.ndarray, np.ndarray]:
    """Expected cost of searching HLB1 first and of searching HLB2 first,
    conditioned on the LostBot being at one of the two (slides 15-16)."""
    p1, p2 = p[h1], p[h2]
    w1 = p1 / (p1 + p2)
    w2 = 1.0 - w1
    d_c1 = dist[position, h1].astype(np.float64)
    d_c2 = dist[position, h2].astype(np.float64)
    d_12 = dist[h1, h2].astype(np.float64)
    cost_1_first = (move_cost * d_c1 + sense_cost) + w2 * (move_cost * d_12 + sense_cost)
    cost_2_first = (move_cost * d_c2 + sense_cost) + w1 * (move_cost * d_12 + sense_cost)
    return cost_1_first, cost_2_first


def plan_destination(p: np.ndarray, position: int, dist: np.ndarray,
                     rng: np.random.Generator, move_cost: float = 1.0,
                     sense_cost: float = 0.5, n_pairs: int = 64) -> int:
    """Next cell to head for, chosen by sampled pairwise expected-cost plans.

    Returns the cell that most often begins the cheaper order over
    ``n_pairs`` sampled pairs. With only one possible cell left, that cell is
    returned.
    """
    support = np.flatnonzero(p > 0)
    if len(support) == 0:
        raise ValueError("belief has no possible cells")
    if len(support) == 1:
        return int(support[0])
    h1, h2 = sample_pairs(p, n_pairs, rng)
    c1, c2 = pair_plan_costs(position, h1, h2, p, dist, move_cost, sense_cost)
    winners = np.where(c1 <= c2, h1, h2)
    votes = np.bincount(winners, minlength=len(p))
    best = np.flatnonzero(votes == votes.max())
    d = dist[position, best]
    nearest = best[d == d.min()]
    return int(nearest[rng.integers(len(nearest))])
