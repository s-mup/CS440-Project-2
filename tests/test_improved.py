import math
import unittest

import numpy as np

from where_is_it.detectors import DetectorA, DetectorB, Outcome
from where_is_it.knowledge_base import BeliefKB, CandidateSetKB
from where_is_it.scoring import (BeliefScorer, CandidateScorer, get_scorer)
from where_is_it.ship import Ship
from where_is_it.strategies import (FreeMovementA, FreeMovementB, ImprovedA,
                                    ImprovedB, InfoPerCostA, ProbPerCostB)
from where_is_it.strategies.scored import expected_cost_objective
from where_is_it.world import run_trial

OPEN5 = Ship(np.ones((5, 5), dtype=bool))
SMALL = Ship.from_ascii("""
    .....#
    .#....
    ...#..
    #.....
    ..#...
""")


def rngs(seed):
    d, s = np.random.SeedSequence(seed).spawn(2)
    return dict(detector_rng=np.random.default_rng(d),
                strategy_rng=np.random.default_rng(s))


def brute_force_ig(ship, det, p, a):
    """Expected information gain by direct enumeration of the outcomes."""
    def H(q):
        q = q[q > 0]
        return -(q * np.log2(q)).sum()
    L = np.array([0.0 if b == a else det.ping_probability(ship.cell_of(a), ship.cell_of(b))
                  for b in range(ship.n)])
    nl = np.where(np.arange(ship.n) == a, 0.0, 1.0 - L)
    exp_h = 0.0
    for lik in (L, nl):
        z = (p * lik).sum()
        if z > 0:
            exp_h += z * H(p * lik / z)
    return H(p) - exp_h, (p * L).sum()


class TestCandidateScorer(unittest.TestCase):
    def setUp(self):
        self.kb = CandidateSetKB(OPEN5, DetectorA(1))
        self.kb.update(OPEN5.index_of((0, 0)), Outcome.NO_PING)   # |S| = 21
        self.sc = CandidateScorer(OPEN5, DetectorA(1)).score(self.kb.possible)

    def test_design_example_information_gain(self):
        centre, edge = OPEN5.index_of((2, 2)), OPEN5.index_of((0, 2))
        self.assertAlmostEqual(self.sc.info_gain[centre], 1.166, places=3)
        self.assertAlmostEqual(self.sc.info_gain[edge], 0.857, places=3)
        self.assertAlmostEqual(self.sc.expected_distance[centre], 48 / 21, places=5)
        self.assertAlmostEqual(self.sc.expected_distance[edge], 72 / 21, places=5)
        self.assertAlmostEqual(self.sc.entropy, math.log2(21))

    def test_outcome_probabilities_sum_to_one(self):
        total = self.sc.p_found + self.sc.p_ping + self.sc.p_no_ping
        np.testing.assert_allclose(total, 1.0)

    def test_resensing_eliminated_area_has_zero_gain(self):
        # Zero up to rounding; strategies treat IG <= INFO_EPS (1e-9) as zero.
        self.assertLess(self.sc.info_gain[OPEN5.index_of((0, 0))], 1e-9)

    def test_cost_per_bit_tradeoff(self):
        # Design example: the centre needs 2 more moves (and is 0.857 farther
        # in total path length) but leaves 0.308 fewer expected bits. It wins
        # exactly when c > 0.857 / 0.308 ~ 2.78.
        cur = OPEN5.index_of((0, 0))
        centre, edge = OPEN5.index_of((2, 2)), OPEN5.index_of((0, 2))
        d = OPEN5.bfs_dist[cur].astype(float)
        for c, centre_better in [(2.0, False), (4.0, True)]:
            J = expected_cost_objective(self.sc, d, c=c, gamma=1.0,
                                        move_cost=1.0, sense_cost=0.5)
            self.assertEqual(J[centre] < J[edge], centre_better, c)


class TestBeliefScorer(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(0)
        for alpha in (0.05, 0.5, 3.0):
            det = DetectorB(alpha)
            p = rng.random(SMALL.n) ** 3
            p[rng.integers(SMALL.n, size=4)] = 0.0       # some eliminated cells
            p /= p.sum()
            sc = BeliefScorer(SMALL, det).score(p)
            for a in range(SMALL.n):
                ig, p_ping = brute_force_ig(SMALL, det, p, a)
                self.assertAlmostEqual(sc.info_gain[a], ig, places=4)
                self.assertAlmostEqual(sc.p_ping[a], p_ping, places=5)
            np.testing.assert_allclose(sc.p_found + sc.p_ping + sc.p_no_ping,
                                       1.0, atol=1e-5)

    def test_point_mass_has_zero_information(self):
        p = np.zeros(SMALL.n)
        p[3] = 1.0
        sc = BeliefScorer(SMALL, DetectorB(0.3)).score(p)
        self.assertTrue(np.all(sc.info_gain < 1e-6))
        self.assertEqual(sc.p_found[3], 1.0)

    def test_scorer_cache_is_reused_and_bounded(self):
        ship = Ship(np.ones((4, 4), dtype=bool))
        s1 = get_scorer(ship, DetectorB(0.3))
        self.assertIs(get_scorer(ship, DetectorB(0.3)), s1)
        for a in (0.1, 0.2, 0.4):
            get_scorer(ship, DetectorB(a))
        self.assertLessEqual(len(ship._scorer_cache), 1)


class TestImprovedStrategiesTerminate(unittest.TestCase):
    def check_all(self, make, detector, max_senses=None):
        for lostbot in range(1, SMALL.n):
            r = run_trial(SMALL, detector, make(), 0, lostbot, **rngs(lostbot),
                          max_actions=5000)
            self.assertTrue(r.found, (make, detector, lostbot))
            if max_senses is not None:
                self.assertLessEqual(r.senses, max_senses)

    def test_type_a_strategies(self):
        for k in (0, 1, 3, 10):
            for make in (lambda: ImprovedA(10, 1.0), lambda: ImprovedA(3, 0.0),
                         InfoPerCostA, FreeMovementA):
                # Every reading shrinks S, so there are at most n readings.
                self.check_all(make, DetectorA(k), max_senses=SMALL.n)

    def test_type_b_strategies(self):
        for alpha in (0.01, 0.3, 3.0):
            for make in (lambda: ImprovedB(3), lambda: ImprovedB(100),
                         ProbPerCostB, FreeMovementB):
                self.check_all(make, DetectorB(alpha))

    def test_improved_b_never_buys_worthless_resense(self):
        # Belief concentrated on the four neighbours of a: re-sensing at a
        # always pings and teaches nothing, so ImprovedB must move.
        ship = Ship(np.ones((5, 5), dtype=bool))
        a = ship.index_of((2, 2))
        strat = ImprovedB(c=3)
        strat.reset(ship, DetectorB(0.5), a, np.random.default_rng(0))
        for _ in range(30):
            strat.kb.update(a, Outcome.PING)
        strat.observe(a, Outcome.PING)
        self.assertTrue(len(strat.plan) > 0)

    def test_information_hiding_prefix_property(self):
        # Same check as for the baseline: with k = 0 the observations before
        # discovery do not depend on where the LostBot is.
        from where_is_it.strategies import Move, Sense  # noqa: F401
        traces = []
        for lostbot in (5, 20):
            strat = ImprovedA(10, 1.0)
            actions = []
            orig = strat.next_action

            def rec(pos, orig=orig, actions=actions):
                act = orig(pos)
                actions.append(act)
                return act
            strat.next_action = rec
            run_trial(SMALL, DetectorA(0), strat, 0, lostbot, **rngs(3))
            traces.append(actions)
        short, long_ = sorted(traces, key=len)
        self.assertEqual(short, long_[:len(short)])


class TestValidation(unittest.TestCase):
    def test_wrong_detector_rejected(self):
        with self.assertRaises(TypeError):
            ImprovedA().reset(SMALL, DetectorB(0.3), 0, np.random.default_rng(0))
        with self.assertRaises(TypeError):
            ImprovedB().reset(SMALL, DetectorA(1), 0, np.random.default_rng(0))
        with self.assertRaises(TypeError):
            ProbPerCostB().reset(SMALL, DetectorA(1), 0, np.random.default_rng(0))

    def test_negative_hyperparameters_rejected(self):
        with self.assertRaises(ValueError):
            ImprovedA(c=-1)
        with self.assertRaises(ValueError):
            ImprovedB(c=-1)


if __name__ == "__main__":
    unittest.main()
