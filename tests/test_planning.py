import unittest

import numpy as np

from where_is_it.planning import pair_plan_costs, plan_destination, sample_pairs
from where_is_it.detectors import DetectorA, DetectorB
from where_is_it.ship import Ship
from where_is_it.strategies import (ImprovedA, ImprovedB, LookaheadA,
                                    LookaheadB, SlidesPureA, SlidesPureB)
from where_is_it.world import run_trial

# The slides' four rooms in a row: A - B - C - D (indices 0..3), start at D.
ROOMS = Ship.from_ascii("....")
A, B, C, D = range(4)
P = np.array([0.3, 0.2, 0.4, 0.1])


class TestPairPlanCosts(unittest.TestCase):
    def test_slide_example_pair_c_a(self):
        # Slides 15-16 with every action costing 1 (the slides' setting).
        # Conditioned on {C, A}: w_C = 0.4/0.7, w_A = 0.3/0.7.
        # C first: (1 move + 1 search) + w_A * (2 moves + 1 search) = 2 + 9/7.
        # A first: (3 moves + 1 search) + w_C * (2 moves + 1 search) = 4 + 12/7.
        c_first, a_first = pair_plan_costs(D, np.array([C]), np.array([A]), P,
                                           ROOMS.bfs_dist, 1.0, 1.0)
        self.assertAlmostEqual(c_first[0], 2 + 9 / 7)
        self.assertAlmostEqual(a_first[0], 4 + 12 / 7)

    def test_project_action_weights(self):
        # Same pair with the project's costs: move = 1, reading = 1/2.
        c_first, a_first = pair_plan_costs(D, np.array([C]), np.array([A]), P,
                                           ROOMS.bfs_dist, 1.0, 0.5)
        self.assertAlmostEqual(c_first[0], 1.5 + (3 / 7) * 2.5)
        self.assertAlmostEqual(a_first[0], 3.5 + (4 / 7) * 2.5)

    def test_free_movement_is_highest_probability_first(self):
        # Slides 7-8: with free movement the more likely cell goes first.
        for h1, h2 in [(A, B), (C, D), (B, C), (A, C)]:
            c1, c2 = pair_plan_costs(D, np.array([h1]), np.array([h2]), P,
                                     ROOMS.bfs_dist, 0.0, 1.0)
            self.assertEqual(c1[0] <= c2[0], P[h1] >= P[h2], (h1, h2))

    def test_movement_can_beat_probability(self):
        # Slide 11: a nearby cell may be worth checking before a more likely
        # distant one. From D, D (p = 0.1, 0 moves) vs A (p = 0.3, 3 moves).
        d_first, a_first = pair_plan_costs(D, np.array([D]), np.array([A]), P,
                                           ROOMS.bfs_dist, 1.0, 0.5)
        self.assertLess(d_first[0], a_first[0])


class TestSampling(unittest.TestCase):
    def test_pairs_are_distinct_and_from_support(self):
        p = np.array([0.0, 0.5, 0.3, 0.2])
        h1, h2 = sample_pairs(p, 500, np.random.default_rng(0))
        self.assertTrue(np.all(h1 != h2))
        self.assertTrue(np.all(p[h1] > 0) and np.all(p[h2] > 0))

    def test_sampling_follows_belief(self):
        p = np.array([0.7, 0.1, 0.1, 0.1])
        h1, _ = sample_pairs(p, 4000, np.random.default_rng(1))
        self.assertAlmostEqual(np.mean(h1 == 0), 0.7, delta=0.03)

    def test_extremely_concentrated_belief_terminates(self):
        # One cell holds nearly all the mass (as at large alpha): every pair
        # must still pair it with the other possible cell, without looping.
        p = np.array([1.0, 1e-200, 0.0, 0.0])
        h1, h2 = sample_pairs(p, 100, np.random.default_rng(0))
        self.assertTrue(np.all(h1 != h2))
        self.assertEqual(set(h1) | set(h2), {0, 1})

    def test_needs_two_cells(self):
        with self.assertRaises(ValueError):
            sample_pairs(np.array([1.0, 0, 0, 0]), 10, np.random.default_rng(0))


class TestPlanDestination(unittest.TestCase):
    def test_slide_example_choice(self):
        # Belief only on {C, A}: every sampled pair is {C, A}, and C first is
        # cheaper, so the planner heads for C.
        p = np.array([0.3, 0.0, 0.4, 0.0])
        self.assertEqual(plan_destination(p, D, ROOMS.bfs_dist,
                                          np.random.default_rng(0)), C)

    def test_single_possible_cell(self):
        p = np.array([0.0, 1.0, 0.0, 0.0])
        self.assertEqual(plan_destination(p, D, ROOMS.bfs_dist,
                                          np.random.default_rng(0)), B)

    def test_reproducible(self):
        ship = Ship(np.ones((9, 9), dtype=bool))
        p = np.random.default_rng(3).random(ship.n)
        p /= p.sum()
        a = plan_destination(p, 0, ship.bfs_dist, np.random.default_rng(5))
        b = plan_destination(p, 0, ship.bfs_dist, np.random.default_rng(5))
        self.assertEqual(a, b)


class TestPlanningStrategiesFinish(unittest.TestCase):
    SHIP = Ship.from_ascii("""
        .....#
        .#....
        ...#..
        #.....
        ..#...
    """)

    def _check(self, make, detector):
        for seed in range(6):
            r = np.random.default_rng(seed)
            start, lost = r.choice(self.SHIP.n, size=2, replace=False)
            d, s = np.random.SeedSequence(seed).spawn(2)
            res = run_trial(self.SHIP, detector, make(), int(start), int(lost),
                            detector_rng=np.random.default_rng(d),
                            strategy_rng=np.random.default_rng(s),
                            max_actions=20_000)
            self.assertTrue(res.found, (make, seed))

    def test_type_a_variants_find_lostbot(self):
        for make in (ImprovedA, LookaheadA, SlidesPureA):
            for k in (0, 1, 3):
                self._check(make, DetectorA(k))

    def test_type_b_variants_find_lostbot(self):
        for make in (ImprovedB, LookaheadB, SlidesPureB):
            for alpha in (0.1, 1.0):
                self._check(make, DetectorB(alpha))


if __name__ == "__main__":
    unittest.main()
