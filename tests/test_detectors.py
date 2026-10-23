import math
import unittest

import numpy as np

from where_is_it.detectors import DetectorA, DetectorB, Outcome
from where_is_it.ship import Ship

OPEN5 = Ship(np.ones((5, 5), dtype=bool))


class TestDetectorA(unittest.TestCase):
    def test_k0_only_reports_found(self):
        det = DetectorA(0)
        self.assertIs(det.sample((2, 2), (2, 2)), Outcome.FOUND)
        for b in [(2, 3), (3, 3), (0, 0)]:
            self.assertIs(det.sample((2, 2), b), Outcome.NO_PING)

    def test_ping_inside_and_on_boundary(self):
        det = DetectorA(2)
        for b in [(5, 6), (6, 6), (7, 7), (3, 3), (7, 3), (5, 3)]:
            self.assertIs(det.sample((5, 5), b), Outcome.PING, b)

    def test_no_ping_outside(self):
        det = DetectorA(2)
        for b in [(8, 5), (5, 2), (8, 8), (2, 7), (0, 0)]:
            self.assertIs(det.sample((5, 5), b), Outcome.NO_PING, b)

    def test_coverage_clipped_at_ship_edges(self):
        a = OPEN5.index_of((0, 0))
        self.assertEqual(DetectorA(1).coverage(OPEN5, a).sum(), 4)
        self.assertEqual(DetectorA(4).coverage(OPEN5, a).sum(), 25)
        centre = OPEN5.index_of((2, 2))
        self.assertEqual(DetectorA(1).coverage(OPEN5, centre).sum(), 9)

    def test_coverage_ignores_blocked_cells(self):
        ship = Ship.from_ascii("""
            .#.
            ###
            ...
        """)
        a = ship.index_of((0, 0))
        self.assertEqual(DetectorA(1).coverage(ship, a).sum(), 1)  # only a

    def test_invalid_k(self):
        for k in [-1, 1.5, True, "2"]:
            with self.assertRaises(ValueError):
                DetectorA(k)


class TestDetectorB(unittest.TestCase):
    def test_euclidean_distance(self):
        self.assertEqual(DetectorB.distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(DetectorB.distance((1, 1), (2, 2)), math.sqrt(2))

    def test_ping_probability_formula(self):
        det = DetectorB(0.5)
        self.assertAlmostEqual(det.ping_probability((0, 0), (3, 4)), math.exp(-2.0))
        self.assertEqual(det.ping_probability((0, 0), (0, 1)), 1.0)
        self.assertAlmostEqual(det.ping_probability((0, 0), (1, 1)),
                               math.exp(-0.5 * (math.sqrt(2) - 1)))

    def test_probability_in_unit_interval(self):
        rng = np.random.default_rng(0)
        for alpha in [1e-3, 0.05, 1.0, 10.0, 200.0]:
            det = DetectorB(alpha)
            for _ in range(200):
                a, b = map(tuple, rng.integers(0, 60, size=(2, 2)))
                if a != b:
                    p = det.ping_probability(a, b)
                    self.assertTrue(0.0 <= p <= 1.0)

    def test_alpha_validation(self):
        for alpha in [0, -0.1, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                DetectorB(alpha)

    def test_same_cell_reports_found_not_probability(self):
        det = DetectorB(0.3)
        with self.assertRaises(ValueError):
            det.ping_probability((4, 4), (4, 4))
        rng = np.random.default_rng(0)
        for _ in range(20):
            self.assertIs(det.sample((4, 4), (4, 4), rng), Outcome.FOUND)

    def test_adjacent_always_pings(self):
        det, rng = DetectorB(5.0), np.random.default_rng(0)
        self.assertTrue(all(det.sample((3, 3), (3, 4), rng) is Outcome.PING
                            for _ in range(1000)))

    def test_fixed_seed_reproducibility(self):
        det = DetectorB(0.2)
        g1, g2 = np.random.default_rng(42), np.random.default_rng(42)
        s1 = [det.sample((0, 0), (5, 5), g1) for _ in range(300)]
        s2 = [det.sample((0, 0), (5, 5), g2) for _ in range(300)]
        self.assertEqual(s1, s2)
        self.assertEqual(len(set(s1)), 2)  # both outcomes actually occur

    def test_empirical_frequency_matches_formula(self):
        det, rng = DetectorB(0.3), np.random.default_rng(7)
        p = det.ping_probability((0, 0), (0, 3))  # exp(-0.6) ~ 0.549
        hits = sum(det.sample((0, 0), (0, 3), rng) is Outcome.PING
                   for _ in range(20000))
        self.assertAlmostEqual(hits / 20000, p, delta=0.02)

    def test_log_likelihood_vectors(self):
        det = DetectorB(0.7)
        a = OPEN5.index_of((2, 2))
        ll_p = det.log_likelihood(OPEN5, a, Outcome.PING)
        ll_n = det.log_likelihood(OPEN5, a, Outcome.NO_PING)
        p = det.ping_probabilities(OPEN5, a)
        for b in range(OPEN5.n):
            if b == a:
                self.assertEqual(ll_p[b], -np.inf)
                self.assertEqual(ll_n[b], -np.inf)
                continue
            q = det.ping_probability(OPEN5.cell_of(a), OPEN5.cell_of(b))
            self.assertAlmostEqual(math.exp(ll_p[b]), q)
            self.assertAlmostEqual(math.exp(ll_n[b]), 1 - q)
            self.assertAlmostEqual(p[b], q)
        for nb in OPEN5.neighbors[a]:  # orthogonal neighbours: P(no ping) = 0
            self.assertEqual(ll_n[nb], -np.inf)


if __name__ == "__main__":
    unittest.main()
