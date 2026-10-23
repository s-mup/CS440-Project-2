import math
import unittest

import numpy as np

from where_is_it.detectors import DetectorA, DetectorB, Outcome
from where_is_it.knowledge_base import (BeliefKB, CandidateSetKB,
                                        InconsistentObservationError)
from where_is_it.ship import Ship

OPEN5 = Ship(np.ones((5, 5), dtype=bool))


def cells(ship, mask):
    return {ship.cell_of(i) for i in np.flatnonzero(mask)}


class TestCandidateSetKB(unittest.TestCase):
    def setUp(self):
        self.kb = CandidateSetKB(OPEN5, DetectorA(1))
        self.corner = OPEN5.index_of((0, 0))

    def test_no_ping_eliminates_square(self):
        # Worked example from the design: 25 -> 21 candidates.
        self.kb.update(self.corner, Outcome.NO_PING)
        self.assertEqual(self.kb.count, 21)
        for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            self.assertFalse(self.kb.is_possible(OPEN5.index_of(c)))

    def test_ping_intersects_square_and_removes_sensing_cell(self):
        self.kb.update(self.corner, Outcome.PING)
        self.assertEqual(cells(OPEN5, self.kb.possible), {(0, 1), (1, 0), (1, 1)})

    def test_two_pings_intersect(self):
        self.kb.update(OPEN5.index_of((1, 1)), Outcome.PING)
        self.kb.update(OPEN5.index_of((2, 2)), Outcome.PING)
        # Square(1,1) ∩ Square(2,2), minus both sensing cells.
        self.assertEqual(cells(OPEN5, self.kb.possible),
                         {(1, 2), (2, 1)})

    def test_repeated_reading_changes_nothing(self):
        self.kb.update(self.corner, Outcome.NO_PING)
        before = self.kb.possible.copy()
        self.kb.update(self.corner, Outcome.NO_PING)
        np.testing.assert_array_equal(before, self.kb.possible)

    def test_excluded_start_and_found(self):
        kb = CandidateSetKB(OPEN5, DetectorA(1), excluded=[self.corner])
        self.assertEqual(kb.count, 24)
        target = OPEN5.index_of((3, 3))
        kb.update(target, Outcome.FOUND)
        self.assertEqual(kb.candidates().tolist(), [target])

    def test_inconsistent_observation_raises(self):
        only_far = [i for i in range(OPEN5.n) if i != OPEN5.index_of((4, 4))]
        kb = CandidateSetKB(OPEN5, DetectorA(1), excluded=only_far)
        with self.assertRaises(InconsistentObservationError):
            kb.update(self.corner, Outcome.PING)
        with self.assertRaises(InconsistentObservationError):
            kb.update(self.corner, Outcome.FOUND)

    def test_requires_detector_a(self):
        with self.assertRaises(TypeError):
            CandidateSetKB(OPEN5, DetectorB(0.5))


class TestBeliefKB(unittest.TestCase):
    # Cells (0,0),(0,1),(0,2),(0,4): from a=(0,0) the others are at distance 1, 2, 4.
    GAP = Ship.from_ascii("...#.")

    def test_ping_matches_worked_example(self):
        kb = BeliefKB(self.GAP, DetectorB(0.5))
        a = self.GAP.index_of((0, 0))
        kb.update(a, Outcome.PING)
        p = kb.probabilities()
        self.assertEqual(p[a], 0.0)
        np.testing.assert_allclose(p[1:], [0.5466, 0.3315, 0.1220], atol=5e-4)

    def test_no_ping_matches_worked_example(self):
        kb = BeliefKB(self.GAP, DetectorB(0.5))
        a = self.GAP.index_of((0, 0))
        kb.update(a, Outcome.NO_PING)
        p = kb.probabilities()
        np.testing.assert_allclose(p, [0.0, 0.0, 0.3362, 0.6638], atol=5e-4)

    def test_update_equals_bayes_rule(self):
        det = DetectorB(0.4)
        kb = BeliefKB(OPEN5, det)
        a = OPEN5.index_of((1, 3))
        prior = kb.probabilities()
        kb.update(a, Outcome.PING)
        lik = np.array([0.0 if b == a else
                        det.ping_probability(OPEN5.cell_of(a), OPEN5.cell_of(b))
                        for b in range(OPEN5.n)])
        expected = prior * lik / (prior * lik).sum()
        np.testing.assert_allclose(kb.probabilities(), expected, rtol=1e-12)

    def test_normalised_after_many_updates(self):
        det, rng = DetectorB(0.3), np.random.default_rng(0)
        kb = BeliefKB(OPEN5, det)
        true_b = OPEN5.index_of((4, 1))
        for _ in range(60):
            a = int(rng.integers(OPEN5.n))
            if a == true_b:
                continue
            kb.update(a, det.sample(OPEN5.cell_of(a), OPEN5.cell_of(true_b), rng))
            self.assertAlmostEqual(kb.probabilities().sum(), 1.0, places=12)
        self.assertTrue(kb.possible()[true_b])

    def test_elimination_of_sensed_cell_and_neighbours(self):
        kb = BeliefKB(OPEN5, DetectorB(0.5))
        a = OPEN5.index_of((2, 2))
        kb.update(a, Outcome.NO_PING)
        p = kb.probabilities()
        self.assertEqual(p[a], 0.0)
        for nb in OPEN5.neighbors[a]:
            self.assertEqual(p[nb], 0.0)
        # Eliminated cells stay eliminated, even after a ping.
        kb.update(OPEN5.index_of((0, 0)), Outcome.PING)
        self.assertEqual(kb.probabilities()[a], 0.0)

    def test_repeated_pings_concentrate_belief(self):
        kb = BeliefKB(OPEN5, DetectorB(0.5))
        a = OPEN5.index_of((2, 2))
        near = OPEN5.index_of((2, 3))
        history = []
        for _ in range(5):
            kb.update(a, Outcome.PING)
            history.append(kb.probabilities()[near])
        self.assertTrue(all(x < y for x, y in zip(history, history[1:])))

    def test_numerical_stability_under_extreme_likelihoods(self):
        corridor = Ship(np.ones((1, 200), dtype=bool))
        kb = BeliefKB(corridor, DetectorB(3.0))
        a = corridor.index_of((0, 0))
        for _ in range(300):
            kb.update(a, Outcome.PING)
        # Far cell: likelihood e^{-3*198} per ping, so ~e^{-178000} in linear
        # space, which would underflow to 0. Log space keeps it nonzero.
        self.assertTrue(np.isfinite(kb.log_p[-1]))
        self.assertTrue(np.isfinite(kb.log_p[1:]).all())
        self.assertFalse(np.isnan(kb.log_p).any())
        self.assertAlmostEqual(kb.probabilities().sum(), 1.0, places=12)

    def test_tiny_alpha_no_ping_is_handled(self):
        corridor = Ship(np.ones((1, 30), dtype=bool))
        kb = BeliefKB(corridor, DetectorB(1e-8))
        kb.update(0, Outcome.NO_PING)  # P(no ping) ~ 1e-8 * (d - 1)
        p = kb.probabilities()
        self.assertAlmostEqual(p.sum(), 1.0, places=12)
        self.assertGreater(p[-1], p[2])  # farther cells more likely

    def test_impossible_observation_raises(self):
        a = OPEN5.index_of((2, 2))
        nb = OPEN5.index_of((2, 3))
        kb = BeliefKB(OPEN5, DetectorB(0.5),
                      excluded=[i for i in range(OPEN5.n) if i != nb])
        with self.assertRaises(InconsistentObservationError):
            kb.update(a, Outcome.NO_PING)   # neighbours always ping

    def test_most_likely_ties_and_entropy(self):
        kb = BeliefKB(OPEN5, DetectorB(0.5), excluded=[0])
        self.assertEqual(len(kb.most_likely()), 24)
        self.assertAlmostEqual(kb.entropy(), math.log2(24))
        kb.update(OPEN5.index_of((2, 2)), Outcome.PING)
        best = {OPEN5.cell_of(i) for i in kb.most_likely()}
        self.assertEqual(best, {(1, 2), (3, 2), (2, 1), (2, 3)})


if __name__ == "__main__":
    unittest.main()
