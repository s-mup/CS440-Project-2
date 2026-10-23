import inspect
import unittest

import numpy as np

from where_is_it.detectors import DetectorA, DetectorB, Outcome
from where_is_it.ship import Ship, generate_ship
from where_is_it.strategies import BaselineA, BaselineB, Move, Sense, Strategy
from where_is_it.world import (FREE_MOVEMENT_COST, ORIGINAL_COST,
                               IllegalMoveError, World, run_trial,
                               sample_scenario)

SMALL = Ship.from_ascii("""
    ....#
    .#...
    ...#.
    #....
""")
CORRIDOR = Ship(np.ones((1, 7), dtype=bool))


def rngs(seed):
    det_ss, strat_ss = np.random.SeedSequence(seed).spawn(2)
    return dict(detector_rng=np.random.default_rng(det_ss),
                strategy_rng=np.random.default_rng(strat_ss))


class Tracing(Strategy):
    """Wraps a strategy and records every action it takes."""

    def __init__(self, inner):
        self.inner, self.trace = inner, []

    def reset(self, ship, detector, start, rng):
        self.trace = []
        self.inner.reset(ship, detector, start, rng)

    def next_action(self, position):
        action = self.inner.next_action(position)
        self.trace.append(action)
        return action

    def observe(self, position, outcome):
        self.inner.observe(position, outcome)


class Pacer(Strategy):
    """Faulty strategy: walks back and forth forever and never senses."""

    def reset(self, ship, detector, start, rng):
        self.ship = ship

    def next_action(self, position):
        return Move(self.ship.neighbors[position][0])

    def observe(self, position, outcome):
        pass


class TestWorld(unittest.TestCase):
    def test_initialisation(self):
        w = World(SMALL, DetectorA(1), 0, 5, np.random.default_rng(0))
        self.assertEqual((w.position, w.moves, w.senses, w.found), (0, 0, 0, False))
        with self.assertRaises(ValueError):
            World(SMALL, DetectorA(1), 0, SMALL.n, np.random.default_rng(0))

    def test_move_counting_and_validity(self):
        w = World(CORRIDOR, DetectorA(0), 0, 6, np.random.default_rng(0))
        w.move(1)
        w.move(2)
        w.move(1)
        self.assertEqual((w.position, w.moves), (1, 3))
        with self.assertRaises(IllegalMoveError):
            w.move(5)                       # not adjacent
        with self.assertRaises(IllegalMoveError):
            w.move(1)                       # staying put is not a move

    def test_blocked_cells_not_enterable(self):
        start = SMALL.index_of((0, 1))
        w = World(SMALL, DetectorA(0), start, 0, np.random.default_rng(0))
        legal = {SMALL.cell_of(i) for i in SMALL.neighbors[start]}
        self.assertEqual(legal, {(0, 0), (0, 2)})   # (1,1) is blocked

    def test_sense_counting_and_found_terminates(self):
        w = World(CORRIDOR, DetectorA(0), 0, 1, np.random.default_rng(0))
        self.assertIs(w.sense(), Outcome.NO_PING)
        w.move(1)
        self.assertIs(w.sense(), Outcome.FOUND)
        self.assertTrue(w.found)
        self.assertEqual((w.moves, w.senses), (1, 2))
        with self.assertRaises(RuntimeError):
            w.sense()

    def test_cost_models(self):
        self.assertEqual(ORIGINAL_COST.cost(moves=3, senses=4), 5.0)
        self.assertEqual(FREE_MOVEMENT_COST.cost(moves=3, senses=4), 4.0)


class TestRunTrial(unittest.TestCase):
    def test_baseline_a_finds_every_location(self):
        for k in [0, 1, 2, 5]:
            for lostbot in range(SMALL.n):
                if lostbot == 0:
                    continue
                r = run_trial(SMALL, DetectorA(k), BaselineA(), 0, lostbot, **rngs(k))
                self.assertTrue(r.found)
                self.assertEqual(r.cost, r.moves + 0.5 * r.senses)
                self.assertLessEqual(r.senses, SMALL.n)

    def test_baseline_b_finds_every_location(self):
        for alpha in [0.05, 0.5, 3.0]:
            for lostbot in range(1, SMALL.n):
                r = run_trial(SMALL, DetectorB(alpha), BaselineB(), 0, lostbot,
                              **rngs(lostbot))
                self.assertTrue(r.found)
                # Baseline B always senses at a not-yet-eliminated cell, and
                # each reading eliminates that cell, so senses <= n.
                self.assertLessEqual(r.senses, SMALL.n)

    def test_cost_model_applied(self):
        r1 = run_trial(SMALL, DetectorA(1), BaselineA(), 0, 10, **rngs(0))
        r2 = run_trial(SMALL, DetectorA(1), BaselineA(), 0, 10, **rngs(0),
                       cost_model=FREE_MOVEMENT_COST)
        self.assertEqual((r1.moves, r1.senses), (r2.moves, r2.senses))
        self.assertEqual(r2.cost, r2.senses)

    def test_reproducible_trials(self):
        ship = generate_ship(21, np.random.default_rng(5))
        start, lostbot = sample_scenario(ship, np.random.default_rng(6))
        for det, strat in [(DetectorA(2), BaselineA), (DetectorB(0.2), BaselineB)]:
            a = run_trial(ship, det, strat(), start, lostbot, **rngs(9))
            b = run_trial(ship, det, strat(), start, lostbot, **rngs(9))
            self.assertEqual(a, b)

    def test_action_cap_prevents_infinite_loop(self):
        r = run_trial(CORRIDOR, DetectorA(0), Pacer(), 0, 6, **rngs(0),
                      max_actions=500)
        self.assertFalse(r.found)
        self.assertTrue(r.hit_action_cap)
        self.assertEqual(r.moves, 500)

    def test_scenarios_distinct_and_reproducible(self):
        s1 = [sample_scenario(SMALL, g) for g in [np.random.default_rng(3)] * 50]
        s2 = [sample_scenario(SMALL, g) for g in [np.random.default_rng(3)] * 50]
        self.assertEqual(s1, s2)
        self.assertTrue(all(a != b for a, b in s1))


class TestInformationHiding(unittest.TestCase):
    def test_strategy_api_never_receives_lostbot(self):
        for method in ("reset", "next_action", "observe"):
            params = inspect.signature(getattr(Strategy, method)).parameters
            self.assertFalse(any("lost" in p for p in params))
        strat = BaselineB()
        run_trial(SMALL, DetectorB(0.3), strat, 0, 9, **rngs(0))
        self.assertFalse(any(isinstance(v, World) for v in vars(strat).values()))

    def test_actions_depend_only_on_observations(self):
        # With k = 0, every reading before discovery is NO_PING wherever the
        # LostBot is. If the strategy cannot see the hidden location, the run
        # that finds the LostBot earlier must be an exact prefix of the other.
        traces = {}
        for lostbot in (3, 11):
            t = Tracing(BaselineA())
            run_trial(SMALL, DetectorA(0), t, 0, lostbot, **rngs(1))
            traces[lostbot] = t.trace
        short, long_ = sorted(traces.values(), key=len)
        self.assertLess(len(short), len(long_))
        self.assertEqual(short, long_[:len(short)])


class TestBaselineBehaviour(unittest.TestCase):
    def test_baseline_a_moves_to_nearest_candidate(self):
        t = Tracing(BaselineA())
        run_trial(CORRIDOR, DetectorA(0), t, 3, 0, **rngs(2))
        self.assertIsInstance(t.trace[0], Sense)
        self.assertIn(t.trace[1], (Move(2), Move(4)))   # distance-1 candidates
        self.assertIsInstance(t.trace[2], Sense)         # sense on arrival

    def test_baseline_a_walks_full_path_before_sensing(self):
        # k = 1 at the left end of a corridor: NO_PING clears cells 0 and 1,
        # so the nearest candidate is cell 2, two moves away.
        t = Tracing(BaselineA())
        run_trial(CORRIDOR, DetectorA(1), t, 0, 6, **rngs(0))
        self.assertEqual(t.trace[:4], [Sense(), Move(1), Move(2), Sense()])

    def test_baseline_b_targets_most_likely_then_nearest(self):
        ship = Ship(np.ones((7, 7), dtype=bool))
        strat = BaselineB()
        strat.reset(ship, DetectorB(0.5), ship.index_of((3, 3)),
                    np.random.default_rng(0))
        a = ship.index_of((3, 3))
        strat.observe(a, Outcome.PING)
        target = strat.plan[-1]
        best = strat.kb.most_likely()
        self.assertIn(target, best)
        self.assertEqual(ship.bfs_dist[a, target], ship.bfs_dist[a, best].min())


if __name__ == "__main__":
    unittest.main()
