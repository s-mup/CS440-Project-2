import argparse
import json
import tempfile
import unittest
from pathlib import Path

from main import make_demo_strategy
from where_is_it.strategies import (BaselineA, BaselineB, FreeMovementA,
                                    FreeMovementB, ImprovedA, ImprovedB)
from where_is_it.world import FREE_MOVEMENT_COST, ORIGINAL_COST


def demo_args(**kw):
    base = dict(detector="A", k=3, alpha=0.1, strategy="baseline",
                tuned="missing.json", free_movement=False)
    base.update(kw)
    return argparse.Namespace(**base)


class TestDemoStrategy(unittest.TestCase):
    def test_baselines_by_default(self):
        _, s, cost = make_demo_strategy(demo_args())
        self.assertIsInstance(s, BaselineA)
        self.assertIs(cost, ORIGINAL_COST)
        _, s, _ = make_demo_strategy(demo_args(detector="B"))
        self.assertIsInstance(s, BaselineB)

    def test_final_uses_tuned_settings(self):
        tuned = {"A": {"13": {"kwargs": {"c": 10, "gamma": 0.0}}},
                 "B": {"0.1": {"kwargs": {"c": 30}}}}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tuned.json"
            path.write_text(json.dumps(tuned))
            _, s, cost = make_demo_strategy(
                demo_args(k=13, strategy="final", tuned=str(path)))
            self.assertIsInstance(s, ImprovedA)
            self.assertEqual((s.c, s.gamma), (10.0, 0.0))
            self.assertIs(cost, ORIGINAL_COST)
            _, s, _ = make_demo_strategy(
                demo_args(detector="B", alpha=0.1, strategy="final", tuned=str(path)))
            self.assertIsInstance(s, ImprovedB)
            self.assertEqual(s.c, 30.0)

    def test_final_falls_back_to_defaults(self):
        _, s, _ = make_demo_strategy(demo_args(k=7, strategy="final"))
        self.assertIsInstance(s, ImprovedA)

    def test_free_strategy_uses_free_cost(self):
        _, s, cost = make_demo_strategy(demo_args(strategy="free"))
        self.assertIsInstance(s, FreeMovementA)
        self.assertIs(cost, FREE_MOVEMENT_COST)
        _, s, _ = make_demo_strategy(demo_args(detector="B", strategy="free"))
        self.assertIsInstance(s, FreeMovementB)


if __name__ == "__main__":
    unittest.main()
