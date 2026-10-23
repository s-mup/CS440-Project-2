"""Command-line entry point.

Examples:
    python main.py demo --detector A --k 3
    python main.py demo --detector B --alpha 0.1 --trials 20 --D 41 --seed 7
    python main.py demo --detector A --k 2 --D 15 --show
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

# Parallelism comes from worker processes, so each process uses one BLAS
# thread. This must be set before numpy is imported; workers inherit it.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402

from where_is_it.detectors import DetectorA, DetectorB
from where_is_it.ship import generate_ship
from where_is_it.strategies import BaselineA, BaselineB
from where_is_it.world import (FREE_MOVEMENT_COST, ORIGINAL_COST, run_trial,
                               sample_scenario)


def cmd_demo(args: argparse.Namespace) -> None:
    # Independent, reproducible random streams derived from one master seed.
    ship_ss, scenario_ss, trials_ss = np.random.SeedSequence(args.seed).spawn(3)
    ship = generate_ship(args.D, np.random.default_rng(ship_ss))

    if args.detector == "A":
        detector, strategy = DetectorA(args.k), BaselineA()
    else:
        detector, strategy = DetectorB(args.alpha), BaselineB()
    cost_model = FREE_MOVEMENT_COST if args.free_movement else ORIGINAL_COST

    t0 = time.perf_counter()
    _ = ship.bfs_dist  # one-time all-pairs BFS, shared by every trial
    print(f"{ship!r}  diameter={ship.diameter}  "
          f"(all-pairs BFS: {time.perf_counter() - t0:.2f}s)")
    print(f"{detector!r}  strategy={strategy.name}  cost model={cost_model.name}")

    scenario_rng = np.random.default_rng(scenario_ss)
    results = []
    t0 = time.perf_counter()
    for i, trial_ss in enumerate(trials_ss.spawn(args.trials)):
        start, lostbot = sample_scenario(ship, scenario_rng)
        det_ss, strat_ss = trial_ss.spawn(2)
        if args.show and i == 0:
            print(ship.render({start: "S", lostbot: "L"}))
        r = run_trial(ship, detector, strategy, start, lostbot,
                      detector_rng=np.random.default_rng(det_ss),
                      strategy_rng=np.random.default_rng(strat_ss),
                      cost_model=cost_model)
        results.append(r)
        if args.trials <= 10:
            print(f"trial {i}: start={r.start} lostbot={r.lostbot} "
                  f"found={r.found} moves={r.moves} senses={r.senses} "
                  f"cost={r.cost:g}")
    elapsed = time.perf_counter() - t0

    costs = [r.cost for r in results]
    print(f"\n{len(results)} trials in {elapsed:.2f}s | "
          f"found {sum(r.found for r in results)}/{len(results)} | "
          f"mean cost {statistics.fmean(costs):.2f} | "
          f"median {statistics.median(costs):.2f} | "
          f"std {statistics.pstdev(costs):.2f} | "
          f"mean moves {statistics.fmean(r.moves for r in results):.1f} | "
          f"mean senses {statistics.fmean(r.senses for r in results):.1f}")


def cmd_experiment(args: argparse.Namespace) -> None:
    import json
    from dataclasses import replace
    from pathlib import Path

    from where_is_it import experiments as ex

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tune_setup, eval_setup = ex.TUNE_SETUP, ex.EVAL_SETUP
    k_values, alpha_values = ex.K_VALUES, ex.ALPHA_VALUES
    if args.quick:  # small and fast, for debugging the pipeline only
        tune_setup = replace(tune_setup, D=21, n_ships=2, scenarios_per_ship=4)
        eval_setup = replace(eval_setup, D=21, n_ships=2, scenarios_per_ship=5)
        k_values, alpha_values = [0, 2, 5, 20], [0.05, 0.3, 2.0]
    if args.ships:
        eval_setup = replace(eval_setup, n_ships=args.ships)
    if args.scenarios:
        eval_setup = replace(eval_setup, scenarios_per_ship=args.scenarios)

    t0 = time.perf_counter()
    if args.stage in ("tune", "all"):
        tuned = ex.tune(out, tune_setup, args.processes, k_values, alpha_values)
        print(f"tuning done ({time.perf_counter() - t0:.0f}s)")
    else:
        with open(out / "tuned_params.json") as f:
            tuned = json.load(f)
    if args.stage in ("evaluate", "all"):
        ex.evaluate(out, tuned, eval_setup, args.processes, k_values, alpha_values)
        print(f"evaluation done ({time.perf_counter() - t0:.0f}s)")


def cmd_analyze(args: argparse.Namespace) -> None:
    from pathlib import Path

    from where_is_it import analysis, visualization

    out = Path(args.out)
    analysis.write_summaries(out)
    visualization.make_all_figures(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 2: Where is it?")
    sub = parser.add_subparsers(dest="command", required=True)

    exp = sub.add_parser("experiment", help="tune and/or run the main sweeps")
    exp.add_argument("--stage", choices=["tune", "evaluate", "all"], default="all")
    exp.add_argument("--out", default="results")
    exp.add_argument("--processes", type=int, default=None)
    exp.add_argument("--ships", type=int, default=None, help="override ship count")
    exp.add_argument("--scenarios", type=int, default=None,
                     help="override scenarios per ship")
    exp.add_argument("--quick", action="store_true", help="tiny debug run")
    exp.set_defaults(func=cmd_experiment)

    ana = sub.add_parser("analyze", help="summary tables and figures from CSVs")
    ana.add_argument("--out", default="results")
    ana.set_defaults(func=cmd_analyze)

    demo = sub.add_parser("demo", help="run a baseline strategy on random trials")
    demo.add_argument("--detector", choices=["A", "B"], default="A")
    demo.add_argument("--k", type=int, default=3, help="Type A square radius")
    demo.add_argument("--alpha", type=float, default=0.1, help="Type B decay rate")
    demo.add_argument("--D", type=int, default=41, help="ship is D x D")
    demo.add_argument("--trials", type=int, default=5)
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--free-movement", action="store_true",
                      help="report cost = number of detector actions")
    demo.add_argument("--show", action="store_true",
                      help="print the ship map for the first trial")
    demo.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
