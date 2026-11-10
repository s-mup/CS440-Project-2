"""Reproducible, parallel experiment runner.

Randomness
----------
Every random stream is derived from fixed integers with numpy's SeedSequence:
    ship i          : SeedSequence([seed, 0, i])
    scenarios on i  : SeedSequence([seed, 1, i])  -> (start, LostBot) pairs
    trial (i, j)    : SeedSequence([seed, 2, i, j]).spawn(2) -> detector noise,
                                                              strategy tie-breaks
For a given scenario, every strategy and parameter value gets the same ship, the
same start and LostBot cells, and the same detector-noise stream ("common random
numbers"), so comparisons between strategies are paired.

Tuning runs use a different master seed from evaluation runs, so the tuned
hyper-parameters are never scored on the trials they were chosen with.

Each trial row stores both the moves and the readings, so both cost models
(original, and free movement) can be computed from the same runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .detectors import DetectorA, DetectorB
from .ship import generate_ship
from .strategies import (BaselineA, BaselineB, FreeMovementA, FreeMovementB,
                         ImprovedA, ImprovedB, InfoPerCostA, LookaheadA,
                         LookaheadB, ProbPerCostB, SlidesPureA, SlidesPureB)
from .world import FREE_MOVEMENT_COST, ORIGINAL_COST, run_trial

K_VALUES = [0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40]
ALPHA_VALUES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0]

# Hyper-parameter grids for the main improved strategies (tuned per k / alpha).
TUNE_GRID_A = [{"c": c, "gamma": g} for c in (1, 3, 10, 30, 100) for g in (0.0, 1.0)]
TUNE_GRID_B = [{"c": c} for c in (1, 3, 10, 30, 100)]

STRATEGIES = {
    "baseline_a": BaselineA,
    "improved_a": ImprovedA,
    "lookahead_a": LookaheadA,
    "slides_pure_a": SlidesPureA,
    "info_per_cost_a": InfoPerCostA,
    "free_movement_a": FreeMovementA,
    "baseline_b": BaselineB,
    "improved_b": ImprovedB,
    "lookahead_b": LookaheadB,
    "slides_pure_b": SlidesPureB,
    "prob_per_cost_b": ProbPerCostB,
    "free_movement_b": FreeMovementB,
}


@dataclass(frozen=True)
class Setup:
    D: int = 41
    n_ships: int = 10
    scenarios_per_ship: int = 20
    seed: int = 2026
    max_actions: int = 100_000

    @property
    def trials(self) -> int:
        return self.n_ships * self.scenarios_per_ship


EVAL_SETUP = Setup()
TUNE_SETUP = Setup(n_ships=3, scenarios_per_ship=8, seed=7777)


def make_detector(kind: str, param):
    return DetectorA(int(param)) if kind == "A" else DetectorB(float(param))


def make_ship(setup: Setup, ship_id: int):
    return generate_ship(setup.D, np.random.default_rng(
        np.random.SeedSequence([setup.seed, 0, ship_id])))


def make_scenarios(setup: Setup, ship, ship_id: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(np.random.SeedSequence([setup.seed, 1, ship_id]))
    out = []
    for _ in range(setup.scenarios_per_ship):
        start, lostbot = rng.choice(ship.n, size=2, replace=False)
        out.append((int(start), int(lostbot)))
    return out


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
_SHIP_CACHE: dict = {}


def _get_ship(setup: Setup, ship_id: int):
    key = (setup.D, setup.seed, ship_id)
    if key not in _SHIP_CACHE:
        if len(_SHIP_CACHE) >= 1:   # one ship per worker keeps memory low
            _SHIP_CACHE.pop(next(iter(_SHIP_CACHE)))
        ship = make_ship(setup, ship_id)
        _ = ship.bfs_dist
        _SHIP_CACHE[key] = (ship, make_scenarios(setup, ship, ship_id))
    return _SHIP_CACHE[key]


def run_task(task: dict) -> list[dict]:
    """Run one (strategy, parameter, ship) cell over all scenarios on that ship."""
    setup = Setup(**task["setup"])
    ship, scenarios = _get_ship(setup, task["ship_id"])
    detector = make_detector(task["kind"], task["param"])
    kwargs = task.get("kwargs", {})
    rows = []
    for j, (start, lostbot) in enumerate(scenarios):
        det_ss, strat_ss = np.random.SeedSequence(
            [setup.seed, 2, task["ship_id"], j]).spawn(2)
        strategy = STRATEGIES[task["strategy"]](**kwargs)
        r = run_trial(ship, detector, strategy, start, lostbot,
                      detector_rng=np.random.default_rng(det_ss),
                      strategy_rng=np.random.default_rng(strat_ss),
                      max_actions=setup.max_actions)
        rows.append({
            "experiment": task["experiment"],
            "D": setup.D, "seed": setup.seed,
            "ship_id": task["ship_id"], "n_open": ship.n, "scenario_id": j,
            "detector": task["kind"], "param": task["param"],
            "strategy": task["strategy"],
            "hyper": json.dumps(kwargs, sort_keys=True),
            "start_r": r.start[0], "start_c": r.start[1],
            "lost_r": r.lostbot[0], "lost_c": r.lostbot[1],
            "start_lost_bfs": int(ship.bfs_dist[start, lostbot]),
            "found": int(r.found), "capped": int(r.hit_action_cap),
            "moves": r.moves, "senses": r.senses,
            "cost_original": ORIGINAL_COST.cost(r.moves, r.senses),
            "cost_free": FREE_MOVEMENT_COST.cost(r.moves, r.senses),
        })
    return rows


def _init_worker():
    # One BLAS thread per process: parallelism comes from the process pool.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _task_key(task: dict) -> str:
    return hashlib.sha1(json.dumps(task, sort_keys=True).encode()).hexdigest()[:20]


def _run_task_keyed(task: dict) -> tuple[str, list[dict]]:
    return _task_key(task), run_task(task)


def run_tasks(tasks: list[dict], processes: int | None = None,
              progress: bool = True, cache_dir: Path | None = None) -> list[dict]:
    """Run tasks in parallel. If ``cache_dir`` is given, each finished task's
    rows are saved there (keyed by a hash of the task), and tasks that are
    already cached are not rerun. An interrupted run therefore resumes where
    it stopped. Results are deterministic, so cached and fresh rows are
    interchangeable."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    processes = processes or max(1, (os.cpu_count() or 2) - 1)
    rows: list[dict] = []
    todo = []
    for t in tasks:
        f = cache_dir / f"{_task_key(t)}.json" if cache_dir else None
        if f is not None and f.exists():
            rows.extend(json.loads(f.read_text()))
        else:
            todo.append(t)
    # Group by ship: concurrent workers then mostly share the ship they have
    # already built, so each worker needs only one ship in memory.
    todo.sort(key=lambda t: (t["ship_id"], t["kind"], float(t["param"])))
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {len(tasks) - len(todo)} tasks cached, {len(todo)} to run", flush=True)

    def done(i, key, out):
        rows.extend(out)
        if cache_dir:
            (cache_dir / f"{key}.json").write_text(json.dumps(out))
        if progress and ((i + 1) % max(1, len(todo) // 20) == 0 or i + 1 == len(todo)):
            print(f"  {i + 1}/{len(todo)} tasks", flush=True)

    if processes == 1:
        for i, t in enumerate(todo):
            done(i, *_run_task_keyed(t))
        return rows
    with get_context("spawn").Pool(processes, initializer=_init_worker) as pool:
        for i, (key, out) in enumerate(pool.imap_unordered(_run_task_keyed, todo)):
            done(i, key, out)
    return rows


def build_tasks(setup: Setup, experiment: str, kind: str, params, strategy: str,
                kwargs_for=lambda p: {}) -> list[dict]:
    """One task per (parameter, ship); the expensive strategies go first, so
    the slowest tasks start early."""
    return [{"setup": asdict(setup), "experiment": experiment, "kind": kind,
             "param": p, "strategy": strategy, "kwargs": kwargs_for(p),
             "ship_id": i}
            for p in params for i in range(setup.n_ships)]


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["strategy"], r["hyper"], float(r["param"]),
                                       r["ship_id"], r["scenario_id"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("D", "seed", "ship_id", "n_open", "scenario_id", "start_r",
                  "start_c", "lost_r", "lost_c", "start_lost_bfs", "found",
                  "capped", "moves", "senses"):
            r[k] = int(r[k])
        for k in ("param", "cost_original", "cost_free"):
            r[k] = float(r[k])
    return rows


# --------------------------------------------------------------------------- #
# Experiment definitions
# --------------------------------------------------------------------------- #
def tune(out_dir: Path, setup: Setup = TUNE_SETUP, processes=None,
         k_values=K_VALUES, alpha_values=ALPHA_VALUES) -> dict:
    """Grid-search (c, gamma) per k for Type A and c per alpha for Type B, on
    tuning seeds only, separately for the final strategies (improved_*, the
    slides' planner) and for the earlier rule they replaced (lookahead_*).
    The criterion is mean original cost; a capped trial counts at its capped
    cost, which is a heavy penalty."""
    tasks = []
    for strat in ("improved", "lookahead"):
        for kw in TUNE_GRID_B:
            tasks += build_tasks(setup, "tune", "B", alpha_values, f"{strat}_b",
                                 lambda p, kw=kw: kw)
        for kw in TUNE_GRID_A:
            tasks += build_tasks(setup, "tune", "A", k_values, f"{strat}_a",
                                 lambda p, kw=kw: kw)
    print(f"tuning: {len(tasks)} tasks, {len(tasks) * setup.scenarios_per_ship} trials")
    rows = run_tasks(tasks, processes, cache_dir=out_dir / "cache")
    write_csv(rows, out_dir / "tuning.csv")
    select_tuned([r for r in rows if r["strategy"].startswith("lookahead")],
                 out_dir / "tuned_params_lookahead.json")
    return select_tuned([r for r in rows if r["strategy"].startswith("improved")],
                        out_dir / "tuned_params.json")


def select_tuned(rows: list[dict], path: Path) -> dict:
    groups: dict = {}
    for r in rows:
        key = (r["detector"], r["param"], r["hyper"])
        groups.setdefault(key, []).append(r["cost_original"])
    best: dict = {"A": {}, "B": {}}
    for (kind, param, hyper), costs in groups.items():
        m = float(np.mean(costs))
        cur = best[kind].get(str(param))
        if cur is None or m < cur["mean_cost"]:
            best[kind][str(param)] = {"kwargs": json.loads(hyper), "mean_cost": m}
    with open(path, "w") as f:
        json.dump(best, f, indent=2, sort_keys=True)
    return best


def evaluate(out_dir: Path, tuned: dict, setup: Setup = EVAL_SETUP,
             processes=None, k_values=K_VALUES, alpha_values=ALPHA_VALUES) -> None:
    """Main sweeps. Every strategy runs on the same scenarios; both cost models
    are stored per trial (Question 6 reuses these runs)."""
    with open(out_dir / "tuned_params_lookahead.json") as f:
        tuned_look = json.load(f)

    def tuned_kw(kind, table=tuned):
        return lambda p: table[kind][str(p)]["kwargs"]

    tasks_b = (build_tasks(setup, "type_b", "B", alpha_values, "improved_b", tuned_kw("B"))
               + build_tasks(setup, "type_b", "B", alpha_values, "free_movement_b")
               + build_tasks(setup, "type_b", "B", alpha_values, "baseline_b")
               + build_tasks(setup, "type_b", "B", alpha_values, "prob_per_cost_b")
               + build_tasks(setup, "type_b", "B", alpha_values, "lookahead_b",
                             tuned_kw("B", tuned_look))
               + build_tasks(setup, "type_b", "B", alpha_values, "slides_pure_b"))
    tasks_a = (build_tasks(setup, "type_a", "A", k_values, "improved_a", tuned_kw("A"))
               + build_tasks(setup, "type_a", "A", k_values, "info_per_cost_a")
               + build_tasks(setup, "type_a", "A", k_values, "free_movement_a")
               + build_tasks(setup, "type_a", "A", k_values, "baseline_a")
               + build_tasks(setup, "type_a", "A", k_values, "lookahead_a",
                             tuned_kw("A", tuned_look))
               + build_tasks(setup, "type_a", "A", k_values, "slides_pure_a"))
    for name, tasks in (("type_a", tasks_a), ("type_b", tasks_b)):
        print(f"{name}: {len(tasks)} tasks, {len(tasks) * setup.scenarios_per_ship} trials")
        write_csv(run_tasks(tasks, processes, cache_dir=out_dir / "cache"),
                  out_dir / f"{name}.csv")
