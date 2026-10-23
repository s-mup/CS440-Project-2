# Project 2: Where is it? (CS 01:198:440, Fall 2026)

A LocatorBot searches a ship for a stationary LostBot using one of two detectors.
It tries to minimise **cost = moves + ½ · detector uses**.

## Setup

Requires Python 3.10+.

```
pip install -r requirements.txt
```

## Running

```
# Baseline A, k = 3, five trials on a 41x41 ship (master seed 0)
python main.py demo --detector A --k 3

# Baseline B, alpha = 0.1, 20 trials, different seed
python main.py demo --detector B --alpha 0.1 --trials 20 --seed 7

# Small ship with the map printed (S = start, L = LostBot)
python main.py demo --detector A --k 2 --D 15 --show

# Report cost as detector uses only (free movement)
python main.py demo --detector B --alpha 0.2 --free-movement
```

## Reproducing the experiments

```
# Tune on separate seeds, then run every sweep (41x41 ships, 200 paired trials
# per point). Resumable: finished tasks are cached in results/cache/.
# On a 15 W laptop CPU this took about 6.2 hours (tuning 1.7 h + sweeps 4.4 h).
python main.py experiment --stage all --processes 6

# Summary tables (CSV + Markdown) and all figures, from the saved CSVs
python main.py analyze

# Tiny end-to-end check of the pipeline (about 30 s)
python main.py experiment --quick --out results_quick
```

Outputs in `results/`: per-trial CSVs (`tuning.csv`, `type_a.csv`, `type_b.csv`),
`tuned_params.json`, summary CSVs, `summary.md`, and `figures/fig1..fig7*.png`.

## Tests

The tests use only the standard library's `unittest`; nothing extra needs installing.

```
python -m unittest discover -s tests -v
```

## Layout

| Module | Responsibility |
|---|---|
| `where_is_it/ship.py` | Grid, open-cell indexing, cached pairwise BFS/Euclidean/Chebyshev distances, Project 1 ship generator |
| `where_is_it/pathfinding.py` | BFS distances, all-pairs BFS, shortest-path reconstruction |
| `where_is_it/detectors.py` | Detector models A (square, radius k) and B (`exp(-α(d-1))`); outcomes FOUND / PING / NO_PING |
| `where_is_it/knowledge_base.py` | `CandidateSetKB` (Type A set) and `BeliefKB` (Type B log-space Bayesian belief) |
| `where_is_it/world.py` | Hidden LostBot location, action execution, cost counters, `run_trial`, scenario sampling |
| `where_is_it/scoring.py` | Vectorised evaluation of every sensing cell: P(found / ping / no ping), expected posterior entropy, info gain, expected distance |
| `where_is_it/strategies/` | `base.py` (interface), `baseline_a/b.py`, `improved_a/b.py` (main strategies + ablations), `scored.py` (shared objective), `free_movement_a/b.py` |
| `where_is_it/experiments.py` | Seeded, paired, parallel, resumable experiment runner |
| `where_is_it/analysis.py` | Statistics, paired comparisons, best configurations, detector-information profile |
| `where_is_it/visualization.py` | Figures 1-7 |
| `main.py` | Command-line interface |
| `tests/` | Unit tests |

### How the modules interact

```
run_trial ──creates──> World (holds hidden LostBot + detector RNG)
    │                      ▲
    │  next_action(pos)    │ Move / Sense executed here
    ▼                      │
Strategy ──owns──> KnowledgeBase ──uses──> Detector model (public formulas)
    └── uses ship.bfs_dist + pathfinding.shortest_path to plan moves
```

A strategy never receives the `World`. It only sees the map, its own position,
and the outcome of each reading it asks for.

## Modelling decisions

* **Ship.** Generated with the Project 1 algorithm: grow a maze from one cell,
  then open one neighbour of half of the dead ends. Movement is 4-connected.
  The default size is D = 41.
* **Placement.** The start cell and the LostBot cell are distinct and uniformly
  random. Both knowledge bases start uniform over every open cell except the start.
* **Baselines.** "Move to" the target means the bot walks the whole BFS
  shortest path (1 cost per step) and then senses (0.5 cost).
* **Detector A.** Pings when the LostBot is within Chebyshev distance k. A
  reading taken on the LostBot's own cell reports FOUND.
* **Detector B.** Readings are independent given the LostBot's cell. The update
  is done in log space with log-sum-exp normalisation. Detector B uses
  Euclidean distance; movement always uses BFS distance.
* **Termination.** Each trial has a generous action cap
  (10 · n · (diameter + 1)). Both baselines provably finish well within it.

## Reproducibility

One master seed is split with `numpy.random.SeedSequence.spawn` into independent
streams for the ship, the scenarios, and, per trial, the detector noise and the
strategy's tie-breaking. Every strategy compared on the same scenario gets the
same ship, start, and LostBot cell.
