"""Summary statistics computed from the experiment CSVs.

All numbers in the report come from the following files:
    summary_type_a.csv / summary_type_b.csv   per (strategy, k|alpha) statistics
    paired_type_a.csv  / paired_type_b.csv    paired comparison vs the baseline
    best_configs.csv                          most efficient k / alpha per strategy
    detector_information.csv                  first-reading information vs k and alpha
    summary.md                                the same tables as Markdown

Statistics: mean, median, sample std, min, max, and a 95% confidence interval
for the mean (normal approximation, 1.96 * s / sqrt(n); n = 200 per point).
Comparisons with the baseline are *paired*: each trial of strategy X shares its
ship, start, LostBot, and detector-noise seed with a baseline trial. The CI of
the mean paired difference is therefore much tighter than comparing two
independent means.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .detectors import DetectorA, DetectorB
from .experiments import EVAL_SETUP, make_ship, read_csv
from .scoring import BeliefScorer, CandidateScorer

Z95 = 1.96
COSTS = ("cost_original", "cost_free")
BASELINE = {"A": "baseline_a", "B": "baseline_b"}
LABELS = {
    "baseline_a": "Baseline A",
    "improved_a": "Improved A (slides' sampled planning + sense on the way)",
    "lookahead_a": "Tried A0 (global argmin-J lookahead)",
    "slides_pure_a": "Tried A-S (slides' planning only)",
    "info_per_cost_a": "Ablation A1 (info gain per cost)",
    "free_movement_a": "Free-movement A (max info gain)",
    "baseline_b": "Baseline B",
    "improved_b": "Improved B (slides' sampled planning + sense on the way)",
    "lookahead_b": "Tried B0 (global argmin-J lookahead)",
    "slides_pure_b": "Tried B-S (slides' planning only)",
    "prob_per_cost_b": "Ablation B3 (probability per cost)",
    "free_movement_b": "Free-movement B (max info gain)",
}


def describe(x) -> dict:
    x = np.asarray(x, dtype=float)
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = Z95 * sd / np.sqrt(len(x))
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "std": sd, "min": float(x.min()), "max": float(x.max()),
            "ci_low": float(x.mean() - half), "ci_high": float(x.mean() + half)}


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["strategy"], r["param"])].append(r)
    for v in g.values():
        v.sort(key=lambda r: (r["ship_id"], r["scenario_id"]))
    return g


def summarise(rows) -> list[dict]:
    out = []
    for (strategy, param), rs in sorted(group(rows).items()):
        rec = {"detector": rs[0]["detector"], "strategy": strategy, "param": param,
               "hyper": rs[0]["hyper"],
               "found_rate": np.mean([r["found"] for r in rs]),
               "capped": int(sum(r["capped"] for r in rs)),
               "mean_moves": np.mean([r["moves"] for r in rs]),
               "mean_senses": np.mean([r["senses"] for r in rs]),
               "std_moves": np.std([r["moves"] for r in rs], ddof=1),
               "std_senses": np.std([r["senses"] for r in rs], ddof=1),
               "mean_oracle_excess": np.mean(
                   [r["cost_original"] - (r["start_lost_bfs"] + 0.5) for r in rs])}
        for cost in COSTS:
            for k, v in describe([r[cost] for r in rs]).items():
                rec[f"{cost}_{k}"] = v
        out.append(rec)
    return out


def paired(rows, kind: str) -> list[dict]:
    g = group(rows)
    base = BASELINE[kind]
    out = []
    for (strategy, param), rs in sorted(g.items()):
        if strategy == base:
            continue
        bs = g[(base, param)]
        assert [(r["ship_id"], r["scenario_id"]) for r in rs] == \
               [(r["ship_id"], r["scenario_id"]) for r in bs], "unpaired data"
        for cost in COSTS:
            s = np.array([r[cost] for r in rs])
            b = np.array([r[cost] for r in bs])
            d = describe(b - s)               # positive = cheaper than baseline
            out.append({
                "detector": kind, "strategy": strategy, "param": param,
                "cost_model": cost, "baseline_mean": b.mean(), "strategy_mean": s.mean(),
                "improvement_pct": 100 * (b.mean() - s.mean()) / b.mean(),
                "mean_saving": d["mean"], "saving_ci_low": d["ci_low"],
                "saving_ci_high": d["ci_high"],
                "win_rate": float(np.mean(s < b)), "tie_rate": float(np.mean(s == b)),
            })
    return out


def best_configs(summaries: list[dict]) -> list[dict]:
    out = []
    by_strategy = defaultdict(list)
    for s in summaries:
        by_strategy[s["strategy"]].append(s)
    for strategy, recs in sorted(by_strategy.items()):
        for cost in COSTS:
            best = min(recs, key=lambda r: r[f"{cost}_mean"])
            out.append({
                "strategy": strategy, "cost_model": cost, "detector": best["detector"],
                "best_param": best["param"], "hyper": best["hyper"],
                "mean_cost": best[f"{cost}_mean"], "ci_low": best[f"{cost}_ci_low"],
                "ci_high": best[f"{cost}_ci_high"], "median_cost": best[f"{cost}_median"],
                "std_cost": best[f"{cost}_std"], "mean_moves": best["mean_moves"],
                "mean_senses": best["mean_senses"], "std_moves": best["std_moves"],
                "std_senses": best["std_senses"]})
    return out


def detector_information(k_values=range(0, 41), alpha_values=None) -> list[dict]:
    """Expected information (bits) from the first reading, averaged over all
    sensing cells, under a uniform belief on evaluation ship 0. Used to justify
    the k and alpha ranges independently of any strategy."""
    if alpha_values is None:
        alpha_values = np.logspace(-3, 1, 41)
    ship = make_ship(EVAL_SETUP, 0)
    uniform_mask = np.ones(ship.n, dtype=bool)
    uniform_p = np.full(ship.n, 1.0 / ship.n)
    out = []
    for k in k_values:
        sc = CandidateScorer(ship, DetectorA(int(k))).score(uniform_mask)
        out.append({"detector": "A", "param": int(k),
                    "mean_info_bits": float(sc.info_gain.mean()),
                    "max_info_bits": float(sc.info_gain.max()),
                    "mean_p_ping": float(sc.p_ping.mean())})
    for a in alpha_values:
        sc = BeliefScorer(ship, DetectorB(float(a))).score(uniform_p)
        out.append({"detector": "B", "param": float(a),
                    "mean_info_bits": float(sc.info_gain.mean()),
                    "max_info_bits": float(sc.info_gain.max()),
                    "mean_p_ping": float(sc.p_ping.mean())})
    return out


def write_rows(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, (float, np.floating)) else v)
                        for k, v in r.items()})


def _md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_summaries(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    md = ["# Experimental results (generated by `python main.py analyze`)\n"]
    all_summ = []
    for kind, name in (("A", "type_a"), ("B", "type_b")):
        rows = read_csv(out_dir / f"{name}.csv")
        summ = summarise(rows)
        pair = paired(rows, kind)
        write_rows(summ, out_dir / f"summary_{name}.csv")
        write_rows(pair, out_dir / f"paired_{name}.csv")
        all_summ += summ
        pname = "k" if kind == "A" else "alpha"
        for cost in COSTS:
            strategies = sorted({s["strategy"] for s in summ},
                                key=lambda s: list(LABELS).index(s))
            params = sorted({s["param"] for s in summ})
            look = {(s["strategy"], s["param"]): s for s in summ}
            body = []
            for p in params:
                cells = [f"{p:g}"]
                for st in strategies:
                    s = look[(st, p)]
                    half = (s[f"{cost}_ci_high"] - s[f"{cost}_ci_low"]) / 2
                    cells.append(f"{s[f'{cost}_mean']:.1f} ± {half:.1f}")
                body.append(cells)
            md.append(f"## Type {kind}: mean {cost} ± 95% CI\n")
            md.append(_md_table([pname] + strategies, body) + "\n")
        for cost in COSTS:
            body = [[f"{r['param']:g}", r["strategy"], f"{r['improvement_pct']:.1f}%",
                     f"{r['mean_saving']:.1f} [{r['saving_ci_low']:.1f}, {r['saving_ci_high']:.1f}]",
                     f"{100 * r['win_rate']:.0f}%"]
                    for r in pair if r["cost_model"] == cost]
            md.append(f"## Type {kind}: paired saving vs baseline ({cost})\n")
            md.append(_md_table([pname, "strategy", "improvement", "mean saving [95% CI]",
                                 "trials cheaper"], body) + "\n")
        cap = sum(s["capped"] for s in summ)
        md.append(f"Trials hitting the action cap (Type {kind}): {cap}\n")

    best = best_configs(all_summ)
    write_rows(best, out_dir / "best_configs.csv")
    body = [[r["strategy"], r["cost_model"], f"{r['best_param']:g}",
             f"{r['mean_cost']:.1f} [{r['ci_low']:.1f}, {r['ci_high']:.1f}]",
             f"{r['median_cost']:.1f}", f"{r['std_cost']:.1f}",
             f"{r['mean_moves']:.1f}", f"{r['mean_senses']:.1f}"] for r in best]
    md.append("## Most efficient parameter per strategy\n")
    md.append(_md_table(["strategy", "cost model", "best k/alpha", "mean [95% CI]",
                         "median", "std", "moves", "senses"], body) + "\n")

    info = detector_information()
    write_rows(info, out_dir / "detector_information.csv")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote summaries to {out_dir}")
