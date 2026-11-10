"""Matplotlib figures for the report (PNG). Run via ``python main.py analyze``.

Fig 1  Type A: mean discovery cost vs k (baseline vs improved, plus ablation)
Fig 2  Type B: mean discovery cost vs alpha (log axis)
Fig 3  Most efficient Type A vs most efficient Type B: moves, readings, cost
Fig 4  Free movement: mean number of detector readings vs k and vs alpha
Fig 5  Detector B: P(ping) vs Euclidean distance for several alpha
Fig 6  Information from a first reading vs k and vs alpha (range justification)
Fig 7  Example search state: Type A candidate set and Type B belief heatmap
Fig 8  Techniques tried: baseline, earlier argmin-J rule, slides' planning
       alone, and the final strategy (slides' planning + sense on the way)

Colour follows the strategy role in a fixed order (baseline, improved,
ablation, free-movement) using a colour-blind-validated categorical palette.
Fig 5 encodes alpha (a magnitude) with a single-hue sequential ramp.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from .analysis import LABELS  # noqa: E402
from .detectors import DetectorA, DetectorB  # noqa: E402
from .experiments import EVAL_SETUP, make_scenarios, make_ship  # noqa: E402
from .strategies import ImprovedA, ImprovedB, Move  # noqa: E402
from .world import World  # noqa: E402

SERIES = {"baseline": "#2a78d6", "improved": "#eb6834",
          "ablation": "#1baf7a", "free": "#eda100",
          "lookahead": "#7c5cc4", "slides": "#8a8984"}
TEXT, TEXT2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"
BLUES = ["#c9def5", "#94bdeb", "#5b98dd", "#2a78d6", "#1d5aa6", "#123c73"]


def role(strategy: str) -> str:
    if strategy.startswith("baseline"):
        return "baseline"
    if strategy.startswith("improved"):
        return "improved"
    if strategy.startswith("free"):
        return "free"
    if strategy.startswith("lookahead"):
        return "lookahead"
    if strategy.startswith("slides"):
        return "slides"
    return "ablation"


def _setup_rc():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlecolor": TEXT, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
        "ytick.color": TEXT2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9, "text.color": TEXT,
    })


def _style(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _read(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return rows


def _series(summary, strategy, cost):
    rs = sorted((r for r in summary if r["strategy"] == strategy), key=lambda r: r["param"])
    x = np.array([r["param"] for r in rs])
    return (x, np.array([r[f"{cost}_mean"] for r in rs]),
            np.array([r[f"{cost}_ci_low"] for r in rs]),
            np.array([r[f"{cost}_ci_high"] for r in rs]))


def _cost_lines(ax, summary, strategies, cost, logx=False):
    for st in strategies:
        x, m, lo, hi = _series(summary, st, cost)
        c = SERIES[role(st)]
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, m, color=c, lw=2, marker="o", ms=5, label=LABELS[st],
                markeredgecolor=SURFACE, markeredgewidth=1)
    if logx:
        ax.set_xscale("log")
    _style(ax)


def fig_cost_vs_param(summary, kind, out: Path):
    strategies = (["baseline_a", "improved_a", "info_per_cost_a"] if kind == "A"
                  else ["baseline_b", "improved_b", "prob_per_cost_b"])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    _cost_lines(ax, summary, strategies, "cost_original", logx=(kind == "B"))
    if kind == "A":
        ax.set_xlabel("k (detector square is (2k+1) x (2k+1); ship is 41 x 41)")
        ax.set_title("Detector A: mean discovery cost vs k")
    else:
        ax.set_xlabel("alpha (log scale)")
        ax.set_title("Detector B: mean discovery cost vs alpha")
    ax.set_ylabel("Mean cost = moves + 0.5 x readings")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper center" if kind == "A" else "upper right")
    n = int(summary[0]["cost_original_n"])
    fig.text(0.01, 0.005, f"Shaded band: 95% CI of the mean, {n} paired trials per point.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_techniques(summary_a, summary_b, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, summ, kind in ((axes[0], summary_a, "A"), (axes[1], summary_b, "B")):
        k = kind.lower()
        sts = [s for s in (f"baseline_{k}", f"lookahead_{k}", f"slides_pure_{k}",
                           f"improved_{k}") if any(r["strategy"] == s for r in summ)]
        _cost_lines(ax, summ, sts, "cost_original", logx=(kind == "B"))
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylabel("Mean cost (log scale)")
        if kind == "A":
            ax.set_xlabel("k")
            ax.set_title("Detector A: techniques tried")
        else:
            ax.set_xlabel("alpha (log scale)")
            ax.set_title("Detector B: techniques tried")
        ax.legend(loc="upper center" if kind == "A" else "upper left", fontsize=8)
    n = int(summary_a[0]["cost_original_n"])
    fig.text(0.01, 0.005, f"Shaded band: 95% CI of the mean, {n} paired trials per point.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_best_comparison(best, summaries, out: Path):
    picks = ["baseline_a", "improved_a", "baseline_b", "improved_b"]
    recs = {r["strategy"]: r for r in best if r["cost_model"] == "cost_original"}
    look = {(s["strategy"], s["param"]): s for s in summaries}
    names, colors, rows = [], [], []
    for st in picks:
        r = recs[st]
        s = look[(st, r["best_param"])]
        p = "k" if st.endswith("_a") else "alpha"
        names.append(f"{LABELS[st].split(' (')[0]}\n{p}={r['best_param']:g}")
        colors.append(SERIES[role(st)])
        n = s["cost_original_n"]
        rows.append([(s["mean_moves"], Z(s["std_moves"], n)),
                     (s["mean_senses"], Z(s["std_senses"], n)),
                     (s["cost_original_mean"],
                      (s["cost_original_ci_high"] - s["cost_original_ci_low"]) / 2)])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    titles = ["Mean moves", "Mean detector readings", "Mean total cost"]
    x = np.arange(len(picks))
    for j, ax in enumerate(axes):
        vals = [r[j][0] for r in rows]
        errs = [r[j][1] for r in rows]
        ax.bar(x, vals, width=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=TEXT2, elinewidth=1, capsize=3)
        for xi, v, e in zip(x, vals, errs):
            ax.annotate(f"{v:.0f}", (xi, v + e), ha="center", va="bottom", fontsize=8,
                        color=TEXT2, xytext=(0, 3), textcoords="offset points")
        ax.set_xticks(x, names, fontsize=8)
        ax.set_title(titles[j], fontsize=11)
        _style(ax)
    fig.suptitle("Most efficient tested k (Detector A) vs alpha (Detector B)",
                 x=0.01, ha="left", fontweight="semibold", color=TEXT)
    fig.text(0.01, 0.005, "Each strategy at its own best parameter. Error bars: 95% CI.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def Z(sd, n):
    return 1.96 * sd / np.sqrt(n)


def fig_free(summary_a, summary_b, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    _cost_lines(axes[0], summary_a, ["baseline_a", "improved_a", "free_movement_a"], "cost_free")
    axes[0].set_xlabel("k")
    axes[0].set_title("Detector A")
    _cost_lines(axes[1], summary_b, ["baseline_b", "improved_b", "free_movement_b"],
                "cost_free", logx=True)
    axes[1].set_xlabel("alpha (log scale)")
    axes[1].set_title("Detector B")
    for ax in axes:
        ax.set_ylabel("Mean detector readings (movement free)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(loc="upper center", fontsize=8)
    fig.suptitle("Free movement: cost = number of detector readings (log scale)",
                 x=0.01, ha="left", fontweight="semibold", color=TEXT)
    n = int(summary_a[0]["cost_original_n"])
    fig.text(0.01, 0.005, f"Same {n} paired trials per point as Figs 1-2. Band: 95% CI.",
             color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ping_probability(out: Path):
    ship = make_ship(EVAL_SETUP, 0)
    iu = np.triu_indices(ship.n, 1)
    mean_d = float(ship.euclid_dist[iu].mean())
    d = np.linspace(1, 57, 400)
    alphas = [0.01, 0.05, 0.1, 0.3, 1.0, 3.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for a, c in zip(alphas, BLUES):
        ax.plot(d, np.exp(-a * (d - 1)), color=c, lw=2, label=f"alpha = {a:g}")
    ax.axvline(mean_d, color=MUTED, lw=1, ls="--")
    ax.annotate(f"mean distance between\nopen cells = {mean_d:.1f}", (mean_d, 0.9),
                xytext=(6, 0), textcoords="offset points", color=TEXT2, fontsize=8)
    ax.set_xlabel("Euclidean distance between LocatorBot and LostBot (cells)")
    ax.set_ylabel("P(ping)")
    ax.set_title("Detector B: P(ping) = exp(-alpha (d - 1))")
    ax.set_ylim(0, 1.03)
    _style(ax, "both")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_information(info, out: Path):
    a_rows = sorted((r for r in info if r["detector"] == "A"), key=lambda r: r["param"])
    b_rows = sorted((r for r in info if r["detector"] == "B"), key=lambda r: r["param"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, rows, logx, xl, title in (
            (axes[0], a_rows, False, "k", "Detector A"),
            (axes[1], b_rows, True, "alpha (log scale)", "Detector B")):
        x = [r["param"] for r in rows]
        ax.plot(x, [r["mean_info_bits"] for r in rows], color=SERIES["baseline"], lw=2,
                label="mean over sensing cells")
        ax.plot(x, [r["max_info_bits"] for r in rows], color=SERIES["improved"], lw=2,
                label="best sensing cell")
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Expected information (bits)")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right" if not logx else "upper left")
        _style(ax)
    fig.suptitle("Information from one reading under a uniform belief (41 x 41 ship)",
                 x=0.01, ha="left", fontweight="semibold", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _replay(ship, detector, strategy, start, lostbot, stop_after_senses):
    world = World(ship, detector, start, lostbot, np.random.default_rng(0))
    strategy.reset(ship, detector, start, np.random.default_rng(1))
    path = [start]
    while not world.found and world.senses < stop_after_senses:
        act = strategy.next_action(world.position)
        if isinstance(act, Move):
            world.move(act.target)
            path.append(world.position)
        else:
            strategy.observe(world.position, world.sense())
    return world, path


def _tuned_kwargs(tuned: dict, kind: str, param: float) -> dict:
    """Tuned hyper-parameters for the tuned value closest to ``param``."""
    key = min(tuned[kind], key=lambda s: abs(float(s) - param))
    return tuned[kind][key]["kwargs"]


def fig_example_state(out_dir: Path, out: Path):
    with open(out_dir / "tuned_params.json") as f:
        tuned = json.load(f)
    ship = make_ship(EVAL_SETUP, 0)
    start, lostbot = make_scenarios(EVAL_SETUP, ship, 0)[0]
    k, alpha = 5, 0.1
    sa = ImprovedA(**_tuned_kwargs(tuned, "A", k))
    sb = ImprovedB(**_tuned_kwargs(tuned, "B", alpha))
    wa, pa = _replay(ship, DetectorA(k), sa, start, lostbot, 6)
    wb, pb = _replay(ship, DetectorB(alpha), sb, start, lostbot, 10)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    blocked = np.where(ship.grid, np.nan, 1.0)
    for ax, world, path, title in (
            (axes[0], wa, pa, f"Detector A, k={k}: candidate set after {wa.senses} readings"),
            (axes[1], wb, pb, f"Detector B, alpha={alpha}: belief after {wb.senses} readings")):
        grid = np.full(ship.shape, np.nan)
        if ax is axes[0]:
            vals = sa.kb.possible.astype(float)
            grid[tuple(ship.cells.T)] = vals
            ax.imshow(grid, cmap=ListedColormap(["#efeeea", "#94bdeb"]), vmin=0, vmax=1)
        else:
            p = sb.kb.probabilities()
            grid[tuple(ship.cells.T)] = p / p.max()
            cmap = LinearSegmentedColormap.from_list("seq", ["#f4f3ef"] + BLUES[1:])
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label("P(cell) / max P", color=TEXT2)
        ax.imshow(blocked, cmap=ListedColormap(["#3a3a38"]), vmin=0, vmax=1)
        rc = np.array([ship.cell_of(i) for i in path])
        ax.plot(rc[:, 1], rc[:, 0], color=SERIES["improved"], lw=1.6, label="path walked")
        ax.plot(*ship.cell_of(start)[::-1], "o", color=SERIES["improved"], ms=8,
                markeredgecolor=SURFACE, label="start")
        ax.plot(*ship.cell_of(world.position)[::-1], "s", color=TEXT, ms=8,
                markeredgecolor=SURFACE, label="LocatorBot now")
        ax.plot(*ship.cell_of(lostbot)[::-1], "*", color="#e34948", ms=14,
                markeredgecolor=SURFACE, label="LostBot (hidden)")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.05, -0.02), ncol=4, fontsize=8)
    fig.suptitle("Example search on evaluation ship 0 (dark = blocked; blue = possible / likely)",
                 x=0.01, ha="left", fontweight="semibold", color=TEXT)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_all_figures(out_dir: Path) -> None:
    _setup_rc()
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    sa = _read(out_dir / "summary_type_a.csv")
    sb = _read(out_dir / "summary_type_b.csv")
    best = _read(out_dir / "best_configs.csv")
    info = _read(out_dir / "detector_information.csv")
    fig_cost_vs_param(sa, "A", fig_dir / "fig1_type_a_cost_vs_k.png")
    fig_cost_vs_param(sb, "B", fig_dir / "fig2_type_b_cost_vs_alpha.png")
    fig_best_comparison(best, sa + sb, fig_dir / "fig3_best_a_vs_best_b.png")
    fig_free(sa, sb, fig_dir / "fig4_free_movement.png")
    fig_ping_probability(fig_dir / "fig5_ping_probability.png")
    fig_information(info, fig_dir / "fig6_information_vs_parameter.png")
    fig_example_state(out_dir, fig_dir / "fig7_example_search_state.png")
    fig_techniques(sa, sb, fig_dir / "fig8_techniques_tried.png")
    print(f"wrote figures to {fig_dir}")
