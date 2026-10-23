"""Breadth-first search on the (unweighted, 4-connected) ship graph.

Functions operate on the adjacency lists ``ship.neighbors`` (cell indices), so
they do not depend on the Ship class. Every move costs the same, so BFS gives
true shortest paths.

This is *movement* distance through open cells. It is a different quantity from
the Euclidean distance used by Detector B or the Chebyshev distance used by
Detector A.
"""
from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

Adjacency = Sequence[Sequence[int]]


class UnreachableError(RuntimeError):
    """Raised when no path exists between two cells."""


def bfs_distances(neighbors: Adjacency, source: int) -> np.ndarray:
    """Single-source BFS. Returns int32 array of move counts (-1 = unreachable)."""
    dist = [-1] * len(neighbors)
    dist[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        du = dist[u] + 1
        for v in neighbors[u]:
            if dist[v] < 0:
                dist[v] = du
                queue.append(v)
    return np.array(dist, dtype=np.int32)


def all_pairs_bfs(neighbors: Adjacency) -> np.ndarray:
    """(n, n) matrix of BFS distances; row s is ``bfs_distances(neighbors, s)``.

    O(n * (n + edges)) time. Computed once per ship and reused for every trial
    and every decision, so no shortest path is recomputed during a search.
    """
    n = len(neighbors)
    out = np.empty((n, n), dtype=np.int32)
    for s in range(n):
        out[s] = bfs_distances(neighbors, s)
    return out


def shortest_path(neighbors: Adjacency, dist: np.ndarray,
                  start: int, goal: int) -> list[int]:
    """Shortest path start -> goal using a precomputed distance matrix.

    Returns the cells to step into, in order: ``start`` is excluded and
    ``goal`` is included (so ``len(path) == dist[start, goal]``). Each step goes
    to the first neighbour, in N4 order, that is one move closer to ``goal``.
    """
    to_goal = dist[goal]
    if to_goal[start] < 0:
        raise UnreachableError(f"cell {goal} is unreachable from cell {start}")
    path = []
    cur = start
    while cur != goal:
        want = to_goal[cur] - 1
        for v in neighbors[cur]:
            if to_goal[v] == want:
                cur = v
                break
        else:  # pragma: no cover - impossible with a consistent dist matrix
            raise RuntimeError("distance matrix inconsistent with adjacency")
        path.append(cur)
    return path


def bfs_path(neighbors: Adjacency, start: int, goal: int) -> list[int] | None:
    """Standalone BFS with parent pointers (no matrix). Same path format as
    ``shortest_path``; returns None if ``goal`` is unreachable."""
    parent = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == goal:
            break
        for v in neighbors[u]:
            if v not in parent:
                parent[v] = u
                queue.append(v)
    if goal not in parent:
        return None
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]
