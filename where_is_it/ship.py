"""Ship layout: grid representation and the Project 1 ship generator.

The ship is a 2-D grid of open (True) and blocked (False) cells. Open cells are
referred to by an integer index 0..n-1 (row-major order); ``ship.cells[i]`` is
the (row, col) coordinate of index i. Movement is 4-connected (up, down, left,
right), exactly as in Project 1.

The layout is public knowledge: the LocatorBot knows the map of the ship.
Only the LostBot's location is hidden (see ``world.py``).
"""
from __future__ import annotations

from functools import cached_property

import numpy as np

# Orthogonal moves only; diagonal movement is not permitted.
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Ship:
    """Static ship layout plus lazily computed pairwise geometry."""

    def __init__(self, grid: np.ndarray):
        grid = np.asarray(grid, dtype=bool)
        if grid.ndim != 2:
            raise ValueError("ship grid must be 2-dimensional")
        if not grid.any():
            raise ValueError("ship has no open cells")
        self.grid = grid.copy()
        self.grid.setflags(write=False)
        self.shape = grid.shape

        rows, cols = np.nonzero(grid)
        self.cells = np.stack([rows, cols], axis=1).astype(np.int64)
        self.cells.setflags(write=False)
        self.n = len(self.cells)

        self.index_grid = np.full(grid.shape, -1, dtype=np.int64)
        self.index_grid[rows, cols] = np.arange(self.n)
        self.index_grid.setflags(write=False)

        H, W = grid.shape
        nbrs = []
        for r, c in self.cells:
            adj = []
            for dr, dc in N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr, nc]:
                    adj.append(int(self.index_grid[nr, nc]))
            nbrs.append(tuple(adj))
        self.neighbors: tuple[tuple[int, ...], ...] = tuple(nbrs)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_ascii(cls, text: str) -> "Ship":
        """Build a ship from rows of '.' (open) and '#' (blocked)."""
        lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
        if not lines or len({len(ln) for ln in lines}) != 1:
            raise ValueError("ASCII ship rows must be non-empty and equal length")
        bad = set("".join(lines)) - {".", "#"}
        if bad:
            raise ValueError(f"unexpected characters in ASCII ship: {bad}")
        return cls(np.array([[ch == "." for ch in ln] for ln in lines]))

    # ------------------------------------------------------------------ #
    # Index / coordinate conversion
    # ------------------------------------------------------------------ #
    def index_of(self, cell: tuple[int, int]) -> int:
        r, c = cell
        H, W = self.shape
        if not (0 <= r < H and 0 <= c < W) or not self.grid[r, c]:
            raise ValueError(f"{cell} is not an open ship cell")
        return int(self.index_grid[r, c])

    def cell_of(self, index: int) -> tuple[int, int]:
        r, c = self.cells[index]
        return int(r), int(c)

    def is_open(self, r: int, c: int) -> bool:
        H, W = self.shape
        return 0 <= r < H and 0 <= c < W and bool(self.grid[r, c])

    # ------------------------------------------------------------------ #
    # Pairwise geometry (computed once per ship, then cached)
    # ------------------------------------------------------------------ #
    @cached_property
    def bfs_dist(self) -> np.ndarray:
        """(n, n) int32 shortest-path move counts; -1 if unreachable."""
        from .pathfinding import all_pairs_bfs

        d = all_pairs_bfs(self.neighbors)
        d.setflags(write=False)
        return d

    @cached_property
    def euclid_dist(self) -> np.ndarray:
        """(n, n) float64 Euclidean distances (used only by Detector B)."""
        diff = self.cells[:, None, :] - self.cells[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2).astype(np.float64))
        d.setflags(write=False)
        return d

    @cached_property
    def cheb_dist(self) -> np.ndarray:
        """(n, n) int32 Chebyshev distances (used only by Detector A)."""
        diff = np.abs(self.cells[:, None, :] - self.cells[None, :, :])
        d = diff.max(axis=2).astype(np.int32)
        d.setflags(write=False)
        return d

    @cached_property
    def diameter(self) -> int:
        """Largest finite shortest-path distance between two open cells."""
        return int(self.bfs_dist.max())

    def is_connected(self) -> bool:
        """True if every open cell is reachable from every other."""
        from .pathfinding import bfs_distances

        return bool((bfs_distances(self.neighbors, 0) >= 0).all())

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def render(self, marks: dict[int, str] | None = None) -> str:
        """ASCII picture; ``marks`` maps cell index -> single character."""
        rows = [["." if v else "#" for v in row] for row in self.grid]
        for idx, ch in (marks or {}).items():
            r, c = self.cell_of(idx)
            rows[r][c] = ch
        return "\n".join("".join(row) for row in rows)

    def __repr__(self) -> str:
        return f"Ship(shape={self.shape}, open_cells={self.n})"


class _IndexedSet:
    """Set of cells with O(1) add, remove and uniform random choice."""

    def __init__(self):
        self.items: list[tuple[int, int]] = []
        self.pos: dict[tuple[int, int], int] = {}

    def __len__(self):
        return len(self.items)

    def __contains__(self, cell):
        return cell in self.pos

    def add(self, cell):
        if cell not in self.pos:
            self.pos[cell] = len(self.items)
            self.items.append(cell)

    def discard(self, cell):
        i = self.pos.pop(cell, None)
        if i is None:
            return
        last = self.items.pop()
        if i < len(self.items):
            self.items[i] = last
            self.pos[last] = i

    def choice(self, rng: np.random.Generator):
        return self.items[int(rng.integers(len(self.items)))]


def generate_ship(D: int, rng: np.random.Generator,
                  dead_end_fraction: float = 0.5) -> Ship:
    """Generate a D x D ship with the Project 1 algorithm (team version).

    1. Start with every cell blocked; open one random interior cell.
    2. Repeatedly open a uniformly random blocked cell that has exactly one
       open orthogonal neighbour, until no such cell exists (a loop-free maze).
    3. Dead-end reduction, as in the team's Project 1 ship generator
       (Sahas's implementation). Let n0 be the number of dead ends (open cells with
       exactly one open neighbour) after step 2. While more than
       int(n0 * (1 - dead_end_fraction)) dead ends remain (n0 // 2 with the
       default 0.5), pick a uniformly random *current* dead end and open one
       random blocked neighbour of it. Dead ends are recounted after every
       opening, so ones created by an opening also count, and the final number
       really is at most half of the original.

    This gives the same distribution of ships as Project 1. The only
    difference is speed: the eligible cells and the dead ends are kept up to
    date incrementally, instead of rescanning the grid after every opening.
    """
    if D < 3:
        raise ValueError("D must be at least 3")
    if not 0.0 <= dead_end_fraction <= 1.0:
        raise ValueError("dead_end_fraction must be in [0, 1]")

    grid = np.zeros((D, D), dtype=bool)
    open_nbrs = np.zeros((D, D), dtype=np.int8)  # open orthogonal neighbours
    frontier = _IndexedSet()                     # blocked, exactly 1 open nbr
    dead_ends = _IndexedSet()                    # open, exactly 1 open nbr

    def nbrs(r, c):
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < D and 0 <= nc < D:
                yield nr, nc

    def refresh(r, c):
        """Re-file one cell into the frontier / dead-end sets."""
        one = open_nbrs[r, c] == 1
        if grid[r, c]:
            frontier.discard((r, c))
            (dead_ends.add if one else dead_ends.discard)((r, c))
        else:
            (frontier.add if one else frontier.discard)((r, c))

    def open_cell(r, c):
        grid[r, c] = True
        for nr, nc in nbrs(r, c):
            open_nbrs[nr, nc] += 1
            refresh(nr, nc)
        refresh(r, c)

    open_cell(int(rng.integers(1, D - 1)), int(rng.integers(1, D - 1)))
    while frontier:
        open_cell(*frontier.choice(rng))

    target = int(len(dead_ends) * (1.0 - dead_end_fraction))
    guard = 50 * D * D          # never reached in practice; prevents a hang
    while len(dead_ends) > target and guard > 0:
        guard -= 1
        r, c = dead_ends.choice(rng)
        blocked = [(nr, nc) for nr, nc in nbrs(r, c) if not grid[nr, nc]]
        if blocked:
            open_cell(*blocked[int(rng.integers(len(blocked)))])

    return Ship(grid)
