"""Detector models.

A detector object is a *public model* of the sensor: the LocatorBot knows the
formulas below and uses them to update its knowledge base. Detector objects
hold no hidden state. To produce an actual reading, ``World.sense`` calls
``sample(a, b, rng)`` with the hidden LostBot cell ``b`` and the world's private
random generator.

Every reading has three possible outcomes:
  FOUND    - the LocatorBot is standing on the LostBot's cell (search ends)
  PING     - the LostBot is elsewhere and the detector fired
  NO_PING  - the LostBot is elsewhere and the detector stayed silent
"""
from __future__ import annotations

import math
from enum import Enum
from numbers import Integral, Real

import numpy as np

Cell = tuple[int, int]


class Outcome(str, Enum):
    FOUND = "found"
    PING = "ping"
    NO_PING = "no_ping"


def _euclid_row(ship, a_index: int) -> np.ndarray:
    """Euclidean distance from cell a to every open cell (O(n), no matrix)."""
    diff = (ship.cells - ship.cells[a_index]).astype(np.float64)
    return np.hypot(diff[:, 0], diff[:, 1])


class DetectorA:
    """Deterministic square detector of 'radius' k.

    Pings iff the LostBot lies in the (2k+1) x (2k+1) square centred on the
    LocatorBot, i.e. max(|dr|, |dc|) <= k (Chebyshev distance). The square may
    extend over blocked cells or past the hull; only open cells can hold the
    LostBot, so only those matter.
    """

    kind = "A"

    def __init__(self, k: int):
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
            raise ValueError(f"k must be a non-negative integer, got {k!r}")
        self.k = int(k)

    def pings(self, a: Cell, b: Cell) -> bool:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= self.k

    def sample(self, a: Cell, b: Cell, rng: np.random.Generator | None = None) -> Outcome:
        """Reading taken at ``a`` when the LostBot is at ``b`` (rng unused)."""
        if tuple(a) == tuple(b):
            return Outcome.FOUND
        return Outcome.PING if self.pings(a, b) else Outcome.NO_PING

    def coverage(self, ship, a_index: int) -> np.ndarray:
        """Boolean mask over open cells inside the square centred on cell a.

        This is D_k(a), the open cells a ping would be consistent with
        (it includes a itself).
        """
        return np.abs(ship.cells - ship.cells[a_index]).max(axis=1) <= self.k

    def __repr__(self) -> str:
        return f"DetectorA(k={self.k})"


class DetectorB:
    """Probabilistic detector: P(ping | a, b) = exp(-alpha * (dist(a, b) - 1)).

    dist is the Euclidean distance between cell coordinates, and a != b.
    Distinct cells are at least 1 apart, so the probability lies in (0, 1].
    Orthogonal neighbours (dist = 1) always ping. Repeated readings are
    independent given the LostBot's location.
    """

    kind = "B"

    def __init__(self, alpha: float):
        if isinstance(alpha, bool) or not isinstance(alpha, Real) \
                or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be a finite number > 0, got {alpha!r}")
        self.alpha = float(alpha)

    @staticmethod
    def distance(a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def ping_probability(self, a: Cell, b: Cell) -> float:
        if tuple(a) == tuple(b):
            raise ValueError("ping probability is undefined for a == b "
                             "(the detector reports FOUND instead)")
        return math.exp(-self.alpha * (self.distance(a, b) - 1.0))

    def sample(self, a: Cell, b: Cell, rng: np.random.Generator) -> Outcome:
        """Reading taken at ``a`` when the LostBot is at ``b``."""
        if tuple(a) == tuple(b):
            return Outcome.FOUND
        p = self.ping_probability(a, b)
        return Outcome.PING if rng.random() < p else Outcome.NO_PING

    def ping_probabilities(self, ship, a_index: int) -> np.ndarray:
        """P(ping | a, b) for every open cell b. The entry for b = a is 0,
        because a reading there yields FOUND, never PING."""
        p = np.exp(-self.alpha * (_euclid_row(ship, a_index) - 1.0))
        p[a_index] = 0.0
        return p

    def log_likelihood(self, ship, a_index: int, outcome: Outcome) -> np.ndarray:
        """log P(outcome | LocatorBot at a, LostBot at b) for every open cell b.

        Computed directly in log space for numerical stability:
          log P(ping)    = -alpha (d - 1)
          log P(no ping) = log(1 - e^{-alpha (d-1)}) = log(-expm1(-alpha (d-1)))
        expm1 is accurate when alpha (d-1) is tiny. At d = 1 it gives log 0 = -inf,
        which correctly rules out the orthogonal neighbours after a NO_PING. The
        entry for b = a is -inf, because a non-FOUND reading proves b != a.
        """
        x = self.alpha * (_euclid_row(ship, a_index) - 1.0)
        if outcome is Outcome.PING:
            ll = -x
        elif outcome is Outcome.NO_PING:
            with np.errstate(divide="ignore"):
                ll = np.log(-np.expm1(-np.maximum(x, 0.0)))
        else:
            raise ValueError("log_likelihood is only defined for PING / NO_PING")
        ll = np.array(ll, dtype=np.float64)
        ll[a_index] = -np.inf
        return ll

    def __repr__(self) -> str:
        return f"DetectorB(alpha={self.alpha:g})"
