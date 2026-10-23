"""Knowledge bases: what the LocatorBot believes about the LostBot's location.

Type A: a set S of cells that could still contain the LostBot.
Type B: a probability distribution P over cells, stored as log-probabilities.

Both update only from (sensing position a, observed outcome). Neither ever sees
the LostBot's true location.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .detectors import DetectorA, DetectorB, Outcome

# Two log-probabilities within this tolerance count as a tie. It equals a
# relative tolerance of about 1e-9 on probabilities, and absorbs rounding
# differences between cells that are tied in exact arithmetic.
TIE_LOG_TOL = 1e-9


class InconsistentObservationError(RuntimeError):
    """An observation has zero probability under the current knowledge base.

    This cannot happen with readings from the real detector, because the true
    location is always kept as consistent. Seeing it means there is a bug.
    """


class CandidateSetKB:
    """Type A knowledge base: the set S of possible LostBot cells.

    Update rules after sensing at cell a (when the reading is not FOUND, the
    LostBot is not at a):
        PING    : S <- (S ∩ D_k(a)) \\ {a}
        NO_PING : S <- S \\ D_k(a)          (D_k(a) contains a)
        FOUND   : S <- {a}
    where D_k(a) is the set of open cells in the (2k+1)x(2k+1) square around a.
    """

    def __init__(self, ship, detector: DetectorA, excluded: Iterable[int] = ()):
        if not isinstance(detector, DetectorA):
            raise TypeError("CandidateSetKB requires a DetectorA")
        self.ship = ship
        self.detector = detector
        self.possible = np.ones(ship.n, dtype=bool)
        self.possible[list(excluded)] = False
        if not self.possible.any():
            raise ValueError("initial candidate set is empty")

    @property
    def count(self) -> int:
        return int(self.possible.sum())

    def candidates(self) -> np.ndarray:
        return np.flatnonzero(self.possible)

    def is_possible(self, index: int) -> bool:
        return bool(self.possible[index])

    def probabilities(self) -> np.ndarray:
        """Uniform distribution over S (each remaining cell is equally likely)."""
        return self.possible / self.possible.sum()

    def update(self, a: int, outcome: Outcome) -> None:
        if outcome is Outcome.FOUND:
            if not self.possible[a]:
                raise InconsistentObservationError("FOUND at an eliminated cell")
            new = np.zeros_like(self.possible)
            new[a] = True
        elif outcome is Outcome.PING:
            new = self.possible & self.detector.coverage(self.ship, a)
            new[a] = False
        elif outcome is Outcome.NO_PING:
            new = self.possible & ~self.detector.coverage(self.ship, a)
        else:
            raise ValueError(f"unknown outcome {outcome!r}")
        if not new.any():
            raise InconsistentObservationError(
                f"{outcome.value} at cell {a} leaves no possible location")
        self.possible = new


class BeliefKB:
    """Type B knowledge base: posterior P(LostBot at b | readings so far).

    Bayes' rule after a reading o at cell a:
        P'(b) = P(b) * P(o | a, b) / Z,    Z = sum_b' P(b') P(o | a, b')
    with P(ping | a, b) = exp(-alpha (d - 1)), P(no ping | a, b) = 1 - that,
    and P(o | a, a) = 0 for o in {PING, NO_PING}, since a non-FOUND reading at a
    proves the LostBot is not at a.

    Numerics: the belief is kept as log-probabilities. An update adds the
    log-likelihood vector, then renormalises with log-sum-exp (subtract the
    maximum before exponentiating). Very small likelihoods therefore never
    underflow to an accidental zero, however many readings are combined.
    Cells that are definitively eliminated have log-probability -inf.
    """

    def __init__(self, ship, detector: DetectorB, excluded: Iterable[int] = ()):
        if not isinstance(detector, DetectorB):
            raise TypeError("BeliefKB requires a DetectorB")
        self.ship = ship
        self.detector = detector
        log_p = np.zeros(ship.n, dtype=np.float64)
        log_p[list(excluded)] = -np.inf
        self.log_p = self._normalise(log_p)

    @staticmethod
    def _normalise(log_unnorm: np.ndarray) -> np.ndarray:
        m = log_unnorm.max()
        if not np.isfinite(m):
            raise InconsistentObservationError(
                "observation has zero probability under the current belief")
        shifted = log_unnorm - m
        return shifted - np.log(np.exp(shifted).sum())

    def update(self, a: int, outcome: Outcome) -> None:
        if outcome is Outcome.FOUND:
            if self.log_p[a] == -np.inf:
                raise InconsistentObservationError("FOUND at an eliminated cell")
            new = np.full(self.ship.n, -np.inf)
            new[a] = 0.0
            self.log_p = new
            return
        if outcome not in (Outcome.PING, Outcome.NO_PING):
            raise ValueError(f"unknown outcome {outcome!r}")
        ll = self.detector.log_likelihood(self.ship, a, outcome)
        self.log_p = self._normalise(self.log_p + ll)

    def probabilities(self) -> np.ndarray:
        return np.exp(self.log_p)

    def possible(self) -> np.ndarray:
        """Mask of cells not definitively ruled out."""
        return self.log_p > -np.inf

    def most_likely(self, tol: float = TIE_LOG_TOL) -> np.ndarray:
        """Indices of all cells tied (within ``tol`` in log space) for max P."""
        return np.flatnonzero(self.log_p >= self.log_p.max() - tol)

    def entropy(self) -> float:
        """Shannon entropy of the belief, in bits."""
        p = self.probabilities()
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
