"""Vectorised evaluation of every possible sensing position.

For each open cell a, if the LocatorBot sensed at a right now, we compute:

    p_found(a)           P(outcome = FOUND)            = P(B = a)
    p_ping(a)            P(outcome = PING)
    p_no_ping(a)         P(outcome = NO_PING)
    expected_entropy(a)  sum_o P(o) * H(B | o)  (bits; H(B | FOUND) = 0)
    info_gain(a)         H(B) - expected_entropy(a)   (bits)
    expected_distance(a) E[d_BFS(a, B)] under the current belief

Efficient exact computation:
  * Both detectors depend only on the displacement a - b. A sum such as
    sum_b p_b L(a, b) is therefore a 2-D convolution of the belief grid with
    a fixed kernel.
    - Type A: the kernel is the (2k+1)x(2k+1) box, so the counts come from a
      summed-area table (2-D prefix sums) in O(n), as exact integers.
    - Type B: the kernels are L, 1-L, L ln L and (1-L) ln(1-L). They are applied
      with FFTs on a zero-padded grid, so there is no wrap-around. This is exact
      up to floating-point rounding, about 1e-13 here.
  * Maze (BFS) distance is *not* a function of displacement, so
    E[d(a, B)] = (D_bfs @ p)_a is one float32 matrix-vector product.

The scorers use only the knowledge base (what the bot believes), never the
true location.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detectors import DetectorA, DetectorB

LN2 = np.log(2.0)


def _f32(x: np.ndarray) -> np.ndarray:
    """Cast to float32 and round magnitudes below 1e-30 to zero.

    Values below ~1e-38 are float32 denormals, and x86 arithmetic on denormals
    is 50-100x slower. Their contribution to any score is far below float32
    resolution, so dropping them changes no decision."""
    out = np.asarray(x, dtype=np.float64).astype(np.float32)
    out[np.abs(out) < 1e-30] = 0.0
    return out


def xlogx(x: np.ndarray) -> np.ndarray:
    """Elementwise x ln x with the convention 0 ln 0 = 0 (also for x <= 0)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    pos = x > 0
    out[pos] = x[pos] * np.log(x[pos])
    return out


@dataclass
class SensingScores:
    p_found: np.ndarray
    p_ping: np.ndarray
    p_no_ping: np.ndarray
    expected_entropy: np.ndarray
    info_gain: np.ndarray
    expected_distance: np.ndarray
    entropy: float


class _GridScorer:
    def __init__(self, ship):
        self.shape = ship.shape
        self.rows, self.cols = ship.cells[:, 0], ship.cells[:, 1]
        self.bfs = ship.bfs_dist.astype(np.float32)

    def to_grid(self, v: np.ndarray) -> np.ndarray:
        g = np.zeros(self.shape, dtype=np.float64)
        g[self.rows, self.cols] = v
        return g

    def expected_distance(self, p: np.ndarray) -> np.ndarray:
        return (self.bfs @ _f32(p)).astype(np.float64)


class CandidateScorer(_GridScorer):
    """Type A: the LostBot is uniform over the candidate set S (|S| = N).

    For sensing position a, let n_cov(a) = |S ∩ D_k(a)|. Then
        n_found = [a in S],  n_ping = n_cov - n_found,  n_no = N - n_cov,
    and each outcome leaves a uniform posterior over that many cells, so
        E[H_post](a) = (n_ping log2 n_ping + n_no log2 n_no) / N.
    n_cov is a box sum, read off a summed-area table.
    """

    def __init__(self, ship, detector: DetectorA):
        super().__init__(ship)
        H, W = ship.shape
        k = detector.k
        self.r0 = np.clip(self.rows - k, 0, H)
        self.r1 = np.clip(self.rows + k + 1, 0, H)
        self.c0 = np.clip(self.cols - k, 0, W)
        self.c1 = np.clip(self.cols + k + 1, 0, W)

    def coverage_counts(self, possible: np.ndarray) -> np.ndarray:
        H, W = self.shape
        sat = np.zeros((H + 1, W + 1))
        sat[1:, 1:] = self.to_grid(possible.astype(np.float64)).cumsum(0).cumsum(1)
        return (sat[self.r1, self.c1] - sat[self.r0, self.c1]
                - sat[self.r1, self.c0] + sat[self.r0, self.c0])

    def score(self, possible: np.ndarray) -> SensingScores:
        N = float(possible.sum())
        n_cov = self.coverage_counts(possible)
        n_found = possible.astype(np.float64)
        n_ping = n_cov - n_found
        n_no = N - n_cov
        exp_h = (xlogx(n_ping) + xlogx(n_no)) / (N * LN2)
        h = np.log2(N)
        return SensingScores(
            p_found=n_found / N,
            p_ping=n_ping / N,
            p_no_ping=n_no / N,
            expected_entropy=exp_h,
            info_gain=np.maximum(h - exp_h, 0.0),
            expected_distance=self.expected_distance(n_found / N),
            entropy=float(h),
        )


class BeliefScorer(_GridScorer):
    """Type B: belief vector p and likelihood L(a, b) = P(ping | a, b), with
    L(a, a) = 0 (b = a gives FOUND, never PING).

        Z_ping(a) = sum_b p_b L_ab          Z_no(a) = sum_b p_b (1 - L_ab)
    The posterior after a ping is p_b L_ab / Z_ping, and its entropy satisfies
        Z_ping H_ping = Z_ping ln Z_ping - S_ping,
        S_ping = sum_b p_b L_ab ln(p_b L_ab)
               = [L * (p ln p)](a) + [(L ln L) * p](a)    (* = convolution)
    The no-ping case is the same with 1 - L. So
        E[H_post](a) = Z_ping ln Z_ping - S_ping + Z_no ln Z_no - S_no.
    """

    def __init__(self, ship, detector: DetectorB):
        super().__init__(ship)
        H, W = ship.shape
        dr = np.arange(-(H - 1), H)[:, None]
        dc = np.arange(-(W - 1), W)[None, :]
        x = detector.alpha * (np.hypot(dr, dc) - 1.0)
        L = np.exp(-x)
        Lbar = -np.expm1(-np.maximum(x, 0.0))   # accurate 1 - L for small x
        L[H - 1, W - 1] = 0.0                   # displacement 0: FOUND instead
        Lbar[H - 1, W - 1] = 0.0
        # Circular convolution of size M >= 2H-1 per axis is enough: we only
        # read outputs m = a + (H-1) for a in [0, H-1], and wrap-around terms
        # come from linear-convolution indices m + M >= 3H-2, beyond its
        # support [0, 3H-3]. So these outputs equal the linear convolution.
        self.fshape = (2 * H - 1, 2 * W - 1)
        f = lambda K: np.fft.rfft2(K, s=self.fshape)  # noqa: E731
        self.FL, self.FLbar = f(L), f(Lbar)
        self.FLlogL, self.FLbarlogLbar = f(xlogx(L)), f(xlogx(Lbar))
        # Output index m = a + (H-1, W-1) holds the sum for sensing cell a.
        self.out_rows = self.rows + H - 1
        self.out_cols = self.cols + W - 1

    def _at_cells(self, F: np.ndarray) -> np.ndarray:
        full = np.fft.irfft2(F, s=self.fshape)
        return full[self.out_rows, self.out_cols]

    def score(self, p: np.ndarray) -> SensingScores:
        plogp = xlogx(p)
        Fp = np.fft.rfft2(self.to_grid(p), s=self.fshape)
        Fq = np.fft.rfft2(self.to_grid(plogp), s=self.fshape)
        z_ping = np.maximum(self._at_cells(self.FL * Fp), 0.0)
        z_no = np.maximum(self._at_cells(self.FLbar * Fp), 0.0)
        s_ping = self._at_cells(self.FL * Fq + self.FLlogL * Fp)
        s_no = self._at_cells(self.FLbar * Fq + self.FLbarlogLbar * Fp)
        exp_h = (xlogx(z_ping) - s_ping + xlogx(z_no) - s_no) / LN2
        exp_h = np.maximum(exp_h, 0.0)
        h = float(-plogp.sum() / LN2)
        return SensingScores(
            p_found=p.copy(),
            p_ping=z_ping,
            p_no_ping=z_no,
            expected_entropy=exp_h,
            info_gain=np.maximum(h - exp_h, 0.0),
            expected_distance=self.expected_distance(p),
            entropy=h,
        )


_CACHE_PER_SHIP = 1


def get_scorer(ship, detector):
    """Build (or reuse) the scorer for this ship and detector setting.

    It depends only on the ship and on k or alpha, so it is cached on the ship
    and shared by every trial there. At most ``_CACHE_PER_SHIP`` settings are
    kept, to bound memory (the float32 BFS matrix is ~6 MB at 41 x 41).
    """
    cache = ship.__dict__.setdefault("_scorer_cache", {})
    key = (detector.kind, getattr(detector, "k", None), getattr(detector, "alpha", None))
    if key not in cache:
        while len(cache) >= _CACHE_PER_SHIP:
            cache.pop(next(iter(cache)))
        cls = CandidateScorer if isinstance(detector, DetectorA) else BeliefScorer
        cache[key] = cls(ship, detector)
    return cache[key]
