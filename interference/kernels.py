from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.special import exp1, kv
from numpy.polynomial.legendre import leggauss


def theis_well_function(u: np.ndarray | float) -> np.ndarray:
    """Dimensionless line-source interference response.

    u = t_D / r_D**2. Returns p_D = 0.5 * E1(1/(4u)).
    For u<=0 the causal response is zero.
    """
    arr = np.asarray(u, dtype=float)
    out = np.zeros_like(arr)
    mask = arr > 0.0
    if np.any(mask):
        out[mask] = 0.5 * exp1(1.0 / (4.0 * arr[mask]))
    return out


def stehfest_coefficients(n: int = 12) -> np.ndarray:
    """Gaver-Stehfest coefficients. n must be positive and even."""
    if n <= 0 or n % 2:
        raise ValueError("Stehfest order n must be a positive even integer.")
    v = np.zeros(n, dtype=float)
    n2 = n // 2
    for k in range(1, n + 1):
        s = 0.0
        jmin = (k + 1) // 2
        jmax = min(k, n2)
        for j in range(jmin, jmax + 1):
            num = j**n2 * math.factorial(2 * j)
            den = (
                math.factorial(n2 - j)
                * math.factorial(j)
                * math.factorial(j - 1)
                * math.factorial(k - j)
                * math.factorial(2 * j - k)
            )
            s += num / den
        v[k - 1] = ((-1) ** (k + n2)) * s
    return v


def invert_stehfest(F: Callable[[float], float], t: float, n: int = 12) -> float:
    """Numerically invert a Laplace-domain scalar function F(s) at t>0."""
    if t <= 0:
        return 0.0
    v = stehfest_coefficients(n)
    ln2 = math.log(2.0)
    total = 0.0
    for k in range(1, n + 1):
        s = k * ln2 / t
        total += v[k - 1] * F(s)
    return ln2 / t * total


def sandal_active_storage_skin_laplace(s: float, rD: float, CD: float, skin: float) -> float:
    """Finite-radius active-well storage/skin interference kernel in Laplace space.

    Based on Sandal et al. (1978), pressure in the formation at r_D from an
    active well with constant surface rate, wellbore storage and steady-state skin.

    Negative skin can introduce a positive-real-axis pole in this formulation;
    callers should not use Stehfest blindly for negative skin.
    """
    if s <= 0 or rD < 1 or CD < 0:
        raise ValueError("Require s>0, rD>=1 and CD>=0.")
    z = math.sqrt(s)
    k0z = kv(0, z)
    k1z = kv(1, z)
    denom = s * (z * k1z + CD * s * (k0z + skin * z * k1z))
    return float(kv(0, rD * z) / denom)


def sandal_active_storage_skin(tD: np.ndarray | float, rD: float, CD: float, skin: float,
                               n_stehfest: int = 12) -> np.ndarray:
    """Real-space Sandal active-well storage/skin response."""
    if skin < 0:
        raise ValueError(
            "Negative skin is not inverted with Stehfest here because the transformed "
            "solution may contain a positive-axis pole. Use a finite skin-zone model or "
            "an explicitly validated negative-skin treatment."
        )
    arr = np.asarray(tD, dtype=float)
    out = np.zeros_like(arr)
    for i, tt in np.ndenumerate(arr):
        if tt > 0:
            out[i] = invert_stehfest(
                lambda s: sandal_active_storage_skin_laplace(s, rD, CD, skin),
                float(tt), n_stehfest
            )
    return out


@lru_cache(maxsize=8)
def _legendre_nodes_weights(nq: int):
    x, w = leggauss(nq)
    theta = math.pi * (x + 1.0)
    weight = math.pi * w
    return theta, weight


def tongpenyai_two_well_laplace(
    p: float,
    rD: float,
    CD1: float,
    s1: float,
    CD2: float,
    s2: float,
    pwD1_initial: float = 0.0,
    n_quad: int = 64,
) -> float:
    """Tongpenyai & Raghavan (1981) observation-well response in Laplace space.

    Well 1 = observation well; Well 2 = active well. Equal well radii are assumed,
    matching the derivation in the paper. This implements Appendix A, including the
    angular integrals beta1 and beta2 and Eqs. A-12 through A-14.
    """
    if p <= 0 or rD <= 2.0:
        # rD >> 1 in practical interference tests. rD<=2 causes near-touching finite wells.
        if p <= 0:
            raise ValueError("Laplace variable p must be > 0.")
        if rD <= 2.0:
            raise ValueError("Require rD>2 for separated finite-radius wells.")
    if CD1 < 0 or CD2 < 0:
        raise ValueError("Dimensionless wellbore storage must be nonnegative.")

    rootp = math.sqrt(p)
    theta, weights = _legendre_nodes_weights(n_quad)
    c = np.cos(theta)
    alpha1 = np.sqrt(1.0 + rD * rD - 2.0 * rD * c)
    C1 = (1.0 - rD * c) / alpha1
    beta1 = float(np.sum(weights * C1 * kv(1, alpha1 * rootp)) / (2.0 * math.pi))
    beta2 = float(np.sum(weights * kv(0, alpha1 * rootp)) / (2.0 * math.pi))

    k0 = float(kv(0, rootp))
    k1 = float(kv(1, rootp))
    p32 = p * rootp

    a3 = CD1 * p * k0 + CD1 * s1 * p32 * k1 + rootp * k1
    a4 = CD1 * p * beta2 + CD1 * s1 * p32 * beta1 + rootp * beta1
    a5 = CD2 * p * beta2 + CD2 * s2 * p32 * beta1 + rootp * beta1
    a6 = CD2 * p * k0 + CD2 * s2 * p32 * k1 + rootp * k1

    det = p * (a3 * a6 - a4 * a5)
    if det == 0 or not np.isfinite(det):
        return np.nan

    B1 = (CD1 * p * a6 * pwD1_initial - a4) / det
    B2 = (a3 - CD1 * p * a5 * pwD1_initial) / det

    return float(
        B1 * (k0 + s1 * rootp * k1)
        + B2 * (beta2 + s1 * rootp * beta1)
    )


def tongpenyai_two_well(
    tD: np.ndarray | float,
    rD: float,
    CD1: float,
    s1: float,
    CD2: float,
    s2: float,
    n_stehfest: int = 12,
    n_quad: int = 64,
) -> np.ndarray:
    """Numerically inverted Tongpenyai two-well storage/skin response."""
    if s1 < 0 or s2 < 0:
        raise ValueError(
            "This implementation intentionally blocks negative skin for Stehfest inversion. "
            "The literature documents positive-axis pole problems for negative skin."
        )
    arr = np.asarray(tD, dtype=float)
    out = np.zeros_like(arr)
    for i, tt in np.ndenumerate(arr):
        if tt > 0:
            out[i] = invert_stehfest(
                lambda p: tongpenyai_two_well_laplace(
                    p, rD, CD1, s1, CD2, s2, 0.0, n_quad
                ),
                float(tt), n_stehfest
            )
    return out

# -----------------------------------------------------------------------------
# Additional kernels: leaky aquifer and double-porosity interference
# -----------------------------------------------------------------------------

def hantush_well_function(u: np.ndarray | float, beta: np.ndarray | float) -> np.ndarray:
    """Hantush-Jacob well function W(u, beta) for a leaky aquifer.

    W(u,beta) = integral_u^inf exp(-y - beta**2/(4y)) / y dy.

    Parameters
    ----------
    u : float or array
        Classical Theis variable r^2/(4 eta t), u>=0.  At t<=0 callers
        should use a causal zero response rather than u=inf.
    beta : float or array
        Leakage-distance ratio r/B, where B is the leakage factor.

    Notes
    -----
    beta=0 reduces exactly to the Theis well function E1(u).
    """
    from scipy.integrate import quad

    ua, ba = np.broadcast_arrays(np.asarray(u, float), np.asarray(beta, float))
    out = np.empty_like(ua, dtype=float)
    for idx in np.ndindex(ua.shape):
        uu = float(ua[idx]); bb = abs(float(ba[idx]))
        if uu < 0:
            raise ValueError("u must be nonnegative.")
        if bb == 0.0:
            out[idx] = float(exp1(uu)) if uu > 0 else np.inf
            continue
        if uu == 0.0:
            out[idx] = float(2.0 * kv(0, bb))
            continue
        f = lambda y: math.exp(-y - bb*bb/(4.0*y)) / y
        out[idx] = quad(f, uu, np.inf, epsabs=1e-11, epsrel=1e-9, limit=200)[0]
    return out


def double_porosity_f_pss(s: float, omega: float, lam: float) -> float:
    """Warren-Root/Deruyck pseudo-steady interporosity transfer function f(s)."""
    if s <= 0 or not (0 < omega <= 1) or lam <= 0:
        raise ValueError("Require s>0, 0<omega<=1, lambda>0.")
    return (omega * (1.0 - omega) * s + lam) / ((1.0 - omega) * s + lam)


def double_porosity_f_transient_slab(s: float, omega: float, lam: float) -> float:
    """Deruyck et al. transient-interporosity f(s) for slab-shaped matrix blocks."""
    if s <= 0 or not (0 < omega <= 1) or lam <= 0:
        raise ValueError("Require s>0, 0<omega<=1, lambda>0.")
    if omega == 1.0:
        return 1.0
    z = math.sqrt(3.0 * (1.0 - omega) * s / lam)
    return omega + math.sqrt(lam * (1.0 - omega) / (3.0 * s)) * math.tanh(z)


def _coth(x: float) -> float:
    if abs(x) < 1e-5:
        # series coth(x)=1/x+x/3-x^3/45+...
        return 1.0/x + x/3.0 - x**3/45.0
    return 1.0 / math.tanh(x)


def double_porosity_f_transient_sphere(s: float, omega: float, lam: float) -> float:
    """Deruyck et al. transient-interporosity f(s) for spherical matrix blocks."""
    if s <= 0 or not (0 < omega <= 1) or lam <= 0:
        raise ValueError("Require s>0, 0<omega<=1, lambda>0.")
    if omega == 1.0:
        return 1.0
    z = math.sqrt(15.0 * (1.0 - omega) * s / lam)
    return omega + (lam / (5.0 * s)) * (z * _coth(z) - 1.0)


def double_porosity_line_source_laplace(
    s: float,
    rD: float,
    omega: float,
    lam: float,
    interporosity: str = "pss",
) -> float:
    """Laplace-domain observation-well line-source response for double porosity.

    Implements the line-source form used by Deruyck et al. for constant-rate
    interference tests: pbar_D = K0(r_D * sqrt(s f(s))) / s.
    """
    if rD <= 0:
        raise ValueError("rD must be positive.")
    if interporosity == "pss":
        ff = double_porosity_f_pss(s, omega, lam)
    elif interporosity in {"slab", "transient_slab"}:
        ff = double_porosity_f_transient_slab(s, omega, lam)
    elif interporosity in {"sphere", "transient_sphere"}:
        ff = double_porosity_f_transient_sphere(s, omega, lam)
    else:
        raise ValueError("interporosity must be 'pss', 'slab', or 'sphere'.")
    return float(kv(0, rD * math.sqrt(s * ff)) / s)


def double_porosity_line_source(
    tD: np.ndarray | float,
    rD: float,
    omega: float,
    lam: float,
    interporosity: str = "pss",
    n_stehfest: int = 12,
) -> np.ndarray:
    """Real-space double-porosity interference response p_D(t_D,r_D)."""
    arr = np.asarray(tD, dtype=float)
    if omega == 1.0:
        # Exact homogeneous limit; avoids unnecessary Laplace-inversion noise.
        out = np.zeros_like(arr)
        mask = arr > 0
        out[mask] = 0.5 * exp1((rD*rD) / (4.0 * arr[mask]))
        return out
    out = np.zeros_like(arr)
    for idx, tt in np.ndenumerate(arr):
        if tt > 0:
            out[idx] = invert_stehfest(
                lambda s: double_porosity_line_source_laplace(
                    s, rD, omega, lam, interporosity
                ),
                float(tt), n_stehfest
            )
    return out
