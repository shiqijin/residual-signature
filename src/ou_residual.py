"""Exact population centering for a specified finite-step tensor scheme.

The observed Markov chain is U[0]=epsilon[0],
U[j]=a*U[j-1]+epsilon[j].  Its first observation is NOT a deterministic
origin: the signature is based on U[j]-U[0], with length-1 increments.
The two canonical innovation coordinates are independent, identically
distributed, centered and symmetric.  Through degree four only variance
and fourth moment are required.  Gaussian and Rademacher laws are supported.

For each observed cell the precise feature convention is

    P_j = R_{j-1} tensor exp_tensor(U[j]-U[j-1]),
    R_j = P_j tensor (1 + delta_Gamma2 + delta_Gamma3 + delta_Gamma4).

The second operation is a right Euler kick, not exp_tensor(delta_Gamma).
Its clocks are determined in increasing degree by E[R_j^w]=0.  Thus this
module exactly calibrates the old reproduction's finite-step update, but
does NOT identify it with the continuous-FV canonical RDE at finite mesh.
In particular Brownian Gamma4 can contain a nonzero discretization term;
Brownian R is NOT forcibly replaced by numerical left-point Ito features.

Calibration uses a deterministic Markov moment recursion on the 99 states
E[R^w U1^p U2^q], |w|+p+q<=4.  There is no reference sample, fitting, target,
or Monte Carlo approximation.  A fixed linear transformation L transports
both features and clocks by L^{tensor degree}; a scalar drift commutes with L.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numba import njit


LEVEL = 4
DIMENSION = 2
OFFSETS = np.array([0, 1, 3, 7, 15], dtype=np.int64)
FEATURE_DIMENSION = 31
SCHEME = "exact_PL_then_right_Euler_Gamma_population_centered"


def _moment(power: int, variance: float, fourth_factor: float) -> float:
    if power == 0:
        return 1.0
    if power == 2:
        return variance
    if power == 4:
        return fourth_factor * variance**2
    if power in (1, 3):
        return 0.0
    raise ValueError("Only moments through degree four are available.")


def _states():
    states = []
    for degree in range(5):
        for word in range(2**degree):
            for p in range(5 - degree):
                for q in range(5 - degree - p):
                    states.append((degree, word, p, q))
    return states, {state: i for i, state in enumerate(states)}


STATES, STATE_INDEX = _states()


def _transition_polynomial(dx_power, current_power, coefficient, variance,
                           fourth_factor):
    """Coefficients of E_eps[((a-1)U+eps)^p (aU+eps)^q]."""
    result = np.zeros(dx_power + current_power + 1)
    for left in range(dx_power + 1):
        for right in range(current_power + 1):
            u_power = left + right
            eps_power = dx_power + current_power - u_power
            result[u_power] += (
                math.comb(dx_power, left) * math.comb(current_power, right)
                * (coefficient - 1.0)**left * coefficient**right
                * _moment(eps_power, variance, fourth_factor)
            )
    return result


def _transition_matrix(coefficient, variance, fourth_factor):
    transition = np.zeros((len(STATES), len(STATES)))
    polys = {
        (p, q): _transition_polynomial(p, q, coefficient, variance, fourth_factor)
        for p in range(5) for q in range(5 - p)
    }
    for row, (degree, word, p, q) in enumerate(STATES):
        for suffix_degree in range(degree + 1):
            suffix = word & ((1 << suffix_degree) - 1)
            count2 = suffix.bit_count()
            count1 = suffix_degree - count2
            prefix = word >> suffix_degree
            factor = 1.0 / math.factorial(suffix_degree)
            poly1, poly2 = polys[count1, p], polys[count2, q]
            for i, coeff1 in enumerate(poly1):
                for j, coeff2 in enumerate(poly2):
                    if coeff1 != 0.0 and coeff2 != 0.0:
                        col = STATE_INDEX[degree - suffix_degree, prefix, i, j]
                        transition[row, col] += factor * coeff1 * coeff2
    return transition


def _validate_law(coefficient, length, innovation_variance,
                  innovation_fourth_factor):
    if isinstance(length, bool) or int(length) != length or length < 2:
        raise ValueError("length must be an integer >=2, counting observed states.")
    if not np.isfinite(coefficient):
        raise ValueError("coefficient must be finite.")
    if not np.isfinite(innovation_variance) or innovation_variance <= 0:
        raise ValueError("innovation_variance must be finite and positive.")
    if (not np.isfinite(innovation_fourth_factor)
            or innovation_fourth_factor < 1.0):
        raise ValueError("fourth_factor=E[eps^4]/Var(eps)^2 must be >=1.")


@lru_cache(maxsize=64)
def _calibrate(coefficient, length, innovation_variance,
               innovation_fourth_factor):
    _validate_law(coefficient, length, innovation_variance,
                  innovation_fourth_factor)
    transition = _transition_matrix(coefficient, innovation_variance,
                                    innovation_fourth_factor)
    moments = np.zeros(len(STATES))
    for row, (degree, _, p, q) in enumerate(STATES):
        if degree == 0:
            moments[row] = (
                _moment(p, innovation_variance, innovation_fourth_factor)
                * _moment(q, innovation_variance, innovation_fourth_factor)
            )
    gammas = {degree: np.zeros((length, 2**degree)) for degree in (2, 3, 4)}
    means = np.zeros((length, 31))
    means[:, 0] = 1.0
    for step in range(1, length):
        geometric = transition @ moments
        delta = {degree: np.zeros(2**degree) for degree in (2, 3, 4)}
        for degree in (2, 3, 4):
            for word in range(2**degree):
                value = geometric[STATE_INDEX[degree, word, 0, 0]]
                for r in range(2, degree):
                    prefix = word >> r
                    suffix = word & ((1 << r) - 1)
                    value += (geometric[STATE_INDEX[degree-r, prefix, 0, 0]]
                              * delta[r][suffix])
                delta[degree][word] = -value
            gammas[degree][step] = gammas[degree][step-1] + delta[degree]
        moments = geometric.copy()
        for row, (degree, word, p, q) in enumerate(STATES):
            for r in range(2, degree + 1):
                prefix = word >> r
                suffix = word & ((1 << r) - 1)
                moments[row] += (
                    geometric[STATE_INDEX[degree-r, prefix, p, q]]
                    * delta[r][suffix]
                )
            if p == 0 and q == 0:
                means[step, int(OFFSETS[degree]) + word] = moments[row]
    if not all(np.isfinite(x).all() for x in gammas.values()):
        raise FloatingPointError("Moment recursion overflowed; unsupported numerical scale.")
    # Immutable cached arrays: public callers receive independent copies.
    for value in gammas.values():
        value.setflags(write=False)
    means.setflags(write=False)
    return gammas, means


def exact_markov_gammas(coefficient: float, length: int = 100,
                       innovation_variance: float = 1.0,
                       innovation_fourth_factor: float = 3.0):
    """Return deterministic prefix clocks {2:(L,4),3:(L,8),4:(L,16)}.

    U[0] has the same independent-coordinate law as each innovation.  For
    Gaussian AR use fourth_factor=3; independent Rademacher RW uses 1.
    This API does not silently assume a stationary or deterministic U[0].
    """
    _validate_law(coefficient, length, innovation_variance,
                  innovation_fourth_factor)
    gamma, _ = _calibrate(float(coefficient), int(length),
                          float(innovation_variance),
                          float(innovation_fourth_factor))
    return {degree: value.copy() for degree, value in gamma.items()}


def exact_ou_gammas(kappa: float, length: int = 100):
    """Author Euler OU: a=1-kappa/length, Var(eps)=1/length, observed U0=eps0."""
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("OU kappa must be finite and nonnegative.")
    _validate_law(0.0, length, 1.0, 3.0)
    return exact_markov_gammas(1.0-kappa/length, length, 1.0/length, 3.0)


@njit(cache=True)
def _batch_features(paths, gamma_increments, save_prefix):
    n_paths, length, _ = paths.shape
    width = length if save_prefix else 1
    result = np.zeros((n_paths, width, 31))
    offsets = np.array([0, 1, 3, 7, 15])
    for row in range(n_paths):
        state = np.zeros(31)
        state[0] = 1.0
        if save_prefix:
            result[row, 0] = state
        for step in range(1, length):
            dx0 = paths[row, step, 0] - paths[row, step-1, 0]
            dx1 = paths[row, step, 1] - paths[row, step-1, 1]
            local = np.zeros(31)
            local[0] = 1.0
            for degree in range(1, 5):
                for word in range(1 << degree):
                    increment = dx0 if (word & 1) == 0 else dx1
                    local[offsets[degree]+word] = (
                        local[offsets[degree-1]+(word >> 1)]*increment/degree
                    )
            geometric = np.zeros(31)
            for degree in range(5):
                for word in range(1 << degree):
                    for r in range(degree + 1):
                        geometric[offsets[degree]+word] += (
                            state[offsets[degree-r]+(word >> r)]
                            * local[offsets[r]+(word & ((1 << r)-1))]
                        )
            state = geometric.copy()
            for degree in range(2, 5):
                for word in range(1 << degree):
                    for r in range(2, degree+1):
                        state[offsets[degree]+word] += (
                            geometric[offsets[degree-r]+(word >> r)]
                            * gamma_increments[step, offsets[r]+(word & ((1 << r)-1))]
                        )
            if save_prefix:
                result[row, step] = state
        if not save_prefix:
            result[row, 0] = state
    return result


def _validate_arrays(paths, gammas):
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 3 or paths.shape[2] != 2 or paths.shape[1] < 2:
        raise ValueError("paths must have shape (n, length>=2, 2).")
    if not np.isfinite(paths).all():
        raise ValueError("paths must be finite.")
    if set(gammas) - {2, 3, 4}:
        raise ValueError("Gamma degrees must belong to {2,3,4}.")
    increments = np.zeros((paths.shape[1], 31))
    for degree in (2, 3, 4):
        value = np.asarray(gammas.get(degree, np.zeros((paths.shape[1], 2**degree))),
                           dtype=np.float64)
        if value.shape != (paths.shape[1], 2**degree) or not np.isfinite(value).all():
            raise ValueError(f"Gamma{degree} has an invalid shape or nonfinite values.")
        if np.any(value[0] != 0.0):
            raise ValueError("Gamma must be zero at the first observed path point.")
        begin = int(OFFSETS[degree])
        increments[1:, begin:begin+2**degree] = np.diff(value, axis=0)
    return np.ascontiguousarray(paths), increments


def residual_features(canonical_paths, gammas):
    """n by 31 terminal features, empty word first, lexicographic word order.

    Apply clocks to paths in the same coordinates.  For canonical clocks on
    independent channels, compute these features before transporting by L.
    Values at the first point may be random; only successive increments enter.
    """
    paths, increments = _validate_arrays(canonical_paths, gammas)
    return _batch_features(paths, increments, False)[:, 0]

