"""Two-driver Gaussian laws for the unchanged d=2, K=4 Guo experiment.

Canonical driver 1 has exact-grid fGn innovations with Hurst H; driver 2
has independent Brownian innovations. Both use a=1-kappa/length. Observed
U[0] is the FIRST innovation, not a prepended zero. The signature has
length-1 increments; calibration retains all correlations with U[0].

This module changes only the input law. It neither samples spatial Wishart
factors nor constructs responses, scales columns, selects supports, or fits.
The caller uses the original Guo Cholesky/tensor-transport convention.
After spatial mixing the observed coordinates need not be pure fBM/BM.

The retained finite convention is exact PL then right Euler Gamma kick,
population-calibrated through degree four; no continuous-lift equality is
claimed. Full latent rank is retained, including the random initial point.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.linalg import cholesky

from fractional_features import (
    CALIBRATION, SCHEME, calibrate_gaussian_load, innovation_covariance,
)

VERSION = "mixed_fGn_BM_full_Gaussian_load_K4_v1"


@lru_cache(maxsize=64)
def _prepare_cached(H: float, kappa: float, length: int) -> dict:
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    coefficient = 1.0 - kappa / length
    factors = np.stack([
        cholesky(innovation_covariance(h, length), lower=True, check_finite=True)
        for h in (H, .5)
    ])
    load = np.zeros((length, 2, 2 * length))
    covariances = np.empty((2, length, length))
    max_factor_error = 0.0
    for channel, h in enumerate((H, .5)):
        one = factors[channel].copy()
        for j in range(1, length):
            one[j] += coefficient * one[j - 1]
        # Do NOT discard epsilon[0] or make it independent of future fGn.
        load[:, channel, channel * length:(channel + 1) * length] = one
        covariances[channel] = one @ one.T
        max_factor_error = max(max_factor_error, float(np.max(np.abs(
            factors[channel] @ factors[channel].T
            - innovation_covariance(h, length)))))
    gammas, witness = calibrate_gaussian_load(load)
    metadata = dict(
        version=VERSION, H=H, H1=H, H2=.5, kappa=kappa,
        length=length, dt=1.0 / length, signature_intervals=length - 1,
        ar_coefficient=coefficient, innovation_distribution="Gaussian",
        innovation_variances=[length ** (-2 * H), 1.0 / length],
        path_law="mixed_fBm_BM_grid" if kappa == 0 else "mixed_Euler_fOU_OU_grid",
        canonical_channels="independent; Hurst=(H,0.5); common Euler coefficient",
        initial_observation="epsilon[0]; full initial/future covariance retained",
        latent_initial_zero_included=False, full_innovation_rank=2 * length,
        latent_rank_truncated=False, calibration=CALIBRATION, scheme=SCHEME,
        Gamma3_identically_zero=True, calibration_uses_labels=False,
        calibration_uses_Monte_Carlo=False,
        exact_continuous_fractional_lift_claim=False,
        cholesky_covariance_max_error=max_factor_error,
        recursion_mean_max_error=float(np.max(np.abs(witness[:, 1:]))),
        spatial_mixing="caller applies original Guo Wishart Cholesky tensor transport",
    )
    if max_factor_error > 1e-11 or not np.isfinite(load).all():
        raise FloatingPointError("invalid finite Gaussian load")
    for value in (factors, load, covariances, witness, *gammas.values()):
        value.setflags(write=False)
    return dict(innovation_cholesky=factors, observation_load=load,
                observation_covariances=covariances, gammas=gammas,
                mean_witness=witness, metadata=metadata)


def prepare_mixed_law(H: float, kappa: float, length: int = 100) -> dict:
    """Return law-only resources with independent mutable array copies.

    H1=H, H2=.5. ``observation_load[t,i,r]`` maps 2*length independent
    standard normals into the UNANCHORED observed canonical coordinates.
    ``gammas`` are anchored at the first observation by the shared calibrator.
    """
    innovation_covariance(H, length)  # Validate before integer coercion/cache.
    law = _prepare_cached(float(H), float(kappa), int(length))
    return dict(
        innovation_cholesky=law["innovation_cholesky"].copy(),
        observation_load=law["observation_load"].copy(),
        observation_covariances=law["observation_covariances"].copy(),
        gammas={k: v.copy() for k, v in law["gammas"].items()},
        mean_witness=law["mean_witness"].copy(), metadata=law["metadata"].copy(),
    )


def sample_mixed_canonical(n: int, rng: np.random.Generator, law: dict) -> np.ndarray:
    """Sample (n,length,2), with original Guo channel-then-time RNG layout."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    length = law["metadata"]["length"]
    normals = rng.standard_normal((int(n), 2, length))
    result = np.empty((int(n), length, 2))
    for channel in range(2):
        result[:, :, channel] = normals[:, channel] @ law["innovation_cholesky"][channel].T
    coefficient = law["metadata"]["ar_coefficient"]
    for j in range(1, length):
        result[:, j] += coefficient * result[:, j - 1]
    return np.ascontiguousarray(result)


__all__ = ["prepare_mixed_law", "sample_mixed_canonical", "VERSION"]
