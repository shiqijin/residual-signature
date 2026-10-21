"""Exact finite Gaussian observation laws for the standalone G/R experiment.

The first observation is stationary and random.  ``observation_load`` maps a
200-dimensional standard Gaussian into all 100 two-dimensional observations;
the shared Gaussian calibrator, and the signature kernel, anchor at observation
zero.  No initial covariance is discarded.  Spatial Wishart mixing belongs to
the caller and is not applied here.

The rotating OU convention is dU=(-kappa I + omega J)U dt + sqrt(2 kappa)dB,
J=[[0,-1],[1,0]], with U(0)~N(0,I) in stationarity.  Matérn marginal variance
is one, and its length parameter has the usual sqrt(2 nu)/ell convention.
Only the input law changes: these resources do not select supports, generate
responses, calibrate Gamma, or fit regressions.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import math

import numpy as np
from scipy.linalg import cholesky, eigvalsh

VERSION = "stationary_matern_rotating_ou_full_observation_covariance_v1"


def _grid(length: int, dt: float) -> np.ndarray:
    if isinstance(length, bool) or int(length) != length or length < 2:
        raise ValueError("length must be an integer >= 2 counting observations")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and strictly positive")
    return np.arange(int(length), dtype=np.float64) * float(dt)


def matern_covariance(times: np.ndarray, nu: float, ell: float) -> np.ndarray:
    """Unit-variance stationary Matérn covariance for nu=1/2,3/2,5/2.

    nu=1/2 is exposed for the OU boundary validation, not in the declared grid.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("times must be a finite one-dimensional observation grid")
    if nu not in (.5, 1.5, 2.5) or not np.isfinite(ell) or ell <= 0:
        raise ValueError("nu must be .5, 1.5, or 2.5 and ell strictly positive")
    distance = np.abs(times[:, None] - times[None, :])
    z = math.sqrt(2 * nu) * distance / float(ell)
    if nu == .5:
        polynomial = np.ones_like(z)
    elif nu == 1.5:
        polynomial = 1 + z
    else:
        polynomial = 1 + z + z * z / 3
    return polynomial * np.exp(-z)


def rotating_ou_covariance(times: np.ndarray, kappa: float,
                           omega: float) -> np.ndarray:
    """Return E[U(s) U(t)^T] in time-major/channel-minor flattened order.

    A block at (s,t) is exp(-kappa*abs(s-t))*R(omega*(s-t)).  In particular,
    Cov(U_1(0), U_2(h)) is positive when sin(omega*h)>0.  This explicit sign
    convention matters: the within-time covariance is I, but cross-lag blocks
    are generally not symmetric.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
        raise ValueError("times must be a finite one-dimensional observation grid")
    if not np.isfinite(kappa) or kappa <= 0 or not np.isfinite(omega):
        raise ValueError("kappa must be strictly positive and omega finite")
    lag = times[:, None] - times[None, :]
    decay = np.exp(-float(kappa) * np.abs(lag))
    cosine, sine = np.cos(float(omega) * lag), np.sin(float(omega) * lag)
    covariance = np.empty((len(times), 2, len(times), 2), dtype=np.float64)
    covariance[:, 0, :, 0] = decay * cosine
    covariance[:, 1, :, 1] = decay * cosine
    covariance[:, 0, :, 1] = -decay * sine
    covariance[:, 1, :, 0] = decay * sine
    return np.ascontiguousarray(covariance.reshape(2 * len(times), -1))


def build_laws(length: int = 100, dt: float = .01) -> list[dict]:
    """Return the ordered 12-law grid from the shared experiment registry."""
    from registry import laws
    _grid(length, dt)
    return laws(length=length, dt=dt)


def _cache_key(law: dict) -> tuple:
    if not isinstance(law, dict):
        raise TypeError("law must be a specification dictionary")
    length, dt = law.get("length", 100), law.get("dt", .01)
    _grid(length, dt)
    family = law.get("family")
    if family == "matern":
        nu, scale = float(law["nu"]), float(law["ell_over_T"])
        if nu not in (1.5, 2.5) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Matérn law requires nu=1.5 or 2.5 and ell_over_T>0")
        params = (nu, scale)
    elif family == "rotating_ou":
        kappa_T, omega_T = float(law["kappa_T"]), float(law["omega_T"])
        if not np.isfinite(kappa_T) or kappa_T <= 0 or not np.isfinite(omega_T):
            raise ValueError("rotating OU law requires kappa_T>0 and finite omega_T")
        params = (kappa_T, omega_T)
    else:
        raise ValueError(f"unknown Gaussian law family: {family!r}")
    return family, int(length), float(dt), *params


@lru_cache(maxsize=64)
def _prepare_cached(key: tuple) -> dict:
    family, length, dt, parameter1, parameter2 = key
    times = _grid(length, dt)
    horizon = float(times[-1] - times[0])
    if family == "matern":
        nu, ell_over_T = parameter1, parameter2
        ell = ell_over_T * horizon
        one_covariance = matern_covariance(times, nu, ell)
        covariance = np.kron(one_covariance, np.eye(2))
        parameters = dict(nu=nu, ell_over_T=ell_over_T, ell=ell)
        channels = "independent canonical channels, identical Matérn temporal covariance"
        formula = "(1+z)exp(-z) for nu=1.5; (1+z+z^2/3)exp(-z) for nu=2.5; z=sqrt(2nu)|s-t|/ell"
    else:
        kappa_T, omega_T = parameter1, parameter2
        kappa, omega = kappa_T / horizon, omega_T / horizon
        covariance = rotating_ou_covariance(times, kappa, omega)
        parameters = dict(kappa_T=kappa_T, omega_T=omega_T,
                          omega_T_over_pi=omega_T / math.pi, kappa=kappa, omega=omega)
        channels = "coupled canonical channels with antisymmetric cross-lag covariance"
        formula = "Cov(U(s),U(t))=exp(-kappa|s-t|)R(omega(s-t)); J=[[0,-1],[1,0]]"

    # The full covariance includes the stationary first observation.  There is
    # no diagonal jitter, spectral truncation, or independent-start substitute.
    eigenvalues = eigvalsh(covariance, check_finite=True)
    if eigenvalues[0] <= 0:
        raise np.linalg.LinAlgError(
            f"non-positive covariance eigenvalue {eigenvalues[0]:.17g}; no jitter applied")
    factor = cholesky(covariance, lower=True, check_finite=True)
    reconstructed = factor @ factor.T
    error = float(np.max(np.abs(reconstructed - covariance)))
    relative_error = float(np.linalg.norm(reconstructed - covariance, ord="fro")
                           / np.linalg.norm(covariance, ord="fro"))
    if error > 2e-12 or not np.isfinite(factor).all():
        raise FloatingPointError(f"Gaussian covariance factorization mismatch: {error}")
    load = np.ascontiguousarray(factor.reshape(length, 2, 2 * length))
    metadata = dict(
        version=VERSION, family=family, **parameters, length=length, dt=dt,
        horizon=horizon, signature_intervals=length - 1,
        distribution="centered Gaussian", observation_law="stationary exact finite-grid covariance",
        initial_observation="stationary random U(0) with covariance I; all initial/future covariance retained",
        stationary_initialization=True, marginal_variance=1.,
        canonical_channels=channels, covariance_formula=formula,
        flattening_order="time-major, channel-minor", latent_rank=2 * length,
        latent_rank_truncated=False, diagonal_jitter=0.,
        spatial_mixing="caller applies existing Wishart Cholesky tensor transport",
        anchoring="generic calibrate_gaussian_load and signature kernel subtract observation zero",
        calibration_uses_same_observation_load_as_sampling=True,
        covariance_min_eigenvalue=float(eigenvalues[0]),
        covariance_max_eigenvalue=float(eigenvalues[-1]),
        covariance_condition_number_2=float(eigenvalues[-1] / eigenvalues[0]),
        cholesky_covariance_max_error=error,
        cholesky_covariance_relative_frobenius_error=relative_error,
        observation_covariance_sha256=hashlib.sha256(covariance.tobytes()).hexdigest(),
        observation_load_sha256=hashlib.sha256(load.tobytes()).hexdigest(),
    )
    for array in (times, load, covariance):
        array.setflags(write=False)
    return dict(observation_load=load, observation_covariance=covariance,
                times=times, metadata=metadata)


def prepare_law(law: dict) -> dict:
    """Return deterministic law-only resources, with independent mutable copies.

    ``observation_load[t,i,r]`` maps independent standard normals to UNANCHORED
    observations.  Call ``calibrate_gaussian_load(resources['observation_load'])``
    to obtain all mixed degree-four clocks. Caching is in memory only.
    """
    resources = _prepare_cached(_cache_key(law))
    result = {name: value.copy() for name, value in resources.items()
              if isinstance(value, np.ndarray)}
    result["metadata"] = deepcopy(resources["metadata"])
    result["metadata"]["law_key"] = law.get("law_key", "custom")
    return result


def sample_canonical(n: int, rng: np.random.Generator, resources: dict) -> np.ndarray:
    """Sample unanchored (n,length,2) paths using exactly observation_load.

    Latent RNG layout is (sample,time-major/channel-minor latent index).
    This is shared by both methods; it is not an Euler time discretization.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    load = np.asarray(resources["observation_load"], dtype=np.float64)
    if load.ndim != 3 or load.shape[1] != 2 or not np.isfinite(load).all():
        raise ValueError("finite observation_load with shape(length,2,rank) required")
    normal = rng.standard_normal((int(n), load.shape[2]))
    paths = normal @ load.reshape(-1, load.shape[2]).T
    return np.ascontiguousarray(paths.reshape(int(n), load.shape[0], 2))


__all__ = ["VERSION", "build_laws", "prepare_law", "sample_canonical",
           "matern_covariance", "rotating_ou_covariance"]
