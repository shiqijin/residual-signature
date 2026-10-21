"""Exact Gaussian-law calibration of the frozen PL/right-Euler-kick scheme.

Two independent canonical coordinates have correlated fGn innovations.  The
first stored observation is epsilon[0], NOT an added deterministic zero.  An
Euler filter U[j] = (1-kappa/length)*U[j-1] + epsilon[j] defines the fOU grid.
All correlations with that first observation are retained in calibration.

The finite feature convention is precisely
  P[j] = R[j-1] tensor exp_tensor(U[j]-U[j-1]),
  R[j] = P[j] tensor (1 + delta_Gamma2[j] + delta_Gamma4[j]).
It is not a Wick signature, terminal mean subtraction, a within-cell continuous
Gamma equation, or an assertion about a continuous fractional rough lift.

Calibration uses the COMPLETE Gaussian innovation Cholesky factor and Gaussian
moment identities.  Q2 stores the homogeneous quadratic coefficient of R2;
T3 stores E[R3 tensor z].  This closes at K4 without a Markov assumption,
Monte Carlo calibration, labels, or a latent-rank approximation.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import cholesky

from ou_residual import residual_features

FROZEN_CODE = Path(__file__).resolve().parent

SCHEME = 'exact_PL_then_right_Euler_Gamma_population_centered'
CALIBRATION = 'full_Gaussian_latent_IBP_K4_v1'


def innovation_covariance(H: float, length: int) -> np.ndarray:
    """Exact covariance of length equal-grid fBm increments on [0,1]."""
    if not np.isfinite(H) or not 0 < H < 1:
        raise ValueError('H must lie strictly between zero and one')
    if isinstance(length, bool) or int(length) != length or length < 2:
        raise ValueError('length must be an integer >=2 counting observations')
    lag = np.arange(int(length))[:, None] - np.arange(int(length))[None, :]
    return (0.5 * (np.abs(lag + 1)**(2*H) + np.abs(lag - 1)**(2*H)
                   - 2*np.abs(lag)**(2*H)) / float(length)**(2*H))


def _filtered_load(innovation_cholesky: np.ndarray, coefficient: float) -> np.ndarray:
    """Return full anchored load(nt,2,2*nt), no eigenvalue/rank truncation."""
    length = len(innovation_cholesky)
    one = innovation_cholesky.copy()
    for j in range(1, length):
        one[j] += coefficient * one[j-1]
    one -= one[0:1].copy()
    load = np.zeros((length, 2, 2*length))
    load[:, 0, :length] = one
    load[:, 1, length:] = one
    return load


def calibrate_gaussian_load(load: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Full mixed K4 clocks for an arbitrary centered Gaussian path load.

    load[j,i,r] represents observed X[j,i] = sum_r load[j,i,r]*z[r],
    with z standard Gaussian.  A random initial observation is anchored here;
    it must not be deleted from the latent innovations before this operation.
    Returned mean witness is obtained from the moment recursion, not from a
    reference sample. Independent Gaussian cubature is needed to test it.
    """
    load = np.asarray(load, dtype=float)
    if load.ndim != 3 or load.shape[0] < 2 or load.shape[1] != 2:
        raise ValueError('load must have shape(observations>=2,2,latent_rank)')
    if not np.isfinite(load).all() or load.shape[2] < 1:
        raise ValueError('finite nonempty Gaussian load required')
    load = load - load[0:1]
    nt, _, rank = load.shape
    q2 = np.zeros((2, 2, rank, rank))
    mean2 = np.zeros((2, 2))
    mean4 = np.zeros((2, 2, 2, 2))
    first3 = np.zeros((2, 2, 2, rank))
    clocks = {k: np.zeros((nt, 2**k)) for k in (2, 3, 4)}
    means = np.zeros((nt, 31)); means[:, 0] = 1.
    for j in range(1, nt):
        x, v = load[j-1], load[j] - load[j-1]
        xv, vv = x @ v.T, v @ v.T
        mean_p2 = mean2 + xv + vv/2
        delta2 = -mean_p2

        # E[R2_ij v_k z_r], with the deterministic part represented by mean2.
        r2vz = np.einsum('ij,kr->ijkr', mean2, v)
        r2vz += np.einsum('ijrs,ks->ijkr', q2 + q2.swapaxes(-1, -2), v,
                         optimize=True)
        r2vv = np.einsum('ijkr,lr->ijkl', r2vz, v, optimize=True)
        xvvz = (np.einsum('ij,kr->ijkr', xv, v)
                 + np.einsum('ik,jr->ijkr', xv, v)
                 + np.einsum('jk,ir->ijkr', vv, x))
        vvvz = (np.einsum('ij,kr->ijkr', vv, v)
                 + np.einsum('ik,jr->ijkr', vv, v)
                 + np.einsum('jk,ir->ijkr', vv, v))
        mean_p4 = (mean4 + np.einsum('ijkr,lr->ijkl', first3, v)
                   + r2vv/2
                   + np.einsum('ijkr,lr->ijkl', xvvz, v)/6
                   + np.einsum('ijkr,lr->ijkl', vvvz, v)/24)
        delta4 = -mean_p4 - np.einsum('ij,kl->ijkl', mean_p2, delta2)

        # The right-end kick acts on P1, not on the old anchored X.
        first_p3 = first3 + r2vz + xvvz/2 + vvvz/6
        first3 = first_p3 + np.einsum('ir,jk->ijkr', x + v, delta2)
        q2 += (np.einsum('ir,js->ijrs', x, v)
               + np.einsum('ir,js->ijrs', v, v)/2)
        mean2 = mean_p2 + delta2
        mean4 = mean_p4 + np.einsum('ij,kl->ijkl', mean_p2, delta2) + delta4
        clocks[2][j] = clocks[2][j-1] + delta2.reshape(4)
        clocks[4][j] = clocks[4][j-1] + delta4.reshape(16)
        means[j, 3:7] = mean2.reshape(4)
        means[j, 15:31] = mean4.reshape(16)
    if not np.isfinite(means).all() or not all(np.isfinite(c).all() for c in clocks.values()):
        raise FloatingPointError('nonfinite Gaussian calibration')
    return clocks, means


@lru_cache(maxsize=64)
def _prepare_cached(H: float, kappa: float, length: int) -> dict:
    if not np.isfinite(kappa) or kappa < 0:
        raise ValueError('kappa must be finite and nonnegative')
    cov = innovation_covariance(H, length)
    factor = cholesky(cov, lower=True, check_finite=True)
    coefficient = 1. - kappa/length
    load = _filtered_load(factor, coefficient)
    gammas, witness = calibrate_gaussian_load(load)
    factor_error = float(np.max(np.abs(factor @ factor.T - cov)))
    scale = float(np.max(np.abs(cov)))
    if factor_error > max(1e-14, 1e-10*scale):
        raise FloatingPointError('innovation Cholesky covariance mismatch')
    metadata = dict(
        H=H, kappa=kappa, length=length, signature_intervals=length-1,
        dt=1./length, ar_coefficient=coefficient,
        innovation_variance=float(cov[0, 0]), innovation_distribution='Gaussian',
        innovation_covariance='exact_equal_grid_fractional_Gaussian_noise',
        path_law='fBm_grid' if kappa == 0 else 'Euler_fOU_grid',
        initial_observation='epsilon[0], correlated with future innovations',
        latent_initial_zero_included=False, full_innovation_rank=2*length,
        latent_rank_truncated=False, spatial_channels='independent canonical',
        calibration=CALIBRATION, scheme=SCHEME, Gamma3_identically_zero=True,
        calibration_uses_labels=False, calibration_uses_Monte_Carlo=False,
        exact_continuous_fractional_lift_claim=False,
        cholesky_covariance_max_error=factor_error,
        recursion_mean_max_error=float(np.max(np.abs(witness[:, 1:]))),
        frozen_feature_kernel=str(FROZEN_CODE / 'ou_residual.py'),
        frozen_feature_kernel_sha256=hashlib.sha256(
            (FROZEN_CODE / 'ou_residual.py').read_bytes()).hexdigest())
    for value in [factor, witness, *gammas.values()]:
        value.setflags(write=False)
    return dict(innovation_cholesky=factor, gammas=gammas, mean_witness=witness,
                metadata=metadata)


def prepare_law(H: float, kappa: float, length: int = 100) -> dict:
    """Prepare deterministic law-only resources; callers get independent arrays."""
    innovation_covariance(H, length)  # Validate before cache-key integer conversion.
    law = _prepare_cached(float(H), float(kappa), int(length))
    return dict(innovation_cholesky=law['innovation_cholesky'].copy(),
                gammas={k: v.copy() for k, v in law['gammas'].items()},
                mean_witness=law['mean_witness'].copy(), metadata=law['metadata'].copy())


def filter_innovations(innovations: np.ndarray, coefficient: float) -> np.ndarray:
    """Input/output (n,observations,2), retaining the random first observation."""
    paths = np.asarray(innovations, dtype=float).copy()
    if paths.ndim != 3 or paths.shape[2] != 2 or paths.shape[1] < 2:
        raise ValueError('innovations must have shape(n,observations>=2,2)')
    if not np.isfinite(paths).all() or not np.isfinite(coefficient):
        raise ValueError('finite innovations and coefficient required')
    for j in range(1, paths.shape[1]):
        paths[:, j] += coefficient * paths[:, j-1]
    return np.ascontiguousarray(paths)


def sample_canonical(n: int, rng: np.random.Generator, law: dict) -> np.ndarray:
    """Sample exact declared finite Gaussian grid; no external covariance draws."""
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be numpy.random.Generator')
    if isinstance(n, bool) or int(n) != n or n < 1:
        raise ValueError('n must be a positive integer')
    length = law['metadata']['length']
    eps = rng.standard_normal((int(n), 2, length)) @ law['innovation_cholesky'].T
    return filter_innovations(eps.transpose(0, 2, 1), law['metadata']['ar_coefficient'])


__all__ = ['prepare_law', 'sample_canonical', 'residual_features',
           'innovation_covariance', 'calibrate_gaussian_load', 'filter_innovations']
