"""Gaussian path sampling, geometric signature features, and Lasso recovery."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True
import json
import math
from pathlib import Path
import warnings
import numpy as np
from numba import njit

LEVEL = 4
DIMENSION = 2
WIDTH = 31
OFFSETS = (0, 1, 3, 7, 15)


def sample_ou(n_paths: int, kappa: float, rng: np.random.Generator,
              length: int = 100, cholesky: np.ndarray | None = None,
              random_covariance: bool = True) -> dict:
    """Euler OU with the author's observation indexing and Wishart(2,I) law.

    For j=0,...,length-1: U_j = a U_{j-1} + epsilon_j, U_{-1}=0,
    a=1-kappa/length, epsilon_j~N(0,I/length). All n_paths in this call
    share one covariance factor. Negative kappa is permitted for source replay.
    Wishart is sampled as A A.T for a 2x2 iid Gaussian A, without jitter,
    rejection, trace normalization, or trimming nearly singular draws.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError('rng must be numpy.random.Generator')
    if not isinstance(n_paths, (int,np.integer)) or n_paths < 1:
        raise ValueError('n_paths must be a positive integer')
    if not isinstance(length, (int,np.integer)) or length < 2:
        raise ValueError('length must be an integer >=2')
    if not np.isfinite(kappa):
        raise ValueError('kappa must be finite')
    if cholesky is None:
        if random_covariance:
            a_gaussian = rng.standard_normal((2,2))
            omega = a_gaussian @ a_gaussian.T
            factor = np.linalg.cholesky(omega)
        else:
            factor = np.eye(2)
            omega = np.eye(2)
    else:
        factor = np.asarray(cholesky,dtype=np.float64)
        if factor.shape != (2,2) or not np.isfinite(factor).all():
            raise ValueError('cholesky must be a finite 2x2 lower-triangular factor')
        if not np.array_equal(factor,np.tril(factor)) or np.any(np.diag(factor)<=0):
            raise ValueError('cholesky must be lower triangular with positive diagonal')
        omega = factor @ factor.T
    dt = 1.0/length
    ar = 1.0-float(kappa)*dt
    # Layout matches the author within-path channel-then-time innovation order.
    z = rng.standard_normal((n_paths,2,length))*math.sqrt(dt)
    for j in range(1,length):
        z[:,:,j] += ar*z[:,:,j-1]
    canonical = np.ascontiguousarray(z.transpose(0,2,1))
    paths = np.ascontiguousarray(canonical @ factor.T)
    return dict(paths=paths,canonical_paths=canonical,omega=omega,cholesky=factor.copy(),
                dt=dt,ar_coefficient=ar,kappa=float(kappa),length=int(length),
                signature_intervals=int(length-1),latent_initial_zero_included=False,
                process='ou',parameter=float(kappa),innovation_variance=dt)


def sample_process(n_paths: int, process: str, parameter: float,
                   rng: np.random.Generator, length: int = 100,
                   cholesky: np.ndarray | None = None,
                   random_covariance: bool = True) -> dict:
    """Gaussian OU or AR(1) observations with one random spatial covariance."""
    name=process.lower()
    if name=='ou':
        return sample_ou(n_paths,parameter,rng,length,cholesky,random_covariance)
    if not isinstance(rng,np.random.Generator):
        raise TypeError('rng must be numpy.random.Generator')
    if not isinstance(n_paths,(int,np.integer)) or n_paths<1 or not isinstance(length,(int,np.integer)) or length<2:
        raise ValueError('positive n_paths and integer length>=2 required')
    if not np.isfinite(parameter):
        raise ValueError('parameter must be finite')
    if name in ('ar','ar1'):
        phi=float(parameter)
        if abs(phi)>1:
            raise ValueError('author AR range is -1<=phi<=1; visible panel uses 0<=phi<=1')
        if cholesky is None:
            if random_covariance:
                a_gaussian=rng.standard_normal((2,2))
                omega=a_gaussian@a_gaussian.T
                factor=np.linalg.cholesky(omega)
            else:
                omega=np.eye(2);factor=np.eye(2)
        else:
            factor=np.asarray(cholesky,dtype=np.float64)
            if factor.shape!=(2,2) or not np.isfinite(factor).all() or not np.array_equal(factor,np.tril(factor)) or np.any(np.diag(factor)<=0):
                raise ValueError('invalid lower-triangular Cholesky factor')
            omega=factor@factor.T
        z=rng.standard_normal((n_paths,2,length))
        for j in range(1,length):
            z[:,:,j] += phi*z[:,:,j-1]
        name='ar1';ar=phi
    else:
        raise ValueError('process must be ou or ar/ar1')
    canonical=np.ascontiguousarray(z.transpose(0,2,1))
    paths=np.ascontiguousarray(canonical@factor.T)
    return dict(paths=paths,canonical_paths=canonical,omega=omega,cholesky=factor.copy(),
                dt=1./length,ar_coefficient=ar,kappa=None,length=int(length),
                signature_intervals=int(length-1),latent_initial_zero_included=False,
                process=name,parameter=float(parameter),innovation_variance=1.)


@njit(cache=False)
def _features(paths):
    n, nt, _ = paths.shape
    geom = np.zeros((n,31))
    offsets = np.array([0,1,3,7,15])
    for i in range(n):
        sg=np.zeros(31)
        sg[0]=1.
        for j in range(nt-1):
            dx=paths[i,j+1]-paths[i,j]
            og=sg.copy()
            increment=np.zeros(31);increment[0]=1.
            for degree in range(1,5):
                off=offsets[degree];low=offsets[degree-1]
                for word in range(1<<degree):
                    before=word//2;letter=word%2
                    increment[off+word]=increment[low+before]*dx[letter]/degree
            for degree in range(1,5):
                for word in range(1<<degree):
                    value=0.
                    for suffix_degree in range(degree+1):
                        modulus=1<<suffix_degree
                        value += og[offsets[degree-suffix_degree]+word//modulus]*increment[offsets[suffix_degree]+word%modulus]
                    sg[offsets[degree]+word]=value
        geom[i]=sg
    return geom


def features(paths: np.ndarray) -> dict[str,np.ndarray]:
    """Compute the exact piecewise-linear geometric signature through degree four."""
    p=np.asarray(paths,dtype=np.float64)
    if p.ndim!=3 or p.shape[2]!=2 or p.shape[0]<1 or p.shape[1]<2:
        raise ValueError('paths must have shape (n>=1, times>=2, 2)')
    if not np.isfinite(p).all():
        raise ValueError('nonfinite path')
    return dict(G_PL=_features(np.ascontiguousarray(p)))


def tensor_transport(feature_array: np.ndarray, factor: np.ndarray) -> np.ndarray:
    """Transport inclusive degree 0..4 tensor coordinates under a linear map."""
    f=np.asarray(feature_array,dtype=np.float64)
    a=np.asarray(factor,dtype=np.float64)
    if f.ndim!=2 or f.shape[1]!=31 or a.shape!=(2,2):
        raise ValueError('expected feature array (n,31) and factor (2,2)')
    if not np.isfinite(f).all() or not np.isfinite(a).all():
        raise ValueError('nonfinite transport input')
    result=np.empty_like(f);result[:,0]=f[:,0]
    power=np.ones((1,1))
    for degree in range(1,5):
        power=np.kron(power,a)
        first=OFFSETS[degree]
        result[:,first:first+(1<<degree)]=f[:,first:first+(1<<degree)] @ power.T
    return result


def column_l2_normalize(feature_array: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Author normalization: sqrt(SUM F^2), not centered SD or RMS."""
    f=np.asarray(feature_array,dtype=np.float64)
    if f.ndim!=2 or f.shape[0]<1 or not np.isfinite(f).all():
        raise ValueError('expected finite two-dimensional feature array')
    scales=np.sqrt(np.sum(f*f,axis=0))
    if np.any(scales<=0) or not np.isfinite(scales).all():
        raise FloatingPointError('zero/nonfinite source column norm; no silent column deletion')
    return f/scales,scales


def adjacent_recovery(coef_path: np.ndarray, support, beta_values) -> dict:
    """Literal author check on sums of adjacent LARS knots, with NO threshold.

    This asks whether some open LARS segment recovers the true signed support.
    The known support is used ONLY here, not to select a deployable lambda.
    """
    c=np.asarray(coef_path,dtype=np.float64)
    s=np.asarray(support,dtype=np.int64)
    b=np.asarray(beta_values,dtype=np.float64)
    if c.ndim!=2 or s.ndim!=1 or b.shape!=s.shape or len(set(s.tolist()))!=len(s):
        raise ValueError('invalid coefficient path/support/beta shapes')
    if np.any(s<0) or np.any(s>=c.shape[0]) or not np.isfinite(c).all() or not np.isfinite(b).all():
        raise ValueError('invalid/nonfinite recovery input')
    truth=set(s.tolist());signed=[];unsigned=[]
    for j in range(c.shape[1]-1):
        middle=c[:,j]+c[:,j+1]
        selected=np.flatnonzero(middle)
        if set(selected.tolist())==truth:
            unsigned.append(j)
            if np.all(b*middle[s]>=0):
                signed.append(j)
    return dict(sign_consistent=bool(signed),support_consistent=bool(unsigned),
                signed_segment_indices=signed,unsigned_segment_indices=unsigned,
                lars_path_length=int(c.shape[1]),support_threshold=0.0,
                selection_rule='exists adjacent-knot segment with exactly the true support and signs')


def lasso_recovery(design: np.ndarray, response: np.ndarray, support, beta_values,
                   max_iter: int = 500, alpha_min: float = 0.0,
                   eps: float = np.finfo(np.float64).eps) -> dict:
    """Original uncentered LARS objective with explicit, auditable defaults.

    Does not suppress warnings or automatically repair/retry failed paths.
    The caller must decide prospectively whether an incomplete path or numerical
    warning invalidates its trial. Sign recovery alone is not a solver certificate.
    """
    from sklearn.linear_model import lars_path
    x=np.asarray(design,dtype=np.float64);y=np.asarray(response,dtype=np.float64)
    if x.ndim!=2 or y.shape!=(x.shape[0],) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('invalid/nonfinite LARS input')
    if max_iter<1 or alpha_min<0 or eps<=0:
        raise ValueError('invalid LARS solver setting')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        alphas,active,coefficients,n_iter=lars_path(
            x,y,method='lasso',alpha_min=alpha_min,max_iter=max_iter,eps=eps,
            copy_X=True,verbose=False,return_path=True,return_n_iter=True)
    if not np.isfinite(alphas).all() or not np.isfinite(coefficients).all():
        raise FloatingPointError('LARS returned nonfinite values')
    result=adjacent_recovery(coefficients,support,beta_values)
    result.update(alphas=alphas,coefficients=coefficients,active=np.asarray(active,dtype=np.int64),
                  n_iter=int(n_iter),max_iter=int(max_iter),alpha_min=float(alpha_min),
                  last_alpha=float(alphas[-1]),reached_requested_alpha=bool(alphas[-1]<=alpha_min),
                  warnings=[dict(category=w.category.__name__,message=str(w.message)) for w in caught],
                  warning_count=len(caught),fit_intercept=False,column_zero_penalized=True)
    return result

