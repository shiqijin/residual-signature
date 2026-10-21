"""Process-local workaround for the observed Accelerate dpotrs SIGSEGV.

Only the small Cholesky solve selected inside sklearn's LARS implementation
is replaced. The regression objective, path algorithm, numerical thresholds,
data, feature definitions and support rule are unchanged. No installed package
is edited. This is a numerical backend change, not a bitwise identity claim.

Crash records: Python-2026-09-05-150106.ips and -151036.ips both terminate in
macOS libBLAS cblas_dtrsm -> libLAPACK dpotrs -> scipy _flapack_dpotrs.
The underlying platform defect is not conclusively diagnosed. We bypass the
observed failing call with ordinary forward/back substitution for p<=31.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from numba import njit

BACKEND = "explicit_cholesky_substitution_v1"
_INSTALLED = False
_CALLS = 0
_METADATA = None


@njit(cache=True)
def _substitute(a, b, lower):
    n, columns = b.shape
    out=b.copy()
    for k in range(columns):
        for i in range(n):
            value=out[i,k]
            for j in range(i):
                value-=(a[i,j] if lower else a[j,i])*out[j,k]
            out[i,k]=value/a[i,i]
        for i in range(n-1,-1,-1):
            value=out[i,k]
            for j in range(i+1,n):
                value-=(a[j,i] if lower else a[i,j])*out[j,k]
            out[i,k]=value/a[i,i]
    return out


def safe_potrs(c, b, lower=0, overwrite_b=0):
    """Supported dpotrs contract: real Cholesky factor, vector/matrix RHS.

    Unused triangle is deliberately ignored. Inputs are never overwritten;
    overwrite_b merely permits overwriting in LAPACK, it does not require it.
    Positive Cholesky diagonal is required by this narrowly scoped adapter.
    """
    global _CALLS
    a=np.asarray(c)
    rhs=np.asarray(b)
    if np.iscomplexobj(a) or np.iscomplexobj(rhs):
        raise TypeError("This LARS workaround supports real inputs only")
    a=np.asarray(a,dtype=np.float64)
    rhs=np.asarray(rhs,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or rhs.ndim not in (1,2) or rhs.shape[0]!=a.shape[0]:
        raise ValueError("Incompatible Cholesky factor/RHS dimensions")
    if np.any(np.diag(a)<=0) or not np.isfinite(np.diag(a)).all() or not np.isfinite(rhs).all():
        raise FloatingPointError("Invalid Cholesky diagonal/RHS; no silent repair")
    was_vector=rhs.ndim==1
    matrix=rhs[:,None] if was_vector else rhs
    out=_substitute(np.ascontiguousarray(a),np.ascontiguousarray(matrix),bool(lower))
    # Preserve the caller's own ill-conditioning / nonfinite-AA handling.
    # potrs does not certify a positive-definite input factor or condition it.
    _CALLS+=1
    return (out[:,0] if was_vector else out), 0


def install_safe_lars():
    """Install only in this process; return version/hash metadata."""
    global _INSTALLED, _METADATA
    if _INSTALLED:
        return dict(_METADATA)
    import sklearn
    import scipy
    import sklearn.linear_model._least_angle as least_angle
    if sklearn.__version__!="1.8.0":
        raise RuntimeError("Workaround validated only against installed sklearn 1.8.0")
    original=least_angle.get_lapack_funcs

    def get_lapack_funcs(names, arrays=(), *args, **kwargs):
        if isinstance(names,str):
            if names=="potrs":
                return safe_potrs
            return original(names,arrays,*args,**kwargs)
        if tuple(names)==("potrs",):
            return (safe_potrs,)
        functions=original(names,arrays,*args,**kwargs)
        return tuple(safe_potrs if name=="potrs" else fn for name,fn in zip(names,functions))

    # This module-global lookup is owned by sklearn's LARS module, not by
    # scipy.linalg.lapack or the installed files. Workers are isolated processes.
    least_angle.get_lapack_funcs=get_lapack_funcs
    safe_potrs(np.eye(2),np.ones(2),lower=True)
    _METADATA=dict(backend=BACKEND,sklearn=sklearn.__version__,scipy=scipy.__version__,
                   numpy=np.__version__,least_angle_source_sha256=hashlib.sha256(Path(least_angle.__file__).read_bytes()).hexdigest(),
                   adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   system_packages_modified=False,objective_changed=False)
    _INSTALLED=True
    return dict(_METADATA)

