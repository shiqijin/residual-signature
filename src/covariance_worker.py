"""Finite-covariance Gaussian experiment worker for Matérn and rotating OU."""
from __future__ import annotations

from runtime import CACHE_ROOT, prepare_imports
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from laws import build_laws, prepare_law

METHODS = ("G_PL", "Residual")
from registry import COVARIANCE_SEED as BASE_SEED
N, P = 100, 31
CACHE = CACHE_ROOT / "calibration"


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@lru_cache(maxsize=12)
def resources(key):
    """Generate law-only clocks once; no fitted data or responses enter calibration."""
    path = CACHE / f"{key}.npz"
    if not path.exists():
        prepare_imports()
        from fractional_features import calibrate_gaussian_load
        law = next(law for law in build_laws() if law["law_key"] == key)
        native = prepare_law(law)
        clocks, witness = calibrate_gaussian_load(native["observation_load"])
        if np.max(np.abs(witness[:, 1:])) > 1e-10:
            raise RuntimeError("Population centering recursion failed")
        CACHE.mkdir(parents=True, exist_ok=True)
        temporary = CACHE / f".{key}.{os.getpid()}.npz"
        np.savez(temporary, observation_load=native["observation_load"],
                 gamma2=clocks[2], gamma3=clocks[3], gamma4=clocks[4])
        os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as arrays:
        return {"observation_load": arrays["observation_load"].copy(),
                "gammas": {k: arrays[f"gamma{k}"].copy() for k in (2, 3, 4)}}


def panel(law,q,repeat,seed_base=BASE_SEED):
    prepare_imports(True)
    from guo_baselines import features,tensor_transport,column_l2_normalize,lasso_recovery
    from ou_residual import residual_features
    from audit_lasso import compact_path_certificate
    law_id=next(i for i,x in enumerate(build_laws(),1) if x['law_key']==law['law_key'])
    seed=[seed_base,law_id,int(q),int(repeat),100]
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    a=rng.standard_normal((2,2)); omega=a@a.T; factor=np.linalg.cholesky(omega)
    native=resources(law['law_key'])
    load=native['observation_load']
    canonical=(rng.standard_normal((N,load.shape[2]))@load.reshape(-1,load.shape[2]).T).reshape(N,100,2)
    canonical=np.ascontiguousarray(canonical)
    fs={'G_PL':features(canonical)['G_PL'],'Residual':residual_features(canonical,native['gammas'])}
    support=rng.choice(P,q,replace=False); beta=rng.standard_normal(q); noise=.01*rng.standard_normal(N)
    base=dict(law_key=law['law_key'],family=law['family'],q=q,repeat=repeat,
              seed_components=seed,support=support,beta=beta,omega=omega,
              canonical_sha256=ahash(canonical),noise_sha256=ahash(noise),
              covariance_condition=float(np.linalg.cond(omega)),
              min_abs_beta=float(np.min(np.abs(beta))),n=N,length=100,degree=4,
              response='own_sparse_paired_support_beta_noise')
    rows=[]
    for method in METHODS:
        x,norms=column_l2_normalize(tensor_transport(fs[method],factor))
        signal=x[:,support]@beta; y=signal+noise
        fit=lasso_recovery(x,y,support,beta,max_iter=500,alpha_min=0,eps=np.finfo(float).eps)
        certificate=compact_path_certificate(x,y,fit)
        fit.pop('alphas'); fit.pop('coefficients')
        row=dict(base,method=method,success=bool(fit['sign_consistent']),
                 certified_success=bool(certificate['certified_sign_consistent']),
                 column_norms=norms,signal_l2=float(np.linalg.norm(signal)),
                 design_sha256=ahash(x),response_sha256=ahash(y),**fit,**certificate)
        rows.append(row)
    return rows


def run_setting(task):
    law, q, repeats, output = task
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{law['law_key']}_q{q}.jsonl"
    temporary = target.with_suffix(".pending")
    fields = ("law_key", "family", "q", "repeat", "method", "success", "certified_success")
    with temporary.open("x", encoding="utf-8") as stream:
        for repeat in range(repeats):
            for row in panel(law, q, repeat):
                stream.write(json.dumps({key: row[key] for key in fields}, sort_keys=True) + "\n")
    temporary.replace(target)
    return {"law_key": law["law_key"], "q": q, "paired_trials": repeats}
