"""Paired geometric/residual recovery for mixed fractional Gaussian paths."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from guo_baselines import features, tensor_transport, column_l2_normalize, lasso_recovery
from audit_lasso import compact_path_certificate
from safe_lars import install_safe_lars
from mixed_law import prepare_mixed_law, sample_mixed_canonical
from ou_residual import residual_features

ROOT = Path(__file__).resolve().parents[1]

METHODS = ('G_PL', 'Residual')
H_GRID = (.3, .4, .6, .7)
K_GRID = (0., 1., 5.)
Q_GRID = (2, 4, 6)


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, default=serialize, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    try:
        if immutable:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def cell_key(H, k, q):
    return f'H{H:g}_k{k:g}_q{q}'


def chunk_path(folder, task):
    H,k,q,start,stop = task
    return Path(folder)/'chunks'/f'{cell_key(H,k,q)}_r{start:04d}-{stop:04d}.jsonl'


def seed_for(config, H, k, q, repeat):
    return [config['seed'],200,round(100*H),round(100*k),q,repeat,config['length']]


def generate_panel(config, H, k, q, repeat, law=None):
    if law is None:
        law = prepare_mixed_law(H,k,config['length'])
    seed = seed_for(config,H,k,q,repeat)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    A = rng.standard_normal((2,2))
    omega = A @ A.T
    factor = np.linalg.cholesky(omega)
    canonical = sample_mixed_canonical(config['n'],rng,law)
    fs = {'G_PL': features(canonical)['G_PL'],
          'Residual': residual_features(canonical,law['gammas'])}
    support = rng.choice(31,q,replace=False)
    beta = rng.standard_normal(q)
    noise = config['noise_sd']*rng.standard_normal(config['n'])
    base = dict(family=config['family'],H=H,kappa=k,q=q,repeat=repeat,stage=config['stage'],
        regime='mixed_fbm_bm' if k==0 else 'mixed_fou_ou',seed_components=seed,
        n=config['n'],length=config['length'],dimension=2,degree=4,
        response=config['response'],support=support,beta=beta,
        min_abs_beta=float(np.min(np.abs(beta))),omega=omega,
        rho=float(omega[0,1]/np.sqrt(omega[0,0]*omega[1,1])),
        covariance_condition=float(np.linalg.cond(omega)),
        canonical_sha256=array_sha(canonical),noise_sha256=array_sha(noise))
    pairs = {}
    for method in METHODS:
        X,norms = column_l2_normalize(tensor_transport(fs[method],factor))
        signal = X[:,support] @ beta
        pairs[method] = (X,signal+noise,norms,float(np.linalg.norm(signal)))
    return base,pairs


def run_chunk(config, task):
    install_safe_lars()
    started = time.perf_counter()
    H,k,q,start,stop = task
    law = prepare_mixed_law(H,k,config['length'])
    folder = ROOT/'results'/config['stage']
    path = chunk_path(folder,task)
    receipt_path = folder/'receipts'/f'{path.stem}.json'
    progress_path = folder/'worker_progress'/f'{path.stem}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(f'.{os.getpid()}.pending')
    diagnostics = dict(warning_fits=0,uncertified_raw_positives=0,iteration_limit_fits=0)
    with temporary.open('x') as stream:
        for repeat in range(start,stop):
            base,pairs = generate_panel(config,H,k,q,repeat,law)
            for method in METHODS:
                X,y,norms,signal_l2 = pairs[method]
                result = lasso_recovery(X,y,base['support'],base['beta'],**config['lars'])
                certificate = compact_path_certificate(X,y,result)
                diagnostics['warning_fits'] += int(result['warning_count']>0)
                diagnostics['iteration_limit_fits'] += int(result['n_iter']>=config['lars']['max_iter'])
                diagnostics['uncertified_raw_positives'] += int(result['sign_consistent'] and not certificate['certified_sign_consistent'])
                result.pop('coefficients'); result.pop('alphas')
                record = dict(**base,method=method,column_norms=norms,signal_l2=signal_l2,
                    design_sha256=array_sha(X),response_sha256=array_sha(y),
                    **result,**certificate)
                stream.write(json.dumps(record,default=serialize,allow_nan=False)+'\n')
            if (repeat-start+1)%10==0 or repeat==stop-1:
                stream.flush()
                atomic_json(progress_path,dict(task=list(task),completed_panels=repeat-start+1,
                    total_panels=stop-start,pid=os.getpid(),updated_unix=time.time(),
                    elapsed_seconds=time.perf_counter()-started,committed=False))
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary,path); temporary.unlink()
    receipt = dict(task=list(task),path=str(path.relative_to(ROOT)),rows=2*(stop-start),
        panels=stop-start,seconds=time.perf_counter()-started,sha256=sha(path),
        diagnostics=diagnostics,law_metadata=law['metadata'],created_unix=time.time())
    atomic_json(receipt_path,receipt,immutable=True)
    atomic_json(progress_path,dict(task=list(task),completed_panels=stop-start,
        total_panels=stop-start,pid=os.getpid(),updated_unix=time.time(),
        elapsed_seconds=receipt['seconds'],committed=True))
    return receipt

