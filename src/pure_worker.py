"""Paired geometric/residual recovery for fractional Gaussian paths."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[key] = '1'

ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from guo_baselines import features, tensor_transport, column_l2_normalize
from safe_lars import install_safe_lars
from fractional_features import prepare_law, sample_canonical, residual_features

METHODS = ('G_PL', 'Residual')


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        if temporary.exists():
            temporary.unlink()


def cell_key(H, kappa, q):
    return f'H{H:g}_k{kappa:g}_q{q}'


def task_list(config):
    # Cover all conditions once before moving to the next repeat chunk.
    return [(H, k, q, start, min(start+config['chunk_size'], config['repeats']))
            for start in range(0, config['repeats'], config['chunk_size'])
            for H in config['H'] for k in config['kappa'] for q in config['q']]


def chunk_path(folder, task):
    H, k, q, start, stop = task
    return Path(folder)/'chunks'/f'{cell_key(H,k,q)}_r{start:04d}-{stop:04d}.jsonl'


def run_chunk(config, task):
    from guo_baselines import lasso_recovery
    from audit_lasso import compact_path_certificate
    install_safe_lars()
    started = time.perf_counter()
    H, kappa, q, start, stop = task
    law = prepare_law(H, kappa, config['length'])
    folder = ROOT/'results'/config['stage']
    path = chunk_path(folder, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(f'.{os.getpid()}.pending')
    progress_path = folder/'worker_progress'/f'{path.stem}.json'
    with temporary.open('x') as stream:
        for repeat in range(start, stop):
            seed = [config['seed'], round(H*100), round(kappa*100), q, repeat, config['length']]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            A = rng.standard_normal((2, 2))
            omega = A @ A.T
            factor = np.linalg.cholesky(omega)
            canonical = sample_canonical(config['n'], rng, law)
            fs = {'G_PL':features(canonical)['G_PL'],
                  'Residual':residual_features(canonical, law['gammas'])}
            support = rng.choice(31, q, replace=False)
            beta = rng.standard_normal(q)
            noise = config['noise_sd']*rng.standard_normal(config['n'])
            base = dict(family=config["family"], H=H, kappa=kappa, process=law['metadata']['path_law'], q=q,
                        repeat=repeat, stage=config['stage'], seed_components=seed,
                        n=config['n'], length=config['length'], support=support, beta=beta,
                        min_abs_beta=float(np.min(np.abs(beta))), omega=omega,
                        rho=float(omega[0,1]/np.sqrt(omega[0,0]*omega[1,1])),
                        covariance_condition=float(np.linalg.cond(omega)),
                        response=config['response'])
            for method in METHODS:
                X, norms = column_l2_normalize(tensor_transport(fs[method], factor))
                signal = X[:, support] @ beta
                y = signal + noise
                result = lasso_recovery(X, y, support, beta)
                certificate = compact_path_certificate(X, y, result)
                result.pop('alphas'); result.pop('coefficients')
                record = dict(**base, method=method, column_norms=norms,
                              signal_l2=float(np.linalg.norm(signal)),
                              **result, **certificate)
                stream.write(json.dumps(record, default=serialize, allow_nan=False)+'\n')
            if (repeat-start+1)%10 == 0 or repeat == stop-1:
                stream.flush()
                atomic_json(progress_path, dict(task=list(task), completed_panels=repeat-start+1,
                                                total_panels=stop-start, pid=os.getpid(),
                                                elapsed_seconds=time.perf_counter()-started,
                                                updated_unix=time.time(), committed=False))
        stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()
    seconds = time.perf_counter()-started
    atomic_json(progress_path, dict(task=list(task), completed_panels=stop-start,
                                    total_panels=stop-start, pid=os.getpid(),
                                    elapsed_seconds=seconds, updated_unix=time.time(), committed=True))
    return dict(task=list(task), rows=2*(stop-start), panels=stop-start,
                seconds=seconds, sha256=sha(path),
                path=str(path.resolve()), law_metadata=law['metadata'])

