"""Paired geometric/residual recovery for Brownian, OU and Gaussian AR paths."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from guo_baselines import sample_process, features, tensor_transport, column_l2_normalize, lasso_recovery

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("G_PL", "Residual")


def serialize(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def residual_setup(process, parameter, length):
    from ou_residual import exact_ou_gammas, exact_markov_gammas
    if process == "ou":
        return exact_ou_gammas(parameter, length=length)
    return exact_markov_gammas(parameter, length=length,
                              innovation_variance=1.0, innovation_fourth_factor=3.0)


def residual_eval(paths, gammas):
    from ou_residual import residual_features
    return residual_features(paths, gammas)


def cell_job(config, parameter, q):
    t0 = time.perf_counter()
    length, repeats = config["length"], config["repeats"]
    process = config["process"]
    gammas = residual_setup(process, parameter, length)
    records = []
    for repeat in range(repeats):
        seed = [config["seed"], {"ou":1,"ar1":2}[process], round((parameter+10)*100), q, repeat, length]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        panel = sample_process(config["n"], process, parameter, rng, length=length)
        canonical = panel["canonical_paths"]
        fs = features(canonical)
        fs["Residual"] = residual_eval(canonical, gammas)
        fs = {m: tensor_transport(f, panel["cholesky"]) for m, f in fs.items()}
        support = rng.choice(31, q, replace=False)
        beta = rng.standard_normal(q)
        eps = 0.01 * rng.standard_normal(config["n"])
        rho = panel["omega"][0, 1] / np.sqrt(panel["omega"][0, 0] * panel["omega"][1, 1])
        base = dict(family=config["family"], process=process, parameter=parameter, display_parameter=1-parameter if process=="ar1" else parameter,
                    q=len(support), repeat=repeat, seed_components=seed,
                    length=length, n=config["n"], response=config["response"],
                    support=support, beta=beta, min_abs_beta=float(np.min(np.abs(beta))),
                    rho=float(rho), covariance_condition=float(np.linalg.cond(panel["omega"])))
        for method in METHODS:
            x, norms = column_l2_normalize(fs[method])
            y = x[:, support] @ beta + eps
            result = lasso_recovery(x, y, support.tolist(), beta)
            from audit_lasso import compact_path_certificate
            certificate = compact_path_certificate(x, y, result)
            # Seeds recreate every path. Keep certificates/candidates, not an
            # uncompressed multi-gigabyte duplicate of all coefficient paths.
            result.pop("coefficients")
            result.pop("alphas")
            records.append({**base, "method": method, **result, **certificate,
                            "signal_l2": float(np.linalg.norm(y-eps))})
    out = ROOT / "results" / config["name"] / f"p{parameter:g}_q{q}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with out.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, default=serialize, allow_nan=False) + "\n")
    return {"process":process, "parameter": parameter, "q": q, "rows": len(records), "seconds": time.perf_counter()-t0}

