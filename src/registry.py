"""Declared Gaussian laws, settings and seeds; no scientific-library imports."""
from copy import deepcopy
import math
from pathlib import Path

FAMILIES = ('BM', 'fBM', 'mixed_fBM_BM', 'OU', 'AR1', 'fOU',
            'mixed_fOU_OU', 'matern', 'rotating_ou')
Q = (2, 4, 6)
H = (.3, .4, .6, .7)
KAPPA = (0., 1., 5.)
COVARIANCE_SEED = 2026102071


def laws(length=100, dt=.01):
    """Ordered six Matérn and six rotating-OU finite-grid specifications."""
    if isinstance(length, bool) or int(length) != length or length < 2:
        raise ValueError('length must be an integer >= 2 counting observations')
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and strictly positive')
    common = dict(length=int(length), dt=float(dt))
    result = []
    for nu in (1.5, 2.5):
        for ell_T in (.1, .3, 1.):
            key = f'matern_nu{nu:g}_ellT{ell_T:g}'.replace('.', 'p')
            result.append(dict(law_key=key, family='matern', nu=nu,
                               ell_over_T=ell_T, **common))
    for kappa_T in (1., 5.):
        for omega_pi in (1., 2., 4.):
            key = f'rotating_ou_kappaT{kappa_T:g}_omegaT{omega_pi:g}pi'
            result.append(dict(law_key=key, family='rotating_ou', kappa_T=kappa_T,
                               omega_T=omega_pi*math.pi, omega_T_over_pi=omega_pi,
                               **common))
    return result


def jobs_for(output, repeats=1000):
    """Return the complete 174-setting workload with the recorded seed streams.

    OU/AR jobs cover one setting each. Fractional and mixed jobs retain chunks
    of 100 trials; covariance jobs cover one setting each. The output path is
    not created here. The dispatcher sets each worker root to output/family.
    """
    if repeats is None:
        repeats = 1000
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer')
    output = str(Path(output).resolve())
    jobs = []

    def add(family, kind, config, task, profile='numpy21'):
        jobs.append(dict(family=family, kind=kind, profile=profile,
                         config=deepcopy(config), task=tuple(task), output=output))

    for process, parameters in [('ou', [i/2 for i in range(11)]),
                                ('ar1', [i/10 for i in range(11)])]:
        config = dict(name='trials', repeats=repeats, length=100, n=100,
                      seed=2026090501, process=process, parameters=parameters,
                      qs=list(Q), jobs=2, response='own_sparse')
        for parameter in parameters:
            family = 'AR1' if process == 'ar1' else ('BM' if parameter == 0 else 'OU')
            config['family'] = family
            for q in Q:
                add(family, 'basic', config, (parameter, q))

    for kind, seed in [('pure', 2026090511), ('mixed', 2026090521)]:
        config = dict(stage='trials', H=list(H), kappa=list(KAPPA), q=list(Q),
                      n=100, length=100, repeats=repeats, chunk_size=100,
                      seed=seed, noise_sd=.01, methods=['G_PL', 'Residual'],
                      jobs=3, response='own_sparse_paired_support_beta_noise',
                      fit_intercept=False, column_norm='Euclidean L2 = 1',
                      constant_penalized=True)
        if kind == 'mixed':
            config.update(dimension=2, degree=4, columns=31,
                          endpoint='adjacent_LARS_path_raw_and_KKT_certified',
                          lars=dict(max_iter=500, alpha_min=0., eps=2.220446049250313e-16),
                          support_threshold=0., no_automatic_retries=True)
        for h in H:
            for kappa in KAPPA:
                family = ('fBM' if kappa == 0 else 'fOU') if kind == 'pure' else (
                    'mixed_fBM_BM' if kappa == 0 else 'mixed_fOU_OU')
                config['family'] = family
                for q in Q:
                    for start in range(0, repeats, config['chunk_size']):
                        add(family, kind, config,
                            (h, kappa, q, start, min(start+config['chunk_size'], repeats)))

    for law in laws():
        family = law['family']
        for q in Q:
            add(family, 'covariance', {},
                (law, q, repeats, str(Path(output)/family)), profile='numpy23')
    return jobs
