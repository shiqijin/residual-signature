# Gaussian signature recovery: 174 settings

Code and complete results for Section 5 of *Wiener Chaos Cancellation for
Gaussian Path Signatures*, by Tomoyuki Ichiba and Qijin Shi.

- `run.py`: the single entry point for setup, checking and simulation.
- `recovery_174.csv`: the reported results, with 1,000 paired trials per setting.
- `src/`: the process registry, shared feature and Lasso routines, simulation
  workers, aggregation code and pinned dependencies.

## Run

Use Python 3.13. From this directory:

```sh
python3.13 run.py setup
python3.13 run.py check
python3.13 run.py run --jobs 2
```

`setup` installs dependencies; `check` verifies the supplied table without
simulation. `run` executes all nine families and writes one combined table to
`runs/full/recovery_174.csv`, with trial records grouped by process family under
`runs/full/trials/`. Use a fresh output directory for each run.

For a quick execution check covering every setting:

```sh
python3.13 run.py run --repeats 1 --jobs 2 --output runs/smoke
```

This checks execution only; the reported experiment uses 174,000 paired datasets
and 348,000 G/R fits. `python3.13 run.py --dry-run` lists the grid without installing
dependencies or fitting models.

The launcher automatically selects the recorded numerical environment for each
model. Two pinned dependency profiles are retained internally to preserve the
reported computations; no manual division of the experiment is needed.
Floating-point solver outcomes can still vary by platform. A full run reports
how many settings reproduce the supplied certified contingency counts.

## Models and settings

Every parameter combination uses q = 2, 4, 6. The full grid and seeds are in
`src/registry.py`.

| Process | Parameters | Settings |
| --- | --- | ---: |
| Brownian motion | OU boundary kappa = 0 | 3 |
| OU | kappa = 0.5, 1, ..., 5 | 30 |
| AR(1) | phi = 0, 0.1, ..., 1 | 33 |
| Fractional Brownian motion | H = 0.3, 0.4, 0.6, 0.7 | 12 |
| Fractional OU | same H; kappa = 1, 5 | 24 |
| Mixed fractional Brownian/Brownian | same H | 12 |
| Mixed fractional OU/OU | same H; kappa = 1, 5 | 24 |
| Matérn | nu = 1.5, 2.5; ell/T = 0.1, 0.3, 1 | 18 |
| Rotating OU | kappa*T = 1, 5; omega*T = pi, 2*pi, 4*pi | 18 |

The AR(1) phi = 1 settings are retained Brownian boundary cases. Matérn and
rotating OU use their stationary finite-grid Gaussian covariance, including
the initial observation.

## Recovery protocol

Each dataset contains 100 two-dimensional paths, each with 100 observations at
spacing 0.01 (99 increments, T = 0.99). A single random lower Cholesky factor of
AA-transpose, with A having independent standard-normal entries, mixes the
canonical channels. G is the geometric signature of the piecewise-linear
interpolation; R uses the same increments with the law-calibrated degree-four
right-Euler clock correction. Calibration arrays are generated automatically.

The 31 features through degree four include the penalised constant. Columns
have Euclidean norm one without centring, and there is no free intercept.
The two methods share a uniform support of size q, independent N(0,1) active
coefficients and N(0,0.01 squared) noise. Each method generates its own response
from its normalised features, so the responses generally differ.

LARS computes the Lasso path with max_iter = 500, alpha_min = 0 and float64
machine epsilon. Raw recovery means exactly the true nonzero positions and
coefficient signs at an adjacent-knot midpoint. Certified recovery additionally
requires a strictly descending segment, a nonincreasing penalty path within
tolerance, and the KKT check with tolerance max(1e-12, 1e-7 times
max(abs(X-transpose y))/n). The support threshold is zero. Recovery rate is the
fraction of trials satisfying the criterion; every trial stays in the
denominator. The paper uses certified recovery. See `src/guo_baselines.py` and
`src/audit_lasso.py` for the implementation.

## Result columns

Each CSV row is one process/parameter/q setting. `parameter` is kappa for
Brownian/OU or phi for AR(1); `H`, `kappa`, `nu`, `ell_over_T`, `kappa_T` and
`omega_T_over_pi` specify the other families. Unused fields are empty.
`law_key` identifies a stationary covariance specification; `bridge` marks the
AR(1) phi = 1 cases.

Prefixes `raw_` and `certified_` distinguish the two recovery criteria.
**Left = G; right = R.** Rates are fractions, and `difference` is R minus G.
`left_only`, `right_only`, `both` and `neither` are paired contingency counts;
`n` is 1,000 in the supplied table. `difference_se` and
`difference_ci95_lower/upper` give the standard error and pointwise normal
95% interval of the paired indicator difference. `exact_mcnemar_p` is the
two-sided exact test on discordant pairs; `holm174_p` adjusts across all
174 settings separately for each criterion.

The design follows Guo et al., *On Consistency of Signature Using Lasso*,
[doi:10.1287/opre.2024.1133](https://doi.org/10.1287/opre.2024.1133), with the
paired G/R protocol above.
