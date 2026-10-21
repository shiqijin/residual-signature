"""KKT certification of adjacent-knot Lasso recovery candidates."""
import numpy as np

ABS_TOL = 1e-12
REL_TOL = 1e-7


def compact_path_certificate(x, y, result):
    """Certificate summary of ALL adjacent midpoints, with no new fits.

    A certified signed candidate is a raw signed-support candidate whose
    midpoint meets KKT tolerance on a strictly descending-alpha segment.
    Existing numerical warnings and incomplete endpoints remain separately
    visible in result.  Existence of one certified candidate can establish a
    positive even if later path points are inaccurate; a failed/incomplete path
    cannot establish a trustworthy negative.  This function does not collapse
    these distinctions into the author's raw decision.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coefficients = np.asarray(result["coefficients"], dtype=float)
    alphas = np.asarray(result["alphas"], dtype=float)
    if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1 or y.shape != (x.shape[0],)
            or coefficients.ndim != 2
            or coefficients.shape != (x.shape[1], len(alphas))
            or alphas.ndim != 1 or len(alphas) == 0
            or not all(np.isfinite(v).all() for v in (x, y, coefficients, alphas))
            or np.any(alphas < 0)):
        raise ValueError("invalid/nonfinite path certificate input")
    count = max(0, len(alphas)-1)
    signed = np.asarray(result.get("signed_segment_indices", []), dtype=int)
    if (signed.ndim != 1 or len(np.unique(signed)) != len(signed)
            or np.any(signed < 0) or np.any(signed >= count)):
        raise ValueError("invalid signed candidate indices")
    alpha0 = float(np.max(np.abs(x.T @ y), initial=0.) / len(y))
    tolerance = max(ABS_TOL, REL_TOL * alpha0)
    nonincreasing = bool(np.all(np.diff(alphas) <= tolerance))
    if count:
        b = (coefficients[:, :-1] + coefficients[:, 1:]) / 2.
        a = (alphas[:-1] + alphas[1:]) / 2.
        corr = x.T @ (y[:, None] - x @ b) / len(y)
        active = b != 0
        violation = np.maximum(np.abs(corr) - a[None, :], 0.)
        violation[active] = np.abs(corr - a[None, :] * np.sign(b))[active]
        per_mid = np.max(violation, axis=0)
        passed = per_mid <= tolerance
        descending = alphas[:-1] > alphas[1:]
    else:
        b = np.empty((x.shape[1], 0))
        a = per_mid = np.empty(0)
        passed = descending = np.empty(0, dtype=bool)
    good = [int(j) for j in signed if passed[j] and descending[j] and nonincreasing]
    first = good[0] if good else None
    candidate_max = float(np.max(per_mid[signed])) if len(signed) else None
    maximum = float(np.max(per_mid)) if count else None
    worst = int(np.argmax(per_mid)) if count else None
    # Candidate coefficients are tiny (31 floats) and make positive replay
    # possible even without keeping a full path in every production record.
    return dict(
        certificate_version="all-adjacent-midpoints-v1",
        kkt_alpha0=alpha0, kkt_tolerance=tolerance,
        midpoint_count=count, path_alpha_nonincreasing=nonincreasing,
        zero_width_segment_count=int(np.sum(np.diff(alphas) == 0)),
        ascending_alpha_segment_count=int(np.sum(np.diff(alphas) > 0)),
        all_midpoint_kkt_max=maximum,
        all_midpoint_kkt_relative=None if maximum is None else maximum/max(alpha0, np.finfo(float).tiny),
        all_midpoint_kkt_failed_count=int(np.sum(~passed)),
        all_midpoint_kkt_passed=bool(count and np.all(passed) and nonincreasing),
        worst_midpoint_index=worst,
        worst_midpoint_alpha=None if worst is None else float(a[worst]),
        signed_candidate_count=int(len(signed)),
        signed_candidate_kkt_max=candidate_max,
        signed_candidate_kkt_relative=None if candidate_max is None else candidate_max/max(alpha0, np.finfo(float).tiny),
        signed_candidate_kkt_failed_count=int(np.sum(~passed[signed])),
        signed_candidate_zero_width_count=int(np.sum(np.diff(alphas)[signed] == 0)),
        certified_signed_candidate_count=len(good),
        certified_sign_consistent=bool(good),
        first_certified_signed_segment=first,
        first_certified_signed_alpha=None if first is None else float(a[first]),
        first_certified_signed_coefficients=None if first is None else b[:, first].tolist(),
    )

