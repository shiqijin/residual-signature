#!/usr/bin/env python3
"""Check the reported 174 settings or aggregate new G/R simulation outputs."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("G_PL", "Residual")
ENDPOINTS = ("raw", "certified")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def identity(row):
    number = lambda name: float(row[name]) if row.get(name) not in (None, "") else None
    return (row["regime"], number("parameter"), number("H"), number("kappa"),
            row.get("law_key", ""), int(row["q"]))


def expected_grid():
    grid = set()
    for q in (2, 4, 6):
        grid.add(("BM", 0.0, None, None, "", q))
        grid.update(("OU", k / 2, None, None, "", q) for k in range(1, 11))
        grid.update(("AR1", k / 10, None, None, "", q) for k in range(11))
        for H in (.3, .4, .6, .7):
            for family in ("fBM", "mixed_fBM_BM"):
                grid.add((family, None, H, 0.0, "", q))
            for family in ("fOU", "mixed_fOU_OU"):
                grid.update((family, None, H, k, "", q) for k in (1.0, 5.0))
        for nu in (1.5, 2.5):
            for ell in (.1, .3, 1.0):
                key = f"matern_nu{nu:g}_ellT{ell:g}".replace(".", "p")
                grid.add(("matern", None, None, None, key, q))
        for k in (1, 5):
            for w in (1, 2, 4):
                grid.add(("rotating_ou", None, None, None,
                          f"rotating_ou_kappaT{k}_omegaT{w}pi", q))
    return grid


def metrics(n, left_only, right_only, both, neither):
    assert n > 0 and min(left_only, right_only, both, neither) >= 0
    assert left_only + right_only + both + neither == n
    gain = (right_only - left_only) / n
    se = math.sqrt(max(0, ((right_only + left_only) - n * gain**2) / (n * (n - 1)))) if n > 1 else None
    return {"n": n, "left_success": left_only + both, "right_success": right_only + both,
            "left_rate": (left_only + both) / n, "right_rate": (right_only + both) / n,
            "difference": gain, "left_only": left_only, "right_only": right_only,
            "both": both, "neither": neither,
            "difference_se": se if se is not None else "",
            "difference_ci95_lower": gain - norm.ppf(.975) * se if se is not None else "",
            "difference_ci95_upper": gain + norm.ppf(.975) * se if se is not None else "",
            "exact_mcnemar_p": float(binomtest(min(left_only, right_only), left_only + right_only).pvalue)
            if left_only + right_only else 1.0}


def adjust(rows):
    for endpoint in ENDPOINTS:
        p = np.array([float(row[endpoint + "_exact_mcnemar_p"]) for row in rows])
        order = np.argsort(p, kind="stable")
        adjusted = np.empty_like(p)
        adjusted[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
        for row, value in zip(rows, adjusted):
            row[endpoint + "_holm174_p"] = float(value)


def check_results(rows):
    assert len(rows) == 174 and {identity(row) for row in rows} == expected_grid()
    computed = []
    for row in rows:
        values = {}
        for endpoint in ENDPOINTS:
            prefix = endpoint + "_"
            counts = [int(row[prefix + field]) for field in ("n", "left_only", "right_only", "both", "neither")]
            assert counts[0] == 1000
            for field, value in metrics(*counts).items():
                assert math.isclose(float(row[prefix + field]), value, rel_tol=2e-11, abs_tol=1e-14), (identity(row), prefix + field)
                values[prefix + field] = value
        computed.append(values)
    adjust(computed)
    for row, values in zip(rows, computed):
        for endpoint in ENDPOINTS:
            field = endpoint + "_holm174_p"
            assert math.isclose(float(row[field]), values[field], rel_tol=2e-11, abs_tol=1e-14)
    return {"status": "PASS", "settings": 174, "paired_trials": 174000,
            "checks": "declared grid, paired counts, rates, intervals, exact tests and Holm adjustment"}


def record_identity(record):
    q = int(record["q"])
    if record.get("family") in ("matern", "rotating_ou"):
        return record["family"], None, None, None, record["law_key"], q
    if record.get("H") is not None:
        H, k = float(record["H"]), float(record["kappa"])
        mixed = str(record.get("regime", "")).lower().startswith("mixed")
        family = ("mixed_fBM_BM" if k == 0 else "mixed_fOU_OU") if mixed else ("fBM" if k == 0 else "fOU")
        return family, None, H, k, "", q
    process, parameter = record["process"], float(record["parameter"])
    assert process in ("ou", "ar1")
    family = "AR1" if process == "ar1" else ("BM" if parameter == 0 else "OU")
    return family, parameter, None, None, "", q


def load_records(paths, repeats):
    bank = {}
    grid = expected_grid()
    for directory in paths:
        files = sorted(Path(directory).rglob("*.jsonl"))
        if not files:
            raise ValueError(f"No trial records in {directory}")
        for path in files:
            with path.open() as stream:
                for line in stream:
                    record = json.loads(line)
                    key, rep, method = record_identity(record), int(record["repeat"]), record["method"]
                    if key not in grid or method not in METHODS or not 0 <= rep < repeats:
                        raise ValueError(f"Unexpected trial in {path}: {(key, rep, method)}")
                    pair = bank.setdefault(key, {}).setdefault(rep, {})
                    if method in pair:
                        raise ValueError(f"Duplicate trial: {(key, rep, method)}")
                    raw = record.get("success", record.get("sign_consistent"))
                    certified = record.get("certified_success", record.get("certified_sign_consistent"))
                    assert isinstance(raw, bool) and isinstance(certified, bool) and (not certified or raw)
                    pair[method] = (raw, certified)
    if set(bank) != grid:
        raise ValueError("The run must contain the complete 174-setting grid")
    for key, trials in bank.items():
        if set(trials) != set(range(repeats)) or any(set(pair) != set(METHODS) for pair in trials.values()):
            raise ValueError(f"Missing paired trials in {key}")
    return bank


def aggregate(bank, reference):
    rows = []
    for original in reference:
        row = {k: value for k, value in original.items() if not k.startswith(("raw_", "certified_"))}
        pairs = list(bank[identity(original)].values())
        for index, endpoint in enumerate(ENDPOINTS):
            counts = [0, 0, 0, 0]
            for pair in pairs:
                g, r = pair["G_PL"][index], pair["Residual"][index]
                counts[2 if g and r else 0 if g else 1 if r else 3] += 1
            row.update({endpoint + "_" + k: v for k, v in metrics(len(pairs), *counts).items()})
        rows.append(row)
    adjust(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the supplied results without simulation")
    parser.add_argument("--input", type=Path, help="Trial directory from the unified run")
    parser.add_argument("--repeats", type=int, default=1000, help="Expected trials per setting; use 1 only for smoke checks")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/recovery_174.csv")
    args = parser.parse_args()
    reference = read_csv(ROOT / "recovery_174.csv")
    if args.check:
        print(json.dumps(check_results(reference), indent=2))
        return
    if args.input is None or args.repeats < 1:
        parser.error("Provide --input and a positive --repeats, or use --check")
    output = args.output.resolve()
    if output == ROOT / "recovery_174.csv":
        parser.error("Do not overwrite the supplied reference results")
    rows = aggregate(load_records([args.input], args.repeats), reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reference[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"settings": len(rows), "paired_trials": args.repeats * len(rows),
              "smoke_only": args.repeats != 1000, "output": str(output)}
    if args.repeats == 1000:
        report["certified_counts_matching_reference"] = sum(
            all(int(row["certified_" + field]) == int(old["certified_" + field])
                for field in ("left_only", "right_only", "both", "neither"))
            for row, old in zip(rows, reference))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
