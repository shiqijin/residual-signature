"""Internal process dispatcher for the unified experiment registry."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
from pathlib import Path
import sys

from runtime import ROOT, prepare_imports


def verify_environment(profile):
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Reproduction requires Python 3.13.")
    requirements = ROOT / "src/environments" / f"{profile}.txt"
    for line in requirements.read_text().splitlines():
        name, expected = line.split("==")
        actual = version(name)
        if actual != expected:
            raise RuntimeError(f"{profile}: {name} {actual}; expected {expected}. Run run.py setup.")
    prepare_imports(True)
    print(json.dumps({"environment": profile, "status": "ready"}), flush=True)


def worker(job):
    prepare_imports(True)
    kind, task = job["kind"], job["task"]
    if kind == "covariance":
        from covariance_worker import run_setting
        result = run_setting(task)
    else:
        import importlib
        module = importlib.import_module({"basic": "basic_worker", "pure": "pure_worker",
                                         "mixed": "mixed_worker"}[kind])
        module.ROOT = Path(job["output"]) / job["family"]
        result = module.cell_job(job["config"], *task) if kind == "basic" else module.run_chunk(job["config"], task)
    trials = result.get("paired_trials", result.get("panels", result.get("rows", 0) // 2))
    return {"family": job["family"], "paired_trials": trials,
            "setting": task[0]["law_key"] + f"_q{task[1]}" if kind == "covariance" else task}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-environment", choices=("numpy21", "numpy23"))
    parser.add_argument("--profile", choices=("numpy21", "numpy23"))
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.check_environment:
        verify_environment(args.check_environment)
        return
    if args.profile is None or args.plan is None:
        parser.error("A plan and numerical profile are required")
    plan = json.loads(args.plan.read_text())
    tasks = [job for job in plan["tasks"] if job["profile"] == args.profile]
    covariance = [job for job in tasks if job["kind"] == "covariance"]
    if covariance:
        from covariance_worker import resources
        for key in dict.fromkeys(job["task"][0]["law_key"] for job in covariance):
            resources(key)
    with ProcessPoolExecutor(max_workers=plan["jobs"]) as pool:
        futures = [pool.submit(worker, job) for job in tasks]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
