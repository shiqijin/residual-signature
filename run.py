#!/usr/bin/env python3
"""Set up, check, or reproduce all 174 Gaussian signature settings."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
PROFILES = ("numpy21", "numpy23")
sys.dont_write_bytecode = True
sys.path.insert(0, str(SRC))
from registry import jobs_for


def python_for(profile):
    override = os.environ.get("RESIDUAL_SIGNATURE_PYTHON_" + profile.upper())
    if override:
        return Path(override).expanduser().absolute()
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return ROOT / ".venvs" / profile / suffix


def call(python, *arguments, env=None):
    subprocess.run([str(python), *map(str, arguments)], check=True, env=env)


def check_environment(profile):
    python = python_for(profile)
    if not python.is_file():
        raise RuntimeError("Run `python3.13 run.py setup` first.")
    call(python, SRC / "execute.py", "--check-environment", profile)
    return python


def setup():
    if sys.version_info[:2] != (3, 13):
        raise RuntimeError("Use Python 3.13, the version used for the reported experiments.")
    for profile in PROFILES:
        python = python_for(profile)
        if not python.is_file():
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venvs" / profile)
        call(python, "-m", "pip", "install", "-r", SRC / "environments" / f"{profile}.txt")
        check_environment(profile)
    print("Setup complete. Run `python3.13 run.py check` or `python3.13 run.py run`.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("setup", "check", "run"), default="run")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum simultaneous simulation workers")
    parser.add_argument("--repeats", type=int, default=1000, help="Trials per setting; use 1 for a smoke check")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/full", help="New run directory")
    parser.add_argument("--dry-run", action="store_true", help="List the complete experiment without fitting")
    args = parser.parse_args()
    if args.jobs < 1 or not 1 <= args.repeats <= 1000:
        parser.error("Require --jobs >= 1 and --repeats in 1..1000")
    if args.command == "setup":
        setup()
        return
    if args.command == "check":
        python = check_environment("numpy23")
        call(python, SRC / "summarize.py", "--check")
        return

    output = args.output.resolve()
    tasks = jobs_for(output / "trials", args.repeats)
    # Fractional settings are split into repeat chunks; count datasets, not jobs.
    counts = Counter()
    for job in tasks:
        size = job["task"][4] - job["task"][3] if job["kind"] in ("pure", "mixed") else args.repeats
        counts[job["family"]] += size
    inventory = {"families": {family: n // args.repeats for family, n in counts.items()},
                 "settings": sum(counts.values()) // args.repeats,
                 "paired_trials": sum(counts.values()), "output": str(output)}
    print(json.dumps(inventory, indent=2), flush=True)
    if args.dry_run:
        return
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    interpreters = {profile: check_environment(profile) for profile in PROFILES}
    output.mkdir(parents=True)
    plan = output / "settings.json"
    plan.write_text(json.dumps({"inventory": inventory, "repeats": args.repeats,
                               "jobs": args.jobs, "tasks": tasks}, indent=2) + "\n")
    env = os.environ.copy()
    env["RESIDUAL_SIGNATURE_CACHE"] = str(output / ".cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for profile, python in interpreters.items():
        call(python, SRC / "execute.py", "--plan", plan, "--profile", profile, env=env)
    call(interpreters["numpy23"], SRC / "summarize.py", "--input", output / "trials",
         "--repeats", args.repeats, "--output", output / "recovery_174.csv", env=env)


if __name__ == "__main__":
    main()
