"""Run all p_AR branches from the shared trunks at inherited pure-BD LRs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import COMPUTE_BUDGETS, MODEL_SPECS, ROOT, budget_slug
from curriculum_config import P_AR_VALUES, is_feasible, phase_steps_for


RESULTS = ROOT / "results_curriculum"
SHARED = RESULTS / "shared_ar"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def complete(path):
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def execute(run, device):
    run["output"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        "--budget",
        str(run["budget"]),
        "--size",
        run["spec"].label,
        "--p-ar",
        str(run["p_ar"]),
        "--trunk-dir",
        str(run["trunk_dir"]),
        "--output",
        str(run["output"]),
        "--device",
        device,
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (run["output"].parent / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    return run, process


def main() -> None:
    args = parse_args()
    runs = []
    for budget in COMPUTE_BUDGETS:
        for spec in MODEL_SPECS:
            for p_ar in P_AR_VALUES:
                if not is_feasible(budget, spec, p_ar):
                    continue
                slug = f"{p_ar:.1f}".replace(".", "p")
                runs.append(
                    {
                        "budget": budget,
                        "spec": spec,
                        "p_ar": p_ar,
                        "trunk_dir": SHARED / budget_slug(budget) / spec.label,
                        "output": (
                            RESULTS
                            / "runs"
                            / budget_slug(budget)
                            / spec.label
                            / f"p_ar_{slug}"
                            / "result.json"
                        ),
                    }
                )
    todo = [run for run in runs if not complete(run["output"])]
    print(f"curriculum_runs={len(runs)} remaining={len(todo)}")
    if args.dry_run:
        for run in todo:
            ar_steps, bd_steps = phase_steps_for(
                run["budget"],
                run["spec"],
                run["p_ar"],
            )
            print(
                f"C={run['budget']:.0e} N={run['spec'].label} "
                f"p={run['p_ar']:.1f} AR={ar_steps} BD={bd_steps}"
            )
        return
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, run, args.device): run
            for run in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            run, process = future.result()
            if process.returncode:
                failures.append(str(run["output"].parent))
                print(
                    f"[{index}/{len(todo)}] FAILED C={run['budget']:.0e} "
                    f"N={run['spec'].label} p={run['p_ar']:.1f}: "
                    f"{process.stderr[-1200:]}",
                    flush=True,
                )
                continue
            result = json.loads(run["output"].read_text())
            print(
                f"[{index}/{len(todo)}] C={run['budget']:.0e} "
                f"N={run['spec'].label} p={run['p_ar']:.1f} "
                f"val={result['val_nelbo']:.5f} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{len(failures)} curriculum runs failed: {failures}")


if __name__ == "__main__":
    main()
