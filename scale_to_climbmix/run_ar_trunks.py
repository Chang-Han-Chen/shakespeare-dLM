"""Build one restart-safe shared AR trunk and pure-AR endpoint per BD point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import COMPUTE_BUDGETS, MODEL_SPECS, ROOT, budget_slug, is_feasible


ROOT_OUTPUT = ROOT / "results_curriculum" / "shared_ar"


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


def execute(point, device):
    budget, spec, output = point
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "curriculum_ar_trunk.py"),
        "--budget",
        str(budget),
        "--size",
        spec.label,
        "--output-dir",
        str(output),
        "--device",
        device,
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / "train.log").write_text(
        process.stdout + ("\n[stderr]\n" + process.stderr if process.stderr else "")
    )
    return point, process


def main() -> None:
    args = parse_args()
    points = [
        (
            budget,
            spec,
            ROOT_OUTPUT / budget_slug(budget) / spec.label,
        )
        for budget in COMPUTE_BUDGETS
        for spec in MODEL_SPECS
        if is_feasible(budget, spec)
    ]
    todo = [
        point
        for point in points
        if not (
            complete(point[2] / "trunk_result.json")
            and complete(point[2] / "pure_ar_result.json")
        )
    ]
    print(f"AR trunk points={len(points)} remaining={len(todo)}")
    if args.dry_run:
        for budget, spec, _ in todo:
            print(f"C={budget:.0e} N={spec.label}")
        return
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(execute, point, args.device): point
            for point in todo
        }
        for index, future in enumerate(as_completed(futures), start=1):
            point, process = future.result()
            budget, spec, output = point
            if process.returncode:
                failures.append(str(output))
                print(
                    f"[{index}/{len(todo)}] FAILED C={budget:.0e} N={spec.label}: "
                    f"{process.stderr[-1200:]}",
                    flush=True,
                )
                continue
            result = json.loads((output / "pure_ar_result.json").read_text())
            print(
                f"[{index}/{len(todo)}] C={budget:.0e} N={spec.label} "
                f"AR_val={result['val_ar_ce']:.5f} "
                f"seconds={result['training_duration_seconds']:.1f} "
                f"elapsed_h={(time.monotonic() - started) / 3600:.2f}",
                flush=True,
            )
    if failures:
        raise SystemExit(f"{len(failures)} AR trunks failed: {failures}")


if __name__ == "__main__":
    main()
