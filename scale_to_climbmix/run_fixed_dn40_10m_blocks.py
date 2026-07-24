"""Run the fixed-D/N 10M ClimbMix curriculum block-size comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import MODEL_BY_LABEL, ROOT, TOKENS_PER_STEP, realized_flops
from curriculum_config import realized_flops as curriculum_realized_flops
from train import atomic_json_dump


SIZE = "10M"
SPEC = MODEL_BY_LABEL[SIZE]
TOKEN_PARAMETER_RATIO = 40.0
LEARNING_RATE = 9e-4
P_AR_VALUES = (0.0, 0.1, 0.3, 0.5, 0.7)
BLOCK_LENGTHS = (4, 32)
RESULTS_DIR = ROOT / "results_fixed_dn40_10M"
TRUNK_DIR = RESULTS_DIR / "shared_ar"


def p_slug(p_ar: float) -> str:
    return f"p_ar_{p_ar:.1f}".replace(".", "p")


def total_steps() -> int:
    return round(TOKEN_PARAMETER_RATIO * SPEC.n_params / TOKENS_PER_STEP)


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def trunk_is_complete() -> bool:
    path = TRUNK_DIR / "trunk_result.json"
    if not is_complete(path):
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("fixed_total_steps") != total_steps():
        return False
    if tuple(row.get("p_ar_values", ())) != P_AR_VALUES[1:]:
        return False
    return all(Path(path).exists() for path in row["checkpoints_by_step"].values())


def trunk_command(device: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "curriculum_ar_trunk.py"),
        "--budget",
        str(realized_flops(total_steps(), SPEC)),
        "--size",
        SIZE,
        "--total-steps",
        str(total_steps()),
        "--skip-pure-ar-endpoint",
        "--lr",
        str(LEARNING_RATE),
        "--output-dir",
        str(TRUNK_DIR),
        "--device",
        device,
    ]
    for p_ar in P_AR_VALUES[1:]:
        command.extend(("--p-ar", str(p_ar)))
    return command


def build_runs() -> list[dict]:
    steps = total_steps()
    clean_tokens = steps * TOKENS_PER_STEP
    runs = []
    for p_ar in P_AR_VALUES:
        ar_steps = round(p_ar * steps)
        bd_steps = steps - ar_steps
        budget = (
            realized_flops(steps, SPEC)
            if p_ar == 0.0
            else curriculum_realized_flops(ar_steps, bd_steps, SPEC)
        )
        for block_len in BLOCK_LENGTHS:
            run_dir = RESULTS_DIR / f"block_{block_len}" / p_slug(p_ar)
            runs.append(
                {
                    "size": SIZE,
                    "n_params": SPEC.n_params,
                    "p_ar": p_ar,
                    "block_len": block_len,
                    "steps": steps,
                    "ar_steps": ar_steps,
                    "bd_steps": bd_steps,
                    "clean_tokens": clean_tokens,
                    "realized_d_over_n": clean_tokens / SPEC.n_params,
                    "budget": budget,
                    "run_dir": run_dir,
                    "result": run_dir / "result.json",
                }
            )
    return runs


def command_for(run: dict, device: str) -> list[str]:
    common = [
        "--budget",
        str(run["budget"]),
        "--size",
        SIZE,
        "--output",
        str(run["result"]),
        "--block-len",
        str(run["block_len"]),
        "--device",
        device,
    ]
    if run["p_ar"] == 0.0:
        return [
            sys.executable,
            str(ROOT / "train.py"),
            *common,
            "--lr",
            str(LEARNING_RATE),
            "--steps",
            str(run["steps"]),
        ]
    return [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        *common,
        "--p-ar",
        str(run["p_ar"]),
        "--total-steps",
        str(run["steps"]),
        "--trunk-dir",
        str(TRUNK_DIR),
        "--bd-lr",
        str(LEARNING_RATE),
    ]


def execute_run(run: dict, device: str):
    run["run_dir"].mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        command_for(run, device),
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    (run["run_dir"] / "train.log").write_text(
        process.stdout
        + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
        encoding="utf-8",
    )
    return run, process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    runs = build_runs()
    config = {
        "size": SIZE,
        "n_params": SPEC.n_params,
        "n_layer": SPEC.n_layer,
        "d_model": SPEC.d_model,
        "n_head": SPEC.n_head,
        "head_dim": SPEC.head_dim,
        "d_ff": SPEC.d_ff,
        "token_parameter_ratio_target": TOKEN_PARAMETER_RATIO,
        "total_steps": total_steps(),
        "clean_tokens": total_steps() * TOKENS_PER_STEP,
        "realized_d_over_n": (
            total_steps() * TOKENS_PER_STEP / SPEC.n_params
        ),
        "effective_train_epochs": (
            total_steps() * TOKENS_PER_STEP / 1_606_235_727
        ),
        "learning_rate": LEARNING_RATE,
        "learning_rate_source": (
            "reuse 9e-4 selected for the largest prior ClimbMix models"
        ),
        "weight_decay": 0.1,
        "p_ar_values": list(P_AR_VALUES),
        "block_lengths": list(BLOCK_LENGTHS),
        "planned_runs": len(runs),
        "shared_ar_trunk": True,
    }
    atomic_json_dump(config, RESULTS_DIR / "experiment_config.json")
    pending = [
        run
        for run in runs
        if args.force or not is_complete(run["result"])
    ]
    print(
        f"N={SPEC.n_params:,} steps={total_steps():,} "
        f"D/N={config['realized_d_over_n']:.4f} "
        f"runs={len(runs)} pending={len(pending)}",
        flush=True,
    )
    for run in runs:
        print(
            f"block={run['block_len']:>2} p={run['p_ar']:.1f} "
            f"steps={run['steps']:,} ({run['ar_steps']:,}+{run['bd_steps']:,})",
            flush=True,
        )
    if args.dry_run:
        return

    if args.force or not trunk_is_complete():
        TRUNK_DIR.mkdir(parents=True, exist_ok=True)
        print("training shared AR trunk", flush=True)
        started = time.monotonic()
        process = subprocess.run(
            trunk_command(args.device),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        (TRUNK_DIR / "train.log").write_text(
            process.stdout
            + ("\n[stderr]\n" + process.stderr if process.stderr else ""),
            encoding="utf-8",
        )
        if process.returncode:
            raise RuntimeError(f"AR trunk failed: {process.stderr[-2000:]}")
        print(
            f"shared AR trunk complete elapsed_min="
            f"{(time.monotonic() - started) / 60:.1f}",
            flush=True,
        )
    if not pending:
        return

    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(execute_run, run, args.device): run
            for run in pending
        }
        completed = 0
        for future in as_completed(futures):
            run, process = future.result()
            completed += 1
            if process.returncode:
                failures.append(
                    {
                        "result": str(run["result"]),
                        "returncode": process.returncode,
                        "stderr": process.stderr[-2000:],
                    }
                )
                state = "FAILED"
                loss = ""
            else:
                row = json.loads(run["result"].read_text(encoding="utf-8"))
                state = "complete"
                loss = f" val={float(row['val_nelbo']):.5f}"
            print(
                f"[{completed}/{len(pending)}] {state} "
                f"block={run['block_len']} p={run['p_ar']:.1f}{loss} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )
    if failures:
        atomic_json_dump(failures, RESULTS_DIR / "failures.json")
        raise SystemExit(f"{len(failures)} runs failed")
    print(
        f"all branches complete elapsed_min="
        f"{(time.monotonic() - started) / 60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
