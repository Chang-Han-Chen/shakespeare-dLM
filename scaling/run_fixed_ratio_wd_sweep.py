"""Run a fixed-D/N AR-weight-decay sweep on TinyShakespeare."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import BLOCK_LEN, MODEL_BY_LABEL, ROOT, TOKENS_PER_STEP, realized_flops
from curriculum_config import realized_flops as curriculum_realized_flops
from train import atomic_json_dump


DEFAULT_TOKEN_PARAMETER_RATIO = 120.0
SIZE_TO_LR = {
    "0.4M": 3e-3,
    "0.8M": 1e-3,
}
P_AR_VALUES = (0.0, 0.1, 0.3, 0.5, 0.7)
AR_WEIGHT_DECAYS = (0.1, 0.2, 0.4, 0.8)
BD_WEIGHT_DECAY = 0.1


def ratio_slug(token_parameter_ratio: float) -> str:
    return f"{token_parameter_ratio:g}".replace(".", "p")


def results_dir_for(
    token_parameter_ratio: float,
    block_len: int = BLOCK_LEN,
) -> Path:
    suffix = "" if block_len == BLOCK_LEN else f"_bl{block_len}"
    return ROOT / f"results_fixed_dn{ratio_slug(token_parameter_ratio)}{suffix}"


def p_slug(p_ar: float) -> str:
    return f"p_ar_{p_ar:.1f}".replace(".", "p")


def wd_slug(weight_decay: float) -> str:
    return f"ar_wd_{weight_decay:.1f}".replace(".", "p")


def total_steps_for(
    size: str,
    token_parameter_ratio: float = DEFAULT_TOKEN_PARAMETER_RATIO,
) -> int:
    spec = MODEL_BY_LABEL[size]
    return round(token_parameter_ratio * spec.n_params / TOKENS_PER_STEP)


def is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, json.JSONDecodeError):
        return False


def build_runs(
    token_parameter_ratio: float = DEFAULT_TOKEN_PARAMETER_RATIO,
    block_len: int = BLOCK_LEN,
) -> list[dict]:
    results_dir = results_dir_for(token_parameter_ratio, block_len)
    runs = []
    for size, lr in SIZE_TO_LR.items():
        spec = MODEL_BY_LABEL[size]
        steps = total_steps_for(size, token_parameter_ratio)
        clean_tokens = steps * TOKENS_PER_STEP
        common = {
            "size": size,
            "spec": spec,
            "lr": lr,
            "steps": steps,
            "clean_tokens": clean_tokens,
            "realized_d_over_n": clean_tokens / spec.n_params,
            "block_len": block_len,
        }
        baseline_dir = results_dir / size / p_slug(0.0)
        runs.append(
            {
                **common,
                "p_ar": 0.0,
                "ar_weight_decay": None,
                "budget": realized_flops(steps, spec),
                "run_dir": baseline_dir,
                "result": baseline_dir / "result.json",
            }
        )
        for p_ar in P_AR_VALUES[1:]:
            for ar_weight_decay in AR_WEIGHT_DECAYS:
                run_dir = (
                    results_dir
                    / size
                    / p_slug(p_ar)
                    / wd_slug(ar_weight_decay)
                )
                runs.append(
                    {
                        **common,
                        "p_ar": p_ar,
                        "ar_weight_decay": ar_weight_decay,
                        "budget": curriculum_realized_flops(steps, spec, p_ar),
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
        run["size"],
        "--lr",
        str(run["lr"]),
        "--steps",
        str(run["steps"]),
        "--block-len",
        str(run["block_len"]),
        "--output",
        str(run["result"]),
        "--device",
        device,
    ]
    if run["p_ar"] == 0.0:
        return [sys.executable, str(ROOT / "train.py"), *common]
    return [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        *common,
        "--p-ar",
        str(run["p_ar"]),
        "--lr-source",
        "prior_dense_attention_full_run",
        "--ar-weight-decay",
        str(run["ar_weight_decay"]),
        "--bd-weight-decay",
        str(BD_WEIGHT_DECAY),
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
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_TOKEN_PARAMETER_RATIO,
        help="Target clean-token to counted-parameter ratio D/N.",
    )
    parser.add_argument("--block-len", type=int, default=BLOCK_LEN)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.ratio <= 0:
        raise ValueError("ratio must be positive")
    if args.block_len < 1 or 256 % args.block_len:
        raise ValueError("block-len must be a positive divisor of 256")
    results_dir = results_dir_for(args.ratio, args.block_len)
    all_runs = build_runs(args.ratio, args.block_len)
    pending = [
        run
        for run in all_runs
        if args.force or not is_complete(run["result"])
    ]
    config = {
        "token_parameter_ratio_target": args.ratio,
        "block_len": args.block_len,
        "p_ar_values": list(P_AR_VALUES),
        "ar_weight_decays": list(AR_WEIGHT_DECAYS),
        "bd_weight_decay": BD_WEIGHT_DECAY,
        "lr_policy": "reuse prior dense-attention full-run selections",
        "sizes": [
            {
                "size": size,
                "n_params": MODEL_BY_LABEL[size].n_params,
                "learning_rate": lr,
                "total_steps": total_steps_for(size, args.ratio),
                "clean_tokens": (
                    total_steps_for(size, args.ratio) * TOKENS_PER_STEP
                ),
                "realized_d_over_n": (
                    total_steps_for(size, args.ratio)
                    * TOKENS_PER_STEP
                    / MODEL_BY_LABEL[size].n_params
                ),
            }
            for size, lr in SIZE_TO_LR.items()
        ],
        "planned_runs": len(all_runs),
    }
    atomic_json_dump(config, results_dir / "experiment_config.json")
    print(
        f"planned={len(all_runs)} pending={len(pending)} "
        f"steps={sum(run['steps'] for run in pending):,}",
        flush=True,
    )
    for run in all_runs:
        wd = (
            "n/a"
            if run["ar_weight_decay"] is None
            else f"{run['ar_weight_decay']:.1f}"
        )
        print(
            f"{run['size']:>4} p={run['p_ar']:.1f} ar_wd={wd:>3} "
            f"bl={run['block_len']:>2} steps={run['steps']:>4} "
            f"D/N={run['realized_d_over_n']:.4f} "
            f"lr={run['lr']:.1e}",
            flush=True,
        )
    if args.dry_run or not pending:
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
                        "stderr": process.stderr[-1600:],
                    }
                )
                state = "FAILED"
                loss = ""
            else:
                row = json.loads(run["result"].read_text(encoding="utf-8"))
                state = "complete"
                loss = f" val={float(row['val_nelbo']):.5f}"
            print(
                f"[{completed}/{len(pending)}] {state} {run['size']} "
                f"p={run['p_ar']:.1f} wd={run['ar_weight_decay']}{loss} "
                f"elapsed_min={(time.monotonic() - started) / 60:.1f}",
                flush=True,
            )
    if failures:
        atomic_json_dump(failures, results_dir / "failures.json")
        raise SystemExit(f"{len(failures)} runs failed")
    print(
        f"all complete elapsed_min={(time.monotonic() - started) / 60:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
