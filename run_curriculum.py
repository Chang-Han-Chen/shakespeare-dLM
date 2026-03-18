"""
run_curriculum.py -- AR warm-start experiments for block MDLM.

Runs a curriculum of:
  phase 1: autoregressive pretraining
  phase 2: block MDLM finetuning from the AR checkpoint

The pure diffusion baseline is represented by p=0.0, which skips phase 1
and trains block MDLM from scratch for the full budget.

Outputs:
  curriculum/p0.3_1M/
    phase1_ar/ckpt.pt
    phase1_ar/loss.pkl
    ckpt.pt
    loss.pkl

Usage:
  python run_curriculum.py
  python run_curriculum.py --dry_run
  python run_curriculum.py --sizes 1M --p_values 0.0 0.3 0.5
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time

from experiment_config import build_command, get_optimal_lr


DEFAULT_SIZES = ["1M", "3M"]
DEFAULT_P_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7]

BATCH_SIZE = 128
BLOCK_SIZE = 256
BLOCK_LEN = 4
TOTAL_STEPS = 1200
EVAL_INTERVAL = 0
EVAL_ITERS = 50
AR_WARMUP_FRAC = 0.20
BD_WARMUP_ITERS = 50
TIMEOUT_S = 7200


def format_p_value(p_value):
    return f"{p_value:.1f}"


def out_dir_for(size, p_value):
    return os.path.join("curriculum", f"p{format_p_value(p_value)}_{size}")


def tail_lines(text, n=10):
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return "\n".join(lines[-n:])


def read_losses(loss_path):
    if not os.path.exists(loss_path):
        return None, None
    with open(loss_path, "rb") as f:
        log = pickle.load(f)
    train_final = log["train"][-1][1] if log.get("train") else None
    val_final = log["val"][-1][1] if log.get("val") else None
    return train_final, val_final


def run_subprocess(cmd, *, cwd, timeout_s, dry_run):
    if dry_run:
        return {"returncode": 0, "stdout": "", "stderr": "", "runtime_s": 0.0}

    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    runtime_s = time.time() - t0
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "runtime_s": runtime_s,
    }


def curriculum_steps(total_steps, p_value):
    ar_steps = int(round(total_steps * p_value))
    bd_steps = total_steps - ar_steps
    return ar_steps, bd_steps


def ar_warmup_iters(ar_steps):
    if ar_steps <= 0:
        return 0
    return max(1, int(round(AR_WARMUP_FRAC * ar_steps)))


def bd_warmup_iters(bd_steps):
    if bd_steps <= 0:
        return 0
    return min(BD_WARMUP_ITERS, bd_steps)


def write_summary(rows, out_path):
    fieldnames = [
        "size",
        "p_ar",
        "ar_steps",
        "bd_steps",
        "ar_lr",
        "bd_lr",
        "ar_warmup_iters",
        "bd_warmup_iters",
        "final_train_ce",
        "final_val_ce",
        "runtime_s",
        "status",
        "run_dir",
        "stderr_tail",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nSummary written to {out_path}")


def maybe_extend(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def build_phase_commands(size, p_value, run_dir, total_steps, *, create_dirs):
    ar_steps, bd_steps = curriculum_steps(total_steps, p_value)
    ar_lr = get_optimal_lr("ar", size)
    bd_lr = get_optimal_lr("block_mdlm", size)

    if bd_lr is None:
        raise ValueError(f"No optimal LR found for block_mdlm size={size}")
    if ar_steps > 0 and ar_lr is None:
        raise ValueError(f"No optimal LR found for ar size={size}")

    phase1_cmd = None
    phase1_ckpt = None

    if ar_steps > 0:
        phase1_dir = os.path.join(run_dir, "phase1_ar")
        if create_dirs:
            os.makedirs(phase1_dir, exist_ok=True)
        phase1_ckpt = os.path.join(phase1_dir, "ckpt.pt")
        phase1_cmd = build_command(
            "ar",
            size,
            phase1_dir,
            max_iters=ar_steps,
            batch_size=BATCH_SIZE,
            block_size=BLOCK_SIZE,
            lr=ar_lr,
            eval_interval=EVAL_INTERVAL,
            eval_iters=EVAL_ITERS,
            warmup_iters=ar_warmup_iters(ar_steps),
            gpt2_eval_interval=0,
            gpt2_eval_samples=0,
            save_interval=0,
            num_final_samples=0,
            sample_interval=0,
        )

    if create_dirs:
        os.makedirs(run_dir, exist_ok=True)
    phase2_cmd = build_command(
        "block_mdlm",
        size,
        run_dir,
        max_iters=bd_steps,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        block_len=BLOCK_LEN,
        lr=bd_lr,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        warmup_iters=bd_warmup_iters(bd_steps),
        gpt2_eval_interval=0,
        gpt2_eval_samples=0,
        save_interval=0,
        num_final_samples=0,
        sample_interval=0,
    )
    maybe_extend(phase2_cmd, "--resume_from", phase1_ckpt)

    return {
        "phase1_cmd": phase1_cmd,
        "phase2_cmd": phase2_cmd,
        "phase1_ckpt": phase1_ckpt,
        "ar_steps": ar_steps,
        "bd_steps": bd_steps,
        "ar_lr": ar_lr,
        "bd_lr": bd_lr,
        "ar_warmup_iters": ar_warmup_iters(ar_steps),
        "bd_warmup_iters": bd_warmup_iters(bd_steps),
    }


def run_one(size, p_value, *, total_steps, timeout_s, dry_run, skip_existing):
    run_dir = out_dir_for(size, p_value)
    loss_path = os.path.join(run_dir, "loss.pkl")
    if skip_existing and os.path.exists(loss_path):
        train_final, val_final = read_losses(loss_path)
        ar_steps, bd_steps = curriculum_steps(total_steps, p_value)
        return {
            "size": size,
            "p_ar": format_p_value(p_value),
            "ar_steps": ar_steps,
            "bd_steps": bd_steps,
            "ar_lr": get_optimal_lr("ar", size) if ar_steps > 0 else None,
            "bd_lr": get_optimal_lr("block_mdlm", size),
            "ar_warmup_iters": ar_warmup_iters(ar_steps),
            "bd_warmup_iters": bd_warmup_iters(bd_steps),
            "final_train_ce": train_final,
            "final_val_ce": val_final,
            "runtime_s": 0.0,
            "status": "skipped",
            "run_dir": run_dir,
            "stderr_tail": "",
        }

    cmds = build_phase_commands(
        size,
        p_value,
        run_dir,
        total_steps,
        create_dirs=not dry_run,
    )
    label = f"p={format_p_value(p_value)} size={size}"

    if dry_run:
        print(f"\n[{label}]")
        if cmds["phase1_cmd"] is not None:
            print("phase1:", " ".join(cmds["phase1_cmd"]))
        print("phase2:", " ".join(cmds["phase2_cmd"]))
        return {
            "size": size,
            "p_ar": format_p_value(p_value),
            "ar_steps": cmds["ar_steps"],
            "bd_steps": cmds["bd_steps"],
            "ar_lr": cmds["ar_lr"],
            "bd_lr": cmds["bd_lr"],
            "ar_warmup_iters": cmds["ar_warmup_iters"],
            "bd_warmup_iters": cmds["bd_warmup_iters"],
            "final_train_ce": None,
            "final_val_ce": None,
            "runtime_s": 0.0,
            "status": "dry_run",
            "run_dir": run_dir,
            "stderr_tail": "",
        }

    total_runtime_s = 0.0

    if cmds["phase1_cmd"] is not None:
        print(f"Running phase 1 AR for {label} ({cmds['ar_steps']} steps)")
        phase1 = run_subprocess(
            cmds["phase1_cmd"],
            cwd=os.getcwd(),
            timeout_s=timeout_s,
            dry_run=dry_run,
        )
        total_runtime_s += phase1["runtime_s"]
        if phase1["returncode"] != 0:
            return {
                "size": size,
                "p_ar": format_p_value(p_value),
                "ar_steps": cmds["ar_steps"],
                "bd_steps": cmds["bd_steps"],
                "ar_lr": cmds["ar_lr"],
                "bd_lr": cmds["bd_lr"],
                "ar_warmup_iters": cmds["ar_warmup_iters"],
                "bd_warmup_iters": cmds["bd_warmup_iters"],
                "final_train_ce": None,
                "final_val_ce": None,
                "runtime_s": total_runtime_s,
                "status": "phase1_fail",
                "run_dir": run_dir,
                "stderr_tail": tail_lines(phase1["stderr"] or phase1["stdout"]),
            }

    print(f"Running phase 2 block_mdlm for {label} ({cmds['bd_steps']} steps)")
    phase2 = run_subprocess(
        cmds["phase2_cmd"],
        cwd=os.getcwd(),
        timeout_s=timeout_s,
        dry_run=dry_run,
    )
    total_runtime_s += phase2["runtime_s"]

    train_final, val_final = read_losses(loss_path)
    status = "ok" if phase2["returncode"] == 0 else "phase2_fail"
    stderr_tail = ""
    if phase2["returncode"] != 0:
        stderr_tail = tail_lines(phase2["stderr"] or phase2["stdout"])

    return {
        "size": size,
        "p_ar": format_p_value(p_value),
        "ar_steps": cmds["ar_steps"],
        "bd_steps": cmds["bd_steps"],
        "ar_lr": cmds["ar_lr"],
        "bd_lr": cmds["bd_lr"],
        "ar_warmup_iters": cmds["ar_warmup_iters"],
        "bd_warmup_iters": cmds["bd_warmup_iters"],
        "final_train_ce": train_final,
        "final_val_ce": val_final,
        "runtime_s": total_runtime_s,
        "status": status,
        "run_dir": run_dir,
        "stderr_tail": stderr_tail,
    }


def main():
    parser = argparse.ArgumentParser(description="Run AR warm-start curriculum for block MDLM")
    parser.add_argument("--sizes", nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--p_values", nargs="+", type=float, default=DEFAULT_P_VALUES)
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--timeout_s", type=int, default=TIMEOUT_S)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    os.makedirs("curriculum", exist_ok=True)

    rows = []
    total = len(args.sizes) * len(args.p_values)
    print(f"Curriculum runs: {total}")

    for size in args.sizes:
        for p_value in args.p_values:
            if not 0.0 <= p_value < 1.0:
                raise ValueError(f"p_values must be in [0.0, 1.0), got {p_value}")
            row = run_one(
                size,
                p_value,
                total_steps=args.total_steps,
                timeout_s=args.timeout_s,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
            rows.append(row)
            train_ce = row["final_train_ce"]
            val_ce = row["final_val_ce"]
            train_str = f"{train_ce:.4f}" if train_ce is not None else "N/A"
            val_str = f"{val_ce:.4f}" if val_ce is not None else "N/A"
            print(
                f"Completed p={row['p_ar']} size={size} | "
                f"train {train_str} | val {val_str} | status {row['status']}"
            )

    if args.dry_run:
        print("\nDry run complete.")
        return

    write_summary(rows, os.path.join("curriculum", "summary.csv"))


if __name__ == "__main__":
    main()
