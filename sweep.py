"""
sweep.py — Grid search over learning rate × model size, per model variant.

Grid:
    model_sizes : 0.1M, 0.3M, 1M, 3M  (n_embd × n_layer)
    learning_rates : 3e-4, 1e-3, 3e-3, 1e-2
    batch_sizes : 64, 128
    models : remasked, mdlm, edit_one_pass, edit_two_pass, ar

Each run launches train.py as a subprocess with GPT2 eval disabled,
eval_iters=20, sample_interval=0 (no printing samples during sweep).

Results are saved under sweep/{model}/{size}_lr{lr}_bs{bs}/loss.pkl
and a summary CSV is written to sweep/summary.csv after all runs.

Usage:
    python sweep.py                          # run full grid sequentially
    python sweep.py --parallel 8             # 8 runs at a time on A100
    python sweep.py --models ar remasked     # subset of models
    python sweep.py --dry_run                # print commands without running
"""

import argparse
import csv
import itertools
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------

MODEL_SIZES = {
    #  tag:   (n_embd, n_layer, approx_params)
    "0.1M":  (64,  2),
    "0.3M":  (96,  3),
    "1M":    (128, 5),
    "3M":    (256, 4),
}

LEARNING_RATES = [3e-4, 1e-3, 3e-3, 1e-2]
BATCH_SIZES = [64, 128]

ALL_MODELS = ["remasked", "mdlm", "edit_one_pass", "edit_two_pass", "ar"]

# Sweep training settings (shorter than full run)
SWEEP_MAX_ITERS = 800
SWEEP_EVAL_INTERVAL = 200
SWEEP_EVAL_ITERS = 20
SWEEP_WARMUP_ITERS = 100
SWEEP_SAVE_INTERVAL = 0  # don't save checkpoints during sweep


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def build_command(model, size_tag, lr, bs, out_dir):
    """Build the train.py command for one sweep run."""
    n_embd, n_layer = MODEL_SIZES[size_tag]
    min_lr = lr / 10.0

    loss_path = os.path.join(out_dir, "loss.pkl")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--n_embd", str(n_embd),
        "--n_layer", str(n_layer),
        "--n_head", "4",
        "--learning_rate", str(lr),
        "--min_lr", str(min_lr),
        "--batch_size", str(bs),
        "--max_iters", str(SWEEP_MAX_ITERS),
        "--eval_interval", str(SWEEP_EVAL_INTERVAL),
        "--eval_iters", str(SWEEP_EVAL_ITERS),
        "--warmup_iters", str(SWEEP_WARMUP_ITERS),
        "--gpt2_eval_interval", "0",       # disable GPT2 eval
        "--gpt2_eval_samples", "0",        # disable GPT2 eval
        "--sample_interval", "0",          # no sample printing
        "--save_interval", str(SWEEP_SAVE_INTERVAL) if SWEEP_SAVE_INTERVAL > 0 else "999999",
        "--num_final_samples", "0",
        "--loss_log_path", loss_path,
        "--checkpoint_path", ckpt_path,
    ]
    return cmd


def read_final_losses(loss_path):
    """Read the loss pickle and return final train/val CE."""
    if not os.path.exists(loss_path):
        return None, None
    with open(loss_path, "rb") as f:
        log = pickle.load(f)

    train_final = log["train"][-1][1] if log["train"] else None
    val_final = log["val"][-1][1] if log["val"] else None
    return train_final, val_final


def write_summary(results, out_path="sweep/summary.csv"):
    """Write a CSV summary of all sweep runs."""
    fieldnames = [
        "model", "size", "n_embd", "n_layer", "lr", "batch_size",
        "final_train_ce", "final_val_ce", "runtime_s", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r["model"], r["size"], r["lr"], r["batch_size"])):
            writer.writerow(r)
    print(f"\nSummary written to {out_path}")


def print_summary_table(results):
    """Print a formatted summary table to stdout."""
    # Group by model, then sort by val_ce
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    print("\n" + "=" * 95)
    print("SWEEP RESULTS")
    print("=" * 95)
    header = f"{'model':<15} {'size':<6} {'lr':<8} {'bs':<4} {'train_ce':>9} {'val_ce':>9} {'time':>7} {'status':<6}"
    print(header)
    print("-" * 95)

    for model in ALL_MODELS:
        if model not in by_model:
            continue
        runs = sorted(by_model[model], key=lambda r: r["final_val_ce"] if r["final_val_ce"] is not None else 1e9)
        for r in runs:
            train_ce = f"{r['final_train_ce']:.4f}" if r["final_train_ce"] is not None else "N/A"
            val_ce = f"{r['final_val_ce']:.4f}" if r["final_val_ce"] is not None else "N/A"
            runtime = f"{r['runtime_s']:.0f}s" if r["runtime_s"] is not None else "N/A"
            print(f"{r['model']:<15} {r['size']:<6} {r['lr']:<8.0e} {r['batch_size']:<4} {train_ce:>9} {val_ce:>9} {runtime:>7} {r['status']:<6}")
        print()


# ---------------------------------------------------------------
# Single-run worker (used by both sequential and parallel modes)
# ---------------------------------------------------------------

def run_one(config):
    """Run a single sweep config. Returns a result dict."""
    model, size, lr, bs = config
    n_embd, n_layer = MODEL_SIZES[size]
    out_dir = f"sweep/{model}/{size}_lr{lr:.0e}_bs{bs}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_command(model, size, lr, bs, out_dir)
    label = f"{model} {size} lr={lr:.0e} bs={bs}"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1h max per run
        )
        runtime = time.time() - t0
        status = "ok" if proc.returncode == 0 else "fail"
        stderr_tail = ""

        if proc.returncode != 0:
            stderr_lines = proc.stderr.strip().split("\n")
            stderr_tail = "\n".join(stderr_lines[-10:])

    except subprocess.TimeoutExpired:
        runtime = time.time() - t0
        status = "timeout"
        stderr_tail = ""

    loss_path = os.path.join(out_dir, "loss.pkl")
    train_final, val_final = read_final_losses(loss_path)

    return {
        "model": model,
        "size": size,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "lr": lr,
        "batch_size": bs,
        "final_train_ce": train_final,
        "final_val_ce": val_final,
        "runtime_s": runtime,
        "status": status,
        "label": label,
        "stderr_tail": stderr_tail,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Which models to sweep")
    parser.add_argument("--sizes", nargs="+", default=list(MODEL_SIZES.keys()),
                        help="Which sizes to sweep")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of runs to launch in parallel (default: 1 = sequential)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    os.makedirs("sweep", exist_ok=True)

    # Build all run configs
    configs = list(itertools.product(args.models, args.sizes, LEARNING_RATES, BATCH_SIZES))
    total = len(configs)
    print(f"Sweep: {total} runs ({len(args.models)} models × {len(args.sizes)} sizes × "
          f"{len(LEARNING_RATES)} LRs × {len(BATCH_SIZES)} BSs)")
    if args.parallel > 1:
        print(f"Parallelism: {args.parallel} concurrent runs")

    if args.dry_run:
        for i, (model, size, lr, bs) in enumerate(configs):
            out_dir = f"sweep/{model}/{size}_lr{lr:.0e}_bs{bs}"
            cmd = build_command(model, size, lr, bs, out_dir)
            print(f"[{i+1}/{total}] {model} {size} lr={lr:.0e} bs={bs}:")
            print(f"  {' '.join(cmd)}")
        return

    results = []
    t_sweep_start = time.time()

    if args.parallel <= 1:
        # Sequential
        for i, config in enumerate(configs):
            label = f"{config[0]} {config[1]} lr={config[2]:.0e} bs={config[3]}"
            print(f"\n[{i+1}/{total}] {label}")
            result = run_one(config)
            val_str = f"{result['final_val_ce']:.4f}" if result["final_val_ce"] is not None else "N/A"
            print(f"  -> val_ce={val_str}  ({result['runtime_s']:.0f}s, {result['status']})")
            if result["stderr_tail"]:
                for line in result["stderr_tail"].split("\n"):
                    print(f"  stderr: {line}")
            results.append(result)
    else:
        # Parallel
        completed = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as executor:
            future_to_config = {
                executor.submit(run_one, config): config
                for config in configs
            }
            for future in as_completed(future_to_config):
                completed += 1
                result = future.result()
                val_str = f"{result['final_val_ce']:.4f}" if result["final_val_ce"] is not None else "N/A"
                print(f"[{completed}/{total}] {result['label']}  "
                      f"val_ce={val_str}  ({result['runtime_s']:.0f}s, {result['status']})")
                if result["stderr_tail"]:
                    for line in result["stderr_tail"].split("\n"):
                        print(f"  stderr: {line}")
                results.append(result)

    t_sweep_total = time.time() - t_sweep_start
    print(f"\nTotal sweep time: {t_sweep_total:.0f}s ({t_sweep_total/60:.1f}min)")

    write_summary(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()
