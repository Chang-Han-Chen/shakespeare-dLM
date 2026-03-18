"""
run_sweep.py — Find best (LR, block_len) per (model, size).

Grid: 4 block models × 4 sizes × 4 LRs × 3 block_lens = 192 runs.
Outputs: all losses, then best (LR, block_len) per (model, size).

Usage:
    python run_sweep.py                    # full grid, sequential
    python run_sweep.py --parallel 8       # 8 at a time
    python run_sweep.py --dry_run          # preview configs
"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_config import (
    BLOCK_MODELS,
    MODEL_SIZES,
    build_command,
    dropout_for_model,
)

SWEEP_SIZES = ["0.1M", "0.3M", "1M", "3M"]
LEARNING_RATES = [3e-4, 1e-3, 3e-3, 1e-2]
BLOCK_LENS = [1, 4, 16]

SWEEP_MAX_ITERS = 800
SWEEP_EVAL_INTERVAL = 200
SWEEP_EVAL_ITERS = 20
SWEEP_WARMUP_ITERS = 100


def out_dir_for(model, size, lr, block_len):
    return f"sweep/{model}_bl{block_len}/{size}_lr{lr:.0e}"


def read_final_val_loss(odir):
    p = os.path.join(odir, "loss.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        log = pickle.load(f)
    if log.get("val"):
        return log["val"][-1][1]
    return None


def run_one(config):
    model, size, lr, block_len = config
    odir = out_dir_for(model, size, lr, block_len)
    os.makedirs(odir, exist_ok=True)

    cmd = build_command(
        model, size, odir,
        max_iters=SWEEP_MAX_ITERS,
        block_len=block_len,
        dropout=dropout_for_model(model),
        lr=lr,
        eval_interval=SWEEP_EVAL_INTERVAL,
        eval_iters=SWEEP_EVAL_ITERS,
        warmup_iters=SWEEP_WARMUP_ITERS,
        gpt2_eval_interval=0,
        gpt2_eval_samples=0,
        save_interval=0,
        num_final_samples=0,
        sample_interval=0,
    )

    t0 = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        pass
    runtime = time.time() - t0

    val_loss = read_final_val_loss(odir)
    return (model, size, lr, block_len, val_loss, runtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    configs = [
        (model, size, lr, bl)
        for model in BLOCK_MODELS
        for size in SWEEP_SIZES
        for lr in LEARNING_RATES
        for bl in BLOCK_LENS
    ]
    total = len(configs)
    print(f"Sweep: {total} runs ({len(BLOCK_MODELS)} models × {len(SWEEP_SIZES)} sizes × {len(LEARNING_RATES)} LRs × {len(BLOCK_LENS)} block_lens)")

    if args.dry_run:
        for model, size, lr, bl in configs:
            print(f"  {model} {size} lr={lr:.0e} bl={bl}")
        return

    os.makedirs("sweep", exist_ok=True)
    results = []
    t_start = time.time()

    if args.parallel <= 1:
        for i, config in enumerate(configs):
            model, size, lr, bl = config
            print(f"[{i+1}/{total}] {model} {size} lr={lr:.0e} bl={bl}", end=" ", flush=True)
            result = run_one(config)
            val = result[4]
            print(f"val={val:.4f}" if val is not None else "val=N/A")
            results.append(result)
    else:
        print(f"Running {args.parallel} parallel jobs")
        done = 0
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(run_one, c): c for c in configs}
            for fut in as_completed(futs):
                done += 1
                r = fut.result()
                model, size, lr, bl, val, _ = r
                v = f"{val:.4f}" if val is not None else "N/A"
                print(f"[{done}/{total}] {model} {size} lr={lr:.0e} bl={bl} val={v}")
                results.append(r)

    print(f"\nTotal time: {time.time() - t_start:.0f}s")

    # --- All losses ---
    print("\n--- All losses ---")
    print(f"{'model':<28} {'size':<6} {'lr':<8} {'bl':>3} {'val_ce':>9}")
    for model, size, lr, bl, val, _ in sorted(results):
        v = f"{val:.4f}" if val is not None else "N/A"
        print(f"{model:<28} {size:<6} {lr:<8.0e} {bl:>3} {v:>9}")

    # --- Best (LR, block_len) per (model, size) ---
    best = {}
    for model, size, lr, bl, val, _ in results:
        if val is None:
            continue
        key = (model, size)
        if key not in best or val < best[key][2]:
            best[key] = (lr, bl, val)

    print(f"\n--- Best (LR, block_len) per (model, size): {len(best)} combos ---")
    print(f"{'model':<28} {'size':<6} {'lr':<8} {'bl':>3} {'val_ce':>9}")
    for (model, size) in sorted(best):
        lr, bl, val = best[(model, size)]
        print(f"{model:<28} {size:<6} {lr:<8.0e} {bl:>3} {val:.4f}")


if __name__ == "__main__":
    main()
