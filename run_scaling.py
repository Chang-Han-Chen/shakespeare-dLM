"""
run_scaling.py — Full training runs with optimal LRs from sweep.

Uses the best (model, size) → LR mapping, all at bs=128.
Runs 4000 steps with eval enabled for apples-to-apples comparison.

For block models, each (model, size) is run at every block_len in
DEFAULT_BLOCK_LENS.

Results saved to scaling/{model}/{size}/loss.pkl
or               scaling/{model}_bl{block_len}/{size}/loss.pkl

Usage:
    python run_scaling.py                          # all runs sequentially
    python run_scaling.py --parallel 4             # 4 at a time
    python run_scaling.py --models ar remasked     # subset
    python run_scaling.py --dry_run
    python run_scaling.py --summary_only
"""

import argparse
import csv
import os
import pickle
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiment_config import (
    ALL_MODELS,
    ALL_SIZES,
    DEFAULT_BLOCK_LENS,
    MODEL_SIZES,
    build_command,
    dropout_for_model,
    get_optimal_lr,
    is_block_model,
)


# ---------------------------------------------------------------
# Fixed training settings
# ---------------------------------------------------------------

BATCH_SIZE = 128
MAX_ITERS = 4000
EVAL_INTERVAL = 300
EVAL_ITERS = 50
GPT2_EVAL_INTERVAL = 0
GPT2_EVAL_SAMPLES = 0
WARMUP_ITERS = 100
SAVE_INTERVAL = 1000


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def out_dir_for(model, size, block_len=None):
    if block_len is not None:
        return f"scaling/{model}_bl{block_len}/{size}"
    return f"scaling/{model}/{size}"


def read_losses(loss_path):
    if not os.path.exists(loss_path):
        return None, None, None
    with open(loss_path, "rb") as f:
        log = pickle.load(f)
    train_final = log["train"][-1][1] if log["train"] else None
    val_final = log["val"][-1][1] if log["val"] else None
    gpt2_final = log["gpt2_ce"][-1][1] if log.get("gpt2_ce") else None
    return train_final, val_final, gpt2_final


def run_one(config):
    model, size, block_len = config
    n_embd, n_layer, _ = MODEL_SIZES[size]
    lr = get_optimal_lr(model, size)
    dropout = dropout_for_model(model)
    odir = out_dir_for(model, size, block_len)
    os.makedirs(odir, exist_ok=True)

    cmd = build_command(
        model, size, odir,
        max_iters=MAX_ITERS,
        batch_size=BATCH_SIZE,
        block_len=block_len,
        dropout=dropout,
        lr=lr,
        eval_interval=EVAL_INTERVAL,
        eval_iters=EVAL_ITERS,
        warmup_iters=WARMUP_ITERS,
        gpt2_eval_interval=GPT2_EVAL_INTERVAL,
        gpt2_eval_samples=GPT2_EVAL_SAMPLES,
        save_interval=SAVE_INTERVAL,
        num_final_samples=5,
    )

    bl_str = f" bl={block_len}" if block_len is not None else ""
    label = f"{model} {size} lr={lr:.0e}{bl_str}"

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
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

    loss_path = os.path.join(odir, "loss.pkl")
    train_final, val_final, gpt2_final = read_losses(loss_path)

    return {
        "model": model,
        "size": size,
        "block_len": block_len,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "lr": lr,
        "dropout": dropout,
        "final_train_ce": train_final,
        "final_val_ce": val_final,
        "final_gpt2_ce": gpt2_final,
        "runtime_s": runtime,
        "status": status,
        "label": label,
        "stderr_tail": stderr_tail,
    }


def enumerate_configs(models, sizes, block_lens):
    """Build (model, size, block_len) list, interleaved by size."""
    configs_by_size = defaultdict(list)
    for model in models:
        for size in sizes:
            if get_optimal_lr(model, size) is None:
                continue
            if is_block_model(model):
                for blen in block_lens:
                    configs_by_size[size].append((model, size, blen))
            else:
                configs_by_size[size].append((model, size, None))

    configs = []
    max_per = max(len(v) for v in configs_by_size.values()) if configs_by_size else 0
    for i in range(max_per):
        for size in sizes:
            if i < len(configs_by_size.get(size, [])):
                configs.append(configs_by_size[size][i])
    return configs


def write_summary(results, out_path="scaling/summary.csv"):
    fieldnames = [
        "model", "size", "block_len", "n_embd", "n_layer", "lr", "dropout",
        "final_train_ce", "final_val_ce", "final_gpt2_ce",
        "runtime_s", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r["model"], r.get("block_len") or 0, r["size"])):
            writer.writerow(r)
    print(f"\nSummary written to {out_path}")


def print_summary_table(results):
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    print("\n" + "=" * 115)
    print(f"SCALING RESULTS ({MAX_ITERS} steps, bs={BATCH_SIZE}, optimal LR per config)")
    print("=" * 115)
    header = (f"{'model':<22} {'size':<6} {'bl':>4} {'lr':<8} {'d':>4} "
              f"{'train_ce':>9} {'val_ce':>9} {'gpt2_ce':>9} {'time':>7} {'status':<6}")
    print(header)
    print("-" * 115)

    for model in ALL_MODELS:
        if model not in by_model:
            continue
        runs = sorted(by_model[model], key=lambda r: (r.get("block_len") or 0, r["size"]))
        for r in runs:
            train_ce = f"{r['final_train_ce']:.4f}" if r["final_train_ce"] is not None else "N/A"
            val_ce = f"{r['final_val_ce']:.4f}" if r["final_val_ce"] is not None else "N/A"
            gpt2_ce = f"{r['final_gpt2_ce']:.4f}" if r["final_gpt2_ce"] is not None else "N/A"
            runtime = f"{r['runtime_s']:.0f}s" if r["runtime_s"] is not None else "N/A"
            bl_str = str(r["block_len"]) if r["block_len"] is not None else "-"
            print(f"{r['model']:<22} {r['size']:<6} {bl_str:>4} {r['lr']:<8.0e} {r['dropout']:>4} "
                  f"{train_ce:>9} {val_ce:>9} {gpt2_ce:>9} {runtime:>7} {r['status']:<6}")
        print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full scaling runs with optimal LRs")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Which models to run")
    parser.add_argument("--sizes", nargs="+", default=ALL_SIZES,
                        help="Which sizes to run")
    parser.add_argument("--block_lens", nargs="+", type=int,
                        default=DEFAULT_BLOCK_LENS,
                        help="Block lengths for block models")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel runs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true",
                        help="Rebuild summary from existing outputs")
    args = parser.parse_args()

    os.makedirs("scaling", exist_ok=True)

    configs = enumerate_configs(args.models, args.sizes, args.block_lens)
    total = len(configs)
    print(f"Scaling runs: {total} configs")

    if args.dry_run:
        for i, (model, size, block_len) in enumerate(configs):
            lr = get_optimal_lr(model, size)
            dropout = dropout_for_model(model)
            bl_str = f" bl={block_len}" if block_len is not None else ""
            print(f"[{i+1}/{total}] {model} {size}{bl_str} lr={lr:.0e} d={dropout}")
        return

    if args.summary_only:
        results = []
        for model, size, block_len in configs:
            n_embd, n_layer, _ = MODEL_SIZES[size]
            lr = get_optimal_lr(model, size)
            dropout = dropout_for_model(model)
            odir = out_dir_for(model, size, block_len)
            loss_path = os.path.join(odir, "loss.pkl")
            train_f, val_f, gpt2_f = read_losses(loss_path)
            results.append({
                "model": model, "size": size, "block_len": block_len,
                "n_embd": n_embd, "n_layer": n_layer,
                "lr": lr, "dropout": dropout,
                "final_train_ce": train_f, "final_val_ce": val_f,
                "final_gpt2_ce": gpt2_f,
                "runtime_s": None,
                "status": "ok" if os.path.exists(loss_path) else "missing",
                "label": f"{model} {size}", "stderr_tail": "",
            })
        write_summary(results)
        print_summary_table(results)
        return

    results = []
    t_start = time.time()

    if args.parallel <= 1:
        for i, config in enumerate(configs):
            model, size, block_len = config
            lr = get_optimal_lr(model, size)
            bl_str = f" bl={block_len}" if block_len is not None else ""
            print(f"\n[{i+1}/{total}] {model} {size}{bl_str} lr={lr:.0e}")
            result = run_one(config)
            val_str = f"{result['final_val_ce']:.4f}" if result["final_val_ce"] is not None else "N/A"
            gpt2_str = f"{result['final_gpt2_ce']:.4f}" if result["final_gpt2_ce"] is not None else "N/A"
            print(f"  -> val_ce={val_str}  gpt2_ce={gpt2_str}  ({result['runtime_s']:.0f}s)")
            if result["stderr_tail"]:
                for line in result["stderr_tail"].split("\n"):
                    print(f"  stderr: {line}")
            results.append(result)
    else:
        print(f"Parallelism: {args.parallel} (interleaved by size)")
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
                gpt2_str = f"{result['final_gpt2_ce']:.4f}" if result["final_gpt2_ce"] is not None else "N/A"
                print(f"[{completed}/{total}] {result['label']}  "
                      f"val_ce={val_str}  gpt2_ce={gpt2_str}  ({result['runtime_s']:.0f}s, {result['status']})")
                if result["stderr_tail"]:
                    for line in result["stderr_tail"].split("\n"):
                        print(f"  stderr: {line}")
                results.append(result)

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f}min)")

    write_summary(results)
    print_summary_table(results)


if __name__ == "__main__":
    main()
