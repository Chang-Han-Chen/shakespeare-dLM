"""
run_isoflop.py — IsoFLOP scaling experiments.

For each FLOP budget, train every (model, size) combo where
    FLOPs = C * N_params * tokens_processed
is held constant, varying the (model_size, num_steps) tradeoff.

C depends on model type (see experiment_config.flop_multiplier).
tokens_processed = batch_size * block_size * num_steps.

Block models use a single fixed block_len (default: 4, from sweep results).
LRs for block models are sweep-derived per (model, size) in experiment_config.

Results saved to isoflop/{budget_label}/{model}/{size}/loss.pkl
or               isoflop/{budget_label}/{model}_bl{block_len}/{size}/loss.pkl

Usage:
    python run_isoflop.py                          # all runs
    python run_isoflop.py --parallel 4             # 4 at a time
    python run_isoflop.py --budgets C1 C2          # subset of budgets
    python run_isoflop.py --models ar remasked     # subset of models
    python run_isoflop.py --block_len 4            # override block_len
    python run_isoflop.py --dry_run                # preview commands
    python run_isoflop.py --summary_only           # rebuild summary from existing
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
    BLOCK_MODELS,
    DEFAULT_BLOCK_LENS,
    ISOFLOP_BLOCK_LEN,
    MODEL_SIZES,
    build_command,
    dropout_for_model,
    flop_multiplier,
    get_optimal_lr,
    is_block_model,
)


# ---------------------------------------------------------------
# FLOP budgets
# ---------------------------------------------------------------

FLOP_BUDGETS = {
    "C1": 6e13,
    "C2": 2e14,
    "C3": 6e14,
    "C4": 2e15,
}

ALL_BUDGETS = list(FLOP_BUDGETS.keys())


# ---------------------------------------------------------------
# Fixed training settings
# ---------------------------------------------------------------

BATCH_SIZE = 128
BLOCK_SIZE = 256
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE  # 32768

MIN_STEPS = 150
MAX_STEPS = 10000

EVAL_ITERS = 50


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def compute_steps(budget, model, size):
    """Steps needed for this (budget, model, size). Returns None if out of bounds."""
    _, _, N = MODEL_SIZES[size]
    C = flop_multiplier(model)
    steps = int(budget / (C * N * TOKENS_PER_STEP))
    if MIN_STEPS <= steps <= MAX_STEPS:
        return steps
    return None


def out_dir_for(budget_label, model, size, block_len=None):
    """Build the output directory path."""
    if block_len is not None:
        return f"isoflop/{budget_label}/{model}_bl{block_len}/{size}"
    return f"isoflop/{budget_label}/{model}/{size}"


def read_losses(loss_path):
    if not os.path.exists(loss_path):
        return None, None
    with open(loss_path, "rb") as f:
        log = pickle.load(f)
    train_final = log["train"][-1][1] if log["train"] else None
    val_final = log["val"][-1][1] if log["val"] else None
    return train_final, val_final


def run_one(config):
    budget_label, budget, model, size, block_len = config
    steps = compute_steps(budget, model, size)
    if steps is None:
        return None

    _, _, N = MODEL_SIZES[size]
    lr = get_optimal_lr(model, size)
    dropout = dropout_for_model(model)
    odir = out_dir_for(budget_label, model, size, block_len)
    os.makedirs(odir, exist_ok=True)

    # Scale warmup and eval_interval to run length
    warmup = min(100, max(10, steps // 20))
    eval_interval = max(steps // 12, 20)

    cmd = build_command(
        model, size, odir,
        max_iters=steps,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        block_len=block_len,
        dropout=dropout,
        lr=lr,
        eval_interval=eval_interval,
        eval_iters=EVAL_ITERS,
        warmup_iters=warmup,
        gpt2_eval_interval=0,
        gpt2_eval_samples=0,
        save_interval=0,
        num_final_samples=0,
        sample_interval=0,
    )

    actual_flops = flop_multiplier(model) * N * steps * TOKENS_PER_STEP

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
    train_final, val_final = read_losses(loss_path)

    return {
        "budget_label": budget_label,
        "budget": budget,
        "model": model,
        "size": size,
        "block_len": block_len,
        "N_params": N,
        "steps": steps,
        "tokens": steps * TOKENS_PER_STEP,
        "flops": actual_flops,
        "lr": lr,
        "dropout": dropout,
        "final_train_ce": train_final,
        "final_val_ce": val_final,
        "runtime_s": runtime,
        "status": status,
        "label": f"{budget_label} {model} {size} bl={block_len} steps={steps}",
        "stderr_tail": stderr_tail,
    }


def enumerate_configs(budget_labels, models, sizes, block_len=ISOFLOP_BLOCK_LEN):
    """Build (budget_label, budget, model, size, block_len) list, interleaved by size.

    Block models use a single fixed block_len (default: ISOFLOP_BLOCK_LEN=4).
    Non-block models use block_len=None.
    """
    configs = []
    for bl in budget_labels:
        budget = FLOP_BUDGETS[bl]
        by_size = defaultdict(list)
        for model in models:
            for size in sizes:
                steps = compute_steps(budget, model, size)
                if steps is None:
                    continue
                if is_block_model(model):
                    by_size[size].append((bl, budget, model, size, block_len))
                else:
                    by_size[size].append((bl, budget, model, size, None))

        max_per = max((len(v) for v in by_size.values()), default=0)
        for i in range(max_per):
            for size in sizes:
                if size in by_size and i < len(by_size[size]):
                    configs.append(by_size[size][i])

    return configs


def write_summary(results, out_path="isoflop/summary.csv"):
    fieldnames = [
        "budget_label", "budget", "model", "size", "block_len",
        "N_params", "steps", "tokens", "flops", "lr", "dropout",
        "final_train_ce", "final_val_ce", "runtime_s", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r["budget_label"], r["model"], r.get("block_len") or 0, r["size"])):
            writer.writerow(r)
    print(f"\nSummary written to {out_path}")


def print_summary_table(results):
    by_budget = defaultdict(list)
    for r in results:
        by_budget[r["budget_label"]].append(r)

    print("\n" + "=" * 130)
    print("ISOFLOP RESULTS")
    print("=" * 130)
    header = (f"{'budget':<6} {'model':<22} {'size':<6} {'bl':>4} {'steps':>6} "
              f"{'tokens':>10} {'FLOPs':>10} {'lr':>8} "
              f"{'train_ce':>9} {'val_ce':>9} {'gap':>7} {'time':>7}")
    print(header)
    print("-" * 130)

    for bl_label in ALL_BUDGETS:
        if bl_label not in by_budget:
            continue
        runs = sorted(by_budget[bl_label], key=lambda r: (r["model"], r.get("block_len") or 0, r["size"]))
        for r in runs:
            train_ce = f"{r['final_train_ce']:.4f}" if r["final_train_ce"] is not None else "N/A"
            val_ce = f"{r['final_val_ce']:.4f}" if r["final_val_ce"] is not None else "N/A"
            if r["final_train_ce"] is not None and r["final_val_ce"] is not None:
                gap = f"{r['final_val_ce'] - r['final_train_ce']:.3f}"
            else:
                gap = "N/A"
            tok_str = f"{r['tokens']/1e6:.1f}M"
            flop_str = f"{r['flops']:.1e}"
            runtime = f"{r['runtime_s']:.0f}s" if r["runtime_s"] is not None else "N/A"
            bl_str = str(r["block_len"]) if r["block_len"] is not None else "-"
            print(f"{r['budget_label']:<6} {r['model']:<22} {r['size']:<6} {bl_str:>4} {r['steps']:>6} "
                  f"{tok_str:>10} {flop_str:>10} {r['lr']:>8.0e} "
                  f"{train_ce:>9} {val_ce:>9} {gap:>7} {runtime:>7}")
        print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IsoFLOP scaling experiments")
    parser.add_argument("--budgets", nargs="+", default=ALL_BUDGETS,
                        choices=ALL_BUDGETS, help="Which FLOP budgets to run")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Which models to run")
    parser.add_argument("--sizes", nargs="+", default=ALL_SIZES,
                        help="Which sizes to run")
    parser.add_argument("--block_len", type=int, default=ISOFLOP_BLOCK_LEN,
                        help="Fixed block length for block models (default: %(default)s)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel runs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true",
                        help="Rebuild summary from existing outputs")
    args = parser.parse_args()

    os.makedirs("isoflop", exist_ok=True)

    configs = enumerate_configs(args.budgets, args.models, args.sizes, block_len=args.block_len)
    total = len(configs)

    print(f"IsoFLOP experiment: {total} runs")
    print(f"Budgets: {args.budgets}")
    print(f"Models: {args.models}")
    print(f"Sizes: {args.sizes}")
    print(f"Block len (for block models): {args.block_len}")
    print(f"Step bounds: [{MIN_STEPS}, {MAX_STEPS}]")

    if args.dry_run:
        print(f"\n--- Coverage matrix ---")
        for bl_label in args.budgets:
            budget = FLOP_BUDGETS[bl_label]
            print(f"\n  {bl_label} ({budget:.0e} FLOPs):")
            header = f"    {'model':<22}"
            for s in args.sizes:
                header += f" {s:>7}"
            header += "  | #sizes"
            print(header)
            print("    " + "-" * (22 + 8 * len(args.sizes) + 10))
            for m in args.models:
                row = f"    {m:<22}"
                n_valid = 0
                for s in args.sizes:
                    st = compute_steps(budget, m, s)
                    if st is not None:
                        row += f" {st:>7}"
                        n_valid += 1
                    else:
                        row += f" {'—':>7}"
                row += f"  |   {n_valid}"
                print(row)
                if is_block_model(m):
                    print(f"    {'':22} (bl={args.block_len})")

        print(f"\n--- Full command list ({total} runs) ---\n")
        for i, (bl_label, budget, model, size, block_len) in enumerate(configs):
            steps = compute_steps(budget, model, size)
            odir = out_dir_for(bl_label, model, size, block_len)
            lr = get_optimal_lr(model, size)
            dropout = dropout_for_model(model)
            bl_str = f" bl={block_len}" if block_len is not None else ""
            print(f"[{i+1}/{total}] {bl_label} {model} {size}{bl_str} steps={steps} lr={lr:.0e} d={dropout}")
        return

    if args.summary_only:
        results = []
        for bl_label, budget, model, size, block_len in configs:
            steps = compute_steps(budget, model, size)
            _, _, N = MODEL_SIZES[size]
            lr = get_optimal_lr(model, size)
            dropout = dropout_for_model(model)
            odir = out_dir_for(bl_label, model, size, block_len)
            loss_path = os.path.join(odir, "loss.pkl")
            train_f, val_f = read_losses(loss_path)
            actual_flops = flop_multiplier(model) * N * steps * TOKENS_PER_STEP
            results.append({
                "budget_label": bl_label, "budget": budget, "model": model,
                "size": size, "block_len": block_len,
                "N_params": N, "steps": steps,
                "tokens": steps * TOKENS_PER_STEP, "flops": actual_flops,
                "lr": lr, "dropout": dropout,
                "final_train_ce": train_f, "final_val_ce": val_f,
                "runtime_s": None,
                "status": "ok" if os.path.exists(loss_path) else "missing",
                "label": f"{bl_label} {model} {size}", "stderr_tail": "",
            })
        write_summary(results)
        print_summary_table(results)
        return

    results = []
    t_start = time.time()

    if args.parallel <= 1:
        for i, config in enumerate(configs):
            bl_label, budget, model, size, block_len = config
            steps = compute_steps(budget, model, size)
            lr = get_optimal_lr(model, size)
            bl_str = f" bl={block_len}" if block_len is not None else ""
            print(f"\n[{i+1}/{total}] {bl_label} {model} {size}{bl_str} steps={steps} lr={lr:.0e}")
            result = run_one(config)
            if result is None:
                continue
            val_str = f"{result['final_val_ce']:.4f}" if result["final_val_ce"] is not None else "N/A"
            print(f"  -> val_ce={val_str}  ({result['runtime_s']:.0f}s)")
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
                result = future.result()
                if result is None:
                    continue
                completed += 1
                val_str = f"{result['final_val_ce']:.4f}" if result["final_val_ce"] is not None else "N/A"
                print(f"[{completed}/{total}] {result['label']}  "
                      f"val_ce={val_str}  ({result['runtime_s']:.0f}s, {result['status']})")
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
