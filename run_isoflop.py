"""
run_isoflop.py — IsoFLOP scaling experiments.

For each FLOP budget, train every (model, size) combo where
    FLOPs = C * N_params * tokens_processed
is held constant, varying the (model_size, num_steps) tradeoff.

C = 6 for all models except edit_two_pass which costs C = 12 (two forward+backward).
tokens_processed = batch_size * block_size * num_steps.

Results saved to isoflop/{budget_label}/{model}/{size}/loss.pkl

Usage:
    python run_isoflop.py                          # all runs
    python run_isoflop.py --parallel 4             # 4 at a time
    python run_isoflop.py --budgets C1 C2          # subset of budgets
    python run_isoflop.py --models ar remasked     # subset of models
    python run_isoflop.py --dry_run                # preview commands
    python run_isoflop.py --summary_only           # rebuild summary from existing
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------
# Model size configs: label -> (n_embd, n_layer, approx N_params)
# ---------------------------------------------------------------

MODEL_SIZES = {
    "0.1M":  (64,   2,  0.107e6),
    "0.3M":  (96,   3,  0.345e6),
    "0.5M":  (116,  3,  0.501e6),
    "1M":    (128,  5,  1.002e6),
    "2M":    (200,  4,  1.949e6),
    "3M":    (256,  4,  3.182e6),
}

ALL_SIZES = list(MODEL_SIZES.keys())
ALL_MODELS = ["remasked", "mdlm", "edit_one_pass", "edit_two_pass", "ar"]


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
# Optimal LR per (model, size) — from sweep at dropout=0.1/0.2
# New sizes (0.5M, 2M) interpolated from neighbors.
# ---------------------------------------------------------------

OPTIMAL_LR = {
    # remasked: all 3e-3 except 0.1M
    ("remasked",      "0.1M"): 1e-2,
    ("remasked",      "0.3M"): 3e-3,
    ("remasked",      "0.5M"): 3e-3,
    ("remasked",      "1M"):   3e-3,
    ("remasked",      "2M"):   3e-3,
    ("remasked",      "3M"):   3e-3,

    # mdlm: same pattern as remasked
    ("mdlm",          "0.1M"): 1e-2,
    ("mdlm",          "0.3M"): 3e-3,
    ("mdlm",          "0.5M"): 3e-3,
    ("mdlm",          "1M"):   3e-3,
    ("mdlm",          "2M"):   3e-3,
    ("mdlm",          "3M"):   3e-3,

    # edit_one_pass: 1e-2 for small, 3e-3 for large
    ("edit_one_pass",  "0.1M"): 1e-2,
    ("edit_one_pass",  "0.3M"): 1e-2,
    ("edit_one_pass",  "0.5M"): 1e-2,
    ("edit_one_pass",  "1M"):   1e-2,
    ("edit_one_pass",  "2M"):   3e-3,
    ("edit_one_pass",  "3M"):   3e-3,

    # edit_two_pass: all 3e-3 except 0.1M
    ("edit_two_pass",  "0.1M"): 1e-2,
    ("edit_two_pass",  "0.3M"): 3e-3,
    ("edit_two_pass",  "0.5M"): 3e-3,
    ("edit_two_pass",  "1M"):   3e-3,
    ("edit_two_pass",  "2M"):   3e-3,
    ("edit_two_pass",  "3M"):   3e-3,

    # ar (dropout=0.2): 1e-2 for <=1M, 3e-3 for >=2M
    ("ar",            "0.1M"): 1e-2,
    ("ar",            "0.3M"): 1e-2,
    ("ar",            "0.5M"): 1e-2,
    ("ar",            "1M"):   1e-2,
    ("ar",            "2M"):   3e-3,
    ("ar",            "3M"):   3e-3,
}


# ---------------------------------------------------------------
# Fixed training settings
# ---------------------------------------------------------------

BATCH_SIZE = 128
BLOCK_SIZE = 256
TOKENS_PER_STEP = BATCH_SIZE * BLOCK_SIZE  # 32768

MIN_STEPS = 150
MAX_STEPS = 10000

EVAL_ITERS = 50
GPT2_EVAL_INTERVAL = 0
GPT2_EVAL_SAMPLES = 0
SAVE_INTERVAL = 0   # no checkpoints to save disk; set >0 if you want them


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def flop_multiplier(model):
    """edit_two_pass does 2 forward+backward per step."""
    return 12 if model == "edit_two_pass" else 6


def dropout_for_model(model):
    return 0.2 if model == "ar" else 0.1


def compute_steps(budget, model, size):
    """Steps needed for this (budget, model, size). Returns None if out of bounds."""
    _, _, N = MODEL_SIZES[size]
    C = flop_multiplier(model)
    steps = int(budget / (C * N * TOKENS_PER_STEP))
    if MIN_STEPS <= steps <= MAX_STEPS:
        return steps
    return None


def build_command(model, size, steps, out_dir):
    n_embd, n_layer, _ = MODEL_SIZES[size]
    lr = OPTIMAL_LR[(model, size)]
    min_lr = lr / 10.0
    dropout = dropout_for_model(model)

    # Scale warmup and eval_interval to run length
    warmup = min(100, max(10, steps // 20))
    eval_interval = max(steps // 12, 20)  # ~12 evals per run + forced final eval

    loss_path = os.path.join(out_dir, "loss.pkl")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--n_embd", str(n_embd),
        "--n_layer", str(n_layer),
        "--n_head", "4",
        "--dropout", str(dropout),
        "--learning_rate", str(lr),
        "--min_lr", str(min_lr),
        "--batch_size", str(BATCH_SIZE),
        "--block_size", str(BLOCK_SIZE),
        "--max_iters", str(steps),
        "--eval_interval", str(eval_interval),
        "--eval_iters", str(EVAL_ITERS),
        "--warmup_iters", str(warmup),
        "--gpt2_eval_interval", str(GPT2_EVAL_INTERVAL),
        "--gpt2_eval_samples", str(GPT2_EVAL_SAMPLES),
        "--sample_interval", "0",
        "--save_interval", str(SAVE_INTERVAL) if SAVE_INTERVAL > 0 else str(steps + 1),
        "--num_final_samples", "0",
        "--loss_log_path", loss_path,
        "--checkpoint_path", ckpt_path,
    ]
    return cmd


def read_losses(loss_path):
    if not os.path.exists(loss_path):
        return None, None
    with open(loss_path, "rb") as f:
        log = pickle.load(f)
    train_final = log["train"][-1][1] if log["train"] else None
    val_final = log["val"][-1][1] if log["val"] else None
    return train_final, val_final


def run_one(config):
    budget_label, budget, model, size = config
    steps = compute_steps(budget, model, size)
    if steps is None:
        return None

    n_embd, n_layer, N = MODEL_SIZES[size]
    lr = OPTIMAL_LR[(model, size)]
    dropout = dropout_for_model(model)
    out_dir = f"isoflop/{budget_label}/{model}/{size}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_command(model, size, steps, out_dir)
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

    loss_path = os.path.join(out_dir, "loss.pkl")
    train_final, val_final = read_losses(loss_path)

    return {
        "budget_label": budget_label,
        "budget": budget,
        "model": model,
        "size": size,
        "n_embd": n_embd,
        "n_layer": n_layer,
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
        "label": f"{budget_label} {model} {size} steps={steps}",
        "stderr_tail": stderr_tail,
    }


def enumerate_configs(budget_labels, models, sizes):
    """Build (budget_label, budget, model, size) list, interleaved by size."""
    configs = []
    for bl in budget_labels:
        budget = FLOP_BUDGETS[bl]
        # Group by size for interleaving
        by_size = defaultdict(list)
        for model in models:
            for size in sizes:
                steps = compute_steps(budget, model, size)
                if steps is not None:
                    by_size[size].append((bl, budget, model, size))

        # Interleave by size for balanced GPU memory
        max_per = max((len(v) for v in by_size.values()), default=0)
        for i in range(max_per):
            for size in sizes:
                if size in by_size and i < len(by_size[size]):
                    configs.append(by_size[size][i])

    return configs


def write_summary(results, out_path="isoflop/summary.csv"):
    fieldnames = [
        "budget_label", "budget", "model", "size", "n_embd", "n_layer",
        "N_params", "steps", "tokens", "flops", "lr", "dropout",
        "final_train_ce", "final_val_ce", "runtime_s", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r["budget_label"], r["model"], r["size"])):
            writer.writerow(r)
    print(f"\nSummary written to {out_path}")


def print_summary_table(results):
    by_budget = defaultdict(list)
    for r in results:
        by_budget[r["budget_label"]].append(r)

    print("\n" + "=" * 120)
    print("ISOFLOP RESULTS")
    print("=" * 120)
    header = (f"{'budget':<6} {'model':<15} {'size':<6} {'steps':>6} "
              f"{'tokens':>10} {'FLOPs':>10} {'lr':>8} "
              f"{'train_ce':>9} {'val_ce':>9} {'gap':>7} {'time':>7}")
    print(header)
    print("-" * 120)

    for bl in ALL_BUDGETS:
        if bl not in by_budget:
            continue
        runs = sorted(by_budget[bl], key=lambda r: (r["model"], r["size"]))
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
            print(f"{r['budget_label']:<6} {r['model']:<15} {r['size']:<6} {r['steps']:>6} "
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
                        choices=ALL_MODELS, help="Which models to run")
    parser.add_argument("--sizes", nargs="+", default=ALL_SIZES,
                        help="Which sizes to run")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel runs")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_only", action="store_true",
                        help="Rebuild summary from existing outputs")
    args = parser.parse_args()

    os.makedirs("isoflop", exist_ok=True)

    configs = enumerate_configs(args.budgets, args.models, args.sizes)
    total = len(configs)

    print(f"IsoFLOP experiment: {total} runs")
    print(f"Budgets: {args.budgets}")
    print(f"Models: {args.models}")
    print(f"Sizes: {args.sizes}")
    print(f"Step bounds: [{MIN_STEPS}, {MAX_STEPS}]")

    if args.dry_run:
        # --- Per-family coverage matrix ---
        print("\nPer-family coverage (steps or — if out of bounds):")
        for bl in args.budgets:
            budget = FLOP_BUDGETS[bl]
            print(f"\n  {bl} ({budget:.0e} FLOPs):")
            header = f"    {'model':<15}"
            for s in args.sizes:
                header += f" {s:>7}"
            header += "  | #sizes"
            print(header)
            print("    " + "-" * (15 + 8 * len(args.sizes) + 10))
            for m in args.models:
                row = f"    {m:<15}"
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

        print(f"\n--- Full command list ({total} runs) ---\n")
        for i, (bl, budget, model, size) in enumerate(configs):
            steps = compute_steps(budget, model, size)
            out_dir = f"isoflop/{bl}/{model}/{size}"
            cmd = build_command(model, size, steps, out_dir)
            lr = OPTIMAL_LR[(model, size)]
            dropout = dropout_for_model(model)
            print(f"[{i+1}/{total}] {bl} {model} {size} steps={steps} lr={lr:.0e} d={dropout}:")
            print(f"  {' '.join(cmd)}")
        return

    if args.summary_only:
        results = []
        for bl, budget, model, size in configs:
            steps = compute_steps(budget, model, size)
            n_embd, n_layer, N = MODEL_SIZES[size]
            lr = OPTIMAL_LR[(model, size)]
            dropout = dropout_for_model(model)
            out_dir = f"isoflop/{bl}/{model}/{size}"
            loss_path = os.path.join(out_dir, "loss.pkl")
            train_f, val_f = read_losses(loss_path)
            actual_flops = flop_multiplier(model) * N * steps * TOKENS_PER_STEP
            results.append({
                "budget_label": bl, "budget": budget, "model": model,
                "size": size, "n_embd": n_embd, "n_layer": n_layer,
                "N_params": N, "steps": steps,
                "tokens": steps * TOKENS_PER_STEP, "flops": actual_flops,
                "lr": lr, "dropout": dropout,
                "final_train_ce": train_f, "final_val_ce": val_f,
                "runtime_s": None,
                "status": "ok" if os.path.exists(loss_path) else "missing",
                "label": f"{bl} {model} {size}", "stderr_tail": "",
            })
        write_summary(results)
        print_summary_table(results)
        return

    results = []
    t_start = time.time()

    if args.parallel <= 1:
        for i, config in enumerate(configs):
            bl, budget, model, size = config
            steps = compute_steps(budget, model, size)
            lr = OPTIMAL_LR[(model, size)]
            print(f"\n[{i+1}/{total}] {bl} {model} {size} steps={steps} lr={lr:.0e}")
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
