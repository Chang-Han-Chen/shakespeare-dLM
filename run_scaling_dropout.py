"""
run_scaling_dropout.py — Re-run scaling experiments with dropout=0.2.

Identical to run_scaling.py but with:
  - dropout=0.2 (up from 0.1)
  - results saved to scaling_dropout02/ (not scaling/)
  - checkpoints saved for later sample generation

Uses the same optimal LRs from the original sweep (dropout=0.1).
If you suspect the LRs need re-tuning, re-run sweep.py with --dropout 0.2.

Usage:
    python run_scaling_dropout.py                          # all 20 runs
    python run_scaling_dropout.py --parallel 4             # 4 at a time
    python run_scaling_dropout.py --models ar remasked     # subset
    python run_scaling_dropout.py --sizes 1M 3M            # subset of sizes
    python run_scaling_dropout.py --dry_run
    python run_scaling_dropout.py --summary_only
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
# Config
# ---------------------------------------------------------------

DROPOUT = 0.2
OUT_ROOT = "scaling_dropout02"

MODEL_SIZES = {
    "0.1M": (64,  2),
    "0.3M": (96,  3),
    "1M":   (128, 5),
    "3M":   (256, 4),
}

# Best LR per (model, size) — carried over from dropout=0.1 sweep
OPTIMAL_LR = {
    ("remasked",      "0.1M"): 1e-2,
    ("remasked",      "0.3M"): 3e-3,
    ("remasked",      "1M"):   3e-3,
    ("remasked",      "3M"):   3e-3,

    ("mdlm",          "0.1M"): 1e-2,
    ("mdlm",          "0.3M"): 3e-3,
    ("mdlm",          "1M"):   3e-3,
    ("mdlm",          "3M"):   3e-3,

    ("edit_one_pass",  "0.1M"): 1e-2,
    ("edit_one_pass",  "0.3M"): 1e-2,
    ("edit_one_pass",  "1M"):   1e-2,
    ("edit_one_pass",  "3M"):   3e-3,

    ("edit_two_pass",  "0.1M"): 1e-2,
    ("edit_two_pass",  "0.3M"): 3e-3,
    ("edit_two_pass",  "1M"):   3e-3,
    ("edit_two_pass",  "3M"):   3e-3,

    ("ar",            "0.1M"): 1e-2,
    ("ar",            "0.3M"): 1e-2,
    ("ar",            "1M"):   1e-2,
    ("ar",            "3M"):   3e-3,
}

ALL_MODELS = ["remasked", "mdlm", "edit_one_pass", "edit_two_pass", "ar"]
ALL_SIZES = list(MODEL_SIZES.keys())

# Training settings (same as original scaling runs)
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

def build_command(model, size, out_dir):
    n_embd, n_layer = MODEL_SIZES[size]
    lr = OPTIMAL_LR[(model, size)]
    min_lr = lr / 10.0

    loss_path = os.path.join(out_dir, "loss.pkl")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")

    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--n_embd", str(n_embd),
        "--n_layer", str(n_layer),
        "--n_head", "4",
        "--dropout", str(DROPOUT),
        "--learning_rate", str(lr),
        "--min_lr", str(min_lr),
        "--batch_size", str(BATCH_SIZE),
        "--max_iters", str(MAX_ITERS),
        "--eval_interval", str(EVAL_INTERVAL),
        "--eval_iters", str(EVAL_ITERS),
        "--warmup_iters", str(WARMUP_ITERS),
        "--gpt2_eval_interval", str(GPT2_EVAL_INTERVAL),
        "--gpt2_eval_samples", str(GPT2_EVAL_SAMPLES),
        "--sample_interval", "0",
        "--save_interval", str(SAVE_INTERVAL),
        "--num_final_samples", "5",
        "--loss_log_path", loss_path,
        "--checkpoint_path", ckpt_path,
    ]
    return cmd


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
    model, size = config
    n_embd, n_layer = MODEL_SIZES[size]
    lr = OPTIMAL_LR[(model, size)]
    out_dir = f"{OUT_ROOT}/{model}/{size}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = build_command(model, size, out_dir)
    label = f"{model} {size} lr={lr:.0e} dropout={DROPOUT}"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200,
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
    train_final, val_final, gpt2_final = read_losses(loss_path)

    return {
        "model": model,
        "size": size,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "lr": lr,
        "dropout": DROPOUT,
        "final_train_ce": train_final,
        "final_val_ce": val_final,
        "final_gpt2_ce": gpt2_final,
        "runtime_s": runtime,
        "status": status,
        "label": label,
        "stderr_tail": stderr_tail,
    }


def write_summary(results, out_path=None):
    if out_path is None:
        out_path = f"{OUT_ROOT}/summary.csv"
    fieldnames = [
        "model", "size", "n_embd", "n_layer", "lr", "dropout",
        "final_train_ce", "final_val_ce", "final_gpt2_ce",
        "runtime_s", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda r: (r["model"], r["size"])):
            writer.writerow(r)
    print(f"\nSummary written to {out_path}")


def print_summary_table(results):
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r)

    print("\n" + "=" * 110)
    print(f"SCALING RESULTS (4000 steps, bs=128, dropout={DROPOUT}, optimal LR per config)")
    print("=" * 110)
    header = (f"{'model':<15} {'size':<6} {'lr':<8} "
              f"{'train_ce':>9} {'val_ce':>9} {'gpt2_ce':>9} {'time':>7} {'status':<6}")
    print(header)
    print("-" * 110)

    for model in ALL_MODELS:
        if model not in by_model:
            continue
        runs = sorted(by_model[model], key=lambda r: list(MODEL_SIZES.keys()).index(r["size"]))
        for r in runs:
            train_ce = f"{r['final_train_ce']:.4f}" if r["final_train_ce"] is not None else "N/A"
            val_ce = f"{r['final_val_ce']:.4f}" if r["final_val_ce"] is not None else "N/A"
            gpt2_ce = f"{r['final_gpt2_ce']:.4f}" if r["final_gpt2_ce"] is not None else "N/A"
            runtime = f"{r['runtime_s']:.0f}s" if r["runtime_s"] is not None else "N/A"
            print(f"{r['model']:<15} {r['size']:<6} {r['lr']:<8.0e} "
                  f"{train_ce:>9} {val_ce:>9} {gpt2_ce:>9} {runtime:>7} {r['status']:<6}")
        print()


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=f"Scaling runs with dropout={DROPOUT} (results in {OUT_ROOT}/)"
    )
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

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Interleave by size for balanced GPU memory
    configs_by_size = defaultdict(list)
    for model in args.models:
        for size in args.sizes:
            if (model, size) in OPTIMAL_LR:
                configs_by_size[size].append((model, size))

    configs = []
    max_per_size = max(len(v) for v in configs_by_size.values()) if configs_by_size else 0
    for i in range(max_per_size):
        for size in args.sizes:
            if i < len(configs_by_size[size]):
                configs.append(configs_by_size[size][i])

    total = len(configs)
    print(f"Scaling runs: {total} configs ({len(args.models)} models × {len(args.sizes)} sizes)")
    print(f"Dropout: {DROPOUT}")
    print(f"Output: {OUT_ROOT}/")

    if args.dry_run:
        for i, (model, size) in enumerate(configs):
            out_dir = f"{OUT_ROOT}/{model}/{size}"
            cmd = build_command(model, size, out_dir)
            lr = OPTIMAL_LR[(model, size)]
            print(f"[{i+1}/{total}] {model} {size} lr={lr:.0e}:")
            print(f"  {' '.join(cmd)}")
        return

    if args.summary_only:
        results = []
        for model, size in configs:
            n_embd, n_layer = MODEL_SIZES[size]
            lr = OPTIMAL_LR[(model, size)]
            out_dir = f"{OUT_ROOT}/{model}/{size}"
            loss_path = os.path.join(out_dir, "loss.pkl")
            train_f, val_f, gpt2_f = read_losses(loss_path)
            results.append({
                "model": model, "size": size, "n_embd": n_embd,
                "n_layer": n_layer, "lr": lr, "dropout": DROPOUT,
                "final_train_ce": train_f, "final_val_ce": val_f,
                "final_gpt2_ce": gpt2_f,
                "runtime_s": None, "status": "ok" if os.path.exists(loss_path) else "missing",
                "label": f"{model} {size}", "stderr_tail": "",
            })
        write_summary(results)
        print_summary_table(results)
        return

    results = []
    t_start = time.time()

    if args.parallel <= 1:
        for i, config in enumerate(configs):
            model, size = config
            lr = OPTIMAL_LR[(model, size)]
            print(f"\n[{i+1}/{total}] {model} {size} lr={lr:.0e} dropout={DROPOUT}")
            result = run_one(config)
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
                completed += 1
                result = future.result()
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
