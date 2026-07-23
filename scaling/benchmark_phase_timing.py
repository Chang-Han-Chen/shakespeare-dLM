"""Single-process H100 timing of AR and block-diffusion optimizer steps."""

from __future__ import annotations

import csv
import json
import statistics
import time

import matplotlib.pyplot as plt
import torch

from config import BATCH_SIZE, MODEL_SPECS, SEED, SEQ_LEN
from curriculum_config import FIGURES_DIR, RESULTS_DIR
from curriculum_train import train_ar_step, train_bd_step
from data import ShakespeareData
from model import BlockDiffusionTransformer
from train import optimizer_for, set_seed


DEVICE = torch.device("cuda")
PEAK_LR = 1e-3
WARMUP_STEPS = 20
TIMED_STEPS = 50
REPEATS = 3


def time_phase(spec, phase, dataset):
    set_seed(SEED)
    model = BlockDiffusionTransformer(spec).to(DEVICE).train()
    optimizer = optimizer_for(model, PEAK_LR)
    step_fn = train_ar_step if phase == "ar" else train_bd_step
    for _ in range(WARMUP_STEPS):
        step_fn(model, optimizer, dataset, DEVICE)
    torch.cuda.synchronize(DEVICE)

    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        for _ in range(TIMED_STEPS):
            step_fn(model, optimizer, dataset, DEVICE)
        torch.cuda.synchronize(DEVICE)
        samples.append((time.perf_counter() - started) / TIMED_STEPS)
    return samples


def save_plot(rows):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    n = [row["n_params"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    time_ax, ratio_ax = axes
    time_ax.plot(
        n,
        [1e3 * row["ar_seconds_per_step"] for row in rows],
        marker="o",
        label="AR measured",
        color="#35618d",
    )
    time_ax.plot(
        n,
        [1e3 * row["bd_seconds_per_step"] for row in rows],
        marker="s",
        label="BD measured",
        color="#d17c37",
    )
    time_ax.set_xscale("log")
    time_ax.set_yscale("log")
    time_ax.set_xlabel("non-embedding parameters N")
    time_ax.set_ylabel("wall time per optimizer step (ms)")
    time_ax.set_title("Single-process H100 step time")
    time_ax.legend(frameon=False)

    ratio_ax.plot(
        n,
        [row["theoretical_flop_ratio_ar_to_bd"] for row in rows],
        marker="s",
        ls="--",
        color="#777777",
        label="theoretical FLOP ratio",
    )
    ratio_ax.plot(
        n,
        [row["measured_wall_ratio_ar_to_bd"] for row in rows],
        marker="o",
        color="#5a9b55",
        label="measured wall-time ratio",
    )
    ratio_ax.set_xscale("log")
    ratio_ax.set_xlabel("non-embedding parameters N")
    ratio_ax.set_ylabel("AR / BD")
    ratio_ax.set_title("AR cost relative to block diffusion")
    ratio_ax.legend(frameon=False)

    fig.suptitle(
        "AR versus block-diffusion training cost — batch 128, sequence 256",
        fontsize=14,
    )
    fig.tight_layout()
    png = FIGURES_DIR / "phase_timing.png"
    pdf = FIGURES_DIR / "phase_timing.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the phase timing benchmark")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dataset = ShakespeareData.load(DEVICE)
    rows = []
    for spec in MODEL_SPECS:
        ar_samples = time_phase(spec, "ar", dataset)
        bd_samples = time_phase(spec, "bd", dataset)
        ar_median = statistics.median(ar_samples)
        bd_median = statistics.median(bd_samples)
        theoretical = (
            spec.autoregressive_training_flops_per_clean_token
            / spec.training_flops_per_clean_token
        )
        row = {
            "size": spec.label,
            "n_params": spec.n_params,
            "ar_training_flops_per_clean_token": (
                spec.autoregressive_training_flops_per_clean_token
            ),
            "bd_training_flops_per_clean_token": (
                spec.training_flops_per_clean_token
            ),
            "theoretical_flop_ratio_ar_to_bd": theoretical,
            "ar_seconds_per_step": ar_median,
            "bd_seconds_per_step": bd_median,
            "measured_wall_ratio_ar_to_bd": ar_median / bd_median,
            "measured_to_theoretical_ratio": (ar_median / bd_median) / theoretical,
            "ar_seconds_per_step_samples": ar_samples,
            "bd_seconds_per_step_samples": bd_samples,
            "warmup_steps": WARMUP_STEPS,
            "timed_steps_per_repeat": TIMED_STEPS,
            "repeats": REPEATS,
            "batch_size": BATCH_SIZE,
            "sequence_length": SEQ_LEN,
            "device": torch.cuda.get_device_name(DEVICE),
        }
        rows.append(row)
        print(
            f"{spec.label:>6} AR={1e3 * ar_median:7.3f}ms "
            f"BD={1e3 * bd_median:7.3f}ms "
            f"wall_ratio={ar_median / bd_median:.3f} "
            f"flop_ratio={theoretical:.3f}",
            flush=True,
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "phase_timing.json"
    json_path.write_text(
        json.dumps(
            {
                "method": (
                    "single process; median of three synchronized 50-step repeats "
                    "after 20 warmup steps; includes data, loss, backward, and AdamW"
                ),
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    csv_path = RESULTS_DIR / "phase_timing.csv"
    csv_fields = [
        key
        for key in rows[0]
        if key not in {"ar_seconds_per_step_samples", "bd_seconds_per_step_samples"}
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    png, pdf = save_plot(rows)
    print(f"saved {json_path}")
    print(f"saved {csv_path}")
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
