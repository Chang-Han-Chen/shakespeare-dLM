"""
Plot training curves from pickled loss logs produced by train.py.

Usage:
    python plot_losses.py loss_remasked.pkl
    python plot_losses.py loss_remasked.pkl loss_mdlm.pkl   # overlay runs
    python plot_losses.py loss_remasked.pkl -o curves.png    # save to file
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def load_log(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_losses(*paths, out_path=None):
    """
    Plot train loss, val loss, and GPT2-CE from one or more loss-log pickles.

    Each pickle is a dict with keys:
        "train":   [(step, loss), ...]
        "val":     [(step, loss), ...]
        "gpt2_ce": [(step, ce),  ...]
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_train, ax_val, ax_gpt2 = axes

    for path in paths:
        label = Path(path).stem
        log = load_log(path)

        if log["train"]:
            steps, vals = zip(*log["train"])
            ax_train.plot(steps, vals, marker="o", markersize=3, label=label)

        if log["val"]:
            steps, vals = zip(*log["val"])
            ax_val.plot(steps, vals, marker="o", markersize=3, label=label)

        if log.get("gpt2_ce"):
            steps, vals = zip(*log["gpt2_ce"])
            ax_gpt2.plot(steps, vals, marker="o", markersize=3, label=label)

    ax_train.set(title="Train CE (fixed-t)", xlabel="step", ylabel="CE")
    ax_val.set(title="Val CE (fixed-t)", xlabel="step", ylabel="CE")
    ax_gpt2.set(title="GPT2-large CE on samples", xlabel="step", ylabel="CE (BPE)")

    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from train.py logs")
    parser.add_argument("logs", nargs="+", help="Path(s) to .pkl loss log files")
    parser.add_argument("-o", "--output", default=None, help="Save plot to file instead of showing")
    args = parser.parse_args()

    plot_losses(*args.logs, out_path=args.output)
