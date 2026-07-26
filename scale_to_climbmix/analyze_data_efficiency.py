"""Summarize the 25M-token, 50M-parameter data-efficiency experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from config import ROOT


RESULTS = ROOT / "results_data_efficiency"
FIGURES = ROOT / "figures_data_efficiency"


def load_complete(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(RESULTS.glob(pattern)):
        row = json.loads(path.read_text())
        if row.get("status") != "complete":
            continue
        row["result_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if not isinstance(value, (dict, list))
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def compact(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        key: value
        for key, value in row.items()
        if key != "train_trace"
    }


def main() -> None:
    full_bd = load_complete("full_bd/endpoints/epoch_*/result.json")
    curricula = load_complete("curriculum/ar_wd_*/result.json")
    if not full_bd:
        raise RuntimeError("No complete full-BD decay endpoints")
    full_bd.sort(key=lambda row: row["total_horizon_epochs"])
    best = min(full_bd, key=lambda row: row["val_nelbo"])
    for row in curricula:
        if row["total_steps"] != best["total_steps"]:
            raise RuntimeError(
                f"Curriculum {row['result_path']} has {row['total_steps']} "
                f"steps; selected full BD has {best['total_steps']}"
            )
        row["absolute_gap_to_full_bd"] = row["val_nelbo"] - best["val_nelbo"]
        row["relative_gap_to_full_bd"] = (
            row["val_nelbo"] / best["val_nelbo"] - 1.0
        )
    curricula.sort(key=lambda row: row["ar_weight_decay"])
    best_curriculum = (
        min(curricula, key=lambda row: row["val_nelbo"])
        if curricula
        else None
    )

    summary = {
        "study": "25M_unique_tokens_50M_parameters",
        "selection_metric": "validation_block_diffusion_nelbo",
        "validation_frequency": "decayed_endpoints_only",
        "best_full_bd": compact(best),
        "best_curriculum": compact(best_curriculum),
        "curriculum_recovers_full_bd": (
            best_curriculum is not None
            and best_curriculum["val_nelbo"] <= best["val_nelbo"]
        ),
        "n_full_bd_endpoints": len(full_bd),
        "n_curriculum_weight_decays": len(curricula),
    }
    if best_curriculum is not None:
        summary["best_curriculum_absolute_gap"] = (
            best_curriculum["val_nelbo"] - best["val_nelbo"]
        )
        summary["best_curriculum_relative_gap"] = (
            best_curriculum["val_nelbo"] / best["val_nelbo"] - 1.0
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "full_bd_endpoints.csv", full_bd)
    write_csv(RESULTS / "curriculum_weight_decay.csv", curricula)
    (RESULTS / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))
    axes[0].plot(
        [row["total_horizon_epochs"] for row in full_bd],
        [row["val_nelbo"] for row in full_bd],
        "o-",
        color="#2878b5",
    )
    axes[0].scatter(
        [best["total_horizon_epochs"]],
        [best["val_nelbo"]],
        marker="*",
        s=140,
        color="#e24a33",
        edgecolor="black",
        linewidth=0.5,
        label="selected endpoint",
        zorder=3,
    )
    axes[0].set_xlabel("total token-exposure horizon (epochs)")
    axes[0].set_ylabel("validation block-diffusion NELBO")
    axes[0].set_title("Full BD: proportional decay endpoints")
    axes[0].legend(frameon=False)

    axes[1].axhline(
        best["val_nelbo"],
        color="#2878b5",
        linestyle="--",
        label="best full BD",
    )
    if curricula:
        curriculum_weight_decays = [
            row["ar_weight_decay"] for row in curricula
        ]
        axes[1].plot(
            curriculum_weight_decays,
            [row["val_nelbo"] for row in curricula],
            "o-",
            color="#e24a33",
            label=r"$p_{\mathrm{AR}}=0.4$",
        )
        axes[1].set_xscale("log", base=2)
        axes[1].set_xticks(curriculum_weight_decays)
        axes[1].set_xticklabels(
            [f"{weight_decay:g}" for weight_decay in curriculum_weight_decays]
        )
        axes[1].scatter(
            [best_curriculum["ar_weight_decay"]],
            [best_curriculum["val_nelbo"]],
            marker="*",
            s=140,
            color="#fbc15e",
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
    axes[1].set_xlabel("AR-phase weight decay")
    axes[1].set_ylabel("validation block-diffusion NELBO")
    axes[1].set_title("Fixed-horizon curriculum recovery")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(
            FIGURES / f"data_efficiency_25m_50m.{extension}",
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
