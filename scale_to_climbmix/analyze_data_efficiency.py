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


def load_seed_replicates(
    best_full_bd: dict,
    curricula: list[dict],
) -> list[dict]:
    plan = json.loads((RESULTS / "plan.json").read_text())
    original_curriculum = next(
        (
            row
            for row in curricula
            if row["ar_weight_decay"] == 0.1
        ),
        None,
    )
    rows = []
    if original_curriculum is not None:
        rows.append(
            {
                "seed": plan["seed"],
                "full_bd_val_nelbo": best_full_bd["val_nelbo"],
                "curriculum_val_nelbo": original_curriculum["val_nelbo"],
                "full_bd_world_size": best_full_bd["world_size"],
                "curriculum_world_size": original_curriculum["world_size"],
                "full_bd_result_path": best_full_bd["result_path"],
                "curriculum_result_path": (
                    original_curriculum["result_path"]
                ),
                "full_bd_wandb_url": best_full_bd["wandb_url"],
                "curriculum_wandb_url": original_curriculum["wandb_url"],
            }
        )

    for seed_directory in sorted((RESULTS / "replicates").glob("seed_*")):
        full_bd_path = seed_directory / "full_bd" / "endpoint" / "result.json"
        curriculum_path = (
            seed_directory / "curriculum_wd0p1" / "result.json"
        )
        if not full_bd_path.exists() or not curriculum_path.exists():
            continue
        full_bd = json.loads(full_bd_path.read_text())
        curriculum = json.loads(curriculum_path.read_text())
        if (
            full_bd.get("status") != "complete"
            or curriculum.get("status") != "complete"
        ):
            continue
        seed = int(seed_directory.name.removeprefix("seed_"))
        if full_bd["seed"] != seed or curriculum["seed"] != seed:
            raise RuntimeError(f"Seed metadata mismatch in {seed_directory}")
        if full_bd["total_steps"] != best_full_bd["total_steps"]:
            raise RuntimeError(
                f"Full-BD replicate seed {seed} has the wrong horizon"
            )
        if curriculum["total_steps"] != best_full_bd["total_steps"]:
            raise RuntimeError(
                f"Curriculum replicate seed {seed} has the wrong horizon"
            )
        if (
            curriculum["p_ar"] != 0.4
            or curriculum["ar_weight_decay"] != 0.1
            or curriculum["bd_weight_decay"] != 0.1
        ):
            raise RuntimeError(
                f"Curriculum replicate seed {seed} is not the untuned setting"
            )
        rows.append(
            {
                "seed": seed,
                "full_bd_val_nelbo": full_bd["val_nelbo"],
                "curriculum_val_nelbo": curriculum["val_nelbo"],
                "full_bd_world_size": full_bd["world_size"],
                "curriculum_world_size": curriculum["world_size"],
                "full_bd_result_path": str(full_bd_path.relative_to(ROOT)),
                "curriculum_result_path": str(
                    curriculum_path.relative_to(ROOT)
                ),
                "full_bd_wandb_url": full_bd["wandb_url"],
                "curriculum_wandb_url": curriculum["wandb_url"],
            }
        )

    for row in rows:
        row["absolute_gap_curriculum_minus_full_bd"] = (
            row["curriculum_val_nelbo"] - row["full_bd_val_nelbo"]
        )
        row["relative_gap_curriculum_minus_full_bd"] = (
            row["curriculum_val_nelbo"] / row["full_bd_val_nelbo"] - 1.0
        )
    rows.sort(key=lambda row: row["seed"])
    return rows


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
    seed_replicates = load_seed_replicates(best, curricula)

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
    if seed_replicates:
        mean_full_bd = sum(
            row["full_bd_val_nelbo"] for row in seed_replicates
        ) / len(seed_replicates)
        mean_curriculum = sum(
            row["curriculum_val_nelbo"] for row in seed_replicates
        ) / len(seed_replicates)
        mean_absolute_gap = sum(
            row["absolute_gap_curriculum_minus_full_bd"]
            for row in seed_replicates
        ) / len(seed_replicates)
        mean_relative_gap = sum(
            row["relative_gap_curriculum_minus_full_bd"]
            for row in seed_replicates
        ) / len(seed_replicates)
        summary["untuned_curriculum_seed_replicates"] = {
            "ar_weight_decay": 0.1,
            "bd_weight_decay": 0.1,
            "seeds": [row["seed"] for row in seed_replicates],
            "n_seeds": len(seed_replicates),
            "mean_full_bd_val_nelbo": mean_full_bd,
            "mean_curriculum_val_nelbo": mean_curriculum,
            "mean_absolute_gap_curriculum_minus_full_bd": (
                mean_absolute_gap
            ),
            "mean_relative_gap_curriculum_minus_full_bd": (
                mean_relative_gap
            ),
            "n_curriculum_wins": sum(
                row["absolute_gap_curriculum_minus_full_bd"] < 0.0
                for row in seed_replicates
            ),
            "paired_gap_sign_consistent": (
                all(
                    row["absolute_gap_curriculum_minus_full_bd"] > 0.0
                    for row in seed_replicates
                )
                or all(
                    row["absolute_gap_curriculum_minus_full_bd"] < 0.0
                    for row in seed_replicates
                )
            ),
            "rows": seed_replicates,
        }

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "full_bd_endpoints.csv", full_bd)
    write_csv(RESULTS / "curriculum_weight_decay.csv", curricula)
    write_csv(RESULTS / "untuned_seed_replicates.csv", seed_replicates)
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

    if seed_replicates:
        repeat_figure, repeat_axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
        colors = ("#2878b5", "#e24a33", "#59a14f", "#b279a2")
        for index, row in enumerate(seed_replicates):
            repeat_axes[0].plot(
                (0, 1),
                (
                    row["full_bd_val_nelbo"],
                    row["curriculum_val_nelbo"],
                ),
                "o-",
                color=colors[index % len(colors)],
                label=f"seed {row['seed']}",
            )
        repeat_axes[0].set_xticks((0, 1), ("full BD", r"$p_{\mathrm{AR}}=0.4$"))
        repeat_axes[0].set_ylabel("validation block-diffusion NELBO")
        repeat_axes[0].set_title("Untuned WD=0.1 paired repeats")
        repeat_axes[0].legend(frameon=False)

        repeat_axes[1].axhline(0.0, color="black", linewidth=1.0)
        repeat_axes[1].bar(
            [str(row["seed"]) for row in seed_replicates],
            [
                100.0 * row["relative_gap_curriculum_minus_full_bd"]
                for row in seed_replicates
            ],
            color=[
                colors[index % len(colors)]
                for index in range(len(seed_replicates))
            ],
        )
        repeat_axes[1].set_xlabel("seed")
        repeat_axes[1].set_ylabel("curriculum gap to full BD (%)")
        repeat_axes[1].set_title("Paired gap changes sign")
        for axis in repeat_axes:
            axis.grid(alpha=0.2)
        repeat_figure.tight_layout()
        for extension in ("png", "pdf"):
            repeat_figure.savefig(
                FIGURES / f"data_efficiency_seed_replicates.{extension}",
                bbox_inches="tight",
            )
        plt.close(repeat_figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
