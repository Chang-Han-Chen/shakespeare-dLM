"""Compare the one-shot 1e19 run with the preregistered rolling forecasts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import ROOT


BUDGET = 1e19
RUN = (
    ROOT
    / "results_matched_p_ar_0p4/runs/1e19/176.0M/"
    "ar_9.0e-04_bd_3.0e-04_bs256_ddp4/result.json"
)
SCALING_SUMMARY = ROOT / "results_matched_p_ar_0p4/summary.json"
OUTPUT = ROOT / "results_matched_p_ar_0p4/one_shot_1e19_comparison.json"
FIGURES = ROOT / "figures_matched_p_ar_0p4"


def predict(law: dict, compute: float) -> float:
    return law["coefficient"] * compute ** law["exponent"]


def main() -> None:
    run = json.loads(RUN.read_text())
    if run.get("status") != "complete":
        raise RuntimeError("The one-shot run is not complete")
    scaling = json.loads(SCALING_SUMMARY.read_text())
    windows = ("rolling_highest3", "rolling_highest4", "rolling_highest5")
    forecasts = []
    for window in windows:
        laws = scaling["laws"][window]
        forecast = {
            "window": window,
            "n_opt": predict(laws["n_opt"], BUDGET),
            "d_opt": predict(laws["d_opt"], BUDGET),
            "loss_min": predict(laws["loss_min"], BUDGET),
        }
        forecast["actual_n_to_forecast"] = run["n_params"] / forecast["n_opt"]
        forecast["actual_d_to_forecast"] = (
            run["clean_tokens"] / forecast["d_opt"]
        )
        forecast["actual_minus_forecast_loss"] = (
            run["val_nelbo"] - forecast["loss_min"]
        )
        forecast["relative_loss_residual"] = (
            run["val_nelbo"] / forecast["loss_min"] - 1.0
        )
        forecasts.append(forecast)

    preferred = next(
        row for row in forecasts if row["window"] == "rolling_highest4"
    )
    result = {
        "status": "complete",
        "comparison": "one_shot_1e19_vs_pre_run_rolling_laws",
        "important_limitation": (
            "One model size tests the predicted loss level but cannot localize "
            "or validate the compute-optimal model-size vertex."
        ),
        "batch_size_changed_from_scaling_profiles": True,
        "scaling_profile_batch_size": 128,
        "one_shot_batch_size": run["batch_size"],
        "actual": {
            "budget": run["budget"],
            "n_params": run["n_params"],
            "clean_tokens": run["clean_tokens"],
            "val_nelbo": run["val_nelbo"],
            "val_masked_ce_t0.5": run["val_masked_ce_t0.5"],
            "realized_flops": run["realized_flops"],
            "realized_to_nominal_compute": run["realized_to_nominal_compute"],
            "accounted_h100_bf16_mfu": run[
                "accounted_h100_bf16_mfu"
            ],
            "duration_seconds": run["duration_seconds"],
            "wandb_url": run["wandb_url"],
            "result_path": str(RUN.relative_to(ROOT)),
        },
        "forecasts": forecasts,
        "preferred_rolling_highest4": preferred,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    labels = ["highest 3", "highest 4", "highest 5"]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(9.7, 4.0))
    axes[0].scatter(
        x,
        [row["loss_min"] for row in forecasts],
        color="#2878b5",
        s=55,
        label="forecast",
    )
    axes[0].axhline(
        run["val_nelbo"],
        color="#e24a33",
        linestyle="--",
        label="one-shot validation",
    )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("validation block-diffusion NELBO")
    axes[0].set_title(r"Loss forecast at $10^{19}$ FLOPs")
    axes[0].legend(frameon=False)

    width = 0.34
    axes[1].bar(
        x - width / 2,
        [row["actual_n_to_forecast"] for row in forecasts],
        width,
        label="parameters N",
        color="#2878b5",
    )
    axes[1].bar(
        x + width / 2,
        [row["actual_d_to_forecast"] for row in forecasts],
        width,
        label="clean tokens D",
        color="#e24a33",
    )
    axes[1].axhline(1.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("actual / forecast")
    axes[1].set_title("Chosen allocation versus rolling laws")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            FIGURES / f"one_shot_1e19_comparison.{extension}",
            bbox_inches="tight",
        )
    plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
