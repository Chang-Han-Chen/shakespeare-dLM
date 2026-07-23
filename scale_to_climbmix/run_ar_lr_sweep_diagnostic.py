"""Run the non-center AR-LR branches for C=3e14, N=0.5M."""

from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BUDGET = 3e14
SIZE = "0.5M"
BD_LR = 2.7e-3
P_VALUES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
AR_TRUNKS = {
    9e-4: ROOT
    / "results_diagnostics"
    / "ar_lr_sweep_3e14_0p5M"
    / "trunks"
    / "ar_lr_9e-4",
    8.1e-3: ROOT
    / "results_diagnostics"
    / "ar_lr_sweep_3e14_0p5M"
    / "trunks"
    / "ar_lr_8p1e-3",
}
OUTPUT = ROOT / "results_diagnostics" / "ar_lr_sweep_3e14_0p5M" / "runs"


def slug(value):
    return f"{value:.1e}".replace(".", "p").replace("+", "")


def run_one(ar_lr, p_ar):
    destination = OUTPUT / f"p_ar_{p_ar:.1f}".replace(".", "p") / f"ar_lr_{slug(ar_lr)}"
    result = destination / "result.json"
    if result.exists() and json.loads(result.read_text()).get("status") == "complete":
        return ar_lr, p_ar, json.loads(result.read_text())["val_nelbo"], "cached"
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "curriculum_train.py"),
        "--budget",
        str(BUDGET),
        "--size",
        SIZE,
        "--p-ar",
        str(p_ar),
        "--trunk-dir",
        str(AR_TRUNKS[ar_lr]),
        "--bd-lr",
        str(BD_LR),
        "--output",
        str(result),
    ]
    with (destination / "train.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    payload = json.loads(result.read_text())
    return ar_lr, p_ar, payload["val_nelbo"], "complete"


def main():
    tasks = [(lr, p) for lr in AR_TRUNKS for p in P_VALUES]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_one, *task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            ar_lr, p_ar, loss, status = future.result()
            print(
                f"[{index:02d}/{len(tasks)}] p={p_ar:.1f} "
                f"AR_LR={ar_lr:.1e} val={loss:.5f} {status}",
                flush=True,
            )


if __name__ == "__main__":
    main()
