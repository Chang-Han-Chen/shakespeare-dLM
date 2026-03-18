import subprocess
import sys
from pathlib import Path

import torch

from backbone import DiffusionBackbone
from model_block_MDLM import Model as BlockMDLM


ROOT = Path(__file__).resolve().parent.parent


def run_repo_cmd(*args, timeout=120):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_single_block_forward_train_ignores_clean_stream():
    torch.manual_seed(0)
    model = DiffusionBackbone(
        vocab_size=10,
        n_embd=8,
        n_head=2,
        n_layer=1,
        head_dim=4,
        block_size=4,
        block_len=4,
        dropout=0.0,
    )
    model.eval()

    xt = torch.tensor([[1, 2, 3, 4]])
    x0_a = torch.tensor([[5, 6, 7, 8]])
    x0_b = torch.tensor([[8, 7, 6, 5]])

    logits_a, _ = model.forward_train(xt, x0_a)
    logits_b, _ = model.forward_train(xt, x0_b)

    assert torch.equal(logits_a, logits_b)


def test_block_train_cli_prints_sample_without_crashing(tmp_path):
    result = run_repo_cmd(
        "train.py",
        "--model",
        "block_mdlm",
        "--max_iters",
        "1",
        "--eval_interval",
        "1",
        "--eval_iters",
        "1",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--block_len",
        "4",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "2",
        "--num_final_samples",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--checkpoint_path",
        str(tmp_path / "block_ckpt.pt"),
        "--loss_log_path",
        str(tmp_path / "block_loss.pkl"),
    )

    assert result.returncode == 0, result.stderr
    assert "Generating sample..." in result.stdout
    assert "TypeError" not in result.stdout
    assert "TypeError" not in result.stderr


def test_train_cli_accepts_save_interval_zero(tmp_path):
    result = run_repo_cmd(
        "train.py",
        "--model",
        "mdlm",
        "--max_iters",
        "2",
        "--eval_interval",
        "0",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "0",
        "--num_final_samples",
        "0",
        "--sample_interval",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--checkpoint_path",
        str(tmp_path / "mdlm_ckpt.pt"),
        "--loss_log_path",
        str(tmp_path / "mdlm_loss.pkl"),
    )

    assert result.returncode == 0, result.stderr
    assert "ZeroDivisionError" not in result.stdout
    assert "ZeroDivisionError" not in result.stderr


def test_train_cli_can_warmstart_block_from_ar_checkpoint(tmp_path):
    ar_ckpt = tmp_path / "ar_ckpt.pt"
    ar_loss = tmp_path / "ar_loss.pkl"
    bd_ckpt = tmp_path / "bd_ckpt.pt"
    bd_loss = tmp_path / "bd_loss.pkl"

    ar_result = run_repo_cmd(
        "train.py",
        "--model",
        "ar",
        "--max_iters",
        "1",
        "--eval_interval",
        "0",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "0",
        "--num_final_samples",
        "0",
        "--sample_interval",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--checkpoint_path",
        str(ar_ckpt),
        "--loss_log_path",
        str(ar_loss),
    )

    assert ar_result.returncode == 0, ar_result.stderr
    assert ar_ckpt.exists()

    bd_result = run_repo_cmd(
        "train.py",
        "--model",
        "block_mdlm",
        "--max_iters",
        "1",
        "--eval_interval",
        "0",
        "--batch_size",
        "2",
        "--block_size",
        "16",
        "--block_len",
        "4",
        "--n_embd",
        "16",
        "--n_head",
        "4",
        "--n_layer",
        "1",
        "--save_interval",
        "0",
        "--num_final_samples",
        "0",
        "--sample_interval",
        "0",
        "--gpt2_eval_interval",
        "0",
        "--use_compile",
        "false",
        "--resume_from",
        str(ar_ckpt),
        "--checkpoint_path",
        str(bd_ckpt),
        "--loss_log_path",
        str(bd_loss),
    )

    assert bd_result.returncode == 0, bd_result.stderr
    assert "Loaded model weights from" in bd_result.stdout
    assert "optimizer_reset=true" in bd_result.stdout
    assert bd_ckpt.exists()


def test_generate_samples_auto_discovers_block_checkpoint(tmp_path):
    text = (ROOT / "data.txt").read_text(encoding="utf-8")
    vocab_size = len(["_"] + sorted(set(text)))

    model = BlockMDLM(
        vocab_size=vocab_size,
        n_embd=16,
        n_head=4,
        n_layer=1,
        head_dim=4,
        block_size=16,
        block_len=4,
        dropout=0.0,
    )

    ckpt_dir = tmp_path / "block_mdlm_bl4" / "0.1M"
    ckpt_dir.mkdir(parents=True)
    torch.save(
        {
            "iter": 1,
            "model_state_dict": model.state_dict(),
            "loss": 4.2,
            "args": {
                "n_embd": 16,
                "n_layer": 1,
                "n_head": 4,
                "block_len": 4,
            },
        },
        ckpt_dir / "ckpt.pt",
    )

    result = run_repo_cmd(
        "generate_samples.py",
        "--models",
        "block_mdlm",
        "--size",
        "0.1M",
        "--num_samples",
        "1",
        "--block_size",
        "16",
        "--prompt_len",
        "8",
        "--ckpt_root",
        str(tmp_path),
    )

    assert result.returncode == 0, result.stderr
    assert "MODEL: block_mdlm (0.1M bl=4)" in result.stdout
    assert "Continuation:" in result.stdout
