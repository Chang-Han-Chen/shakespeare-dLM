import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F


# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------


def validate_block_len(block_size: int, block_len: int) -> None:
    if block_len <= 0:
        raise ValueError(f"block_len must be positive, got {block_len}")
    if block_size % block_len != 0:
        raise ValueError(
            f"block_len={block_len} must divide block_size={block_size}"
        )


def num_blocks(block_size: int, block_len: int) -> int:
    validate_block_len(block_size, block_len)
    return block_size // block_len


# ------------------------------------------------------------------
# BD3-LM masks
# ------------------------------------------------------------------


def make_bd3_train_mask(seq_len: int, block_len: int, device=None) -> torch.Tensor:
    """
    Full 2L x 2L BD3-LM training mask for x_t ⊕ x_0.

    This implements the paper/repo mask

        M_full = [[M_BD, M_OBC],
                  [   0,  M_BC]]

    with
      - M_BD  : block-diagonal attention within noisy blocks x_t
      - M_OBC : attention from noisy block b to clean blocks < b
      - M_BC  : block-causal attention on the clean x_0 stream

    Returned mask is boolean and shaped (1, 1, 2L, 2L), ready for SDPA.
    True means "allowed to attend".
    """
    validate_block_len(seq_len, block_len)
    idx = torch.arange(2 * seq_len, device=device)
    q_idx = idx[:, None]
    k_idx = idx[None, :]

    q_is_x0 = q_idx >= seq_len
    k_is_x0 = k_idx >= seq_len

    q_block = torch.where(q_is_x0, (q_idx - seq_len) // block_len, q_idx // block_len)
    k_block = torch.where(k_is_x0, (k_idx - seq_len) // block_len, k_idx // block_len)

    m_bd = (q_block == k_block) & (~q_is_x0) & (~k_is_x0)
    m_obc = (~q_is_x0) & k_is_x0 & (k_block < q_block)
    m_bc = q_is_x0 & k_is_x0 & (k_block <= q_block)

    mask = m_bd | m_obc | m_bc
    return mask[None, None, :, :]


def make_block_causal_mask(seq_len: int, block_len: int, device=None) -> torch.Tensor:
    """
    One-stream block-causal mask used at sampling time.

    Token i may attend to token j iff block(j) <= block(i), i.e.
    previous blocks are fully visible and the current block is
    bidirectional within itself.
    """
    validate_block_len(seq_len, block_len)
    pos = torch.arange(seq_len, device=device)
    q_block = pos[:, None] // block_len
    k_block = pos[None, :] // block_len
    mask = k_block <= q_block
    return mask[None, None, :, :]


# ------------------------------------------------------------------
# Batch construction
# ------------------------------------------------------------------


def sample_data_chunk(split: str, cfg) -> torch.Tensor:
    data_split = cfg["train_data"] if split == "train" else cfg["val_data"]
    B, L = cfg["batch_size"], cfg["block_size"]
    idx = torch.randint(len(data_split) - L, (B,))
    return torch.stack([data_split[i : i + L] for i in idx])


def sample_block_timesteps(
    batch_size: int,
    block_size: int,
    block_len: int,
    T: int,
    device,
    fixed_t_step: Optional[int] = None,
) -> torch.Tensor:
    nblk = num_blocks(block_size, block_len)
    if fixed_t_step is None:
        t_blocks = torch.randint(1, T + 1, (batch_size, nblk), device=device)
    else:
        fixed_t_step = int(max(1, min(T, fixed_t_step)))
        t_blocks = torch.full((batch_size, nblk), fixed_t_step, device=device, dtype=torch.long)
    return t_blocks


def expand_block_values(values: torch.Tensor, block_len: int, *, seq_len: Optional[int] = None) -> torch.Tensor:
    out = values.repeat_interleave(block_len, dim=1)
    if seq_len is not None:
        out = out[:, :seq_len]
    return out


def make_block_noisy_batch(
    x0: torch.Tensor,
    cfg,
    fixed_t_step: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BD3-style corruption for a simplified masked setup.

    We sample one discrete timestep per block and repeat it across all
    tokens in that block, matching the blockwise t-sampling pattern of
    the BD3-LM repo.
    """
    B, L = x0.shape
    device = x0.device
    block_len = cfg["block_len"]

    t_blocks = sample_block_timesteps(
        B,
        L,
        block_len,
        cfg["T"],
        device=device,
        fixed_t_step=fixed_t_step,
    )
    a_blocks = cfg["survival_prob_tensor"](t_blocks)
    a_tokens = expand_block_values(a_blocks, block_len, seq_len=L)

    token_mask = torch.rand(B, L, device=device) > a_tokens
    xt = x0.clone()
    xt[token_mask] = cfg["mask_token_id"]
    return xt, x0, token_mask


def get_block_batch(split: str, cfg, fixed_t_step: Optional[int] = None):
    x0 = sample_data_chunk(split, cfg)
    xt, x0, token_mask = make_block_noisy_batch(x0, cfg, fixed_t_step=fixed_t_step)
    dev = cfg["device"]
    return xt.to(dev), x0.to(dev), token_mask.to(dev)


# ------------------------------------------------------------------
# Loss / sampling helpers
# ------------------------------------------------------------------


def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor, supervise_mask: torch.Tensor) -> torch.Tensor:
    B, Tseq, V = logits.shape
    per_token_loss = F.cross_entropy(
        logits.reshape(B * Tseq, V),
        targets.reshape(B * Tseq),
        reduction="none",
    ).reshape(B, Tseq)
    supervise_f = supervise_mask.float()
    denom = supervise_f.sum().clamp_min(1.0)
    return (per_token_loss * supervise_f).sum() / denom


@torch.no_grad()
def sample_tokens_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.size(0), logits.size(1))


# ------------------------------------------------------------------
# Blockwise generation helpers
# ------------------------------------------------------------------


def current_block_range(block_idx: int, block_len: int) -> Tuple[int, int]:
    start = block_idx * block_len
    end = start + block_len
    return start, end


def prompt_start_block(prompt_mask: torch.Tensor, block_len: int) -> int:
    """
    First block that is not completely inside the initial prompt.

    This supports both aligned prompts and prompts that end in the middle
    of a block.
    """
    prompt_len = int(prompt_mask[0].sum().item())
    return prompt_len // block_len


def build_generation_prompt_mask(
    full_prompt_mask: torch.Tensor,
    block_start: int,
    block_end: int,
) -> torch.Tensor:
    """
    Prompt mask for one block-diffusion stage.

    All blocks before the current one are frozen and treated like prompt.
    Inside the current block, we preserve the original prompt mask so
    partially-prompted blocks work naturally.
    """
    out = torch.ones(
        full_prompt_mask.size(0),
        block_end,
        dtype=torch.bool,
        device=full_prompt_mask.device,
    )
    out[:, block_start:block_end] = full_prompt_mask[:, block_start:block_end]
    return out


# ------------------------------------------------------------------
# Agreement checks
# ------------------------------------------------------------------


def block_causal_equals_causal_when_block_len_is_one(seq_len: int, device=None) -> bool:
    mask = make_block_causal_mask(seq_len, 1, device=device)[0, 0]
    tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return bool(torch.equal(mask, tri))


def bd3_train_mask_special_cases_ok(seq_len: int, block_len: int, device=None) -> bool:
    mask = make_bd3_train_mask(seq_len, block_len, device=device)[0, 0]
    xt_to_xt = mask[:seq_len, :seq_len]
    xt_to_x0 = mask[:seq_len, seq_len:]
    x0_to_xt = mask[seq_len:, :seq_len]
    x0_to_x0 = mask[seq_len:, seq_len:]

    if block_len == seq_len:
        cond_1 = bool(xt_to_xt.all())
        cond_2 = bool((~xt_to_x0).all())
        cond_3 = bool((~x0_to_xt).all())
        cond_4 = bool(x0_to_x0.all())
        return cond_1 and cond_2 and cond_3 and cond_4

    if block_len == 1:
        eye = torch.eye(seq_len, dtype=torch.bool, device=device)
        tri = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        strict_lower = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-1)
        return (
            bool(torch.equal(xt_to_xt, eye))
            and bool(torch.equal(xt_to_x0, strict_lower))
            and bool((~x0_to_xt).all())
            and bool(torch.equal(x0_to_x0, tri))
        )

    return True
