"""
model_block_edit_one_pass.py — block diffusion with one-pass self-correction.

Training:
  - BD3-style blockwise masking
  - plus corruption of visible tokens

Sampling:
  - rewrite the current block every step
  - remask the least-confident current-block positions
"""

import torch
from torch.nn import functional as F

from backbone import DiffusionBackbone
from block_utils import (
    build_generation_prompt_mask,
    current_block_range,
    get_block_batch,
    prompt_start_block,
)


class Model(DiffusionBackbone):
    pass


# ------------------------------------------------------------------
# Training interface
# ------------------------------------------------------------------


def get_batch(split, cfg):
    xt, x0, token_mask = get_block_batch(split, cfg)

    corrupt_prob = cfg.get("corrupt_prob", 0.15)
    visible = ~token_mask
    edit_mask = visible & (torch.rand_like(visible.float()) < corrupt_prob)
    if edit_mask.any():
        rand_offset = torch.randint(1, cfg["vocab_size"], x0.shape, device=x0.device)
        wrong_tokens = (x0 + rand_offset) % cfg["vocab_size"]
        xt[edit_mask] = wrong_tokens[edit_mask]

    supervise_mask = token_mask | edit_mask
    return xt, x0, supervise_mask



def get_eval_batch(split, cfg, fixed_t_step):
    xt, x0, token_mask = get_block_batch(split, cfg, fixed_t_step=fixed_t_step)
    return xt, x0, token_mask



def compute_loss(model, batch, cfg):
    xt, x0, supervise_mask = batch
    _, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=supervise_mask)
    return loss



def compute_eval_loss(model, xt, x0, mask):
    _, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
    return loss


# ------------------------------------------------------------------
# Reverse diffusion sampling
# ------------------------------------------------------------------


@torch.no_grad()
def generate_from(
    model,
    x,
    prompt_mask,
    *,
    T,
    block_size,
    block_len,
    vocab_size,
    mask_token_id,
    survival_prob_scalar,
    decode,
):
    model.eval()

    num_blk = block_size // block_len
    first_block = prompt_start_block(prompt_mask, block_len)

    for block_idx in range(first_block, num_blk):
        block_start, block_end = current_block_range(block_idx, block_len)
        x_view = x[:, :block_end].clone()
        frozen_mask = build_generation_prompt_mask(prompt_mask, block_start, block_end)

        for t in reversed(range(1, T + 1)):
            logits, _ = model.forward_sample(x_view)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, vocab_size), 1
            ).view_as(x_view)
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            x_view = torch.where(frozen_mask, x_view, sampled)

            if t > 1:
                total_gen = (~frozen_mask).sum(dim=1)
                target_masked_next = torch.floor(
                    (1.0 - survival_prob_scalar(t - 1)) * total_gen.float()
                ).long().clamp_min(0)

                for b in range(x_view.size(0)):
                    k = int(target_masked_next[b].item())
                    if k <= 0:
                        continue
                    conf_b = sampled_conf[b].masked_fill(frozen_mask[b], float("inf"))
                    low_conf_idx = torch.topk(
                        conf_b, k=k, dim=0, largest=False
                    ).indices
                    x_view[b, low_conf_idx] = mask_token_id

        logits, _ = model.forward_sample(x_view)
        final_tokens = torch.argmax(logits, dim=-1)
        x_view = torch.where(frozen_mask, x_view, final_tokens)

        x[:, :block_end] = x_view

    return decode(x[0].tolist())
