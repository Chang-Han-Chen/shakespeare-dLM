"""
model_block_remasked.py — block diffusion with iterative remasking.
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
    return get_block_batch(split, cfg)



def get_eval_batch(split, cfg, fixed_t_step):
    return get_block_batch(split, cfg, fixed_t_step=fixed_t_step)



def compute_loss(model, batch, cfg):
    xt, x0, mask = batch
    _, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
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
