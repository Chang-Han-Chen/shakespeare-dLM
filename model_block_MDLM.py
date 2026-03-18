"""
model_block_MDLM.py — block diffusion version of MDLM.

Training follows the BD3-LM pattern:
  - sample one timestep per block
  - corrupt all blocks independently
  - run a single dual-stream transformer pass on x_t ⊕ x_0

Sampling is block-autoregressive:
  - previous blocks are frozen
  - diffusion happens only inside the current block
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


def _progressive_unmask_step(x, prompt_mask, sampled, sampled_conf, t, *, mask_token_id, survival_prob_scalar):
    current_mask = (x == mask_token_id) & (~prompt_mask)
    if not current_mask.any():
        return x

    if t == 1:
        return torch.where(current_mask, sampled, x)

    total_gen = (~prompt_mask).sum(dim=1)
    current_masked = current_mask.sum(dim=1)
    target_masked_next = torch.floor(
        (1.0 - survival_prob_scalar(t - 1)) * total_gen.float()
    ).long()
    target_masked_next = torch.minimum(target_masked_next, current_masked)
    num_to_unmask = (current_masked - target_masked_next).clamp_min(0)

    for b in range(x.size(0)):
        k = int(num_to_unmask[b].item())
        if k <= 0:
            continue
        conf_b = sampled_conf[b].masked_fill(~current_mask[b], float("-inf"))
        chosen = torch.topk(conf_b, k=k, dim=0, largest=True).indices
        x[b, chosen] = sampled[b, chosen]

    return x


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

            x_view = _progressive_unmask_step(
                x_view,
                frozen_mask,
                sampled,
                sampled_conf,
                t,
                mask_token_id=mask_token_id,
                survival_prob_scalar=survival_prob_scalar,
            )

        if (x_view[:, block_start:block_end] == mask_token_id).any():
            logits, _ = model.forward_sample(x_view)
            final_tokens = torch.argmax(logits, dim=-1)
            x_view = torch.where(x_view == mask_token_id, final_tokens, x_view)

        x[:, :block_end] = x_view

    return decode(x[0].tolist())
