"""
model_block_edit_two_pass.py — block diffusion with a two-pass corrector.
"""

import torch
from torch.nn import functional as F

from backbone import DiffusionBackbone
from block_utils import (
    build_generation_prompt_mask,
    current_block_range,
    get_block_batch,
    masked_ce_loss,
    prompt_start_block,
    sample_tokens_from_logits,
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
    lambda_corr = cfg.get("lambda_corr", 1.0)

    logits_1, _ = model.forward_train(xt, x0)
    loss_denoise = masked_ce_loss(logits_1, x0, mask)

    with torch.no_grad():
        sampled = sample_tokens_from_logits(logits_1.float())
        draft = xt.clone()
        draft[mask] = sampled[mask]
        corr_mask = draft != x0

    if corr_mask.any():
        logits_2, _ = model.forward_train(draft, x0)
        loss_corr = masked_ce_loss(logits_2, x0, corr_mask)
    else:
        loss_corr = torch.zeros((), device=xt.device)

    return loss_denoise + lambda_corr * loss_corr



def compute_eval_loss(model, xt, x0, mask):
    _, loss = model.forward_train(xt, x0, targets=x0, supervise_mask=mask)
    return loss


# ------------------------------------------------------------------
# Reverse diffusion sampling
# ------------------------------------------------------------------


@torch.no_grad()
def corrective_sweep(model, x, prompt_mask, mask_token_id):
    logits_fill, _ = model.forward_sample(x)
    sampled_fill = sample_tokens_from_logits(logits_fill.float())

    draft = x.clone()
    masked_now = draft == mask_token_id
    draft[masked_now] = sampled_fill[masked_now]

    logits_corr, _ = model.forward_sample(draft)
    corrected = torch.argmax(logits_corr, dim=-1)

    visible_non_prompt = (~prompt_mask) & (~masked_now)
    out = x.clone()
    out[visible_non_prompt] = corrected[visible_non_prompt]
    return out


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
    corrector_steps=1,
    num_correction_events=5,
    correct_at_end=True,
):
    model.eval()

    num_blk = block_size // block_len
    first_block = prompt_start_block(prompt_mask, block_len)
    num_correction_events = max(1, int(num_correction_events))
    correct_every = max(1, T // num_correction_events)

    for block_idx in range(first_block, num_blk):
        block_start, block_end = current_block_range(block_idx, block_len)
        x_view = x[:, :block_end].clone()
        frozen_mask = build_generation_prompt_mask(prompt_mask, block_start, block_end)

        for t in reversed(range(1, T + 1)):
            logits, _ = model.forward_sample(x_view)
            sampled = sample_tokens_from_logits(logits.float())

            masked_now = x_view == mask_token_id
            total_masked_now = masked_now.sum(dim=1)

            if t == 1:
                total_masked_next = torch.zeros_like(total_masked_now)
            else:
                total_masked_next = torch.floor(
                    (1.0 - survival_prob_scalar(t - 1)) * (~frozen_mask).sum(dim=1).float()
                ).long()
            num_to_fill = (total_masked_now - total_masked_next).clamp_min(0)

            if masked_now.any():
                probs = F.softmax(logits.float(), dim=-1)
                sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
                sampled_conf = sampled_conf.masked_fill(~masked_now, float("-inf"))

                for b in range(x_view.size(0)):
                    k = int(num_to_fill[b].item())
                    if k <= 0:
                        continue
                    fill_idx = torch.topk(
                        sampled_conf[b], k=k, dim=0, largest=True
                    ).indices
                    x_view[b, fill_idx] = sampled[b, fill_idx]

            do_correction = (
                (t > 1 and t % correct_every == 0)
                or (correct_at_end and t == 1)
            )
            if do_correction:
                for _ in range(corrector_steps):
                    x_view = corrective_sweep(model, x_view, frozen_mask, mask_token_id)

        if (x_view[:, block_start:block_end] == mask_token_id).any():
            logits, _ = model.forward_sample(x_view)
            final_tokens = torch.argmax(logits, dim=-1)
            x_view = torch.where(x_view == mask_token_id, final_tokens, x_view)

        x[:, :block_end] = x_view

    return decode(x[0].tolist())
