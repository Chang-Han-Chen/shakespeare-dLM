"""
model_edit_two_pass.py — Two-pass self-correcting diffusion language model.

Pass 1 (denoising): standard masked-token prediction.
Pass 2 (correction): the model re-reads its own draft and fixes mistakes
on visible (non-mask) positions.

During generation the two passes are interleaved: an MDLM-style
progressive-unmasking step, plus occasional corrective sweeps over the
already-visible tokens.
"""

import torch
from torch.nn import functional as F

from backbone import DiffusionBackbone
from block_utils import masked_ce_loss, sample_tokens_from_logits


class Model(DiffusionBackbone):
    pass


# ------------------------------------------------------------------
# Training interface
# ------------------------------------------------------------------

def get_batch(split, cfg):
    """Standard masked batch (same as remasked / MDLM)."""
    data_split = cfg["train_data"] if split == "train" else cfg["val_data"]
    B, L = cfg["batch_size"], cfg["block_size"]
    idx = torch.randint(len(data_split) - L, (B,))
    x0 = torch.stack([data_split[i : i + L] for i in idx])

    t = torch.randint(1, cfg["T"] + 1, (B,))
    a_t = cfg["survival_prob_tensor"](t).unsqueeze(1)

    token_mask = torch.rand(B, L) > a_t
    xt = x0.clone()
    xt[token_mask] = cfg["mask_token_id"]

    dev = cfg["device"]
    return xt.to(dev), x0.to(dev), token_mask.to(dev)


def compute_loss(model, batch, cfg):
    """Two-pass loss: denoising CE + correction CE on model's own draft."""
    xt, x0, mask = batch
    lambda_corr = cfg.get("lambda_corr", 1.0)

    # Pass 1: denoising
    logits_1, _ = model(xt)
    loss_denoise = masked_ce_loss(logits_1, x0, mask)

    # Build stop-gradient draft
    with torch.no_grad():
        sampled = sample_tokens_from_logits(logits_1.float())
        draft = xt.clone()
        draft[mask] = sampled[mask]
        corr_mask = draft != x0

    # Pass 2: correction on draft
    if corr_mask.any():
        logits_2, _ = model(draft)
        loss_corr = masked_ce_loss(logits_2, x0, corr_mask)
    else:
        loss_corr = torch.zeros((), device=xt.device)

    return loss_denoise + lambda_corr * loss_corr


def compute_eval_loss(model, xt, x0, mask):
    """Standard denoising CE (uses supervise_mask kwarg)."""
    _, loss = model(xt, targets=x0, supervise_mask=mask)
    return loss


# ------------------------------------------------------------------
# Reverse Diffusion Sampling
# Outer loop: MDLM progressive unmasking (carry over revealed tokens)
# Occasional inner loop: corrective sweep over visible non-prompt tokens
# ------------------------------------------------------------------

@torch.no_grad()
def corrective_sweep(model, x, prompt_mask, mask_token_id):
    """
    Fill masked positions with a draft, then correct visible non-prompt
    positions in a second forward pass.
    """
    logits_fill, _ = model(x)
    sampled_fill = sample_tokens_from_logits(logits_fill.float())

    draft = x.clone()
    masked_now = (draft == mask_token_id)
    draft[masked_now] = sampled_fill[masked_now]

    logits_corr, _ = model(draft)
    corrected = torch.argmax(logits_corr, dim=-1)

    visible_non_prompt = (~prompt_mask) & (~masked_now)
    out = x.clone()
    out[visible_non_prompt] = corrected[visible_non_prompt]
    return out


@torch.no_grad()
def generate_from(model, x, prompt_mask, *,
                  T, block_size, vocab_size, mask_token_id,
                  survival_prob_scalar, decode,
                  corrector_steps=1,
                  num_correction_events=5,
                  correct_at_end=True,
                  **kwargs):
    """
    Two-pass sampler.

    Outer loop progressively unmasks tokens (MDLM-style carry-over).
    Corrective sweeps happen every `correct_every` outer steps.
    """
    model.eval()

    num_correction_events = max(1, int(num_correction_events))
    correct_every = max(1, T // num_correction_events)

    for t in reversed(range(1, T + 1)):
        # --- Pass 1: progressive unmasking ---
        logits, _ = model(x)
        sampled = sample_tokens_from_logits(logits.float())

        masked_now = (x == mask_token_id)
        total_masked_now = masked_now.sum(dim=1)

        if t == 1:
            total_masked_next = torch.zeros_like(total_masked_now)
        else:
            total_masked_next = torch.floor(
                (1.0 - survival_prob_scalar(t - 1)) * (~prompt_mask).sum(dim=1).float()
            ).long()
        num_to_fill = (total_masked_now - total_masked_next).clamp_min(0)

        if masked_now.any():
            probs = F.softmax(logits.float(), dim=-1)
            sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            sampled_conf = sampled_conf.masked_fill(~masked_now, float("-inf"))

            for b in range(x.size(0)):
                k = int(num_to_fill[b].item())
                if k <= 0:
                    continue
                fill_idx = torch.topk(
                    sampled_conf[b], k=k, dim=0, largest=True
                ).indices
                x[b, fill_idx] = sampled[b, fill_idx]

        # --- Pass 2: occasional corrective sweeps ---
        should_correct = (t % correct_every == 0)
        if correct_at_end and t == 1:
            should_correct = True

        if should_correct:
            for _ in range(corrector_steps):
                x = corrective_sweep(model, x, prompt_mask, mask_token_id)

    # Final fill for any remaining masked positions
    logits, _ = model(x)
    final_tokens = torch.argmax(logits, dim=-1)
    x = torch.where(x == mask_token_id, final_tokens, x)

    return decode(x[0].tolist())
