"""
model_MDLM.py — Masked Diffusion Language Model (MDLM).

Carries over unmasked tokens with no remasking during reverse diffusion.
At each step, the highest-confidence masked tokens are progressively
unmasked; tokens that have already been revealed are never re-masked.
"""

import torch
from torch.nn import functional as F

from backbone import DiffusionBackbone


class Model(DiffusionBackbone):
    pass


# ------------------------------------------------------------------
# Training interface
# ------------------------------------------------------------------

def get_batch(split, cfg):
    """Standard masked batch — identical to model_remasked."""
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
    """Denoising CE on masked positions."""
    xt, x0, mask = batch
    _, loss = model(xt, targets=x0, mask=mask)
    return loss


def compute_eval_loss(model, xt, x0, mask):
    """Standard denoising CE (called by the shared eval loop)."""
    _, loss = model(xt, targets=x0, mask=mask)
    return loss


# ------------------------------------------------------------------
# Reverse Diffusion Sampling (progressive unmasking, no remasking)
# ------------------------------------------------------------------

def _progressive_unmask_step(x, prompt_mask, sampled, sampled_conf, t, *,
                             mask_token_id, survival_prob_scalar):
    """Unmask the highest-confidence masked tokens for the next timestep."""
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
def generate_from(model, x, prompt_mask, *,
                  T, block_size, vocab_size, mask_token_id,
                  survival_prob_scalar, decode, **kwargs):
    """
    Progressive-unmasking sampler (MDLM style).

    Already-revealed tokens are carried forward unchanged; at each
    reverse step the highest-confidence masked positions are unmasked.
    """
    model.eval()

    for t in reversed(range(1, T + 1)):
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, vocab_size), 1
        ).view_as(x)
        sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        x = _progressive_unmask_step(
            x, prompt_mask, sampled, sampled_conf, t,
            mask_token_id=mask_token_id,
            survival_prob_scalar=survival_prob_scalar,
        )

    # Final cleanup: if any tokens are still masked, greedily decode them
    if (x == mask_token_id).any():
        logits, _ = model(x)
        final_tokens = torch.argmax(logits, dim=-1)
        x = torch.where(x == mask_token_id, final_tokens, x)

    return decode(x[0].tolist())
