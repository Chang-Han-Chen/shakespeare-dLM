"""
model_remasked.py — Simple remasking diffusion language model.

Reverse diffusion re-masks low-confidence tokens at each step based on
the noise schedule, so the model iteratively refines its predictions.
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
    """Standard masked batch — mask tokens according to survival_prob."""
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
# Reverse Diffusion Sampling
# ------------------------------------------------------------------

@torch.no_grad()
def generate_from(model, x, prompt_mask, *,
                  T, block_size, vocab_size, mask_token_id,
                  survival_prob_scalar, decode, **kwargs):
    """
    Iterative remasking sampler.

    At each reverse step the model predicts all tokens, then re-masks
    the least-confident ones according to the noise schedule for the
    next (lower) timestep.
    """
    model.eval()

    for t in reversed(range(1, T + 1)):
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, vocab_size), 1
        ).view(1, block_size)
        sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        x = torch.where(prompt_mask, x, sampled)

        if t > 1:
            gen_positions = (~prompt_mask).sum().item()
            next_mask_ratio = 1.0 - survival_prob_scalar(t - 1)
            k = int(next_mask_ratio * gen_positions)
            if k > 0:
                conf = sampled_conf.masked_fill(prompt_mask, float("inf"))
                low_conf_idx = torch.topk(
                    conf, k=k, dim=1, largest=False
                ).indices
                x.scatter_(1, low_conf_idx, mask_token_id)

    logits, _ = model(x)
    final_tokens = torch.argmax(logits, dim=-1)
    x = torch.where(prompt_mask, x, final_tokens)

    return decode(x[0].tolist())
