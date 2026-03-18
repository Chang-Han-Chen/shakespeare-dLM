"""
model_edit_one_pass.py — Self-correcting diffusion language model (one pass).

Trained with an additional corruption step: a fraction of *visible* tokens
are replaced with wrong tokens, and the model is supervised on both the
masked positions and the corrupted positions (via ``supervise_mask``).

During reverse diffusion, all non-prompt positions are rewritten each step,
then the least-confident ones are re-masked for later revision.
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
    """Masked batch + visible-token corruption (corrupt_prob)."""
    data_split = cfg["train_data"] if split == "train" else cfg["val_data"]
    B, L = cfg["batch_size"], cfg["block_size"]
    idx = torch.randint(len(data_split) - L, (B,))
    x0 = torch.stack([data_split[i : i + L] for i in idx])

    t = torch.randint(1, cfg["T"] + 1, (B,))
    a_t = cfg["survival_prob_tensor"](t).unsqueeze(1)

    token_mask = torch.rand(B, L) > a_t
    xt = x0.clone()
    xt[token_mask] = cfg["mask_token_id"]

    # Corrupt a fraction of *visible* positions with wrong tokens
    corrupt_prob = cfg.get("corrupt_prob", 0.15)
    visible = ~token_mask
    edit_mask = visible & (torch.rand(B, L) < corrupt_prob)
    if edit_mask.any():
        rand_offset = torch.randint(1, cfg["vocab_size"], x0.shape)
        wrong_tokens = (x0 + rand_offset) % cfg["vocab_size"]
        xt[edit_mask] = wrong_tokens[edit_mask]

    supervise_mask = token_mask | edit_mask

    dev = cfg["device"]
    return xt.to(dev), x0.to(dev), supervise_mask.to(dev)


def compute_loss(model, batch, cfg):
    """CE on masked + corrupted positions."""
    xt, x0, supervise_mask = batch
    _, loss = model(xt, targets=x0, supervise_mask=supervise_mask)
    return loss


def compute_eval_loss(model, xt, x0, mask):
    """Standard denoising CE (uses supervise_mask kwarg)."""
    _, loss = model(xt, targets=x0, supervise_mask=mask)
    return loss


# ------------------------------------------------------------------
# Reverse Diffusion Sampling (rewrite all + confidence remasking)
# ------------------------------------------------------------------

@torch.no_grad()
def generate_from(model, x, prompt_mask, *,
                  T, block_size, vocab_size, mask_token_id,
                  survival_prob_scalar, decode, **kwargs):
    """
    Self-correcting sampler.

    Every non-prompt position is rewritten each step, then the
    least-confident positions are re-masked so the model can revise
    them in later steps.
    """
    model.eval()

    for t in reversed(range(1, T + 1)):
        logits, _ = model(x)
        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, vocab_size), 1
        ).view_as(x)
        sampled_conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        # Rewrite all non-prompt positions
        x = torch.where(prompt_mask, x, sampled)

        # Remask least-confident non-prompt positions
        if t > 1:
            total_gen = (~prompt_mask).sum(dim=1)
            target_masked_next = torch.floor(
                (1.0 - survival_prob_scalar(t - 1)) * total_gen.float()
            ).long().clamp_min(0)

            for b in range(x.size(0)):
                k = int(target_masked_next[b].item())
                if k <= 0:
                    continue
                conf_b = sampled_conf[b].masked_fill(prompt_mask[b], float("inf"))
                low_conf_idx = torch.topk(
                    conf_b, k=k, dim=0, largest=False
                ).indices
                x[b, low_conf_idx] = mask_token_id

    # Final greedy pass
    logits, _ = model(x)
    final_tokens = torch.argmax(logits, dim=-1)
    x = torch.where(prompt_mask, x, final_tokens)

    return decode(x[0].tolist())
