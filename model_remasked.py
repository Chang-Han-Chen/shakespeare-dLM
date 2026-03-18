"""
model_remasked.py — Simple remasking diffusion language model.

Reverse diffusion re-masks low-confidence tokens at each step based on
the noise schedule, so the model iteratively refines its predictions.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=3)


# ------------------------------------------------------------------
# Layers
# ------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.dropout = dropout

        self.c_q = nn.Linear(n_embd, n_embd, bias=False)
        self.c_k = nn.Linear(n_embd, n_embd, bias=False)
        self.c_v = nn.Linear(n_embd, n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, cos_sin):
        B, Tseq, C = x.size()

        q = self.c_q(x).view(B, Tseq, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, Tseq, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, Tseq, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        q = norm(q)
        k = norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, Tseq, -1)
        y = self.c_proj(y)
        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_proj(F.relu(self.c_fc(x)).square())
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, head_dim, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_embd, n_head, head_dim, dropout)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, head_dim,
                 block_size, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.emb_dropout = nn.Dropout(dropout)

        self.rotary_seq_len = block_size * 2
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, head_dim, dropout) for _ in range(n_layer)]
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _precompute_rotary_embeddings(seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx, targets=None, mask=None):
        B, Tseq = idx.size()

        x = self.token_emb(idx)
        x = self.emb_dropout(x)
        x = norm(x)

        cos_sin = (self.cos[:, :Tseq], self.sin[:, :Tseq])

        for block in self.blocks:
            x = block(x, cos_sin)

        x = norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if mask is None:
                raise ValueError("mask must be provided when targets is provided")

            per_token_loss = F.cross_entropy(
                logits.reshape(B * Tseq, self.vocab_size),
                targets.reshape(B * Tseq),
                reduction="none",
            ).reshape(B, Tseq)

            mask_f = mask.float()
            loss = (per_token_loss * mask_f).sum() / mask_f.sum().clamp_min(1.0)

        return logits, loss


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
                  survival_prob_scalar, decode):
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
