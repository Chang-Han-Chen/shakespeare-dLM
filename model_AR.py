"""
model_AR.py — Autoregressive (causal) language model baseline.

Standard next-token prediction GPT, using the same architecture as the
diffusion variants (RoPE, QK-norm, ReGLU MLP, RMSNorm) so parameter
counts are directly comparable.
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
            q, k, v,
            is_causal=True,
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
                 block_size, dropout, **kwargs):
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

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, L)  input token ids
            targets: (B, L)  shifted targets (idx[:, 1:] predicts targets[:, 1:])
                     When provided, returns CE loss over all positions.

        Returns:
            logits: (B, L, vocab_size)
            loss:   scalar or None
        """
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
            # Standard next-token CE: logits[:, :-1] predicts targets[:, 1:]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size),
                targets[:, 1:].reshape(-1),
            )

        return logits, loss


# ------------------------------------------------------------------
# Training interface
# ------------------------------------------------------------------

def get_batch(split, cfg):
    """
    Standard AR batch: draw random contiguous chunks from data.
    Returns (x, x, None) — input and target are the same sequence,
    loss is computed via shifted prediction inside compute_loss.
    """
    data_split = cfg["train_data"] if split == "train" else cfg["val_data"]
    B, L = cfg["batch_size"], cfg["block_size"]
    idx = torch.randint(len(data_split) - L, (B,))
    x = torch.stack([data_split[i : i + L] for i in idx])
    dev = cfg["device"]
    return x.to(dev), x.to(dev), None


def get_eval_batch(split, cfg):
    """AR validation should use the same clean next-token setup as training."""
    return get_batch(split, cfg)


def compute_loss(model, batch, cfg):
    """Next-token prediction CE."""
    x, targets, _ = batch
    _, loss = model(x, targets=targets)
    return loss


def compute_eval_loss(model, xt, x0, mask):
    """
    Called by the shared eval loop with (xt, x0, mask) from get_eval_batch.
    For AR, xt == x0 and mask is ignored. We just compute standard AR CE.
    """
    _, loss = model(xt, targets=x0)
    return loss


# ------------------------------------------------------------------
# Generation (autoregressive sampling)
# ------------------------------------------------------------------

@torch.no_grad()
def generate_from(model, x, prompt_mask, *,
                  T, block_size, vocab_size, mask_token_id,
                  survival_prob_scalar, decode, **kwargs):
    """
    Autoregressive generation.  Fills positions left-to-right starting
    from the end of the prompt (first False in prompt_mask).

    Accepts the same keyword signature as the diffusion models so
    train.py can call it uniformly — T, survival_prob_scalar, and
    mask_token_id are unused.
    """
    model.eval()

    # Find where the prompt ends (first non-prompt position)
    prompt_len = int(prompt_mask[0].sum().item())

    # Autoregressively sample one token at a time
    for i in range(prompt_len, block_size):
        logits, _ = model(x[:, :i])
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1)  # (1, 1)
        x[:, i] = next_token[:, 0]

    return decode(x[0].tolist())
