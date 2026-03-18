"""
backbone.py — Shared transformer backbone for all diffusion models.

Supports both standard (single-stream) and BD3-style (dual-stream) operation:

  1. Standard mode (block_len == block_size): bidirectional attention, no mask.
     Used by the original remasked / MDLM / edit_* models.
  2. Block-diffusion mode (block_len < block_size):
     - forward_train():  dual-stream over x_t ⊕ x_0 with the BD3 attention mask
     - forward_sample(): one-stream block-causal attention for blockwise sampling

The objective/sampler still lives in the per-model files.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from block_utils import make_bd3_train_mask, make_block_causal_mask, validate_block_len


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

    def forward(self, x, cos_sin, attn_mask=None):
        B, Tseq, _ = x.size()

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
            attn_mask=attn_mask,
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

    def forward(self, x, cos_sin, attn_mask=None):
        x = x + self.attn(norm(x), cos_sin, attn_mask=attn_mask)
        x = x + self.mlp(norm(x))
        return x


# ------------------------------------------------------------------
# Backbone
# ------------------------------------------------------------------


class DiffusionBackbone(nn.Module):
    """
    Shared backbone for all diffusion models.

    When block_len == block_size (one block = whole sequence), this is
    identical to the original bidirectional transformer: no attention
    mask is applied, and dual-stream buffers are still registered but
    never used by the standard models.

    When block_len < block_size, the BD3 block-diffusion masks are
    active and forward_train / forward_sample use them.
    """

    def __init__(
        self,
        vocab_size,
        n_embd,
        n_head,
        n_layer,
        head_dim,
        block_size,
        dropout,
        block_len=None,
    ):
        super().__init__()
        if block_len is None:
            block_len = block_size
        validate_block_len(block_size, block_len)

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.block_len = block_len
        self._is_single_block = (block_len == block_size)

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.emb_dropout = nn.Dropout(dropout)

        # Rotary embeddings for positions 0..block_size-1
        cos, sin = self._precompute_rotary_embeddings(block_size, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Dual-stream training uses the same 0..L-1 positions on both halves.
        self.register_buffer(
            "cos_dual",
            torch.cat([cos, cos], dim=1),
            persistent=False,
        )
        self.register_buffer(
            "sin_dual",
            torch.cat([sin, sin], dim=1),
            persistent=False,
        )

        # Precompute attention masks.
        # When _is_single_block, the masks degenerate to all-True (train)
        # and all-True (sample), but we skip them entirely for performance.
        if not self._is_single_block:
            train_mask = make_bd3_train_mask(block_size, block_len)
            sample_mask = make_block_causal_mask(block_size, block_len)
            self.register_buffer("train_attn_mask", train_mask, persistent=False)
            self.register_buffer("sample_attn_mask", sample_mask, persistent=False)
        else:
            self.register_buffer("train_attn_mask", None, persistent=False)
            self.register_buffer("sample_attn_mask", None, persistent=False)

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

    def _select_rotary(self, Tseq: int, dual_stream: bool):
        if dual_stream:
            if Tseq % 2 != 0:
                raise ValueError(f"dual_stream expects even length, got {Tseq}")
            return self.cos_dual[:, :Tseq], self.sin_dual[:, :Tseq]
        return self.cos[:, :Tseq], self.sin[:, :Tseq]

    def _select_attn_mask(self, Tseq: int, dual_stream: bool):
        if self._is_single_block:
            return None  # full bidirectional — fastest path
        if dual_stream:
            return self.train_attn_mask[:, :, :Tseq, :Tseq]
        return self.sample_attn_mask[:, :, :Tseq, :Tseq]

    def _forward_core(self, idx, *, dual_stream=False):
        """Run the transformer, return logits. No loss computation."""
        B, Tseq = idx.size()

        x = self.token_emb(idx)
        x = self.emb_dropout(x)
        x = norm(x)

        cos_sin = self._select_rotary(Tseq, dual_stream)
        attn_mask = self._select_attn_mask(Tseq, dual_stream)

        for block in self.blocks:
            x = block(x, cos_sin, attn_mask=attn_mask)

        x = norm(x)
        logits = self.lm_head(x)

        if dual_stream:
            logits = logits[:, : Tseq // 2]

        return logits

    # ----- Public API -----

    def forward(self, idx, targets=None, mask=None, supervise_mask=None):
        """
        Backward-compatible forward for standard (non-block) models.

        Accepts either `mask` or `supervise_mask` as the supervision boolean
        tensor (model_remasked uses `mask`, edit models use `supervise_mask`).
        """
        logits = self._forward_core(idx, dual_stream=False)

        loss = None
        if targets is not None:
            sup = mask if mask is not None else supervise_mask
            if sup is None:
                raise ValueError(
                    "mask or supervise_mask must be provided when targets is provided"
                )
            B, T2, V = logits.shape
            per_token_loss = F.cross_entropy(
                logits.reshape(B * T2, V),
                targets.reshape(B * T2),
                reduction="none",
            ).reshape(B, T2)
            sup_f = sup.float()
            loss = (per_token_loss * sup_f).sum() / sup_f.sum().clamp_min(1.0)

        return logits, loss

    def forward_train(self, xt, x0, targets=None, supervise_mask=None):
        """Dual-stream forward for block-diffusion training.

        When block_len == block_size (single block), the BD3 mask keeps
        x_t and x_0 fully isolated, so we skip the concatenation entirely
        and run single-stream on x_t alone for correctness and speed.
        """
        if self._is_single_block:
            # Single block: x_t cannot see any x_0 (no preceding blocks).
            # Equivalent to standard bidirectional forward on x_t only.
            logits = self._forward_core(xt, dual_stream=False)
        else:
            x_input = torch.cat([xt, x0], dim=1)
            logits = self._forward_core(x_input, dual_stream=True)

        loss = None
        if targets is not None:
            if supervise_mask is None:
                raise ValueError(
                    "supervise_mask must be provided when targets is provided"
                )
            B, T2, V = logits.shape
            per_token_loss = F.cross_entropy(
                logits.reshape(B * T2, V),
                targets.reshape(B * T2),
                reduction="none",
            ).reshape(B, T2)
            sup_f = supervise_mask.float()
            loss = (per_token_loss * sup_f).sum() / sup_f.sum().clamp_min(1.0)

        return logits, loss

    def forward_sample(self, idx, targets=None, supervise_mask=None):
        """Single-stream forward with block-causal mask for generation."""
        logits = self._forward_core(idx, dual_stream=False)

        loss = None
        if targets is not None:
            if supervise_mask is None:
                raise ValueError(
                    "supervise_mask must be provided when targets is provided"
                )
            B, T2, V = logits.shape
            per_token_loss = F.cross_entropy(
                logits.reshape(B * T2, V),
                targets.reshape(B * T2),
                reduction="none",
            ).reshape(B, T2)
            sup_f = supervise_mask.float()
            loss = (per_token_loss * sup_f).sum() / sup_f.sum().clamp_min(1.0)

        return logits, loss
