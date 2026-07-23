"""Minimal Llama-2-style dual-stream block-diffusion transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import BLOCK_LEN, SEQ_LEN, VOCAB_SIZE, ModelSpec


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def make_dual_stream_mask(
    seq_len: int = SEQ_LEN,
    block_len: int = BLOCK_LEN,
) -> torch.Tensor:
    """Allowed attention for concatenated [noisy | clean] streams."""
    if seq_len % block_len:
        raise ValueError("block_len must divide seq_len")
    position = torch.arange(2 * seq_len)
    query = position[:, None]
    key = position[None, :]
    query_clean = query >= seq_len
    key_clean = key >= seq_len
    query_block = torch.where(query_clean, query - seq_len, query) // block_len
    key_block = torch.where(key_clean, key - seq_len, key) // block_len

    noisy_within_block = (~query_clean) & (~key_clean) & (query_block == key_block)
    noisy_to_clean_prefix = (~query_clean) & key_clean & (key_block < query_block)
    clean_block_causal = query_clean & key_clean & (key_block <= query_block)
    return (noisy_within_block | noisy_to_clean_prefix | clean_block_causal)[None, None]


def rotary_frequencies(seq_len: int, head_dim: int, base: float = 10_000.0):
    inverse = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len).float(), inverse)
    return angles.cos()[None, None], angles.sin()[None, None]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin, attention_mask=None, is_causal=False):
        batch, length, width = x.shape
        shape = (batch, length, self.n_head, self.head_dim)
        q = apply_rope(self.q_proj(x).view(shape).transpose(1, 2), cos, sin)
        k = apply_rope(self.k_proj(x).view(shape).transpose(1, 2), cos, sin)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        return self.o_proj(output.transpose(1, 2).contiguous().view(batch, length, width))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.attention_norm = RMSNorm(spec.d_model)
        self.attention = Attention(spec.d_model, spec.n_head)
        self.ffn_norm = RMSNorm(spec.d_model)
        self.feed_forward = SwiGLU(spec.d_model, spec.d_ff)

    def forward(self, x, cos, sin, attention_mask=None, is_causal=False):
        x = x + self.attention(
            self.attention_norm(x),
            cos,
            sin,
            attention_mask,
            is_causal,
        )
        return x + self.feed_forward(self.ffn_norm(x))


class BlockDiffusionTransformer(nn.Module):
    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec
        self.token_embedding = nn.Embedding(VOCAB_SIZE, spec.d_model)
        self.layers = nn.ModuleList([TransformerBlock(spec) for _ in range(spec.n_layer)])
        self.norm = RMSNorm(spec.d_model)
        self.lm_head = nn.Linear(spec.d_model, VOCAB_SIZE, bias=False)

        # Corresponding positions in the noisy and clean streams share RoPE.
        cos, sin = rotary_frequencies(SEQ_LEN, spec.head_dim)
        self.register_buffer("cos", torch.cat((cos, cos), dim=2), persistent=False)
        self.register_buffer("sin", torch.cat((sin, sin), dim=2), persistent=False)
        self.register_buffer("attention_mask", make_dual_stream_mask(), persistent=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, noisy: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        if noisy.shape != clean.shape or noisy.shape[1] != SEQ_LEN:
            raise ValueError(f"Expected matching [batch, {SEQ_LEN}] streams")
        x = self.token_embedding(torch.cat((noisy, clean), dim=1))
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, self.attention_mask)
        return self.lm_head(self.norm(x[:, :SEQ_LEN]))

    def forward_ar(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != SEQ_LEN:
            raise ValueError(f"Expected [batch, {SEQ_LEN}] tokens")
        x = self.token_embedding(tokens)
        cos = self.cos[:, :, :SEQ_LEN]
        sin = self.sin[:, :, :SEQ_LEN]
        for layer in self.layers:
            x = layer(x, cos, sin, is_causal=True)
        return self.lm_head(self.norm(x))

    def counted_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name != "token_embedding.weight"
        )
