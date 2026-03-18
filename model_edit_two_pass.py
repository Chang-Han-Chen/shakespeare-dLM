"""
model_edit_two_pass.py — Two-pass self-correcting diffusion language model.

Pass 1 (denoising): standard masked-token prediction.
Pass 2 (correction): the model re-reads its own draft and fixes mistakes
on visible (non-mask) positions.

During generation the two passes are interleaved: an MDLM-style
progressive-unmasking step, plus occasional corrective sweeps over the
already-visible tokens.
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


def masked_ce_loss(logits, targets, supervise_mask):
    """Cross-entropy averaged over supervised positions only."""
    B, Tseq, V = logits.shape
    per_token_loss = F.cross_entropy(
        logits.reshape(B * Tseq, V),
        targets.reshape(B * Tseq),
        reduction="none",
    ).reshape(B, Tseq)
    supervise_f = supervise_mask.float()
    denom = supervise_f.sum().clamp_min(1.0)
    return (per_token_loss * supervise_f).sum() / denom


@torch.no_grad()
def sample_tokens_from_logits(logits):
    """Multinomial sample from logits; returns token ids shaped (B, L)."""
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(
        probs.view(-1, probs.size(-1)), 1
    ).view(logits.size(0), logits.size(1))


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

    def forward(self, idx, targets=None, supervise_mask=None):
        """
        Parameters
        ----------
        idx            : (B, L) token ids
        targets        : (B, L) ground-truth token ids
        supervise_mask : (B, L) bool — positions the loss covers
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
            if supervise_mask is None:
                raise ValueError(
                    "supervise_mask must be provided when targets is provided"
                )
            loss = masked_ce_loss(logits, targets, supervise_mask)

        return logits, loss


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
                  correct_at_end=True):
    """
    Two-pass sampler.

    Outer loop progressively unmasks tokens (MDLM-style carry-over).
    Corrective sweeps happen only every `correct_every = max(1, T // num_correction_events)`
    outer steps.

    Example:
        T = 100, num_correction_events = 5
        -> correct_every = 20
        -> correction at t = 80, 60, 40, 20, 1  (if correct_at_end=True)
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
            # Final step: unmask everything — don't rely on schedule
            # (t_min clipping would prevent full unmasking otherwise)
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